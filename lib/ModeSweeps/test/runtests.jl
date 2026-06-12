using Test
using LinearAlgebra
using ModeSweeps
using DielectricSmoothing
using MaxwellEigenmodes
using ModeAnalysis

# Setup script shared by the test batches: an analytic dispersive Gaussian-bump
# waveguide parameterized by frequency ω and bump width wx. Written to disk because
# batches reference their problem definition by (copied) script file.
const SETUP_SRC = """
function _eps_profile(ω, wx)
    grid = Grid(6.0, 4.0, 16, 16)
    wx < 0 && error("invalid bump width wx = \$wx")
    xs, ys = x(grid), y(grid)
    Nx, Ny = size(grid)
    eps = zeros(3, 3, Nx, Ny); deps = zeros(3, 3, Nx, Ny); ddeps = zeros(3, 3, Nx, Ny)
    for (iy, yy) in enumerate(ys), (ix, xx) in enumerate(xs)
        G = exp(-(xx^2 / wx^2 + yy^2 / 0.6^2))
        for a in 1:3
            eps[a, a, ix, iy] = (1.5 + 0.4ω^2) + ((4.0 + 2.0ω^2) - (1.5 + 0.4ω^2)) * G
            deps[a, a, ix, iy] = 0.8ω + (4.0ω - 0.8ω) * G
            ddeps[a, a, ix, iy] = 0.8 + (4.0 - 0.8) * G
        end
    end
    return eps, deps, ddeps, grid
end

function make_problem(p)
    eps, deps, ddeps, grid = _eps_profile(p.ω, p.wx)
    return (; ε⁻¹=sliceinv_3x3(eps), ∂ε_∂ω=deps, ∂²ε_∂ω²=ddeps, grid)
end
"""

const TESTDIR = mktempdir()
const SETUP_FILE = joinpath(TESTDIR, "test_setup.jl")
write(SETUP_FILE, SETUP_SRC)

"poll a batch until it completes (or times out); returns the final status"
function wait_for_batch(batch; timeout=600, poll=5)
    t0 = time()
    st = batch_status(batch; verbose=false)
    while st.pending > 0 && (time() - t0) < timeout
        sleep(poll)
        st = batch_status(batch; verbose=false)
    end
    return st
end

@testset "ModeSweeps" begin
    @testset "param_grid" begin
        ps = param_grid(ω=[0.6, 0.65, 0.7], wx=[1.0, 1.2], T=30.0)
        @test length(ps) == 6
        @test ps[1] == (; ω=0.6, wx=1.0, T=30.0)
        @test ps[2].ω == 0.65            # first keyword varies fastest
        @test ps[2].wx == 1.0
        @test ps[4].wx == 1.2
        @test all(p -> p.T == 30.0, ps)  # scalars broadcast
        @test param_grid(ω=0.6) == [(; ω=0.6)]
        # JSON round trip used for cross-session persistence
        ps2 = ModeSweeps._params_from_json(ModeSweeps._params_to_json(ps))
        @test ps2 == ps
    end

    @testset "deployment artifacts (dry run, slurm script)" begin
        cfg = SlurmConfig(time="0:42:00", partition="micro", mem="8G", cpus_per_task=2,
            max_concurrent=10, julia_flags=["-t", "2"])
        dir = joinpath(TESTDIR, "dry")
        batch = deploy_batch(SETUP_FILE, (; ω=[0.6, 0.65], wx=[1.0, 1.2]);
            name="drytest", dir, nev=2, backend=:none, slurm=cfg)
        @test length(batch) == 4
        @test isfile(joinpath(dir, "batch.toml"))
        @test isfile(joinpath(dir, "params.json"))
        @test isfile(joinpath(dir, "setup.jl"))
        @test isfile(joinpath(dir, "runtask.jl"))
        script = read(joinpath(dir, "job.sbatch"), String)
        @test occursin("#SBATCH --array=1-4%10", script)
        @test occursin("#SBATCH --time=0:42:00", script)
        @test occursin("#SBATCH --partition=micro", script)
        @test occursin("#SBATCH --mem=8G", script)
        @test occursin("#SBATCH --cpus-per-task=2", script)
        @test occursin("runtask.jl", script) && occursin("\$SLURM_ARRAY_TASK_ID", script)
        @test occursin("-t 2", script)
        # nothing ran
        st = batch_status(batch; verbose=false)
        @test st.total == 4 && st.done == 0 && st.failed == 0 && st.pending == 4
        @test isempty(gather_batch(batch; partial=true, save=false))
        # re-loadable in a "new session"
        b2 = load_batch(dir)
        @test b2.params == batch.params
        @test b2.manifest["nev"] == 2
    end

    @testset "local-backend batch: run, status, partial & full gather" begin
        dir = joinpath(TESTDIR, "local")
        batch = deploy_batch(SETUP_FILE, param_grid(ω=[1 / 1.6, 1 / 1.5], wx=[1.0, 1.2]);
            name="localtest", dir, nev=2, save_fields=true, backend=:local,
            solver="KrylovKitEigsolve()", solver_kwargs=(; k_tol=1e-10))
        @test length(batch) == 4

        # status & partial gathering are valid at any time while the batch runs
        st0 = batch_status(batch; verbose=false)
        @test st0.total == 4 && st0.done + st0.failed + st0.pending == 4
        partial0 = gather_batch(batch; partial=true, save=false)
        @test length(partial0) >= 2 * st0.done   # tasks may finish between the two calls
        if st0.pending > 0
            # strict gathering refuses an incomplete batch
            @test_throws ErrorException gather_batch(batch; partial=false)
        end

        st = wait_for_batch(batch)
        @test st.done == 4
        @test st.failed == 0

        rows = gather_batch(batch)   # saves summary.{csv,tsv,json} by default
        @test length(rows) == 8      # 4 tasks × 2 bands
        r1 = first(rows)
        for c in (:task, :ω, :wx, :band, :status, :kmag, :neff, :ng, :gvd, :Aeff,
            :pol_x, :pol_y, :pol_z, :pol_axis)
            @test hasproperty(r1, c)
        end
        @test all(r -> r.status == "done", rows)
        @test all(r -> 1.0 < r.neff < 2.2, rows)
        @test all(r -> isfinite(r.ng) && isfinite(r.gvd) && r.Aeff > 0, rows)
        @test all(r -> r.pol_axis in 1:3, rows)
        @test all(r -> max(r.pol_x, r.pol_y, r.pol_z) <= 1.0 + 1e-9, rows)

        # cross-check one row against a direct in-process solve
        include(SETUP_FILE)
        p = batch.params[1]
        prob = Base.invokelatest(make_problem, p)   # make_problem was just include'd
        kk, evk = solve_k(p.ω, copy(prob.ε⁻¹), prob.grid, KrylovKitEigsolve(); nev=2, k_tol=1e-10)
        row11 = only(filter(r -> r.task == 1 && r.band == 1, rows))
        @test row11.kmag ≈ kk[1] rtol = 1e-6
        @test row11.neff ≈ kk[1] / p.ω rtol = 1e-6
        ng_direct = group_index(kk[1], evk[1], p.ω, prob.ε⁻¹, prob.∂ε_∂ω, prob.grid)
        @test row11.ng ≈ ng_direct rtol = 1e-4

        # tabular round trips
        for ext in (".csv", ".tsv", ".json")
            path = joinpath(dir, "summary" * ext)
            @test isfile(path)
            re = load_summary(path)
            @test length(re) == length(rows)
            @test [r.neff for r in re] ≈ [r.neff for r in rows] rtol = 1e-12
            @test [r.task for r in re] == [r.task for r in rows]
        end
        extra = save_summary(rows, joinpath(dir, "alt"); formats=(:csv,))
        @test load_summary(only(extra)) |> length == 8

        # gather from a freshly loaded batch handle ("new session")
        b2 = load_batch(dir)
        rows2 = gather_batch(b2; save=false)
        @test [r.neff for r in rows2] ≈ [r.neff for r in rows]

        # full field data round trip
        fd = load_fields(b2, 1)
        @test fd.ω ≈ p.ω
        @test length(fd.evecs) == 2 && length(fd.Es) == 2
        @test fd.kmags ≈ kk rtol = 1e-6
        M̂ = HelmholtzMap(fd.kmags[1], copy(prob.ε⁻¹), prob.grid)
        @test norm(M̂ * fd.evecs[1] - p.ω^2 .* fd.evecs[1]) / norm(fd.evecs[1]) < 1e-6
        @test size(fd.Es[1]) == (3, 16, 16)
    end

    @testset "failure handling & partial results" begin
        dir = joinpath(TESTDIR, "failing")
        # second task has an invalid parameter (wx < 0) and must fail cleanly
        batch = deploy_batch(SETUP_FILE, [(; ω=1 / 1.55, wx=1.0), (; ω=1 / 1.55, wx=-1.0)];
            name="failtest", dir, nev=1, backend=:local)
        st = wait_for_batch(batch)
        @test st.done == 1
        @test st.failed == 1
        @test st.failed_tasks == [2]
        @test occursin("invalid bump width", read(ModeSweeps._task_base(batch, 2) * ".failed", String))
        # partial gather: good task yields data, failed task yields a NaN row
        rows = gather_batch(batch; partial=true, save=false)
        @test length(rows) == 2
        good = only(filter(r -> r.status == "done", rows))
        bad = only(filter(r -> r.status == "failed", rows))
        @test isfinite(good.neff) && good.task == 1
        @test isnan(bad.neff) && bad.task == 2
        # strict gathering refuses an incomplete/failed batch
        @test_throws ErrorException gather_batch(batch; partial=false)
    end

    @testset "frequency_sweep sugar" begin
        dir = joinpath(TESTDIR, "fsweep")
        batch = frequency_sweep(SETUP_FILE; ω=[0.6, 0.62, 0.64], wx=[1.0, 1.2],
            name="fsweeptest", dir, nev=1, backend=:none)
        @test length(batch) == 6
        # ω varies fastest: contiguous frequency sweeps per geometry value
        @test [p.ω for p in batch.params[1:3]] == [0.6, 0.62, 0.64]
        @test all(p -> p.wx == 1.0, batch.params[1:3])
        @test all(p -> p.wx == 1.2, batch.params[4:6])
        @test batch.manifest["n_tasks"] == 6
    end

    if get(ENV, "OPTIMODE_TEST_SLURM", "false") == "true"
        @testset "SLURM submission (opt-in)" begin
            @test Sys.which("sbatch") !== nothing
            dir = joinpath(TESTDIR, "slurm")
            batch = deploy_batch(SETUP_FILE, param_grid(ω=[1 / 1.55], wx=[1.0]);
                name="slurmtest", dir, nev=1, backend=:slurm,
                slurm=SlurmConfig(time="0:10:00"))
            @test !isempty(batch.manifest["slurm_jobid"])
            st = wait_for_batch(batch; timeout=1200, poll=15)
            @test st.done == 1
            rows = gather_batch(batch)
            @test length(rows) == 1 && isfinite(only(rows).neff)
        end
    end
end
