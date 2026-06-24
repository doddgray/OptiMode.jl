# Dispersion and adjoint sensitivities of a thin-film X-cut LiNbO3 **Bragg waveguide**
# whose width is sinusoidally modulated along the propagation axis.
#
# This exercises the 3D periodic-waveguide adjoint (`solve_k_periodic`, see
# `lib/MaxwellEigenmodes/src/grads/period.jl`) on a realistic, dispersive, anisotropic
# structure. For the fundamental Bragg band we compute, at many wavelengths spanning an
# octave below the first-order Bragg resonance (≈1 μm) up to the band edge:
#
#   * the modal effective index  n_eff = kz/ω,
#   * the group index            n_g  = dkz/dω,           and
#   * the group-velocity dispersion GVD = d²kz/dω²,
#
# with n_g and GVD obtained from finite differences of kz(ω) along ω (so they include BOTH
# the geometric/Bragg dispersion and the LiNbO3 material dispersion — ε(λ=1/ω) is rebuilt at
# every stencil point), and then compares the **adjoint** partial derivatives of all three
# quantities with respect to (i) the absolute Bragg period Λ and (ii) the width-modulation
# amplitude A against brute-force **finite differences**.
#
# The period derivative comes directly from the period adjoint; the amplitude derivative is
# the inverse-permittivity adjoint (ei_bar) contracted with the geometry Jacobian ∂ε⁻¹/∂A
# (forward-mode AD of the smooth structure builder) — the standard inverse-design
# composition. Both are exact to the eigensolver tolerance, at a cost independent of the
# number of parameters.
#
# NOTE on band coverage: because LiNbO3 is only weakly birefringent (n_o ≈ n_e), this
# high-contrast TFLN waveguide guides BOTH polarizations — a quasi-TM mode (band 1, n_o) and
# a quasi-TE mode (band 2, n_e) — each of which folds and opens its own first-order Bragg
# band edge near 1 μm. We validate the adjoint sensitivities on both bands, sweeping from an
# octave below the resonance in toward each band edge (the slow-light region). A literal
# isolated *octave-above* branch is not cleanly guided for a realistic high-contrast TFLN
# waveguide (it is multimode and the polarization bands interleave near the edge), so points
# inside the stop-band / where the fixed-ω solve does not converge are simply dropped; the
# folded band structure is plotted separately for context.
#
# Run (needs MaxwellEigenmodes + Plots, ForwardDiff, Roots, FiniteDifferences in the env):
#   julia --project=… examples/tfln_bragg_waveguide_dispersion_adjoint.jl

using MaxwellEigenmodes
using MaxwellEigenmodes.DielectricSmoothing            # Grid, x, y
using LinearAlgebra, StaticArrays, Printf, Statistics
using ForwardDiff, Plots
import Roots
using ChainRulesCore: ZeroTangent
ENV["GKSwstype"] = "100"; gr()

const solver = KrylovKitEigsolve()

# ---------------------------------------------------------------------------------------
# 1. Structure: X-cut LiNbO3 strip with sinusoidally width-modulated Bragg grating
# ---------------------------------------------------------------------------------------
# X-cut LiNbO3 (congruent, Zelmon et al. 1997), λ in μm
no2(λ) = 1 + 2.6734λ^2/(λ^2-0.01764) + 1.2290λ^2/(λ^2-0.05914) + 12.614λ^2/(λ^2-474.6)
ne2(λ) = 1 + 2.9804λ^2/(λ^2-0.02047) + 0.5981λ^2/(λ^2-0.0666)  + 8.9543λ^2/(λ^2-416.08)
nclad2(λ) = 1.444^2                                     # SiO2-ish cladding (dispersionless)
win(s, hw, e) = 0.5*(tanh((s+hw)/e) - tanh((s-hw)/e))   # smooth top-hat

# Inverse permittivity (3,3,Nx,Ny,Nz) of one Bragg period. Sim frame: x = in-plane
# transverse (crystal Z, extraordinary n_e), y = out-of-plane (crystal X, ordinary n_o),
# z = propagation (crystal Y, ordinary n_o). The width w(z) = w0 + A·cos(2πz/Λ) is defined
# in index space, so the structure stretches with the absolute period Λ. Smooth in all of
# (x, y, z, A, λ) → finite differences and ForwardDiff are well behaved.
function tfln_epsi(grid, λ, A; w0=0.9, t=0.3, ex=0.05, ey=0.05)
    Nx,Ny,Nz = size(grid); xs,ys = x(grid), y(grid)
    εxx, εyy, εzz = ne2(λ), no2(λ), no2(λ); εcl = nclad2(λ)
    epsi = zeros(typeof(A*λ), 3,3,Nx,Ny,Nz)
    for iz in 1:Nz
        w = w0 + A*cospi(2*(iz-1)/Nz)
        for (iy,yy) in enumerate(ys), (ix,xx) in enumerate(xs)
            f = win(xx, w/2, ex) * win(yy, t/2, ey)
            epsi[1,1,ix,iy,iz] = inv(εcl + (εxx-εcl)*f)
            epsi[2,2,ix,iy,iz] = inv(εcl + (εyy-εcl)*f)
            epsi[3,3,ix,iy,iz] = inv(εcl + (εzz-εcl)*f)
        end
    end
    epsi
end

# ---------------------------------------------------------------------------------------
# 2. Per-band fixed-ω solver for the Bloch propagation constant kz(ω)
# ---------------------------------------------------------------------------------------
_grid(g, Λ) = Grid(g.Δx, g.Δy, Λ, g.Nx, g.Ny, g.Nz)

# Band 1 (below the gap): warm-started Newton. Band 2+ near the edge has vanishing group
# velocity (Newton diverges), so solve by a robust bracketed root-find on kz.
function kz_band(ω, epsi, Λ, grid, guess, band)
    g = _grid(grid, Λ)
    if band == 1
        ms = ModeSolver(SVector(0.0,0.0,float(guess)), epsi, g; nev=1)
        return solve_k_single(ms, ω, solver; nev=1, eigind=1, k_tol=1e-11, eig_tol=1e-11, max_eigsolves=40)
    else
        kedge = 0.5/Λ
        function ω2(kz)
            ms = ModeSolver(SVector(0.0,0.0,kz), epsi, g; nev=2)
            ev,_ = solve_ω²(ms, solver; nev=2, tol=1e-10); sqrt(abs(ev[2]))
        end
        kz = Roots.find_zero(kz->ω2(kz)-ω, (0.03*kedge, 0.9999*kedge), Roots.Brent(); xatol=1e-11)
        ms = ModeSolver(SVector(0.0,0.0,kz), epsi, g; nev=2)
        _, evs = solve_ω²(ms, solver; nev=2, tol=1e-11); return kz, evs[2]
    end
end

# Adjoint sensitivities of kz at fixed ω: (∂kz/∂Λ, ∂kz/∂A).
function adj_dΛ_dA(ω, λ, A, Λ, grid, kz, ev)
    g = _grid(grid, Λ); epsi = tfln_epsi(g, λ, A)
    _, ei_bar, Λ_bar = MaxwellEigenmodes._solve_k_period_grads(
        ω, epsi, Λ, g, [kz], [ev], [1.0], [ZeroTangent()], solver; nev=1)
    dε_dA = ForwardDiff.derivative(a -> tfln_epsi(g, λ, a), A)
    return Λ_bar, sum(ei_bar .* dε_dA)
end

# ---------------------------------------------------------------------------------------
# 3. Per-wavelength dispersion + adjoint/FD parameter derivatives
# ---------------------------------------------------------------------------------------
const OFFS = (-1, 0, 1)
c1(h) = Dict(-1=>-1/(2h),  0=>0.0,    1=>1/(2h))      # d/dω   (3-point central)
c2(h) = Dict(-1=>1/h^2,    0=>-2/h^2, 1=>1/h^2)       # d²/dω²

function eval_point(ω0, A, Λ, grid, band, guess; h=0.008, δΛ=2e-4, δA=2e-4, nclad=1.444)
    kzs=Dict{Int,Float64}(); dΛ=Dict{Int,Float64}(); dA=Dict{Int,Float64}(); gp=guess
    for j in OFFS
        ωj = ω0+j*h; λj = 1/ωj; epsi = tfln_epsi(grid, λj, A)
        kz, ev = kz_band(ωj, epsi, Λ, grid, gp, band); neff = kz/ωj
        (neff<nclad || neff>2.4 || !isfinite(neff)) && return nothing
        kzs[j]=kz; gp=kz
        dl,dm = adj_dΛ_dA(ωj, λj, A, Λ, grid, kz, ev); dΛ[j]=dl; dA[j]=dm
    end
    C1=c1(h); C2=c2(h)
    neff=kzs[0]/ω0; ng=sum(C1[j]*kzs[j] for j in OFFS); gvd=sum(C2[j]*kzs[j] for j in OFFS)
    adj=(neff_Λ=dΛ[0]/ω0, ng_Λ=sum(C1[j]*dΛ[j] for j in OFFS), gvd_Λ=sum(C2[j]*dΛ[j] for j in OFFS),
         neff_A=dA[0]/ω0, ng_A=sum(C1[j]*dA[j] for j in OFFS), gvd_A=sum(C2[j]*dA[j] for j in OFFS))
    function disp_at(A2,Λ2)
        kk=Dict{Int,Float64}(); gp=kzs[0]
        for j in OFFS
            ωj=ω0+j*h; λj=1/ωj
            kz,_=kz_band(ωj, tfln_epsi(grid,λj,A2), Λ2, grid, gp, band); kk[j]=kz; gp=kz
        end
        (kk[0]/ω0, sum(C1[j]*kk[j] for j in OFFS), sum(C2[j]*kk[j] for j in OFFS))
    end
    nL,gL,vL=disp_at(A,Λ+δΛ); nLm,gLm,vLm=disp_at(A,Λ-δΛ)
    nA,gA,vA=disp_at(A+δA,Λ); nAm,gAm,vAm=disp_at(A-δA,Λ)
    fd=(neff_Λ=(nL-nLm)/(2δΛ), ng_Λ=(gL-gLm)/(2δΛ), gvd_Λ=(vL-vLm)/(2δΛ),
        neff_A=(nA-nAm)/(2δA), ng_A=(gA-gAm)/(2δA), gvd_A=(vA-vAm)/(2δA))
    (λ=1/ω0, neff=neff, ng=ng, gvd=gvd, kz=kzs[0], band=band, adj=adj, fd=fd)
end

# ---------------------------------------------------------------------------------------
# 4. Sweep wavelengths on both guided Bragg bands + folded band structure
# ---------------------------------------------------------------------------------------
# X-cut LiNbO3's two birefringent polarizations are both well guided (n_o > n_e), so this
# high-contrast TFLN waveguide carries two modes — a quasi-TM (band 1, n_o) and a quasi-TE
# (band 2, n_e) — each with its own first-order Bragg band edge near 1 μm. We sweep both,
# from an octave below the resonance in toward their band edges, dropping points where the
# fixed-ω solve does not cleanly converge (e.g. inside the stop-band region).
const Λ0 = 0.27                                        # period → 1st-order Bragg ≈ 1 μm
const A0 = 0.20                                        # width-modulation amplitude (μm)
const grid = Grid(2.2, 1.6, Λ0, 18, 14, 8)

λ1s = [2.0,1.85,1.7,1.55,1.42,1.30,1.20,1.13,1.09,1.06,1.05]   # band 1 (quasi-TM)
λ2s = [1.6,1.5,1.4,1.3,1.2,1.13,1.08,1.05,1.03]                # band 2 (quasi-TE)
function run_band(λs, band, A, Λ, grid, guess0)
    rows = Any[]; guess = guess0       # warm-started continuation (band 1; band 2 brackets)
    for λ in λs
        r = try eval_point(1/λ, A, Λ, grid, band, guess) catch e; @warn "drop" band λ; nothing end
        r === nothing && continue
        guess = r.kz; push!(rows, r)
        @printf("band%d  λ=%.3f  n_eff=%.4f  n_g=%.3f  GVD=%.3f\n", band, r.λ, r.neff, r.ng, r.gvd)
    end
    rows
end
rows = vcat(run_band(λ1s, 1, A0, Λ0, grid, 0.78),
            run_band(λ2s, 2, A0, Λ0, grid, 1.5))

# folded band structure (lowest 4 bands) at a reference material, for context
kedge = 0.5/Λ0; ks = range(0.02kedge, 0.999kedge, 24); epsi_ref = tfln_epsi(grid, 1.0, A0)
bands = map(ks) do kz
    ev,_ = solve_ω²(ModeSolver(SVector(0.0,0.0,kz), epsi_ref, grid; nev=4), solver; nev=4, tol=1e-9)
    sqrt.(abs.(ev[1:4]))
end

# ---------------------------------------------------------------------------------------
# 5. Plots  (band 1 = quasi-TM, blue; band 2 = quasi-TE, red)
# ---------------------------------------------------------------------------------------
b1 = [r for r in rows if r.band==1]; b1 = b1[sortperm([r.λ for r in b1])]
b2 = [r for r in rows if r.band==2]; b2 = b2[sortperm([r.λ for r in b2])]
λ1=[r.λ for r in b1]; λ2=[r.λ for r in b2]
edge = min(minimum(λ1), isempty(λ2) ? Inf : minimum(λ2))
mark(p) = vline!(p,[edge]; color=:gray, ls=:dash, lw=2, label="")

function dp(field, yl)
    p=plot(xlabel="λ (μm)", ylabel=yl, framestyle=:box, legend=:best)
    plot!(p, λ1, [getfield(r,field) for r in b1]; m=:circle, ms=4, lw=1.7, color=:dodgerblue, label="band 1 (q-TM)")
    isempty(b2) || plot!(p, λ2, [getfield(r,field) for r in b2]; m=:diamond, ms=4, lw=1.7, color=:crimson, label="band 2 (q-TE)")
    mark(p); p
end
fig1 = plot(dp(:neff,"n_eff"), dp(:ng,"group index n_g"), dp(:gvd,"GVD ∂²k/∂ω² (μm)");
            layout=(3,1), size=(860,950),
            plot_title="X-cut LiNbO₃ Bragg waveguide dispersion — two polarization bands (Λ=$(Λ0) μm)")
savefig(fig1, joinpath(@__DIR__, "tfln_bragg_dispersion.png"))

function cp(s, ttl)
    p=plot(xlabel="λ (μm)", ylabel=ttl, framestyle=:box, legend=:best)
    plot!(p, λ1, [getfield(r.adj,s) for r in b1]; lw=2, color=:dodgerblue, label="adjoint b1")
    scatter!(p, λ1, [getfield(r.fd,s) for r in b1]; m=:x, ms=5, color=:navy, label="FD b1")
    if !isempty(b2)
        plot!(p, λ2, [getfield(r.adj,s) for r in b2]; lw=2, color=:crimson, label="adjoint b2")
        scatter!(p, λ2, [getfield(r.fd,s) for r in b2]; m=:+, ms=5, color=:darkred, label="FD b2")
    end
    mark(p); p
end
fig2 = plot(cp(:neff_Λ,"∂n_eff/∂Λ"), cp(:ng_Λ,"∂n_g/∂Λ"), cp(:gvd_Λ,"∂GVD/∂Λ"),
            cp(:neff_A,"∂n_eff/∂A"), cp(:ng_A,"∂n_g/∂A"), cp(:gvd_A,"∂GVD/∂A");
            layout=(2,3), size=(1550,850),
            plot_title="Adjoint vs finite-difference sensitivities — top: period Λ, bottom: width-mod amplitude A")
savefig(fig2, joinpath(@__DIR__, "tfln_bragg_derivatives.png"))

pp = plot(xlabel="finite difference", ylabel="adjoint", framestyle=:box, legend=:topleft,
          title="adjoint vs FD parity (both bands, all λ, all 6 sensitivities)")
allv = Float64[]
for (s,lb) in [(:neff_Λ,"∂nₑ/∂Λ"),(:ng_Λ,"∂n_g/∂Λ"),(:gvd_Λ,"∂GVD/∂Λ"),
               (:neff_A,"∂nₑ/∂A"),(:ng_A,"∂n_g/∂A"),(:gvd_A,"∂GVD/∂A")]
    a=[getfield(r.adj,s) for r in rows]; f=[getfield(r.fd,s) for r in rows]
    append!(allv,a); append!(allv,f); scatter!(pp,f,a; ms=5,label=lb)
end
m=maximum(abs,allv); plot!(pp,[-m,m],[-m,m]; color=:black,ls=:dash,label="y=x")
savefig(pp, joinpath(@__DIR__, "tfln_bragg_parity.png"))

pb = plot(xlabel="Bloch wavevector kz·Λ (cycles)", ylabel="frequency ω (μm⁻¹)",
          title="Folded Bragg band structure (lowest 4 bands)", legend=:topleft, framestyle=:box)
for b in 1:4
    plot!(pb, ks./(2kedge), [bands[i][b] for i in eachindex(ks)]; lw=2, label="band $b")
end
savefig(pb, joinpath(@__DIR__, "tfln_bragg_bandstructure.png"))

# agreement summary (over both bands)
println("\nbands solved: $(length(b1)) (q-TM) + $(length(b2)) (q-TE) = $(length(rows)) points")
println("adjoint vs finite-difference (median relative discrepancy):")
for (s,lb) in [(:neff_Λ,"∂n_eff/∂Λ"),(:ng_Λ,"∂n_g/∂Λ"),(:gvd_Λ,"∂GVD/∂Λ"),
               (:neff_A,"∂n_eff/∂A"),(:ng_A,"∂n_g/∂A"),(:gvd_A,"∂GVD/∂A")]
    a=[getfield(r.adj,s) for r in rows]; f=[getfield(r.fd,s) for r in rows]
    @printf("  %-10s  %.2e\n", lb, median(abs.(a.-f)./(abs.(a).+abs.(f).+1e-30)))
end
println("\nwrote tfln_bragg_{dispersion,derivatives,parity,bandstructure}.png")
