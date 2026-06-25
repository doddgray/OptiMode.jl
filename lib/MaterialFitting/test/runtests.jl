using Test
using MaterialFitting
using MaterialDispersion
using MaterialDispersion: ε_fn
using Plots                      # exercises the plotting extension + auto-saved fit plots
using LinearAlgebra

const DATADIR = joinpath(@__DIR__, "data")
const OUTDIR = mktempdir()

# reference fused-silica (Malitson) Sellmeier, used to synthesize exact data
sio2_ref(λ) = sqrt(1 + 0.6961663λ^2/(λ^2 - 0.0684043^2) +
                       0.4079426λ^2/(λ^2 - 0.1162414^2) +
                       0.8974794λ^2/(λ^2 - 9.896161^2))

@testset "MaterialFitting" begin

    @testset "IndexDataset" begin
        λ = collect(0.3:0.05:2.0)
        ds = index_dataset(λ, sio2_ref.(λ); axis="iso", label="SiO₂ synthetic")
        @test length(ds) == length(λ)
        @test ds.λ[1] ≈ 0.3
        @test_throws ArgumentError index_dataset([1.0], [1.4])           # need ≥2 points
        @test_throws ArgumentError index_dataset([1.0, 2.0], [1.4])      # length mismatch
        # unsorted input is sorted
        ds2 = index_dataset([2.0, 0.5, 1.0], [1.4, 1.46, 1.44])
        @test issorted(ds2.λ)
    end

    @testset "RefractiveIndex.INFO YAML parsing + formula evaluation" begin
        entry = MaterialFitting.refractiveindex_entry(joinpath(DATADIR, "SiO2_Malitson.yml"))
        @test length(entry.blocks) == 1
        @test entry.blocks[1].kind == :formula
        @test entry.blocks[1].formula == 1
        @test occursin("Malitson", entry.references)
        ds = refractiveindex_dataset(joinpath(DATADIR, "SiO2_Malitson.yml");
                                     λ_range=(0.25, 2.5), n_points=120)
        @test length(ds) == 120
        @test extrema(ds.λ) == (0.25, 2.5)
        # the evaluated formula must reproduce the closed-form Sellmeier
        @test all(isapprox.(ds.n, sio2_ref.(ds.λ); rtol=1e-10))
        @test ds.n[findmin(abs.(ds.λ .- 1.55))[2]] ≈ sio2_ref(1.55) rtol = 1e-3
    end

    @testset "URL → YAML candidate mapping" begin
        urls = refractiveindex_url_to_yaml_urls("https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson")
        @test any(u -> occursin("main/SiO2/Malitson.yml", u), urls)
        @test refractiveindex_url_to_yaml_urls("https://example.com/x.yml") == ["https://example.com/x.yml"]
    end

    @testset "Sellmeier fit recovers synthetic data" begin
        λ = collect(range(0.25, 2.5; length=60))
        ds = index_dataset(λ, sio2_ref.(λ); label="SiO₂ synthetic")
        fit = fit_sellmeier(ds; n_terms=3, λ_range=(0.25, 2.5))
        @test fit.n_terms == 3
        @test fit.rms_error < 1e-5             # 3-term fit of 3-term data ⇒ near-exact
        @test sellmeier_n(fit, 1.55) ≈ sio2_ref(1.55) rtol = 1e-5
        # a 2-term fit over a narrow range is still good there
        fit2 = fit_sellmeier(ds; n_terms=2, λ_range=(0.4, 1.7))
        @test fit2.max_error < 1e-3
        @test fit2.λ_range == (0.4, 1.7)
    end

    @testset "auto-saved fit comparison plot" begin
        λ = collect(range(0.25, 2.5; length=40))
        ds = index_dataset(λ, sio2_ref.(λ); label="SiO₂")
        fit = fit_sellmeier(ds; n_terms=3, λ_range=(0.3, 2.0), name="SiO2_test", plotdir=OUTDIR)
        files = filter(f -> endswith(f, ".png"), readdir(OUTDIR))
        @test !isempty(files)                  # a plot was saved when the fit ran
        @test any(f -> occursin("SiO2_test", f), files)
    end

    @testset "build isotropic Material" begin
        λ = collect(range(0.3, 2.0; length=40))
        ds = index_dataset(λ, sio2_ref.(λ))
        fit = fit_sellmeier(ds; n_terms=3, λ_range=(0.3, 2.0), name=:SiO₂_fit)
        mat = build_material(fit)
        @test mat isa Material
        ε = ε_fn(mat)(1.31)
        @test sqrt(ε[1, 1]) ≈ sio2_ref(1.31) rtol = 1e-4
        @test ε[1, 1] ≈ ε[3, 3]                # isotropic
    end

    @testset "anisotropic (uniaxial) Material" begin
        λ = collect(range(0.5, 4.0; length=50))
        # synthetic ordinary/extraordinary indices
        no(λ) = sqrt(1 + 2.6734λ^2/(λ^2 - 0.01764) + 1.2290λ^2/(λ^2 - 0.05914) + 12.614λ^2/(λ^2 - 474.6))
        ne(λ) = sqrt(1 + 2.9804λ^2/(λ^2 - 0.02047) + 0.5981λ^2/(λ^2 - 0.0666) + 8.9543λ^2/(λ^2 - 416.08))
        dso = index_dataset(λ, no.(λ); axis="o", label="nₒ")
        dse = index_dataset(λ, ne.(λ); axis="e", label="nₑ")
        fo = fit_sellmeier(dso; n_terms=3, λ_range=(0.5, 4.0))
        fe = fit_sellmeier(dse; n_terms=3, λ_range=(0.5, 4.0))
        mat = build_material(; o=fo, e=fe, name=:LiNbO₃_fit)
        ε = ε_fn(mat)(1.55)
        @test sqrt(ε[1, 1]) ≈ no(1.55) rtol = 1e-3       # ordinary on x,y
        @test sqrt(ε[3, 3]) ≈ ne(1.55) rtol = 1e-3       # extraordinary on z
        @test !isapprox(ε[1, 1], ε[3, 3])                # genuinely anisotropic
    end

    @testset "temperature-dependent Sellmeier" begin
        # synthetic thermo data: n²(λ,T) = n₀²(λ) + 2·n·dn_dT·(T-T₀), dn_dT = 1e-5 /°C
        T₀ = 25.0; dndT = 1.0e-5
        nT(λ, T) = sqrt(sio2_ref(λ)^2 + 2 * sio2_ref(λ) * dndT * (T - T₀))
        λ = collect(range(0.4, 2.0; length=40))
        dsets = [index_dataset(λ, nT.(λ, T); T=T, label="SiO₂") for T in (15.0, 25.0, 35.0, 55.0)]
        tf = fit_thermo_sellmeier(dsets; n_terms=3, λ_range=(0.4, 2.0), T_poly_order=1, name=:SiO₂_T)
        @test tf.rms_error < 1e-4
        @test thermo_n²(tf, 1.55, 45.0) ≈ nT(1.55, 45.0)^2 rtol = 1e-4
        # recovered thermo-optic slope dn/dT near 1.55 μm
        slope = (sqrt(thermo_n²(tf, 1.55, 60.0)) - sqrt(thermo_n²(tf, 1.55, 20.0))) / 40.0
        @test slope ≈ dndT rtol = 0.05
        matT = build_material(tf)
        @test matT isa Material
        fεT = MaterialDispersion.generate_fn(matT, :ε, :λ, :T)
        @test sqrt(fεT(1.55, 45.0)[1, 1]) ≈ nT(1.55, 45.0) rtol = 1e-3
    end

    @testset "save / reload material model" begin
        λ = collect(range(0.3, 2.0; length=40))
        ds = index_dataset(λ, sio2_ref.(λ))
        fit = fit_sellmeier(ds; n_terms=3, λ_range=(0.3, 2.0), name=:SiO₂_io)
        path = joinpath(OUTDIR, "sio2_fit.jld2")
        save_material_model(fit, path)
        @test isfile(path)
        fit2 = load_material_model(path)
        @test fit2 isa SellmeierFit
        @test fit2.A₀ ≈ fit.A₀
        @test fit2.C ≈ fit.C
        @test sellmeier_n(fit2, 1.55) ≈ sellmeier_n(fit, 1.55)
        mat = build_material(fit2)
        @test sqrt(ε_fn(mat)(1.55)[1, 1]) ≈ sio2_ref(1.55) rtol = 1e-4
    end

    # Opt-in live network test against the real RefractiveIndex.INFO database.
    if get(ENV, "MATERIALFITTING_TEST_NETWORK", "false") == "true"
        @testset "live RefractiveIndex.INFO fetch" begin
            ds = refractiveindex_dataset("https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson";
                                         λ_range=(0.3, 2.0), n_points=50)
            @test sqrt(ε_fn(build_material(fit_sellmeier(ds; n_terms=3, λ_range=(0.3, 2.0))))(1.55)[1,1]) ≈ 1.444 rtol = 1e-2
        end
    end
end
