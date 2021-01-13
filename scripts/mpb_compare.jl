using Revise, LinearAlgebra, StaticArrays, PyCall, FiniteDifferences, OptiMode, Plots, ChainRules, Zygote, Plots, HDF5, Dates  #, GLMakie, AbstractPlotting,
 #FFTW, BenchmarkTools, LinearMaps, IterativeSolvers, Roots, GeometryPrimitives
# pyplot()
# pygui()
SHM3 = SHermitianCompact{3,Float64,6}
mp = pyimport("meep")
mpb = pyimport("meep.mpb")

"""
################################################################################
#																			   #
#		         Define Utility Functions for Comparison of Julia 			   #
#		                        Solver with MPB 	                 		   #
#																			   #
################################################################################
"""


function write_sweep(sw_name;
                    data_dir="/home/dodd/data/OptiMode/grad_ng_p_rwg_dense/",
                    dt_fmt=dateformat"Y-m-d--H-M-S",
                    extension=".h5",
                    kwargs...)
    timestamp = Dates.format(now(),dt_fmt)
    fname = sw_name * "_" *  timestamp * extension
    @show fpath = data_dir * fname
    h5open(fpath, "cw") do file
        for (data_name,data) in kwargs
            write(file, string(data_name), data)
        end
    end
    return fpath
end

function read_sweep(sw_name;
                    data_dir="/home/dodd/data/OptiMode/grad_ng_p_rwg_dense/",
                    dt_fmt=dateformat"Y-m-d--H-M-S",
                    extension=".h5",
                    sw_keys=["ws","ts","p0","p̄_AD","p̄_FD","p̄_SJ"]
                    )
    # choose most recently timestamped file matching sw_name tag and extension
    fname = sort(  filter(x->(prod(split(x,"_")[begin:end-1])==prod(split(sw_name,"_"))),
                        readdir(data_dir));
                by=file->DateTime(split(file[begin:end-length(extension)],"_")[end],dt_fmt)
            )[end]
    @show fpath = data_dir * fname
    ds_data = h5open(fpath, "r") do file
        @show ds_keys = keys(file)
        ds_data = Dict([k=>read(file,k) for k in sw_keys]...)
    end
    return ds_data
end

# Generic data collection/loading fn use examples:
# fpath_test = write_sweep("wt";ws,ts,p0,p̄_AD,p̄_FD,p̄_SJ)
# ds_test = read_sweep("wt")

function get_Δs_mpb(ms)
    Δx, Δy, Δz = [(Δ == 0.0 ? 1.0 : Δ) for Δ in ms.geometry_lattice.size.__array__()]
end

function get_Ns_mpb(ms)
    Nx, Ny, Nz = [max(NN, 1) for NN in Int.(ms.resolution .* ms.geometry_lattice.size)]
end

function get_xyz_mpb(ms)
    Δx, Δy, Δz = get_Δs_mpb(ms) # [ (Δ==0. ? 1. : Δ) for Δ in ms_size.__array__() ]
    Nx, Ny, Nz = get_Ns_mpb(ms)
    x = ((Δx / Nx) .* (0:(Nx-1))) .- Δx / 2.0
    y = ((Δy / Ny) .* (0:(Ny-1))) .- Δy / 2.0
    z = ((Δz / Nz) .* (0:(Nz-1))) .- Δz / 2.0
    return x, y, z
end

function get_ε⁻¹_mpb(ms)
    x, y, z = get_xyz_mpb(ms)
    Nx = length(x)
    Ny = length(y)
    Nz = length(z)
    ε⁻¹ = Array{Float64,5}(undef, (3, 3, Nx, Ny, Nz))
    for i = 1:Nx, j = 1:Ny, k = 1:Nz
        ε⁻¹[:, :, i, j, k] .= real(ms.get_epsilon_inverse_tensor_point(mp.Vector3(
            x[i],
            y[j],
            z[k],
        )).__array__())
    end
    return ε⁻¹
end

function get_ε_mpb(ms)
    x, y, z = get_xyz_mpb(ms)
    Nx = length(x)
    Ny = length(y)
    Nz = length(z)
    ε = Array{Float64,5}(undef, (3, 3, Nx, Ny, Nz))
    for i = 1:Nx, j = 1:Ny, k = 1:Nz
        ε[:, :, i, j, k] .= real(ms.get_epsilon_inverse_tensor_point(mp.Vector3(
            x[i],
            y[j],
            z[k],
        )).inverse().__array__())
    end
    return ε
end

function ε_slices(
    ms;
    yind = 64,
    zind = 1,
    size = (800, 800),
    xlims = nothing,
    ylims = nothing,
    marker = :dot)
    x, y, z = get_xyz_mpb(ms)
    ε = get_ε_mpb(ms)
    p_diag = plot(
        x,
        ε[1, 1, :, yind, zind],
        label = "11",
        xlabel = "x [μm]",
        ylabel = "εᵢᵢ (diagonal elements)";
        xlims,
        ylims,
        marker,
    )
    plot!(p_diag, x, ε[2, 2, :, yind, zind], label = "22"; xlims, ylims, marker)
    plot!(p_diag, x, ε[3, 3, :, yind, zind], label = "33"; xlims, ylims, marker)
    p_offdiag = plot(
        x,
        ε[1, 2, :, yind, zind],
        label = "12",
        xlabel = "x [μm]",
        ylabel = "εᵢⱼ (off-diag. elements)";
        xlims,
        ylims,
        marker,
    )
    plot!(p_offdiag, x, ε[1, 3, :, yind, zind], label = "13"; xlims, ylims, marker)
    plot!(p_offdiag, x, ε[2, 3, :, yind, zind], label = "23"; xlims, ylims, marker)
    l = @layout [
        a
        b
    ]
    plot(p_diag, p_offdiag, layout = l, size = size)
end

"""
################################################################################
#																			   #
#		         Compare MPB and Julia solutions and gradients of 			   #
#		                Ridge Waveguide Dispersion Model 	           		   #
#																			   #
################################################################################
"""

"""
Default design parameters for ridge waveguide. Both MPB and OptiMode functions
should intake data in this format for convenient apples-to-apples comparison.
"""
p_def = [
    1.55,               #   wavelength              `λ`             [μm]
    1.7,                #   top ridge width         `w_top`         [μm]
    0.7,                #   ridge thickness         `t_core`        [μm]
    π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
    2.4,                #   core index              `n_core`        [1]
    1.4,                #   substrate index         `n_subs`        [1]
]

pω_def = [
    1.45,               #   propagation constant    `kz`            [μm⁻¹]
    1.7,                #   top ridge width         `w_top`         [μm]
    0.7,                #   ridge thickness         `t_core`        [μm]
    π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
    2.4,                #   core index              `n_core`        [1]
    1.4,                #   substrate index         `n_subs`        [1]
    0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
]

"""
MPB functions for ridge waveguide data
"""

function ms_rwg_mpb(
    p = p_def;
    Δx = 6.0,
    Δy = 4.0,
    Δz = 1.0,
    Nx = 128,
    Ny = 128,
    Nz = 1,
    edge_gap = 0.5)
    λ, w, t_core, θ, n_core, n_subs = p
    ω = 1 / λ
    nk = 10
    n_bands = 1
    res = mp.Vector3((Nx / Δx), (Ny / Δy), 1) #mp.Vector3(Int(Nx/Δx),Int(Ny/Δy),1) #16
    n_guess = 0.9 * n_core
    n_min = n_subs
    n_max = n_core
    t_subs = (Δy - t_core - edge_gap) / 2.0
    c_subs_y = -Δy / 2.0 + edge_gap / 2.0 + t_subs / 2.0
    # Set up MPB modesolver, use find-k to solve for one eigenmode `H` with prop. const. `k` at specified temporal freq. ω
    k_pts = mp.interpolate(nk, [mp.Vector3(0.05, 0, 0), mp.Vector3(0.05 * nk, 0, 0)])
    lat = mp.Lattice(size = mp.Vector3(Δx, Δy, 0))
    tanθ = tan(θ)
    tcore_tanθ = t_core * tanθ
    w_bottom = w + 2 * tcore_tanθ
    verts = [
        mp.Vector3(w / 2.0, t_core / 2.0, -5.0),
        mp.Vector3(-w / 2.0, t_core / 2.0, -5.0),
        mp.Vector3(-w_bottom / 2.0, -t_core / 2.0, -5.0),
        mp.Vector3(w_bottom / 2.0, -t_core / 2.0, -5.0),
    ]
    # verts = [ mp.Vector3(-w/2., -t_core/2., -5.),mp.Vector3(w, 2*t_core, -5.), mp.Vector3(w, -t_core/2., -5.)  ]
    core = mp.Prism(
        verts,
        10.0,
        axis = mp.Vector3(0.0, 0.0, 1.0),
        material = mp.Medium(index = n_core),
    )
    subs = mp.Block(
        size = mp.Vector3(Δx - edge_gap, t_subs, 10.0),
        center = mp.Vector3(0, c_subs_y, 0),
        material = mp.Medium(index = n_subs),
    )
    ms = mpb.ModeSolver(
        geometry_lattice = lat,
        geometry = [core, subs],
        k_points = k_pts,
        resolution = res,
        num_bands = n_bands,
        default_material = mp.vacuum,
        deterministic = true,
    )
    ms.init_params(mp.NO_PARITY, false)
    return ms
end

function nng_rwg_mpb(
    p = p_def;
    Δx = 6.0,
    Δy = 4.0,
    Δz = 1.0,
    Nx = 128,
    Ny = 128,
    Nz = 1,
    edge_gap = 0.5,
    band_idx = 1,
    tol = 1e-8)
    λ, w, t_core, θ, n_core, n_subs = p
    ω = 1 / λ
    nk = 10
    n_bands = 1
    res = mp.Vector3((Nx / Δx), (Ny / Δy), 1) #mp.Vector3(Int(Nx/Δx),Int(Ny/Δy),1) #16
    n_guess = 0.9 * n_core
    n_min = n_subs
    n_max = n_core
    ms = ms_rwg_mpb(p; Δx, Δy, Δz, Nx, Ny, Nz, edge_gap)
    kz = ms.find_k(
        mp.NO_PARITY,             # parity (meep parity object)
        ω,                    # ω at which to solve for k
        band_idx,                        # band_min (find k(ω) for bands
        band_idx,                        # band_max  band_min:band_max)
        mp.Vector3(0, 0, 1),      # k direction to search
        tol,                     # fractional k error tolerance
        n_guess * ω,              # kmag_guess, |k| estimate
        n_min * ω,                # kmag_min (find k in range
        n_max * ω,               # kmag_max  kmag_min:kmag_max)
    )[1]
    neff = kz / ω
    ng = 1 / ms.compute_one_group_velocity_component(mp.Vector3(0, 0, 1), band_idx)
    return [neff, ng]
end

function ε_rwg_mpb(
    p = p_def;
    Δx = 6.0,
    Δy = 4.0,
    Δz = 1.0,
    Nx = 128,
    Ny = 128,
    Nz = 1,
    edge_gap = 0.5,
    band_idx = 1,
    tol = 1e-8)
    λ, w, t_core, θ, n_core, n_subs = p
    n_guess = 0.9 * n_core
    ms = ms_rwg_mpb(p; Δx, Δy, Δz, Nx, Ny, Nz, edge_gap)
    ms.solve_kpoint(mp.Vector3(0, 0, 1) * n_guess / λ)
    return get_ε_mpb(ms)
end

function ε⁻¹_rwg_mpb(
    p = p_def;
    Δx = 6.0,
    Δy = 4.0,
    Δz = 1.0,
    Nx = 128,
    Ny = 128,
    Nz = 1,
    edge_gap = 0.5,
    band_idx = 1,
    tol = 1e-8)
    λ, w, t_core, θ, n_core, n_subs = p
    n_guess = 0.9 * n_core
    ms = ms_rwg_mpb(p; Δx, Δy, Δz, Nx, Ny, Nz, edge_gap)
    ms.solve_kpoint(mp.Vector3(0, 0, 1) * n_guess / λ)
    return get_ε⁻¹_mpb(ms)
end

∇nng_rwg_mpb_FD(
    p;
    Δx = 6.0,
    Δy = 4.0,
    Δz = 1.0,
    Nx = 128,
    Ny = 128,
    Nz = 1,
    edge_gap = 0.5,
    band_idx = 1,
    tol = 1e-8,
    nFD = 2) = FiniteDifferences.jacobian(
    central_fdm(nFD, 1),
    x -> nng_rwg_mpb(x; Δx, Δy, Δz, Nx, Ny, Nz, edge_gap, band_idx, tol),
    p,)[1]

wt_rwg_mpb(wt; NN = 128, bi = 1, tol = 1e-8) = nng_rwg_mpb(
    [p_def[1], wt[1], wt[2], p_def[4:6]...];
    Nx = NN,
    Ny = NN,
    band_idx = bi,
    tol,
)
∇wt_rwg_mpb_FD(wt; NN = 128, bi = 1, tol = 1e-8, nFD = 2) =
    FiniteDifferences.jacobian(central_fdm(nFD, 1), x -> wt_rwg_mpb(x; NN, bi, tol), wt)[1]

function wt_rwg_mpb_sweep(ws, ts, Ns; bi = 1, tol = 1e-8)
    nw = length(ws)
    nt = length(ts)
    nN = length(Ns)
    nng_mpb = zeros(Float64, (2, nw, nt, nN))
    ∇nng_mpb = zeros(Float64, (2, 2, nw, nt, nN))
    for wind = 1:nw, tind = 1:nt, Nind = 1:nN
        nng_mpb[:, wind, tind, Nind] =
            wt_rwg_mpb([ws[wind], ts[tind]]; NN = Ns[Nind], bi, tol)
        @views ∇nng_mpb[:, :, wind, tind, Nind] .=
            ∇wt_rwg_mpb_FD([ws[wind], ts[tind]]; NN = Ns[Nind], bi, tol)
    end
    return nng_mpb, ∇nng_mpb
end

function plot_ε_rwg_xslices_N(
    Ns,
    p = p_def;
    Δx = 6.0,
    Δy = 4.0,
    Δz = 1.0,
    Nz = 1,
    edge_gap = 0.5,
    size = (800, 800),
    xlims = nothing,
    ylims = nothing,
    marker = :dot,
    cs = (:red, :blue, :green, :purple, :orange, :magenta))

    nN = length(Ns)
    ms = ms_rwg_mpb(p; Δx, Δy, Δz, Nx = Ns[1], Ny = Ns[1], Nz, edge_gap)
    x, y, z = get_xyz_mpb(ms)
    iy = Int(ceil(Ns[1] / 2))
    εs1 = Array{Float64,3}(undef, (3, 3, Ns[1]))
    for ix = 1:Ns[1]
        εs1[:, :, ix] .= real(ms.get_epsilon_inverse_tensor_point(mp.Vector3(
            x[ix],
            y[iy],
            z[1],
        )).inverse().__array__())
    end
    p_diag = plot(
        x,
        εs1[1, 1, :],
        c = cs[1],
        label = "11",
        xlabel = "x [μm]",
        ylabel = "εᵢᵢ (diagonal elements)",
        legend = :bottomright;
        xlims,
        ylims,
        marker,
    )
    plot!(p_diag, x, εs1[2, 2, :], c = cs[2], label = "22"; xlims, ylims, marker)
    plot!(p_diag, x, εs1[3, 3, :], c = cs[3], label = "33"; xlims, ylims, marker)
    p_offdiag = plot(
        x,
        εs1[1, 2, :],
        c = cs[4],
        label = "12",
        xlabel = "x [μm]",
        ylabel = "εᵢⱼ (off-diag. elements)",
        legend = :bottomright;
        xlims,
        ylims,
        marker,
    )
    plot!(p_offdiag, x, εs1[1, 3, :], c = cs[5], label = "13"; xlims, ylims, marker)
    plot!(p_offdiag, x, εs1[2, 3, :], c = cs[6], label = "23"; xlims, ylims, marker)
    for Nind = 2:nN
        NN = Ns[Nind]
        ms = ms_rwg_mpb(p; Δx, Δy, Δz, Nx = NN, Ny = NN, Nz, edge_gap)
        x, y, z = get_xyz_mpb(ms)
        iy = Int(ceil(NN[1] / 2))
        εs = Array{Float64,3}(undef, (3, 3, NN))
        for ix = 1:NN
            εs[:, :, ix] .= real(ms.get_epsilon_inverse_tensor_point(mp.Vector3(
                x[ix],
                y[iy],
                z[1],
            )).inverse().__array__())
        end
        plot!(p_diag, x, εs[1, 1, :], c = cs[1], label = nothing; xlims, ylims, marker)
        plot!(p_diag, x, εs[2, 2, :], c = cs[2], label = nothing; xlims, ylims, marker)
        plot!(p_diag, x, εs[3, 3, :], c = cs[3], label = nothing; xlims, ylims, marker)
        plot!(p_offdiag, x, εs[1, 2, :], c = cs[4], label = nothing; xlims, ylims, marker)
        plot!(p_offdiag, x, εs[1, 3, :], c = cs[5], label = nothing; xlims, ylims, marker)
        plot!(p_offdiag, x, εs[2, 3, :], c = cs[6], label = nothing; xlims, ylims, marker)
    end
    l = @layout [
        a
        b
    ]
    plot(p_diag, p_offdiag, layout = l, size = size)
end

function ε_rwg_xslices_N(
    Ns,
    p = p_def;
    Δx = 6.0,
    Δy = 4.0,
    Δz = 1.0,
    Nz = 1,
    edge_gap = 0.5)
    nN = length(Ns)
    ms = ms_rwg_mpb(p; Δx, Δy, Δz, Nx = Ns[end], Ny = Ns[end], Nz, edge_gap)
    x, y, z = get_xyz_mpb(ms)
    # iy = Int(ceil(Ns[end] / 2))
    εs = Array{Float64,4}(undef, (3, 3, Ns[end], nN))
    for ix = 1:Ns[end]
        εs[:, :, ix, nN] .= real(ms.get_epsilon_inverse_tensor_point(mp.Vector3(
            x[ix],
            0.0,
            0.0,
        )).inverse().__array__())
    end
    for Nind = 1:nN-1
        NN = Ns[Nind]
        ms = ms_rwg_mpb(p; Δx, Δy, Δz, Nx = NN, Ny = NN, Nz, edge_gap)
        # xtemp, y, z = get_xyz_mpb(ms)
        # iy = Int(ceil(NN[1] / 2))
        for ix = 1:Ns[end]
            εs[:, :, ix, Nind] .= real(ms.get_epsilon_inverse_tensor_point(mp.Vector3(
                x[ix],
                0.0,
                0.0,
            )).inverse().__array__())
        end
    end
    return x, εs
end

function ε_rwg_xslices_w(
    ws;
    p = [
        1.55,               #   wavelength              `λ`             [μm]
        1.7,                #   top ridge width         `w_top`         [μm]
        0.7,                #   ridge thickness         `t_core`        [μm]
        π / 10.0,                 #   ridge sidewall angle    `θ`             [radian]
        2.4,                #   core index              `n_core`        [1]
        1.4,                #   substrate index         `n_subs`        [1]
    ],
    Δx = 6.0,
    Δy = 4.0,
    Δz = 1.0,
    Nx = 256,
    Ny = 256,
    Nz = 1,
    edge_gap = 0.5)
    nw = length(ws)
    ms = ms_rwg_mpb([
        1.55,               #   wavelength              `λ`             [μm]
        ws[1],                #   top ridge width         `w_top`         [μm]
        0.7,                #   ridge thickness         `t_core`        [μm]
        π / 10.0,                 #   ridge sidewall angle    `θ`             [radian]
        2.4,                #   core index              `n_core`        [1]
        1.4,                #   substrate index         `n_subs`        [1]
    ]; Δx, Δy, Δz, Nx, Ny, Nz, edge_gap)
    x, y, z = get_xyz_mpb(ms)
    εs = Array{Float64,4}(undef, (3, 3, length(x), nw))
    for ix = 1:Nx
        εs[:, :, ix, 1] .= real(ms.get_epsilon_inverse_tensor_point(mp.Vector3(
            x[ix],
            0.0,
            0.0,
        )).inverse().__array__())
    end
    for wind = 2:length(ws)
        ww = ws[wind]
        ms = ms_rwg_mpb(
            [
                1.55,               #   wavelength              `λ`             [μm]
                ws[wind],                #   top ridge width         `w_top`         [μm]
                0.7,                #   ridge thickness         `t_core`        [μm]
                π / 10.0,                 #   ridge sidewall angle    `θ`             [radian]
                2.4,                #   core index              `n_core`        [1]
                1.4,                #   substrate index         `n_subs`        [1]
            ];
            Δx,
            Δy,
            Δz,
            Nx,
            Ny,
            Nz,
            edge_gap,
        )
        # xtemp, y, z = get_xyz_mpb(ms)
        # iy = Int(ceil(NN[1] / 2))
        for ix = 1:Nx
            εs[:, :, ix, wind] .= real(ms.get_epsilon_inverse_tensor_point(mp.Vector3(
                x[ix],
                0.0,
                0.0,
            )).inverse().__array__())
        end
    end
    return x, εs
end

"""
OptiMode functions for ridge waveguide data
"""
function nngω_rwg_OM(p::Vector{Float64} = pω_def;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 128, #16,
                    Ny = 128, #16,
                    Nz = 1,
                    band_idx = 1,
                    tol = 1e-8)
                    # kz, w, t_core, θ, n_core, n_subs, edge_gap = p
                    # nng_tuple = solve_nω(kz,ridge_wg(w,t_core,θ,edge_gap,n_core,n_subs,Δx,Δy),Δx,Δy,Δz,Nx,Ny,Nz;tol)
                    nng_tuple = solve_nω(p[1],ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),Δx,Δy,Δz,Nx,Ny,Nz;tol)
                    [nng_tuple[1],nng_tuple[2]]
end

function nng_rwg_OM(p = p_def;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 128,
                    Ny = 128,
                    Nz = 1,
                    edge_gap = 0.5,
                    band_idx = 1,
                    tol = 1e-8)
                    λ, w, t_core, θ, n_core, n_subs = p
                    ω = 1/λ
                    # solve_n(ω,ridge_wg(w,t_core,edge_gap,n_core,n_subs,dropgrad(Δx),dropgrad(Δy)),dropgrad(Δx),dropgrad(Δy),dropgrad(Δz),dropgrad(Nx),dropgrad(Ny),dropgrad(Nz))
                    nng_tuple = solve_n(ω,ridge_wg(w,t_core,θ,edge_gap,n_core,n_subs,Δx,Δy),Δx,Δy,Δz,Nx,Ny,Nz;tol)
                    [nng_tuple[1],nng_tuple[2]]
end

function ε_rwg_OM(p = p_def;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 128,
                    Ny = 128,
                    Nz = 1,
                    edge_gap = 0.5,
                    npix_sm = 2 )
                    λ, w, t_core, θ, n_core, n_subs = p
                    # εₛ(ridge_wg(w,t_core,edge_gap,n_core,n_subs,dropgrad(Δx),dropgrad(Δy)),dropgrad(Δx),dropgrad(Δy),dropgrad(Nx),dropgrad(Ny);npix_sm)
                    reshape(
                        permutedims(
                            εₛ(ridge_wg(w,t_core,θ,edge_gap,n_core,n_subs,Δx,Δy),Δx,Δy,Nx,Ny;npix_sm),
                            (3,4,1,2),
                        ),
                        (3,3,Nx,Ny,1),
                    )
end

function ε⁻¹_rwg_OM(p = p_def;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 128,
                    Ny = 128,
                    Nz = 1,
                    edge_gap = 0.5,
                    npix_sm = 2 )
                    λ, w, t_core, θ, n_core, n_subs = p
                    # εₛ⁻¹(ridge_wg(w,t_core,edge_gap,n_core,n_subs,dropgrad(Δx),dropgrad(Δy)),dropgrad(Δx),dropgrad(Δy),dropgrad(Nx),dropgrad(Ny);npix_sm)
                    reshape(
                        permutedims(
                            εₛ⁻¹(ridge_wg(w,t_core,θ,edge_gap,n_core,n_subs,Δx,Δy),Δx,Δy,Nx,Ny;npix_sm),
                            (3,4,1,2),
                        ),
                        (3,3,Nx,Ny,1),
                    )
end

∇nngω_rwg_OM_FD(
    p;
    Δx = 6.0,
    Δy = 4.0,
    Δz = 1.0,
    Nx = 128,
    Ny = 128,
    Nz = 1,
    edge_gap = 0.5,
    band_idx = 1,
    tol = 1e-8,
    nFD = 3) = FiniteDifferences.jacobian(
    central_fdm(nFD, 1),
    x -> nngω_rwg_OM(x; Δx, Δy, Δz, Nx, Ny, Nz, edge_gap, band_idx, tol),
    p,)[1]

∇nng_rwg_OM_FD(
    p;
    Δx = 6.0,
    Δy = 4.0,
    Δz = 1.0,
    Nx = 128,
    Ny = 128,
    Nz = 1,
    edge_gap = 0.5,
    band_idx = 1,
    tol = 1e-8,
    nFD = 2) = FiniteDifferences.jacobian(
    central_fdm(nFD, 1),
    x -> nng_rwg_OM(x; Δx, Δy, Δz, Nx, Ny, Nz, edge_gap, band_idx, tol),
    p,)[1]

wtω_rwg_OM(wt; NN = 128, bi = 1, tol = 1e-8) = nngω_rwg_OM(
    [pω_def[1], wt[1], wt[2], pω_def[4:7]...];
    Nx = NN,
    Ny = NN,
    band_idx = bi,
    tol,
)
∇wtω_rwg_OM_FD(wt; NN = 128, bi = 1, tol = 1e-8, nFD = 3) =
    FiniteDifferences.jacobian(central_fdm(nFD, 1), x -> wtω_rwg_OM(x; NN, bi, tol), wt)[1]

function wtω_rwg_OM_sweep(sw_name, ws, ts, Ns;
    bi = 1,
    tol = 1e-8,
    data_dir="/home/dodd/data/OptiMode/mpb_compare_rwg/",
    dt_fmt=dateformat"Y-m-d--H-M-S",
    extension=".h5",
    )
    timestamp = Dates.format(now(),dt_fmt)
    fname = sw_name * "_" *  timestamp * extension
    @show fpath = data_dir * fname
    nw = length(ws)
    nt = length(ts)
    nN = length(Ns)
    nng_OM = zeros(Float64, (2, nw, nt, nN))
    ∇nng_OM_FD = zeros(Float64, (2, 2, nw, nt, nN))
    ∇nng_OM_AD = zeros(Float64, (2, 2, nw, nt, nN))
    for wind = 1:nw, tind = 1:nt, Nind = 1:nN
        println("wind: $wind of $nw,  tind: $tind of $nt,  Nind: $Nind of $nN")
        wt = [ws[wind], ts[tind]]
        nng,nng_pb = Zygote.pullback(x -> wtω_rwg_OM(x; NN = Ns[Nind], bi, tol), wt)
        nng_OM[:, wind, tind, Nind] = nng
        @views ∇nng_OM_AD[:,:,wind,tind,Nind] .= [ real(nng_pb([1.,0.])[1])'    # = [   ∂n/∂w   ∂n/∂t
                                                   real(nng_pb([0.,1.])[1])' ]  #       ∂ng/∂w  ∂ng/∂t  ]
        @views ∇nng_OM_FD[:, :, wind, tind, Nind] .=
            ∇wtω_rwg_OM_FD([ws[wind], ts[tind]]; NN = Ns[Nind], bi, tol)
        h5open(fpath, "w") do file
            @write file nng_OM
            @write file ws
            @write file ts
            @write file Ns
            @write file tol
            write(file, "band_index", bi)
            write(file, "grad_nng_OM_FD", ∇nng_OM_FD)
            write(file, "grad_nng_OM_AD", ∇nng_OM_AD)
        end
    end
    return nng_OM, ∇nng_OM_AD, ∇nng_OM_FD
end

wt_rwg_OM(wt; NN = 128, bi = 1, tol = 1e-8) = nng_rwg_OM(
    [p_def[1], wt[1], wt[2], p_def[4:6]...];
    Nx = NN,
    Ny = NN,
    band_idx = bi,
    tol,
)
∇wt_rwg_OM_FD(wt; NN = 128, bi = 1, tol = 1e-8, nFD = 3) =
    FiniteDifferences.jacobian(central_fdm(nFD, 1), x -> wt_rwg_OM(x; NN, bi, tol), wt)[1]

function wt_rwg_OM_sweep(ws, ts, Ns; bi = 1, tol = 1e-8)
    nw = length(ws)
    nt = length(ts)
    nN = length(Ns)
    nng_OM = zeros(Float64, (2, nw, nt, nN))
    ∇nng_OM = zeros(Float64, (2, 2, nw, nt, nN))
    for wind = 1:nw, tind = 1:nt, Nind = 1:nN
        nng_OM[:, wind, tind, Nind] =
            wt_rwg_OM([ws[wind], ts[tind]]; NN = Ns[Nind], bi, tol)
        @views ∇nng_OM[:, :, wind, tind, Nind] .=
            ∇wt_rwg_OM_FD([ws[wind], ts[tind]]; NN = Ns[Nind], bi, tol)
    end
    return nng_OM, ∇nng_OM
end

##
# @assert size(ε_rwg_mpb(p_def)) == size(ε_rwg_OM(p_def))
# @assert size(ε⁻¹_rwg_mpb(p_def)) == size(ε⁻¹_rwg_OM(p_def))

# ∇wtω_rwg_OM_FD([1.5, 0.7]; NN = 64, bi = 1, tol = 1e-8, nFD = 3)
# nng_OM, ∇nng_OM_AD, ∇nng_OM_FD = wtω_rwg_OM_sweep(1.3:0.2:1.7, .5:0.2:.7, 2 .^(5:7); bi = 1, tol = 1e-8)
# ∇nng_err_OM = abs.(∇nng_OM_FD .- ∇nng_OM_AD) ./ abs.(∇nng_OM_FD)
##
function compare_ε11_rwg(p=p_def;
    Δx = 6.0,
    Δy = 4.0,
    Δz = 1.0,
    Nx = 128,
    Ny = 128,
    Nz = 1,
    edge_gap = 0.5,
    )
    eps_mpb = ε_rwg_mpb(p;Δx,Δy,Δz,Nx,Ny,Nz,edge_gap)
    eps_OM = ε_rwg_OM(p;Δx,Δy,Δz,Nx,Ny,Nz,edge_gap)
    x = ((Δx / Nx) .* (0:(Nx-1))) .- Δx / 2.0
    y = ((Δy / Ny) .* (0:(Ny-1))) .- Δy / 2.0
    hm_eps11_mpb = heatmap(x,
        y,
        eps_mpb[1,1,:,:,1]',
        xlabel = "x [μm]",
        ylabel = "y [μm]",
        title = "Smoothed ε₁₁, MPB",
        aspect_ratio=:equal,
        )
    hm_eps11_OM = heatmap(x,
        y,
        eps_OM[1,1,:,:,1]',
        xlabel = "x [μm]",
        ylabel = "y [μm]",
        title = "Smoothed ε₁₁, OptiMode",
        aspect_ratio=:equal,
        )
    l = @layout[  a  b  ]
    plot(hm_eps11_mpb,hm_eps11_OM,layout=l)
end
# compare_ε11_rwg(p_def)
##
# Ns = Int.(ceil.(2 .^ collect(6:0.5:9)))
# Ns = Int.(ceil.(2 .^ collect(4.4:0.02:8)))
# x_slices_N, ε_xslices_N = ε_rwg_xslices_N(Ns, [
#     1.55,               #   wavelength              `λ`             [μm]
#     1.7,                #   top ridge width         `w_top`         [μm]
#     0.7,                #   ridge thickness         `t_core`        [μm]
#     π / 10.0,              #   ridge sidewall angle    `θ`             [radian]
#     2.4,                #   core index              `n_core`        [1]
#     1.4,                #   substrate index         `n_subs`        [1]
# ])
# # Makie.surface(x_slices[75:105],(.01 .* Ns),ε_xslices[1,1,75:105,:])
# ixmin, ixmax = 80, 102
# sN11 = Plots.surface(Ns, x_slices_N[ixmin:ixmax], ε_xslices_N[1, 1, ixmin:ixmax, :])
# sN22 = Plots.surface(Ns, x_slices_N[ixmin:ixmax], ε_xslices_N[2, 2, ixmin:ixmax, :])
# sN12 = Plots.surface(Ns, x_slices_N[ixmin:ixmax], ε_xslices_N[1, 2, ixmin:ixmax, :])
# sN13 = Plots.surface(Ns, x_slices_N[ixmin:ixmax], ε_xslices_N[1, 3, ixmin:ixmax, :])
# sN23 = Plots.surface(Ns, x_slices_N[ixmin:ixmax], ε_xslices_N[2, 3, ixmin:ixmax, :])
#
# ws = collect(1.0:0.005:1.2)
# x_slices_w, ε_xslices_w = ε_rwg_xslices_w(ws)
# #ixmin, ixmax = 97, 113 #50,56
# ixmin, ixmax = 1, 256
# # sw11 = Plots.surface(Ns,x_slices_w,ε_xslices_w[1,1,:,:])
# sw11 = Plots.surface(ws, x_slices_w[ixmin:ixmax], ε_xslices_w[1, 1, ixmin:ixmax, :])
# sw22 = Plots.surface(ws, x_slices_w[ixmin:ixmax], ε_xslices_w[2, 2, ixmin:ixmax, :])
# sw33 = Plots.surface(ws, x_slices_w[ixmin:ixmax], ε_xslices_w[3, 3, ixmin:ixmax, :])
# sw12 = Plots.surface(ws, x_slices_w[ixmin:ixmax], ε_xslices_w[1, 2, ixmin:ixmax, :])
# sw13 = Plots.surface(ws, x_slices_w[ixmin:ixmax], ε_xslices_w[1, 3, ixmin:ixmax, :])
# sw23 = Plots.surface(ws, x_slices_w[ixmin:ixmax], ε_xslices_w[2, 3, ixmin:ixmax, :])

#
#
# using Plots
# pyplot()
# pygui()
# p_xslices = ε_rwg_xslices_N(Ns, p_def, xlims = (-1.1, -0.8))
#
# p_slices64 = ε_slices(ms_rwg_mpb(p_def; Nx = 64, Ny = 64), xlims = (-1.1, -0.8), yind = 32)
# p_slices128 =
#     ε_slices(ms_rwg_mpb(p_def; Nx = 128, Ny = 128), xlims = (-1.1, -0.8), yind = 64)
# p_slices256 =
#     ε_slices(ms_rwg_mpb(p_def; Nx = 256, Ny = 256), xlims = (-1.1, -0.8), yind = 128)
#
# p_slices256.subplots[1]
#
# nng_rwg_mpb(p_def)
# ∇nng_rwg_mpb_FD(p_def)
# wt_rwg_mpb([1.0, 0.7]; NN = 64, bi = 1, tol = 1e-5)
# grad_nng = ∇wt_rwg_mpb_FD([1.0, 0.7]; NN = 64, bi = 1, tol = 1e-5)


# function wt_rwg_sweep(ws,ts,Ns;bi=1,tol=1e-5,name="rwg_ng_bm_N",path="/home/dodd/github/OptiMode/test/")
#     bm = [ @benchmark(outs[$i] = ngom_rwg($ks,$w,$t_core,$edge_gap,$n_core,$n_subs,$Δx,$Δy,$Δz,$Ns[$i],$Ns[$i],1))  for i=1:length(Ns) ]
#     ns = [o[1] for o in outs]
#     ngs = [o[2] for o in outs]
#     # bm = [ @benchmark(ngom_rwg(1.6,$w,$t_core,$edge_gap,$n_core,$n_subs,$Δx,$Δy,$Δz,$nn,$nn,1))  for nn=Ns ]
#     df = DataFrame( N = Ns,
#                     Nsq = (Ns.^2),
#                     n = ns,
#                     ng = ngs,
#                     time = (time.(bm).*1e-9),
#                     gctime_frac = (gctime.(bm) ./ time.(bm)),
#                     mem = memory.(bm),
#                     allocs = allocs.(bm),
#                    )
#     CSV.write(*(path,name,".csv"), df)
#     # plot_bm_single(df)
#     df
# end

########################################
##### w-t-N sweep 1 (MPB only) #########
########################################
# ws = collect(0.8:0.1:1.7)
# ts = collect(0.5:0.1:1.3)
# Ns = Int.(2 .^(collect(5:8)))
# nng_mpb, ∇nng_mpb = wt_rwg_mpb_sweep(ws,ts,Ns;bi=1)
# ws1 = collect(0.8:0.1:1.7)
# ts1 = collect(0.5:0.1:1.3)
# Ns1 = Int.(2 .^(collect(5:8)))
# nng_mpb1 = copy(nng_mpb)
# ∇nng_mpb1 = copy(∇nng_mpb)
# n_mpb1 = nng_mpb1[1,:,:,:]
# ng_mpb1 = nng_mpb1[2,:,:,:]
# ∂n∂w_mpb1 = ∇nng_mpb1[1,1,:,:,:]
# ∂n∂t_mpb1 = ∇nng_mpb1[1,2,:,:,:]
# ∂ng∂w_mpb1 = ∇nng_mpb1[2,1,:,:,:]
# ∂ng∂t_mpb1 = ∇nng_mpb1[2,2,:,:,:]
# mpb_fname1="rwg_mpb_wt1.h5"
# # path="/home/dodd/github/OptiMode/test/"
# h5open(mpb_fname1, "cw") do file
#     write(file, "ws", ws1)  # alternatively, say "@write file A"
#     write(file, "ts", ts1)
#     write(file, "Ns", Ns1)
#     write(file, "nng", nng_mpb1)
#     write(file, "∇nng", ∇nng_mpb1)
#     write(file, "∂n∂w", ∂n∂w_mpb1)
#     write(file, "∂n∂t", ∂n∂t_mpb1)
#     write(file, "∂ng∂w", ∂ng∂w_mpb1)
#     write(file, "∂ng∂t", ∂ng∂t_mpb1)
# end
# ∇nng_mpb1_read = h5open(mpb_fname1, "r") do file
#     read(file, "∇nng")
# end



function plot_wt_sweep(ws, ts, Ns, nng, ∇nng; tind_w = 4, size = (800, 800))
    n = nng[1, :, :, :]
    ng = nng[2, :, :, :]
    ∂n∂w = ∇nng[1, 1, :, :, :]
    ∂n∂t = ∇nng[1, 2, :, :, :]
    ∂ng∂w = ∇nng[2, 1, :, :, :]
    ∂ng∂t = ∇nng[2, 2, :, :, :]
    # w slices
    p_n_w = plot(
        ws,
        n[:, tind_w, 1],
        label = "N=$(Ns[1])",
        xlabel = "top width [μm]",
        ylabel = "effective index n",
        legend = :bottomright,
    )
    p_ng_w = plot(
        ws,
        ng[:, tind_w, 1],
        label = "N=$(Ns[1])",
        xlabel = "top width [μm]",
        ylabel = "eff. group index ng",
        legend = false,
    )
    p_∂n∂w_w = plot(
        ws,
        ∂n∂w[:, tind_w, 1],
        label = "N=$(Ns[1])",
        xlabel = "top width [μm]",
        ylabel = "∂n/∂w [μm⁻¹]",
        legend = false,
    )
    p_∂ng∂w_w = plot(
        ws,
        ∂ng∂w[:, tind_w, 1],
        label = "N=$(Ns[1])",
        xlabel = "top width [μm]",
        ylabel = "∂ng/∂w [μm⁻¹]",
        legend = false,
    )
    p_∂n∂t_w = plot(
        ws,
        ∂n∂t[:, tind_w, 1],
        label = "N=$(Ns[1])",
        xlabel = "top width [μm]",
        ylabel = "∂n/∂t [μm⁻¹]",
        legend = false,
    )
    p_∂ng∂t_w = plot(
        ws,
        ∂ng∂t[:, tind_w, 1],
        label = "N=$(Ns[1])",
        xlabel = "top width [μm]",
        ylabel = "∂ng/∂t [μm⁻¹]",
        legend = false,
    )
    for Nind = 2:length(Ns)
        plot!(
            p_n_w,
            ws,
            n[:, tind_w, Nind],
            label = "N=$(Ns[Nind])",
            xlabel = "top width [μm]",
            ylabel = "effective index n",
        )
        plot!(
            p_ng_w,
            ws,
            ng[:, tind_w, Nind],
            label = "N=$(Ns[Nind])",
            xlabel = "top width [μm]",
            ylabel = "eff. group index ng",
        )
        plot!(
            p_∂n∂w_w,
            ws,
            ∂n∂w[:, tind_w, Nind],
            label = "N=$(Ns[Nind])",
            xlabel = "top width [μm]",
            ylabel = "∂n/∂w [μm⁻¹]",
        )
        plot!(
            p_∂ng∂w_w,
            ws,
            ∂ng∂w[:, tind_w, Nind],
            label = "N=$(Ns[Nind])",
            xlabel = "top width [μm]",
            ylabel = "∂ng/∂w [μm⁻¹]",
        )
        plot!(
            p_∂n∂t_w,
            ws,
            ∂n∂t[:, tind_w, Nind],
            label = "N=$(Ns[Nind])",
            xlabel = "top width [μm]",
            ylabel = "∂n/∂t [μm⁻¹]",
        )
        plot!(
            p_∂ng∂t_w,
            ws,
            ∂ng∂t[:, tind_w, Nind],
            label = "N=$(Ns[Nind])",
            xlabel = "top width [μm]",
            ylabel = "∂ng/∂t [μm⁻¹]",
        )
    end
    l_w = @layout [
        a b
        c d
        e f
    ]
    plot(p_n_w, p_ng_w, p_∂n∂w_w, p_∂ng∂w_w, p_∂n∂t_w, p_∂ng∂t_w, layout = l_w, size = size)
end

# plot_wt_sweep(ws,ts,Ns,nng_mpb,∇nng_mpb;tind_w=2)
