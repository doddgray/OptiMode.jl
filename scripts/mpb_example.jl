using LinearAlgebra, StaticArrays, PyCall  #FFTW, BenchmarkTools, LinearMaps, IterativeSolvers, Roots, GeometryPrimitives
SHM3 = SHermitianCompact{3,Float64,6}
# MPB solve for reference
mp = pyimport("meep")
mpb = pyimport("meep.mpb")

# w           = 1.7
# t_core      = 0.7
# θ           = π/12
# edge_gap    = 0.5               # μm
# n_core      = 2.4
# n_subs      = 1.4
# λ           = 1.55              # μm

w = 1.7
t_core = 0.7
θ = π / 14.0
edge_gap = 0.5               # μm
n_core = 2.4
n_subs = 1.4
λ = 1.55                  # μm
Nx = 128
Ny = 128
Nz = 1
ω = 1 / λ
Δx = 6.0                    # μm
Δy = 4.0                    # μm
Δz = 1.0                    # μm  # not used, but don't set to zero

nk = 10
n_bands = 2
res = mp.Vector3((Nx / Δx), (Ny / Δy), 1) #mp.Vector3(Int(Nx/Δx),Int(Ny/Δy),1) #16
##


n_guess = 0.9 * n_core
n_min = n_subs
n_max = n_core
t_subs = (Δy - t_core - edge_gap) / 2.0
c_subs_y = -Δy / 2.0 + edge_gap / 2.0 + t_subs / 2.0



# Set up MPB modesolver, use find-k to solve for one eigenmode `H` with prop. const. `k` at specified temporal freq. ω

k_pts = mp.interpolate(nk, [mp.Vector3(0.05, 0, 0), mp.Vector3(0.05 * nk, 0, 0)])
lat = mp.Lattice(size = mp.Vector3(Δx, Δy, 0))
# core = mp.Block(size=mp.Vector3(w,t_core,10.0,),
#                     center=mp.Vector3(0,0,0,),
#                     material=mp.Medium(index=n_core),
#                    )
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
    resolution = 1.5 * res,
    num_bands = n_bands,
    deterministic = true,
    default_material = mp.vacuum,
)
ms.init_params(mp.NO_PARITY, false)
ms.solve_kpoint(mp.Vector3(0, 0, 1) * n_guess / λ)
ε_mean_mpb = ms.get_epsilon()

hm1 = heatmap(ε_mean_mpb)

w_bottom2 = 2 * w + 2 * tcore_tanθ
verts2 = [
    mp.Vector3(w, t_core / 2.0, -5.0),
    mp.Vector3(-w, t_core / 2.0, -5.0),
    mp.Vector3(-w_bottom2 / 2.0, -t_core / 2.0, -5.0),
    mp.Vector3(w_bottom2 / 2.0, -t_core / 2.0, -5.0),
]
# verts = [ mp.Vector3(-w/2., -t_core/2., -5.),mp.Vector3(w, 2*t_core, -5.), mp.Vector3(w, -t_core/2., -5.)  ]
core2 = mp.Prism(
    verts2,
    10.0,
    axis = mp.Vector3(0.0, 0.0, 1.0),
    material = mp.Medium(index = n_core),
)
geom2 = [core2, subs]
ms.geometry = geom2
ms.init_params(mp.NO_PARITY, false)
ε_mean_mpb2 = ms.get_epsilon()
hm2 = heatmap(ε_mean_mpb2)




nx_mpb = size(ε_mean_mpb)[1]
ny_mpb = size(ε_mean_mpb)[2]
dx_mpb = Δx / nx_mpb
dy_mpb = Δy / ny_mpb
x_mpb = (dx_mpb .* (0:(nx_mpb-1))) .- Δx / 2.0 #(Δx/2. - dx_mpb)
y_mpb = (dy_mpb .* (0:(ny_mpb-1))) .- Δy / 2.0 #(Δy/2. - dy_mpb)
k_mpb = ms.find_k(
    mp.NO_PARITY,             # parity (meep parity object)
    1 / λ,                    # ω at which to solve for k
    1,                        # band_min (find k(ω) for bands
    n_bands,                        # band_max  band_min:band_max)
    mp.Vector3(0, 0, 1),      # k direction to search
    1e-4,                     # fractional k error tolerance
    n_guess / λ,              # kmag_guess, |k| estimate
    n_min / λ,                # kmag_min (find k in range
    n_max / λ,               # kmag_max  kmag_min:kmag_max)
)



1 ./ ms.compute_group_velocity_component(mp.Vector3(0, 0, 1))
# 1 / ms.compute_one_group_velocity_component(mp.Vector3(0, 0, 1),1)

function n_ng_mpb(om, ms)
    kz = ms.find_k(
        mp.NO_PARITY,             # parity (meep parity object)
        om,                    # ω at which to solve for k
        1,                        # band_min (find k(ω) for bands
        n_bands,                        # band_max  band_min:band_max)
        mp.Vector3(0, 0, 1),      # k direction to search
        1e-5,                     # fractional k error tolerance
        n_guess * om,              # kmag_guess, |k| estimate
        n_min * om,                # kmag_min (find k in range
        n_max * om,               # kmag_max  kmag_min:kmag_max)
    )[1]
    neff = kz / om
    ng = 1 / ms.compute_one_group_velocity_component(mp.Vector3(0, 0, 1), 1)
    return neff, ng
end



# nngs_w = [ nng_rwg_mpb([ww,t_core,0.,edge_gap,n_core,n_subs,λ];Δx,Δy,Δz,Nx,Ny,Nz,band_idx=bi) for ww in ws,bi=1:2 ]
# nngs_t = [ nng_rwg_mpb([w,tt,0.,edge_gap,n_core,n_subs,λ];Δx,Δy,Δz,Nx,Ny,Nz,band_idx=bi) for tt in ts,bi=1:2 ]
# ns_w = [nngg[1] for nngg in nngs_w]
# ngs_w = [nngg[2] for nngg in nngs_w]
ns = [nngg[1] for nngg in nngs]
ngs = [nngg[2] for nngg in nngs]



wf_n = surface(ts, ws, ns[:, :, 1], st = :wireframe)
surface!(ts, ws, ns[:, :, 1])

# pyplot()
##
cam_n = (30, 50)

sfc_n = surface(
    ts,
    ws,
    ns[:, :, 1],
    # linecolor=:black,
    c = cgrad(:viridis),
    # st=:wireframe,
    camera = cam_n,
)
surface!(
    sfc_n,
    ts,
    ws,
    ns[:, :, 2],
    # linecolor=:black,
    c = cgrad(:plasma),
    # st=:wireframe,
    camera = cam_n,
)
cam_ng = (30, 50)
sfc_ng = surface(
    ts,
    ws,
    ngs[:, :, 1],
    # linecolor=:black,
    c = cgrad(:viridis),
    # st=:wireframe,
    camera = cam_ng,
)
surface!(
    sfc_ng,
    ts,
    ws,
    ngs[:, :, 2],
    # linecolor=:black,
    c = cgrad(:plasma),
    # st=:wireframe,
    camera = cam_ng,
)


l = @layout [
    a
    b
]
plot(sfc_n, sfc_ng, layout = l, size = (800, 800))
##
surface!(ts, ws, ns[:, :, 1], st = :wireframe, surfacealpha = 0, fillalpha = 0)
# wireframe!(sfc_n,ts,ws,ns[:,:,1])

using Plots
##
plt_n1_w = plot(ws, ns[:, :, 1], label = nothing, c = :blue)
plt_n2_w = plot!(ws, ns[:, :, 2], label = nothing, c = :red)
plt_ng1_w = plot(ws, ngs[:, :, 1], label = nothing, c = :blue)
plt_ng2_w = plot!(ws, ngs[:, :, 2], label = nothing, c = :red)

plot(plt_n1_w, plt_ng1_w)
plt_n_t = plot(ts, ns_t, label = "nt", marker = :dot)
plt_ng_w = plot(ws, ngs_w, label = "ngw", marker = :dot)
plt_ng_t = plot(ts, ngs_t, label = "ngt", marker = :dot)
l = @layout [
    a b
    c d
]
plot(plt_n_w, plt_n_t, plt_ng_w, plt_ng_t, layout = l, size = (800, 800))
# n_ng_mpb(0.6)
##
neff_mpb = k_mpb * λ
ng_mpb = 1 / ms.compute_one_group_velocity_component(mp.Vector3(0, 0, 1), 1)
e_mpb = reshape(ms.get_efield(1), (nx_mpb, ny_mpb, 3))
d_mpb = reshape(ms.get_dfield(1), (nx_mpb, ny_mpb, 3))
h_mpb = reshape(ms.get_hfield(1), (nx_mpb, ny_mpb, 3))
S_mpb = reshape(ms.get_poynting(1), (nx_mpb, ny_mpb, 3))
U_mpb = reshape(ms.get_tot_pwr(1), (nx_mpb, ny_mpb))
# H_mpb = vec(reshape(ms.get_eigenvectors(1,1),(nx_mpb*ny_mpb,2)))
H_mpb_raw = reshape(ms.get_eigenvectors(1, 1), (ny_mpb, nx_mpb, 2))
H_mpb = zeros(ComplexF64, (3, nx_mpb, ny_mpb, 1)) #Array{ComplexF64,4}(undef,(3,ny_mpb,ny_mpb,1))
for i = 1:nx_mpb
    for j = 1:ny_mpb
        H_mpb[1, i, j, 1] = H_mpb_raw[j, i, 1]
        H_mpb[2, i, j, 1] = H_mpb_raw[j, i, 2]
    end
end
SHM3 = SHermitianCompact{3,Float64,6}
z_mpb = [0.0]
nz_mpb = 1
# ε⁻¹_mpb = [SHM3([ms.get_epsilon_inverse_tensor_point(mp.Vector3(x_mpb[i],y_mpb[j],z_mpb[k]))[a][b] for a=1:3,b=1:3]) for i=1:nx_mpb,j=1:ny_mpb,k=1:nz_mpb]
ε⁻¹_mpb = [
    SHM3([
        get(
            get(
                ms.get_epsilon_inverse_tensor_point(mp.Vector3(
                    x_mpb[i],
                    y_mpb[j],
                    z_mpb[k],
                )),
                a - 1,
            ),
            b - 1,
        ) for a = 1:3, b = 1:3
    ]) for i = 1:nx_mpb, j = 1:ny_mpb, k = 1:nz_mpb
]
ε_mpb = zeros(Float64, (3, 3, size(ε⁻¹_mpb)...))
for i = 1:nx_mpb, j = 1:ny_mpb, k = 1:nz_mpb
    ε_mpb[:, :, i, j, k] = inv(ε⁻¹_mpb[i, j, k])
end


function get_Δs_mpb(ms)
    Δx, Δy, Δz = [(Δ == 0.0 ? 1.0 : Δ) for Δ in ms_size.__array__()]
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

x, y, z = get_xyz_mpb(ms)
ε = get_ε_mpb(ms)
ε⁻¹ = get_ε⁻¹_mpb(ms)

##
function ε_slices(
    ms;
    yind = 64,
    zind = 1,
    size = (800, 800),
    xlims = nothing,
    ylims = nothing,
    marker = :dot,
)
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

p_slices = ε_slices(ms, xlims = (-2.0, -1.4))

heatmap(x, y, ε)

#ε_mpb = [SHM3(inv(ε⁻¹_mpb[i,j,k])) for i=1:nx_mpb,j=1:ny_mpb,k=1:nz_mpb]
e_mpb = [SVector(e_mpb[i, j, :]...) for i = 1:nx_mpb, j = 1:ny_mpb]
d_mpb = [SVector(d_mpb[i, j, :]...) for i = 1:nx_mpb, j = 1:ny_mpb]
h_mpb = [SVector(h_mpb[i, j, :]...) for i = 1:nx_mpb, j = 1:ny_mpb]
S_mpb = [SVector(S_mpb[i, j, :]...) for i = 1:nx_mpb, j = 1:ny_mpb]

Nx = nx_mpb
Ny = ny_mpb
Nz = 1
kz = k_mpb
ε⁻¹ = [ε⁻¹_mpb[i, j, k][a, b] for a = 1:3, b = 1:3, i = 1:Nx, j = 1:Ny, k = 1:Nz];
ω_mpb = 1 / λ;
ω = ω_mpb;
Ha = copy(H_mpb[1:2, :, :, :]);
H = copy(vec(Ha));
plot_ε(ε_mpb, x_mpb, y_mpb)
# Nz = 1
# Neigs = 1
# N = *(Nx,Ny,Nz)
# s = ridge_wg(w,t_core,edge_gap,n_core,n_subs,g)
# ε⁻¹ = εₛ⁻¹(s,g)
# ε = SHM3.(inv.(ε⁻¹))
