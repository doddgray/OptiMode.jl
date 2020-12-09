using LinearAlgebra, StaticArrays, PyCall  #FFTW, BenchmarkTools, LinearMaps, IterativeSolvers, Roots, GeometryPrimitives
SHM3 = SHermitianCompact{3,Float64,6}
# MPB solve for reference
mp = pyimport("meep")
mpb = pyimport("meep.mpb")

w           = 1.7
t_core      = 0.7
edge_gap    = 0.5               # μm
n_core      = 2.4
n_subs      = 1.4
λ           = 1.55              # μm

nk          = 10
Δx       = 6.                    # μm
Δy       = 4.                    # μm
Δz       = 1.                    # μm  # not used, but don't set to zero
n_bands     = 1
res         = 16

n_guess = 0.9 * n_core
n_min = n_subs
n_max = n_core
t_subs = (Δy -t_core - edge_gap )/2.
c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.

# Set up MPB modesolver, use find-k to solve for one eigenmode `H` with prop. const. `k` at specified temporal freq. ω

k_pts = mp.interpolate(nk, [mp.Vector3(0.05, 0, 0), mp.Vector3(0.05*nk, 0, 0)] )
lat = mp.Lattice(size=mp.Vector3(Δx, Δy,0))
# core = mp.Block(size=mp.Vector3(w,t_core,10.0,),
#                     center=mp.Vector3(0,0,0,),
#                     material=mp.Medium(index=n_core),
#                    )
verts = [ mp.Vector3(-w/2., -t_core/2., -5.),mp.Vector3(w, 2*t_core, -5.), mp.Vector3(w, -t_core/2., -5.)  ]
core = mp.Prism(verts,
                   10.0,
                   axis=mp.Vector3(0.,0.,1.),
                   material=mp.Medium(index=n_core),
                  )
subs = mp.Block(size=mp.Vector3(Δx-edge_gap, t_subs , 10.0),
                center=mp.Vector3(0, c_subs_y, 0),
                material=mp.Medium(index=n_subs),
                )

ms = mpb.ModeSolver(geometry_lattice=lat,
                    geometry=[core,subs],
                    k_points=k_pts,
                    resolution=res,
                    num_bands=n_bands,
                    default_material=mp.vacuum)

ms.init_params(mp.NO_PARITY, false)
ε_mean_mpb = ms.get_epsilon()
nx_mpb = size(ε_mean_mpb)[1]
ny_mpb = size(ε_mean_mpb)[2]
dx_mpb = Δx / nx_mpb
dy_mpb = Δy / ny_mpb
x_mpb = (dx_mpb .* (0:(nx_mpb-1))) .- Δx/2. #(Δx/2. - dx_mpb)
y_mpb = (dy_mpb .* (0:(ny_mpb-1))) .- Δy/2. #(Δy/2. - dy_mpb)
k_mpb = ms.find_k(mp.NO_PARITY,             # parity (meep parity object)
                  1/λ,                    # ω at which to solve for k
                  1,                        # band_min (find k(ω) for bands
                  n_bands,                        # band_max  band_min:band_max)
                  mp.Vector3(0, 0, 1),      # k direction to search
                  1e-4,                     # fractional k error tolerance
                  n_guess/λ,              # kmag_guess, |k| estimate
                  n_min/λ,                # kmag_min (find k in range
                  n_max/λ,               # kmag_max  kmag_min:kmag_max)
)[1]

function n_ng_mpb(om,ms)
    kz = ms.find_k(mp.NO_PARITY,             # parity (meep parity object)
                      om,                    # ω at which to solve for k
                      1,                        # band_min (find k(ω) for bands
                      n_bands,                        # band_max  band_min:band_max)
                      mp.Vector3(0, 0, 1),      # k direction to search
                      1e-5,                     # fractional k error tolerance
                      n_guess*om,              # kmag_guess, |k| estimate
                      n_min*om,                # kmag_min (find k in range
                      n_max*om,               # kmag_max  kmag_min:kmag_max)
    )[1]
    neff = kz / om
    ng = 1 / ms.compute_one_group_velocity_component(mp.Vector3(0, 0, 1), 1)
    return neff, ng
end

# n_ng_mpb(0.6)

neff_mpb = k_mpb * λ
ng_mpb = 1 / ms.compute_one_group_velocity_component(mp.Vector3(0, 0, 1), 1)
e_mpb = reshape(ms.get_efield(1),(nx_mpb,ny_mpb,3))
d_mpb = reshape(ms.get_dfield(1),(nx_mpb,ny_mpb,3))
h_mpb = reshape(ms.get_hfield(1),(nx_mpb,ny_mpb,3))
S_mpb = reshape(ms.get_poynting(1),(nx_mpb,ny_mpb,3))
U_mpb = reshape(ms.get_tot_pwr(1),(nx_mpb,ny_mpb))
# H_mpb = vec(reshape(ms.get_eigenvectors(1,1),(nx_mpb*ny_mpb,2)))
H_mpb_raw = reshape(ms.get_eigenvectors(1,1), (ny_mpb,nx_mpb,2))
H_mpb = zeros(ComplexF64,(3,nx_mpb,ny_mpb,1)) #Array{ComplexF64,4}(undef,(3,ny_mpb,ny_mpb,1))
for i=1:nx_mpb
    for j=1:ny_mpb
        H_mpb[1,i,j,1] = H_mpb_raw[j,i,1]
        H_mpb[2,i,j,1] = H_mpb_raw[j,i,2]
    end
end
SHM3 = SHermitianCompact{3,Float64,6}
z_mpb = [0.,]
nz_mpb = 1
# ε⁻¹_mpb = [SHM3([ms.get_epsilon_inverse_tensor_point(mp.Vector3(x_mpb[i],y_mpb[j],z_mpb[k]))[a][b] for a=1:3,b=1:3]) for i=1:nx_mpb,j=1:ny_mpb,k=1:nz_mpb]
ε⁻¹_mpb = [SHM3([get(get(ms.get_epsilon_inverse_tensor_point(mp.Vector3(x_mpb[i],y_mpb[j],z_mpb[k])),a-1),b-1) for a=1:3,b=1:3]) for i=1:nx_mpb,j=1:ny_mpb,k=1:nz_mpb]
ε_mpb = [SHM3(inv(ε⁻¹_mpb[i,j,k])) for i=1:nx_mpb,j=1:ny_mpb,k=1:nz_mpb]
e_mpb = [SVector(e_mpb[i,j,:]...) for i=1:nx_mpb,j=1:ny_mpb]
d_mpb = [SVector(d_mpb[i,j,:]...) for i=1:nx_mpb,j=1:ny_mpb]
h_mpb = [SVector(h_mpb[i,j,:]...) for i=1:nx_mpb,j=1:ny_mpb]
S_mpb = [SVector(S_mpb[i,j,:]...) for i=1:nx_mpb,j=1:ny_mpb]

Nx = nx_mpb
Ny = ny_mpb
Nz = 1
kz = k_mpb
ε⁻¹ = [ε⁻¹_mpb[i,j,k][a,b] for a=1:3,b=1:3,i=1:Nx,j=1:Ny,k=1:Nz];
ω_mpb = 1/λ;
ω = ω_mpb;
Ha = copy(H_mpb[1:2,:,:,:]);
H = copy(vec(Ha));
# Nz = 1
# Neigs = 1
# N = *(Nx,Ny,Nz)
# s = ridge_wg(w,t_core,edge_gap,n_core,n_subs,g)
# ε⁻¹ = εₛ⁻¹(s,g)
# ε = SHM3.(inv.(ε⁻¹))
