using Revise, LinearAlgebra, StaticArrays, PyCall  #FFTW, BenchmarkTools, LinearMaps, IterativeSolvers, Roots, GeometryPrimitives
using OptiMode
SHM3 = SHermitianCompact{3,Float64,6}
LinearAlgebra.ldiv!(c,A::LinearMaps.LinearMap,b) = mul!(c,A',b)
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
)

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
ε⁻¹_mpb = [SHM3([ms.get_epsilon_inverse_tensor_point(mp.Vector3(x_mpb[i],y_mpb[j],z_mpb[k]))[a][b] for a=1:3,b=1:3]) for i=1:nx_mpb,j=1:ny_mpb,k=1:nz_mpb]
ε_mpb = [SHM3(inv(ε⁻¹_mpb[i,j,k])) for i=1:nx_mpb,j=1:ny_mpb,k=1:nz_mpb]
e_mpb = [SVector(e_mpb[i,j,:]...) for i=1:nx_mpb,j=1:ny_mpb]
d_mpb = [SVector(d_mpb[i,j,:]...) for i=1:nx_mpb,j=1:ny_mpb]
h_mpb = [SVector(h_mpb[i,j,:]...) for i=1:nx_mpb,j=1:ny_mpb]
S_mpb = [SVector(S_mpb[i,j,:]...) for i=1:nx_mpb,j=1:ny_mpb]

Nx = nx_mpb
Ny = ny_mpb
k = SVector(0.,0.,k_mpb[1])
g = MaxwellGrid(Δx,Δy,Nx,Ny)
ds = MaxwellData(k,g);

##

# Nz = 1
# Neigs = 1
# N = *(Nx,Ny,Nz)
# s = ridge_wg(w,t_core,edge_gap,n_core,n_subs,g)
# ε⁻¹ = εₛ⁻¹(s,g)
# ε = SHM3.(inv.(ε⁻¹))



# U_mpb_plot = heatmap(
#             x_mpb,
#             y_mpb,
#             transpose(U_mpb),
#             c=cgrad(:cherry),
#             aspect_ratio=:equal,
#             legend=false,
#             colorbar = true,
#             # clim = ( 0., max(ε_mpb...) ),
# )






# H1 = randn(ComplexF64,3,Neigs,Nx,Ny,Nz) #Array{ComplexF64}(undef,3,Neigs,Nx,Ny,Nz)
# Iₛₚ = CartesianIndices(size(H1)[3:5])
# HSA = [SVector{3,ComplexF64}(randn(ComplexF64,3)...) for p=1:Neigs,i=1:Nx,j=1:Ny,k=1:Nz];
# KpG = [KVec(SVector{3,Float64}(randn(Float64,3)...)) for p=1:1,i=1:Nx,j=1:Ny,k=1:Nz];
# ε⁻¹ = [SHM3(randn(Float64,6)) for p=1:1,i=1:Nx,j=1:Ny,k=1:Nz];
# P = plan_fft!(reinterpret(ComplexF64,HSA),(2:4));
# iP = plan_ifft!(reinterpret(ComplexF64,HSA),(2:4));
# HSV = vec(copy(HSA))

# HA = [HSA[p,i,j,k][d] for p=1:Neigs,i=1:Nx,j=1:Ny,k=1:Nz,d=1:3];

# 3D FFTW benchmarks for Nx=64,Ny=128,Nz=256 ##

# CPU #
# @btime fft!(H1,(3:5)); # 601.221 ms (28 allocations: 2.89 KiB)
# @btime fft!(H2,(1:3)); # 970.900 ms (28 allocations: 2.89 KiB)
# @btime fft!(reinterpret(ComplexF64,$HSA),(2:4)); #541.107 ms (29 allocations: 2.78 KiB)
# size(reinterpret(ComplexF64,H2M)) # (192, 128, 256, 5) --> mixes spatial and non-spatial dimension, non-spatial dims should go first

# non-Mutating Operators

# function t2c(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     Hout = Array{ComplexF64}(undef,(3,size(Hin)[2:end]...))
#     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @inbounds scale = ds.kpG[i,j,k].mag
#         @inbounds Hout[1,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[1] ) * scale  
#         @inbounds Hout[2,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].m[2] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[2] ) * scale  
#         @inbounds Hout[3,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].m[3] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[3] ) * scale 
#     end
#     return Hout
# end

# function c2t(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     Hout = Array{ComplexF64}(undef,(2,size(Hin)[2:end]...))
#     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @inbounds Hout[1,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].m[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].m[3] 
#         @inbounds Hout[2,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].n[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].n[3] 
#     end
#     return Hout
# end

# function zcross_t2c(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     Hout = zeros(ComplexF64,(3,size(Hin)[2:end]...))
#     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @inbounds Hout[1,i,j,k] = -Hin[1,i,j,k] * ds.kpG[i,j,k].m[2] - Hin[2,i,j,k] * ds.kpG[i,j,k].n[2]   
#         @inbounds Hout[2,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[1]  
#     end
#     return Hout
# end

# function kcross_t2c(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     Hout = Array{ComplexF64}(undef,(3,size(Hin)[2:end]...))
#     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @inbounds scale = -ds.kpG[i,j,k].mag
#         @inbounds Hout[1,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].n[1] - Hin[2,i,j,k] * ds.kpG[i,j,k].m[1] ) * scale  
#         @inbounds Hout[2,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].n[2] - Hin[2,i,j,k] * ds.kpG[i,j,k].m[2] ) * scale  
#         @inbounds Hout[3,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].n[3] - Hin[2,i,j,k] * ds.kpG[i,j,k].m[3] ) * scale 
#     end
#     return Hout
# end

# function kcross_c2t(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     Hout = Array{ComplexF64}(undef,(2,size(Hin)[2:end]...))
#     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @inbounds scale = ds.kpG[i,j,k].mag
#         @inbounds at1 = Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].m[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].m[3]
#         @inbounds at2 = Hin[1,i,j,k] * ds.kpG[i,j,k].n[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].n[3]
#         @inbounds Hout[1,i,j,k] =  -at2 * scale 
#         @inbounds Hout[2,i,j,k] =  at1 * scale 
#     end
#     return Hout
# end

# function kcrossinv_t2c(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     Hout = Array{ComplexF64}(undef,(3,size(Hin)[2:end]...))
#     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @inbounds scale = 1 / ds.kpG[i,j,k].mag
#         @inbounds Hout[1,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].n[1] - Hin[2,i,j,k] * ds.kpG[i,j,k].m[1] ) * scale  
#         @inbounds Hout[2,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].n[2] - Hin[2,i,j,k] * ds.kpG[i,j,k].m[2] ) * scale  
#         @inbounds Hout[3,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].n[3] - Hin[2,i,j,k] * ds.kpG[i,j,k].m[3] ) * scale 
#     end
#     return Hout
# end

# function kcrossinv_c2t(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     Hout = Array{ComplexF64}(undef,(2,size(Hin)[2:end]...))
#     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @inbounds scale = -1 / ds.kpG[i,j,k].mag
#         @inbounds at1 = Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].m[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].m[3]
#         @inbounds at2 = Hin[1,i,j,k] * ds.kpG[i,j,k].n[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].n[3]
#         @inbounds Hout[1,i,j,k] =  -at2 * scale 
#         @inbounds Hout[2,i,j,k] =  at1 * scale 
#     end
#     return Hout
# end

# function ε⁻¹_dot(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     Hout = similar(Hin)
#     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @inbounds Hout[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,1]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,1]*Hin[3,i,j,k]
#         @inbounds Hout[2,i,j,k] =  ε⁻¹[i,j,k][1,2]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,2]*Hin[3,i,j,k]
#         @inbounds Hout[3,i,j,k] =  ε⁻¹[i,j,k][1,3]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,3]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,3]*Hin[3,i,j,k]
#         # @inbounds Hout[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][1,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][1,3]*Hin[3,i,j,k]
#         # @inbounds Hout[2,i,j,k] =  ε⁻¹[i,j,k][2,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][2,3]*Hin[3,i,j,k]
#         # @inbounds Hout[3,i,j,k] =  ε⁻¹[i,j,k][3,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][3,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,3]*Hin[3,i,j,k]
#     end
#     return Hout
# end

# function ε_dot_approx(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     Hout = similar(Hin)
#     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @inbounds ε_ave = 3 / tr(ε⁻¹[i,j,k])
#         @inbounds Hout[1,i,j,k] =  ε_ave * Hin[1,i,j,k]
#         @inbounds Hout[2,i,j,k] =  ε_ave * Hin[2,i,j,k]
#         @inbounds Hout[3,i,j,k] =  ε_ave * Hin[3,i,j,k]
#     end
#     return Hout
# end

# function M(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
#     d = ds.𝓕 * kcross_t2c(Hin,ds);
#     e = ε⁻¹_dot(d,ε⁻¹,ds); # (-1/(π)) .*
#     kcross_c2t(ds.𝓕⁻¹ * e,ds)
# end

# function M(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
#     HinA = reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
#     HoutA = M(HinA,ε⁻¹,ds)
#     return vec(HoutA)
# end

# M̂(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> M(H,ε⁻¹,ds)::AbstractArray{ComplexF64,1},(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=false)

# function P(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
#     e = ds.𝓕⁻¹ * kcrossinv_t2c(Hin,ds);
#     d = ε_dot_approx(e,ε⁻¹,ds); # (-1/(π)) .*
#     kcrossinv_c2t(ds.𝓕 * d,ds)
# end

# function P(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
#     HinA = reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
#     HoutA = P(HinA,ε⁻¹,ds)
#     return vec(HoutA)
# end

# P̂(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> P(H,ε⁻¹,ds)::AbstractArray{ComplexF64,1},(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=false)

# function Mₖ(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
#     d = ds.𝓕 * zcross_t2c(Hin,ds);
#     e = ε⁻¹_dot(d,ε⁻¹,ds); # (-1/(π)) .*
#     kcross_c2t(ds.𝓕⁻¹ * e,ds)
# end

# function Mₖ(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
#     HinA = reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
#     HoutA = Mₖ(HinA,ε⁻¹,ds)
#     return -vec(HoutA)
# end

# M̂ₖ(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> Mₖ(H,ε⁻¹,ds)::AbstractArray{ComplexF64,1},(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=false)

# function solve_ω(k::SVector{3},ε⁻¹::Array{SHM3,3},g::MaxwellGrid;neigs=1,eigind=1,maxiter=3000,tol=1e-7)
#     ds = MaxwellData(k,g)
#     res = IterativeSolvers.lobpcg(M̂(ε⁻¹,ds),false,neigs;P=P̂(ε⁻¹,ds),maxiter,tol)
#     H =  res.X #[:,eigind]                       # eigenmode wavefn. magnetic fields in transverse pol. basis
#     ω =  √(real(res.λ[eigind]))                     # eigenmode temporal freq.,  neff = kz / ω, kz = k[3] 
#     # ωₖ =   real( ( H' * M̂ₖ(ε⁻¹,ds) * H )[1]) / ω       # ωₖ/∂kz = group velocity = c / ng, c = 1 here
#     ωₖ =   real( ( H[:,eigind]' * M̂ₖ(ε⁻¹,ds) * H[:,eigind] )[1]) / ω       # ωₖ/∂kz = group velocity = c / ng, c = 1 here
#     return H, ω, ωₖ
# end

# function kpG(k::SVector{3,Float64},g::MaxwellGrid)::Array{KVec,3}
#     [KVec(k-gx-gy-gz) for gx=g.gx, gy=g.gy, gz=g.gz]
# end

# """
# modified solve_ω version for Newton solver, which wants (x -> f(x), f(x)/f'(x)) as input to solve f(x) = 0
# """
# function _solve_Δω²(k,ωₜ,ε⁻¹::Array{SHM3,3},ds::MaxwellData;neigs=1,eigind=1,maxiter=10000,tol=1e-8)
#     ds.k = SVector(0.,0.,k)
#     ds.kpG .= kpG(SVector(0.,0.,k),ds.grid)
#     # res = IterativeSolvers.lobpcg(M̂(ε⁻¹,ds),false,ds.H⃗;P=P̂(ε⁻¹,ds),maxiter,tol)
#     res = IterativeSolvers.lobpcg(M̂!(ε⁻¹,ds),false,ds.H⃗;P=P̂!(ε⁻¹,ds),maxiter,tol)
#     H =  res.X #[:,eigind]                      # eigenmode wavefn. magnetic fields in transverse pol. basis
#     ω² =  (real(res.λ[eigind]))                # eigenmode temporal freq.,  neff = kz / ωₖ, kz = k[3] 
#     Δω² = ω² - ωₜ^2
#     ω²ₖ =   2 * real( ( H[:,eigind]' * M̂ₖ(ε⁻¹,ds) * H[:,eigind] )[1])       # ωₖ/∂kz = group velocity = c / ng, c = 1 here
#     ds.H⃗ .= H
#     ds.ω² = ω²
#     ds.ω²ₖ = ω²ₖ
#     return Δω² , Δω² / ω²ₖ
# end

# function solve_k(ω::Float64,k₀::Float64,ε⁻¹::Array{SHM3,3},g::MaxwellGrid;neigs=1,eigind=1,maxiter=10000,tol=1e-8)
#     ds = MaxwellData(k₀,g)
#     kz = Roots.find_zero(k -> _solve_Δω²(k,ω,ε⁻¹,ds;neigs,eigind,maxiter,tol), k₀, Roots.Newton())
#     ds.ω = √ds.ω²
#     ds.ωₖ = ds.ω²ₖ / ( 2 * ds.ω )
#     return kz, ds
# end

# function solve_k(ω::Float64,ε⁻¹::Array{SHM3,3},ds::MaxwellData;neigs=1,eigind=1,maxiter=10000,tol=1e-8)
#     kz = Roots.find_zero(k -> _solve_Δω²(k,ω,ε⁻¹,ds;neigs,eigind,maxiter,tol), ds.k[3], Roots.Newton())
#     ds.ω = √ds.ω²
#     ds.ωₖ = ds.ω²ₖ / ( 2 * ds.ω )
#     return kz
# end

# function compare_fields(f_mpb,f,xlim,ylim)
#     hm_f_mpb_real = [ heatmap(x_mpb,y_mpb,[real(f_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
#     hm_f_mpb_imag = [ heatmap(x_mpb,y_mpb,[imag(f_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]
    
#     hm_f_real = [ heatmap(x_mpb,y_mpb,[real(f[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
#     hm_f_imag = [ heatmap(x_mpb,y_mpb,[imag(f[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]
    
#     hm_f_ratio_real = [ heatmap(x_mpb,y_mpb,[real(f[ix,i,j])/real(f_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
#     hm_f_ratio_imag = [ heatmap(x_mpb,y_mpb,[imag(f[ix,i,j])/imag(f_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]
    
#     l = @layout [   a   b   c 
#                     d   e   f
#                     g   h   i
#                     k   l   m    
#                     n   o   p
#                     q   r   s    ]
#     plot(hm_f_mpb_real...,
#         hm_f_mpb_imag...,
#         hm_f_real...,
#         hm_f_imag...,
#         hm_f_ratio_real...,
#         hm_f_ratio_imag...,
#         layout=l,
#         size = (1300,1300),
#     ) 
# end

# # Mutating Operators

# # function t2c!(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
# #     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
# #         @inbounds scale = ds.kpG[i,j,k].mag
# #         @inbounds ds.e[1,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[1] ) * scale  
# #         @inbounds ds.e[2,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].m[2] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[2] ) * scale  
# #         @inbounds ds.e[3,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].m[3] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[3] ) * scale 
# #     end
# #     return ds.e
# # end

# # function c2t!(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
# #     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
# #         @inbounds ds.e[1,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].m[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].m[3] 
# #         @inbounds ds.e[2,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].n[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].n[3]
# #         # @inbounds ds.e[3,i,j,k] =  0.0 
# #     end
# #     return ds.e
# # end

# # function zcross_t2c!(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
# #     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
# #         @inbounds ds.e[1,i,j,k] = -Hin[1,i,j,k] * ds.kpG[i,j,k].m[2] - Hin[2,i,j,k] * ds.kpG[i,j,k].n[2]   
# #         @inbounds ds.e[2,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[1]  
# #     end
# #     return ds.e
# # end


# ###############################################################################
# function kcross_t2c!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         scale = -ds.kpG[i,j,k].mag #-ds.kpG[i,j,k].mag
#         ds.d[1,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[1] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[1] ) * scale  
#         ds.d[2,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[2] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[2] ) * scale  
#         ds.d[3,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[3] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[3] ) * scale 
#     end
#     return ds.d
# end

# function kcross_c2t!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         scale = ds.kpG[i,j,k].mag
#         at1 = ds.e[1,i,j,k] * ds.kpG[i,j,k].m[1] + ds.e[2,i,j,k] * ds.kpG[i,j,k].m[2] + ds.e[3,i,j,k] * ds.kpG[i,j,k].m[3]
#         at2 = ds.e[1,i,j,k] * ds.kpG[i,j,k].n[1] + ds.e[2,i,j,k] * ds.kpG[i,j,k].n[2] + ds.e[3,i,j,k] * ds.kpG[i,j,k].n[3]
#         ds.H[1,i,j,k] =  -at2 * scale 
#         ds.H[2,i,j,k] =  at1 * scale
#     end
#     return ds.H
# end

# function kcrossinv_t2c!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         scale = 1 / ds.kpG[i,j,k].mag
#         ds.e[1,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[1] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[1] ) * scale  
#         ds.e[2,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[2] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[2] ) * scale  
#         ds.e[3,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[3] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[3] ) * scale 
#     end
#     return ds.e
# end

# function kcrossinv_c2t!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         scale = -1 / ds.kpG[i,j,k].mag
#         at1 = ds.d[1,i,j,k] * ds.kpG[i,j,k].m[1] + ds.d[2,i,j,k] * ds.kpG[i,j,k].m[2] + ds.d[3,i,j,k] * ds.kpG[i,j,k].m[3]
#         at2 = ds.d[1,i,j,k] * ds.kpG[i,j,k].n[1] + ds.d[2,i,j,k] * ds.kpG[i,j,k].n[2] + ds.d[3,i,j,k] * ds.kpG[i,j,k].n[3]
#         ds.H[1,i,j,k] =  -at2 * scale 
#         ds.H[2,i,j,k] =  at1 * scale 
#     end
#     return ds.H
# end

# # function kcrossinv_t2c!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
# #     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
# #         scale = 1 / ds.kpG[i,j,k].mag
# #         ds.einv[1,i,j,k] = ( ds.Hinv[1,i,j,k] * ds.kpG[i,j,k].n[1] - ds.Hinv[2,i,j,k] * ds.kpG[i,j,k].m[1] ) * scale  
# #         ds.einv[2,i,j,k] = ( ds.Hinv[1,i,j,k] * ds.kpG[i,j,k].n[2] - ds.Hinv[2,i,j,k] * ds.kpG[i,j,k].m[2] ) * scale  
# #         ds.einv[3,i,j,k] = ( ds.Hinv[1,i,j,k] * ds.kpG[i,j,k].n[3] - ds.Hinv[2,i,j,k] * ds.kpG[i,j,k].m[3] ) * scale 
# #     end
# #     return ds.einv
# # end

# # function kcrossinv_c2t!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
# #     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
# #         scale = -1 / ds.kpG[i,j,k].mag
# #         at1 = ds.dinv[1,i,j,k] * ds.kpG[i,j,k].m[1] + ds.dinv[2,i,j,k] * ds.kpG[i,j,k].m[2] + ds.dinv[3,i,j,k] * ds.kpG[i,j,k].m[3]
# #         at2 = ds.dinv[1,i,j,k] * ds.kpG[i,j,k].n[1] + ds.dinv[2,i,j,k] * ds.kpG[i,j,k].n[2] + ds.dinv[3,i,j,k] * ds.kpG[i,j,k].n[3]
# #         ds.Hinv[1,i,j,k] =  -at2 * scale 
# #         ds.Hinv[2,i,j,k] =  at1 * scale 
# #     end
# #     return ds.Hinv
# # end


# function ε⁻¹_dot!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         ds.e[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*ds.d[1,i,j,k] + ε⁻¹[i,j,k][2,1]*ds.d[2,i,j,k] + ε⁻¹[i,j,k][3,1]*ds.d[3,i,j,k]
#         ds.e[2,i,j,k] =  ε⁻¹[i,j,k][1,2]*ds.d[1,i,j,k] + ε⁻¹[i,j,k][2,2]*ds.d[2,i,j,k] + ε⁻¹[i,j,k][3,2]*ds.d[3,i,j,k]
#         ds.e[3,i,j,k] =  ε⁻¹[i,j,k][1,3]*ds.d[1,i,j,k] + ε⁻¹[i,j,k][2,3]*ds.d[2,i,j,k] + ε⁻¹[i,j,k][3,3]*ds.d[3,i,j,k]
#         # ds.e[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][1,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][1,3]*Hin[3,i,j,k]
#         # ds.e[2,i,j,k] =  ε⁻¹[i,j,k][2,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][2,3]*Hin[3,i,j,k]
#         # ds.e[3,i,j,k] =  ε⁻¹[i,j,k][3,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][3,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,3]*Hin[3,i,j,k]
#     end
#     return ds.e
# end

# # function ε_dot_approx!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
# #     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
# #         ε_ave = 3 / tr(ε⁻¹[i,j,k])
# #         ds.dinv[1,i,j,k] =  ε_ave * ds.einv[1,i,j,k]
# #         ds.dinv[2,i,j,k] =  ε_ave * ds.einv[2,i,j,k]
# #         ds.dinv[3,i,j,k] =  ε_ave * ds.einv[3,i,j,k]
# #     end
# #     return ds.dinv
# # end

# function ε_dot_approx!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         ε_ave = 3 / tr(ε⁻¹[i,j,k])
#         ds.d[1,i,j,k] =  ε_ave * ds.e[1,i,j,k]
#         ds.d[2,i,j,k] =  ε_ave * ds.e[2,i,j,k]
#         ds.d[3,i,j,k] =  ε_ave * ds.e[3,i,j,k]
#     end
#     return ds.d
# end

# function M!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
#     kcross_t2c!(ds);
#     # ds.𝓕! * ds.d;
#     mul!(ds.d,ds.𝓕!,ds.d);
#     ε⁻¹_dot!(ε⁻¹,ds);
#     # ds.𝓕⁻¹! * ds.e;
#     mul!(ds.e,ds.𝓕⁻¹!,ds.e)
#     kcross_c2t!(ds)
# end

# function M!(Hout::AbstractArray{ComplexF64,1},Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
#     # copyto!(ds.H,reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz)))
#     @inbounds ds.H .= reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
#     M!(ε⁻¹,ds);
#     # copyto!(Hout,vec(ds.H))
#     @inbounds Hout .= vec(ds.H)
# end

# function M̂!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)
#     function f!(y::AbstractArray{ComplexF64,1},x::AbstractArray{ComplexF64,1})::AbstractArray{ComplexF64,1}
#         M!(y,x,ε⁻¹,ds)    
#     end
#     return LinearMap{ComplexF64}(f!,(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=true)
# end

# # function P!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
# #     kcrossinv_t2c!(ds);
# #     # ds.𝓕⁻¹! * ds.e;
# #     mul!(ds.einv,ds.𝓕⁻¹!,ds.einv)
# #     ε_dot_approx!(ε⁻¹,ds);
# #     # ds.𝓕! * ds.d;
# #     mul!(ds.dinv,ds.𝓕!,ds.dinv);
# #     kcrossinv_c2t!(ds)
# # end

# # function P!(Hout::AbstractArray{ComplexF64,1},Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
# #     # copyto!(ds.H,reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz)))
# #     @inbounds ds.Hinv .= reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
# #     P!(ε⁻¹,ds);
# #     # copyto!(Hout,vec(ds.H))
# #     @inbounds Hout .= vec(ds.Hinv)
# # end

# function P!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
#     kcrossinv_t2c!(ds);
#     # ds.𝓕⁻¹! * ds.e;
#     mul!(ds.e,ds.𝓕⁻¹!,ds.e)
#     ε_dot_approx!(ε⁻¹,ds);
#     # ds.𝓕! * ds.d;
#     mul!(ds.d,ds.𝓕!,ds.d);
#     kcrossinv_c2t!(ds)
# end

# function P!(Hout::AbstractArray{ComplexF64,1},Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
#     # copyto!(ds.H,reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz)))
#     @inbounds ds.H .= reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
#     P!(ε⁻¹,ds);
#     # copyto!(Hout,vec(ds.H))
#     @inbounds Hout .= vec(ds.H)
# end

# function P̂!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)
#     function fp!(y::AbstractArray{ComplexF64,1},x::AbstractArray{ComplexF64,1})::AbstractArray{ComplexF64,1}
#         P!(y,x,ε⁻¹,ds)    
#     end
#     return LinearMap{ComplexF64}(fp!,(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=true)
# end


# ######################
# ######################3
# ########################
# ##

# eigind = 1;
# k₀ = 1.4 # k_mpb[1] # 1.47625...
# ds = MaxwellData(k₀,g)
# H, ω, ωₖ = solve_ω(SVector(0.,0.,k₀),ε⁻¹_mpb,g;neigs=1,eigind=1,maxiter=3000,tol=1e-7)

# ##

# # HA = reshape(ds.H,(2,ds.Nx,ds.Ny,ds.Nz))
# HA = M!(ε⁻¹_mpb,ds)
# # M!(ε⁻¹_mpb,ds)

# ##
# k₀ = 1.3 # k_mpb[1] # 1.47625...
# ds = MaxwellData(k₀,g)
# ω_mpb = k_mpb[1] / neff_mpb[1]
# @btime solve_k($ω_mpb,$ε⁻¹_mpb,$ds)
# # kz = solve_k(ω_mpb,ε⁻¹_mpb,ds)
# ##
# ω = ds.ω
# ωₖ = ds.ωₖ
# H = copy(reshape(ds.H⃗[:,1],(2,ds.Nx,ds.Ny,ds.Nz)))
# # HAout = similar(HAin)

# # HA = M(HAin,ε⁻¹_mpb,ds)
# e = ε⁻¹_dot(ds.𝓕 * kcross_t2c(HA,ds),ε⁻¹_mpb,ds)
# neff = kz / ω
# neff_err = neff - neff_mpb[1]
# neff_err_rel = neff_err / neff_mpb[1]
# println("k = $kz")
# println("ω = $ωₖ")
# println("ω_mpb = $ω_mpb")
# println("neff = $neff")
# println("neff_mpb = $(neff_mpb[1])")
# println("neff_err = $neff_err")
# println("neff_err_rel = $neff_err_rel")

# ng = 1 / ωₖ
# ng_err = ng - ng_mpb
# ng_err_rel = ng_err / ng_mpb
# println("ng = $ng")
# println("ng_mpb = $ng_mpb")
# println("ng_err = $ng_err")
# println("ng_err_rel = $ng_err_rel")

# compare_fields(e_mpb, e,(-3,3),(-2,2))



# # HAc = copy(HA)
# # MHAc = M(HA,ε⁻¹_mpb,ds)
# # # vec(MHAc ./ HAc)
# # # mean( vec(MHAc ./ HAc))
# # ##
# # HAout = similar(HAc)
# # MHAc2 = M!(HAout,HAc,ε⁻¹_mpb,ds)
# # ##
# # vec(MHAc2 ./ HAc)
# # ##
# # vec(HAout ./ HAc)
# # var( vec(MHAc ./ HAc))


# # abs2.( HAout - MHAc )


# ##

# function testfn2(kz,ε⁻¹_mpb,g)
#     ds = MaxwellData(kz,g)
#     res = IterativeSolvers.lobpcg(M̂!(ε⁻¹_mpb,ds),false,1;P=P̂!(ε⁻¹_mpb,ds),maxiter=1000,tol=1e-7)
# end

# ##

# @btime testfn2(1.4,$ε⁻¹_mpb,$g)

# ##
# k₀ = k_mpb[1] # 1.4 # k_mpb[1] # 1.47625...
# ds = MaxwellData(k₀,g)
# res = IterativeSolvers.lobpcg(M̂!(ε⁻¹_mpb,ds),false,1;P=P̂!(ε⁻¹_mpb,ds),maxiter=1000,tol=1e-7)
# # res = IterativeSolvers.lobpcg(M̂!(ε⁻¹_mpb,ds),false,1;maxiter=3000,tol=1e-7)


# H =  res.X #[:,eigind]                       # eigenmode wavefn. magnetic fields in transverse pol. basis
# ω =  √(abs(real(res.λ[eigind])))                     # eigenmode temporal freq.,  neff = kz / ω, kz = k[3] 
# #ωₖ =   real( ( H[:,eigind]' * M̂ₖ(ε⁻¹,ds) * H[:,eigind] )[1]) / ω       # ωₖ/∂kz = group velocity = c / ng, c = 1 here
# ω_mpb = k_mpb[1] / neff_mpb[1]
# kz = k₀ #solve_k(ω_mpb,ε⁻¹_mpb,ds)
# HA = reshape(H,(2,ds.Nx,ds.Ny,ds.Nz))
# e = ε⁻¹_dot(ds.𝓕 * kcross_t2c(HA,ds),ε⁻¹_mpb,ds)
# neff = kz / ω
# neff_err = neff - neff_mpb[1]
# neff_err_rel = neff_err / neff_mpb[1]
# println("k = $kz")
# println("ω = $ωₖ")
# println("ω_mpb = $ω_mpb")
# println("neff = $neff")
# println("neff_mpb = $(neff_mpb[1])")
# println("neff_err = $neff_err")
# println("neff_err_rel = $neff_err_rel")

# ng = 1 / ωₖ
# ng_err = ng - ng_mpb
# ng_err_rel = ng_err / ng_mpb
# println("ng = $ng")
# println("ng_mpb = $ng_mpb")
# println("ng_err = $ng_err")
# println("ng_err_rel = $ng_err_rel")

# compare_fields(e_mpb, e,(-3,3),(-2,2))


# ##

# # @btime kcross_t2c!($Hout1,$HA,$ds);
# # @btime kcross_c2t!($Hout1,$HA,$ds);
# # @btime epsinv_dot!($Hout1,$HA,$ε⁻¹,$ds);
# # @btime $𝓕 * $Hout1;
# # @btime M!($Hout,$Hin,$ε⁻¹,$𝓕,$𝓕⁻¹,$ds,$Hw1,$Hw2); # 1.335 ms (0 allocations: 0 bytes)
# # @btime M!($Hout_v,$Hin_v,$ε⁻¹,$𝓕,$𝓕⁻¹,$ds,$Hw1,$Hw2); # 1.211 ms (4 allocations: 256 bytes)
# N = 3* ds.Nx * ds.Ny * ds.Nz
# f = (Hout_v,Hin_v) -> M!(Hout_v,Hin_v,ε⁻¹_mpb,𝓕,𝓕⁻¹,ds,Hw1,Hw2)
# Mop = LinearMap{ComplexF64}(f,N,ishermitian=true,ismutating=true) 
# # @btime $Mop * $Hin_v  # 1.342 ms (6 allocations: 768.33 KiB
# res = IterativeSolvers.lobpcg(Mop,false,1;maxiter=3000,tol=1e-7)
# @show λ_sol = res.λ
# dsol_v = res.X[:,1]
# dsol = reshape(dsol_v,(3,size(ds.kpG)...)) # reinterpret(SVector{3,ComplexF64}, ... )[1,:,:,1]

# d =   copy(dsol)
# # H = permutedims(fftshift(reshape(vec(reinterpret(SVector{2,ComplexF64},permutedims(ms.get_eigenvectors(1,1),[2,1,3]))),(ny_mpb,nx_mpb))),[2,1])

# # function d_from_H2(H,ds::MaxwellData)
# #     kpG2 = [KVec(k-SVector(ggx, 0., 0.)-SVector(0., ggy, 0.)) for ggx=( twopi .* fftfreq(Nx,Nx/Δx)), ggy=(twopi  .* fftfreq(Ny,Ny/Δy)) ]
# #     d_recip = [ kcross_t2c(H[i,j],kpG2[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
# #     temp =  (-1/(2π)) .* fft( reinterpret( ComplexF64, reshape( d_recip , (1,ds.Nx,ds.Ny) )), (2,3))
# #     return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
# # end

# # d = d_from_H2( H, ds )

# xlim =(-3,3)
# ylim =(-2,2)

# # xlim =(-0.5,0.5)
# # ylim =(-0.5,0.5)

# hm_d_mpb_real = [ heatmap(x_mpb,y_mpb,[real(d_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_d_mpb_imag = [ heatmap(x_mpb,y_mpb,[imag(d_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# hm_d_real = [ heatmap(x_mpb,y_mpb,[real(d[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_d_imag = [ heatmap(x_mpb,y_mpb,[imag(d[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# hm_d_ratio_real = [ heatmap(x_mpb,y_mpb,[real(d[ix,i,j])/real(d_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_d_ratio_imag = [ heatmap(x_mpb,y_mpb,[imag(d[ix,i,j])/imag(d_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# l = @layout [   a   b   c 
#                 d   e   f
#                 g   h   i
#                 k   l   m    
#                 n   o   p
#                 q   r   s    ]


# plot(
#     hm_d_mpb_real...,
#     hm_d_mpb_imag...,
#     hm_d_real...,
#     hm_d_imag...,
#     hm_d_ratio_real...,
#     hm_d_ratio_imag...,
#     layout=l,
#     size = (1300,1300),
# )

# ##

# function h_from_H!(h::AbstractArray{ComplexF64,4},H::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     t2c!(h,H,ds);
#     𝓕 * h
# end
# h_temp = copy(H_mpb)
# # h =   h_from_H!(h_temp, H_mpb, ds) .* (1/(2π)) 
# h =   h_from_H!(h_temp, Hsol, ds) .* (1/(2π)) 

# xlim =(-3,3)
# ylim =(-2,2)

# # xlim =(-0.5,0.5)
# # ylim =(-0.5,0.5)

# hm_h_mpb_real = [ heatmap(x_mpb,y_mpb,[real(h_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_h_mpb_imag = [ heatmap(x_mpb,y_mpb,[imag(h_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# hm_h_real = [ heatmap(x_mpb,y_mpb,[real(h[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_h_imag = [ heatmap(x_mpb,y_mpb,[imag(h[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# hm_h_ratio_real = [ heatmap(x_mpb,y_mpb,[real(h[ix,i,j])/real(h_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_h_ratio_imag = [ heatmap(x_mpb,y_mpb,[imag(h[ix,i,j])/imag(h_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]


# l = @layout [   a   b   c 
#                 d   e   f
#                 g   h   i
#                 k   l   m    
#                 n   o   p
#                 q   r   s    ]


# plot(
#     hm_h_mpb_real...,
#     hm_h_mpb_imag...,
#     hm_h_real...,
#     hm_h_imag...,
#     hm_h_ratio_real...,
#     hm_h_ratio_imag...,
#     layout=l,
#     size = (1300,1300),
# )


# ##

# # function d_from_H!(d::AbstractArray{ComplexF64,4},H::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
# #     kcross_t2c!(d,H,ds);
# #     𝓕 * d
# # end
# # d_temp = copy(H_mpb)
# # d =   d_from_H!(d_temp, H_mpb, ds) .* (-1/(π)) 

# xlim =(-3,3)
# ylim =(-2,2)

# hm_d_mpb_real = [ heatmap(x_mpb,y_mpb,[real(d_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_d_mpb_imag = [ heatmap(x_mpb,y_mpb,[imag(d_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# hm_d_real = [ heatmap(x_mpb,y_mpb,[real(d[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_d_imag = [ heatmap(x_mpb,y_mpb,[imag(d[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# hm_d_ratio_real = [ heatmap(x_mpb,y_mpb,[real(d[ix,i,j])/real(d_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_d_ratio_imag = [ heatmap(x_mpb,y_mpb,[imag(d[ix,i,j])/imag(d_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# l = @layout [   a   b   c 
#                 d   e   f
#                 g   h   i
#                 k   l   m    
#                 n   o   p
#                 q   r   s    ]


# plot(
#     hm_d_mpb_real...,
#     hm_d_mpb_imag...,
#     hm_d_real...,
#     hm_d_imag...,
#     hm_d_ratio_real...,
#     hm_d_ratio_imag...,
#     layout=l,
#     size = (1300,1300),
# )


# ##

# function e_from_H!(e::AbstractArray{ComplexF64,4},d::AbstractArray{ComplexF64,4},H::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     kcross_t2c!(d,H,ds);
#     𝓕 * d;
#     epsinv_dot!(e,(-1/(π)) .* d,ε⁻¹,ds);
# end

# d_temp = copy(H_mpb)
# e_temp = copy(H_mpb)
# e = e_from_H!(e_temp,d_temp,H_mpb,ε⁻¹_mpb,ds)

# xlim =(-3,3)
# ylim =(-2,2)

# # xlim =(-0.5,0.5)
# # ylim =(-0.5,0.5)

# hm_e_mpb_real = [ heatmap(x_mpb,y_mpb,[real(e_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_e_mpb_imag = [ heatmap(x_mpb,y_mpb,[imag(e_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# hm_e_real = [ heatmap(x_mpb,y_mpb,[real(e[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_e_imag = [ heatmap(x_mpb,y_mpb,[imag(e[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# hm_e_ratio_real = [ heatmap(x_mpb,y_mpb,[real(e[ix,i,j])/real(e_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_e_ratio_imag = [ heatmap(x_mpb,y_mpb,[imag(e[ix,i,j])/imag(e_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# l = @layout [   a   b   c 
#                 d   e   f
#                 g   h   i
#                 k   l   m    
#                 n   o   p
#                 q   r   s    ]
# plot(
#     hm_e_mpb_real...,
#     hm_e_mpb_imag...,
#     hm_e_real...,
#     hm_e_imag...,
#     hm_e_ratio_real...,
#     hm_e_ratio_imag...,
#     layout=l,
#     size = (1300,1300),
# )

# ##

# function d_from_d!(dout::AbstractArray{ComplexF64,4},e::AbstractArray{ComplexF64,4},H::AbstractArray{ComplexF64,4},din::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     epsinv_dot!(e,-din,ε⁻¹,ds);
#     𝓕⁻¹ * e;
#     kcross_c2t!(H,e,ds);
#     kcross_t2c!(dout,H,ds);
#     𝓕 * dout;
#     return dout
# end

# din = copy(d)
# H_temp = copy(H_mpb)
# e_temp = copy(H_mpb)
# dout_temp = copy(H_mpb)
# dout =  d_from_d!(dout_temp,e_temp,H_temp,din,ε⁻¹_mpb,ds)

# xlim =(-3,3)
# ylim =(-2,2)

# # xlim =(-0.5,0.5)
# # ylim =(-0.5,0.5)
# hm_d_mpb_real = [ heatmap(x_mpb,y_mpb,[real(d_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_d_mpb_imag = [ heatmap(x_mpb,y_mpb,[imag(d_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# hm_d_real = [ heatmap(x_mpb,y_mpb,[real(dout[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_d_imag = [ heatmap(x_mpb,y_mpb,[imag(dout[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# hm_d_ratio_real = [ heatmap(x_mpb,y_mpb,[real(dout[ix,i,j])/real(d_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
# hm_d_ratio_imag = [ heatmap(x_mpb,y_mpb,[imag(dout[ix,i,j])/imag(d_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

# l = @layout [   a   b   c 
#                 d   e   f
#                 g   h   i
#                 k   l   m    
#                 n   o   p
#                 q   r   s    ]


# plot(
#     hm_d_mpb_real...,
#     hm_d_mpb_imag...,
#     hm_d_real...,
#     hm_d_imag...,
#     hm_d_ratio_real...,
#     hm_d_ratio_imag...,
#     layout=l,
#     size = (1300,1300),
# )


# ##
# # @btime kcross_t2c!($Hout1,$HA,$ds);
# # @btime kcross_c2t!($Hout1,$HA,$ds);
# # @btime epsinv_dot!($Hout1,$HA,$ε⁻¹,$ds);
# # @btime $𝓕 * $Hout1;
# # @btime M!($Hout,$Hin,$ε⁻¹,$𝓕,$𝓕⁻¹,$ds,$Hw1,$Hw2); # 1.335 ms (0 allocations: 0 bytes)
# # @btime M!($Hout_v,$Hin_v,$ε⁻¹,$𝓕,$𝓕⁻¹,$ds,$Hw1,$Hw2); # 1.211 ms (4 allocations: 256 bytes)
# N = 3* ds.Nx * ds.Ny * ds.Nz
# f = (Hout_v,Hin_v) -> M!(Hout_v,Hin_v,ε⁻¹,𝓕,𝓕⁻¹,ds,Hw1,Hw2)
# Mop = LinearMap{ComplexF64}(f,N,ismutating=true) 
# # @btime $Mop * $Hin_v  # 1.342 ms (6 allocations: 768.33 KiB
# #res = IterativeSolvers.lobpcg(Mop,false,1;maxiter=3000,tol=1e-7)

# ##

# Hsol_v = res.X[:,1]
# Hsol = reshape(Hsol_v,(3,size(ds.kpG)...)) # reinterpret(SVector{3,ComplexF64}, ... )[1,:,:,1]
# dsol = similar(Hsol)
# kcross_t2c!(dsol,Hsol,ds);
# 𝓕 * dsol;

# heatmap([real(dsol[1,i,j]) for i=1:ds.Nx,j=1:ds.Ny])


# ##

# function testfn() #ε⁻¹,ds)
#     N = 3* ds.Nx * ds.Ny * ds.Nz
#     H1 = similar(HA);
#     H2 = similar(HA);
#     xout = similar(HA);
#     𝓕 = plan_fft!(HA,(2:4))
#     𝓕⁻¹ = inv(𝓕) 
#     f = (xout,xin) -> M!(H1,H2,xout,xin,ε⁻¹,𝓕,𝓕⁻¹,ds)
#     Mop = LinearMap{ComplexF64}(f,f,N,N,ishermitian=true,ismutating=true)
#     res = IterativeSolvers.lobpcg(Mop,false,1;maxiter=3000,tol=1e-7)
#     # vals,vecs,info = eigsolve(x -> M!(H1,H2,H3,x,ε⁻¹,𝓕,𝓕⁻¹,ds),HA)
# end
# ##
# # function kcross_t2c!(Hout::Array{ComplexF64,4},Hin::Array{ComplexF64,4},ds::MaxwellData)::Array{ComplexF64,4}
# #     Hout[:,i,j,k] = [ Vector{ComplexF64}( ( Hin[1,i,j,k] * ds.kpG[i,j,k].n - Hin[2,i,j,k] * ds.kpG[i,j,k].m ) * ds.kpG[i,j,k].mag ) for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz ]
# # end

# # in place operations with staticarrays
# function kcross_t2c(v::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     return ( v[1] * k.n - v[2] * k.m ) * k.mag
# end

# function kcross_c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     at1 = a ⋅ k.m
#     at2 = a ⋅ k.n
#     v0 = -at2 * k.mag
#     v1 = at1 * k.mag
#     return SVector(v0,v1,0.0)
# end

# function kcrossinv_t2c(v::SVector{2,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     return ( v[1] * k.n - v[2] * k.m ) * ( -1 / k.mag )
# end

# function kcrossinv_c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{2,ComplexF64}
#     at1 = a ⋅ k.m
#     at2 = a ⋅ k.n
#     v0 = -at2 * (-1 / k.mag )
#     v1 = at1 * ( -1 / k.mag )
#     return SVector(v0,v1)
# end

# function myfft!(H::Array{SVector{3,ComplexF64}})::Nothing
#     P * reinterpret(ComplexF64,H);
#     return nothing
# end

# function myifft!(H::Array{SVector{3,ComplexF64}})::Nothing
#     iP * reinterpret(ComplexF64,H);
#     return nothing
# end

# function flatten(H::Array{SVector{3,ComplexF64}})
#     return vec(reinterpret(ComplexF64,H))
# end

# function unflatten(Hvec::Array{ComplexF64},shape::NTuple{4,Int64})::Array{SVector{3,ComplexF64}}
#     return reinterpret(SVector{3,ComplexF64},reshape(Hvec,(3*shape[1],shape[2:end]...)))  
# end

# function M!(H::Array{SVector{3,ComplexF64}},eps::Array{SHM3},ds::MaxwellData)::Array{SVector{3,ComplexF64}} 
#     H .= kcross_t2c.(H,ds.kpG);
#     ds.𝓕 * reinterpret(ComplexF64,H); # reinterpret(ComplexF64,reshape(H,(1,size(k)...)));
#     H .= eps .* -H;
#     ds.𝓕⁻¹ * reinterpret(ComplexF64,H); # reinterpret(ComplexF64,reshape(H,(1,size(k)...))); 
#     H .= kcross_c2t.(conj.(H),ds.kpG)
# end

# function P!(H::Array{SVector{3,ComplexF64}},eps::Array{SHM3},ds::MaxwellData)::Array{SVector{3,ComplexF64}}
#     H .= kcrossinv_t2c.(H,ds.kpG);
#     ds.𝓕⁻¹ * reinterpret(ComplexF64,H); #reinterpret(ComplexF64,reshape(H,(1,size(k)...)));
#     H .= eps .* -H;
#     ds.𝓕 * reinterpret(ComplexF64,H);  # reinterpret(ComplexF64,reshape(H,(1,size(k)...))); 
#     H .= kcrossinv_c2t.(conj.(H),ds.kpG)
# end

# function M!(Hv::Vector{SVector{3,ComplexF64}},eps::Array{SHM3},ds::MaxwellData)::Vector{SVector{3,ComplexF64}}
#     H = reshape(Hv,size(k))
#     H .= kcross_t2c.(H,ds.kpG);
#     myfft!(H);
#     H .= eps .* H;
#     myifft!(H);
#     H .= kcross_c2t.(H,ds.kpG)
#     return Hv
# end

# function M!(Hv::Vector{ComplexF64},eps::Array{SHM3},ds::MaxwellData)::Vector{ComplexF64}
#     HSv = copy(reinterpret(SVector{3,ComplexF64}, Hv))
#     M!(HSv,eps,ds.kpG);
#     Hv .= copy(reinterpret(ComplexF64, HSv))
# end

# function M!(Hv::Vector{ComplexF64},eps::Array{SHM3},ds::MaxwellData,Hw::Array{SVector{3,ComplexF64}})::Vector{ComplexF64}
#     copyto!(Hw, reinterpret(SVector{3,ComplexF64},reshape(Hv,(3,size(ds.kpG)...)))[1,:,:,:] )
#     M!(Hw,eps,ds.kpG);
#     copyto!(Hv, vec( reinterpret(ComplexF64,Hw) ) )
#     return Hv
# end

# function P!(Hv::Vector{ComplexF64},eps::Array{SHM3},ds::MaxwellData,Hw::Array{SVector{3,ComplexF64}})::Vector{ComplexF64}
#     copyto!(Hw, reinterpret(SVector{3,ComplexF64},reshape(Hv,(3,size(ds.kpG)...)))[1,:,:,:] )
#     P!(Hw,eps,ds.kpG);
#     copyto!(Hv, vec( reinterpret(ComplexF64,Hw) ) )
#     return Hv
# end


# Hv = copy(flatten(HSA));
# H₀ = copy(Hv);
# ##
# function M̂ₖ!(ε⁻¹::Array{SHM3},kpG::Array{KVec},Hw::Array{SVector{3,ComplexF64}})::LinearMaps.FunctionMap{ComplexF64}
#     N = 3 * length(kpG)
#     f = (Hout, Hin) -> ( M!(Hin,ε⁻¹,kpG,Hw), Hin )
#     return LinearMap{ComplexF64}(f,f,N,N,ishermitian=true,ismutating=true)
# end

# function P̂ₖ!(ε::Array{SHM3},ε⁻¹::Array{SHM3},kpG::Array{KVec},Hw::Array{SVector{3,ComplexF64}})::LinearMaps.FunctionMap{ComplexF64}
#     N = 3 * length(kpG)
#     f = (Hout, Hin) -> ( P!(Hin,ε,kpG,Hw), Hin )
#     fc = (Hout, Hin) -> ( M!(Hin,ε⁻¹,kpG,Hw), Hin )
#     return LinearMap{ComplexF64}(f,f,N,N,ishermitian=true,ismutating=true)
# end


# # f = x::Vector{ComplexF64}-> M!(x,ε⁻¹,KpG,Hw)
# # vals,vecs,info = eigsolve(f,H₀)


# function ftest()
#     Hw = copy(HSA)
#     Mop = M̂ₖ!(ε⁻¹,ds.kpG,Hw)
#     Pop = P̂ₖ!(ε,ε⁻¹,ds.kpG,Hw)
#     res = IterativeSolvers.lobpcg(Mop,false,1;maxiter=3000,tol=1e-7)
# end
# ftest()
# ##

# # function foo(x)
# #     x .= M!(x,ε⁻¹,ds.kpG,HSA)
# #     return x
# # end
# @show Hv[1:6]
# # foo(Hv);
# Mop * Hv;
# # M!(Hv,ε⁻¹,ds.kpG,HSA);
# @show Hv[1:6];
# ##
# M!(Hv,ε⁻¹,KpG,HSA) 
# Hv[1:6]
# ##
# res = IterativeSolvers.lobpcg(Mop,false,1;P=Pop,maxiter=3000,tol=1e-7)

# ##
# # f = x::Vector{ComplexF64}-> M!(x,ε⁻¹,ds.kpG,Hw)

# # function foo(x::Vector{ComplexF64})::Vector{ComplexF64}
# #     x .= M!(x,ε⁻¹,ds.kpG,Hw)
# #     return x
# # end

# # vals,vecs,info = eigsolve(Mop,HSA)