using Revise
using OptiMode, BenchmarkTools
# using LinearAlgebra, LinearMaps, IterativeSolvers, FFTW, ChainRulesCore, ChainRules, Zygote, OptiMode, BenchmarkTools
include("mpb_example.jl")
H,kz = solve_k(ω,ε⁻¹,Δx,Δy,Δz)
Hs,ks = solve_k(ωs,ε⁻¹,Δx,Δy,Δz)
solve_ω(kz,ε⁻¹,Δx,Δy,Δz)

Hs,ks = solve_k(ωs,ε⁻¹,Δx,Δy,Δz)

g = MaxwellGrid(Δx,Δy,1.0,Nx,Ny,1);
ds = MaxwellData(kz,g);

ns,ngs = solve_n(ωs,ε⁻¹,Δx,Δy,Δz)

plot(ωs,[ns,ngs],label=["n","ng"])

@btime solve_k($ω,$ε⁻¹,$Δx,$Δy,$Δz)
# 1.056 s (146622 allocations: 242.46 MiB)
solve_n(ω_mpb,ε⁻¹,Δx,Δy,Δz)
@btime solve_n($ω_mpb,$ε⁻¹,$Δx,$Δy,$Δz)
# 981.227 ms (164555 allocations: 242.43 MiB)

n_ng(ωs) = solve_n(ωs,ε⁻¹,Δx,Δy,Δz)

Zygote.gradient(om->solve_n(ωs,ε⁻¹,Δx,Δy,Δz)[1],ωs)

Zygote.gradient(om->solve_n(ωs,ε⁻¹,Δx,Δy,Δz)[2],ωs)

H(om) = solve_k(om,ε⁻¹,Δx,Δy,Δz)[1]
function sum2(op,arr)
    return sum(op,arr)
end

function sum2adj( Δ, op, arr )
    n = length(arr)
    g = x->Δ*Zygote.gradient(op,x)[1]
    return ( nothing, map(g,arr))
end

Zygote.@adjoint function sum2(op,arr)
    return sum2(op,arr),Δ->sum2adj(Δ,op,arr)
end

using FFTW, Zygote, Tullio, FiniteDifferences
x = g.x
y = g.y

using FFTW
size(H)
function Hdep(om)
	H,kz = solve_k(om,ε⁻¹,Δx,Δy,Δz)
	# kpg_mag, kpg_mn = calc_kpg(k,Δx,Δy,1.0,Nx,Ny,1)
	Ha = reshape(H,(2,Nx,Ny,Nz))
	# d = fft( kx_t2c(Ha,kpg_mn,kpg_mag), (2:4))
	@tullio Hd := x[i]^4 * abs2.(Ha)[a,i,j,1] * y[j]^4 nograd=(x,y)
	return Hd
end
Hdep(ω,ε⁻¹)
Hdep(0.6,ε⁻¹)
∂Hd_AD(om) = real(Zygote.gradient(Hdep,om)[1])
# Hd_grad_fd = central_fdm(3, 1)(om->Hdep(om,ε⁻¹),ω)
∂Hd_FD(om) = central_fdm(3, 1)(x->Hdep(x),om)
# ωs = 0.5:0.05:0.7
Hd_grad_AD = ∂Hd_AD.(ωs)
# Hd_grad_FD = ∂Hd_FD.(ωs)
# using Plots: plot, plot!
plot(ωs,Hd_grad_AD,color=:red); plot!(ωs,Hd_grad_FD,color=:Black)

plot(ωs,Hd_grad_AD./Hd_grad_FD,color=:green)

ω
ω / kz
ω / ng_mpb
collect(-ωs ./ 2.)
1 / ω
1 / ω^2

function kdep(om,ε⁻¹)
	H,kz = solve_k(om,ε⁻¹,Δx,Δy,Δz)
	kpg_mag, kpg_mn = calc_kpg(kz,Δx,Δy,1.0,Nx,Ny,1)
	# # Ha = reshape(H,(2,Nx,Ny,Nz))
	# # d = fft( kx_t2c(Ha,kpg_mn,kpg_mag), (2:4))
	@tullio kd := x[i]^2 * kpg_mag[i,j,1] * y[j]^4 nograd=(x,y)
	return kd
end
kdep(ω,ε⁻¹)
kdep(0.6,ε⁻¹)
Zygote.gradient(kdep,ω,ε⁻¹)
central_fdm(3, 1)(om->kdep(om,ε⁻¹),ω)


using Plots
function compare_ω(oms;Δx=6.,Δy=4.,res=16)
	Δz       = 1.                    # μ
	lat = mp.Lattice(size=mp.Vector3(Δx, Δy,0))
	ms = mpb.ModeSolver(geometry_lattice=lat,
	                    geometry=[core,subs],
	                    k_points=k_pts,
	                    resolution=res,
	                    num_bands=n_bands,
	                    default_material=mp.vacuum)
	ms.init_params(mp.NO_PARITY, false)
	ε_mean_mpb = ms.get_epsilon()
	Nx = size(ε_mean_mpb)[1]
	Ny = size(ε_mean_mpb)[2]
	Nz = 1
	dx = Δx / Nx
	dy = Δy / Ny
	x = (dx .* (0:(Nx-1))) .- Δx/2.
	y = (dy .* (0:(Ny-1))) .- Δy/2.
	z = [ 0. ]
	ε⁻¹ = [real(get(get(ms.get_epsilon_inverse_tensor_point(mp.Vector3(x[i],y[j],z[k])),a-1),b-1)) for a=1:3,b=1:3,i=1:Nx,j=1:Ny,k=1:Nz]
	n_ng(om) = solve_n(om,ε⁻¹,Δx,Δy,Δz)
	function n_ng_mpb(om)
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
	ns_ngs = n_ng.(oms); ns_ngs_mpb = n_ng_mpb.(oms)
	ns = [nn[1] for nn in ns_ngs]; ngs = [nn[2] for nn in ns_ngs];
	ns_mpb = [nn[1] for nn in ns_ngs_mpb]; ngs_mpb = [nn[2] for nn in ns_ngs_mpb];
	return ns, ngs, ns_mpb, ngs_mpb
end

function plot_compare_ω!(oms,data,p1, p2, p3, p4;line=:solid)
	ns, ngs, ns_mpb, ngs_mpb = data
	plot!(p1,oms,[ns_mpb ns],color=[:Red :Blue],legend=false,ls=line); xlabel!("ω"); ylabel!("neff")
	plot!(p2,oms,[ngs_mpb ngs],color=[:Red :Blue],legend=false,ls=line); xlabel!("ω"); ylabel!("ng")
	plot!(p3,oms,(ns_mpb .- ns) ./ ns_mpb,color=:Green,legend=false,ls=line); xlabel!("ω"); ylabel!("Δn/n")
	plot!(p4,oms,(ngs_mpb .- ngs) ./ ngs_mpb,color=[:Red :Blue],legend=false,ls=line); xlabel!("ω"); ylabel!("Δng/ng")
	return p1, p2, p3, p4
end

function plot_compare_ω(oms,data;lines=[:solid,:dash])
	ns, ngs, ns_mpb, ngs_mpb = data[1]
	p1 = plot(oms,[ns_mpb ns],color=[:Red :Blue],ls=lines[1],legend=false); xlabel!("ω"); ylabel!("neff")
	p2 = plot(oms,[ngs_mpb ngs],label=["MPB"  "OM"],legend=true,color=[:Red :Blue],ls=lines[1]); xlabel!("ω"); ylabel!("ng")
	p3 = plot(oms,(ns_mpb .- ns) ./ ns_mpb,color=:Green,legend=false,ls=lines[1]); xlabel!("ω"); ylabel!("Δn/n")
	p4 = plot(oms,(ngs_mpb .- ngs) ./ ngs_mpb,color=[:Red :Blue],legend=false,ls=lines[1]); xlabel!("ω"); ylabel!("Δng/ng")
	[ plot_compare_ω!(data[i+1],p1, p2, p3, p4;line=lines[i+1]) for i=1:length(data)-1 ];
	l = @layout [   a   b
					c   d  	]
	plot(p1, p2, p3, p4, layout=l, size=(800,800))
end

ωs = 0.5:0.01:0.7
rs = [ 16 32 ]
# data = [compare_ω(ωs;res=rr) for rr in rs]
plot_compare_ω(ωs,data)

a = get_n(ω)
b = Zygote.gradient(get_n,ω)[1]
c = central_fdm(7, 1)(get_n,ω)
d = get_ng(ω)
(d-a)/ω
-d - c

Zygote.gradient(get_ng,ω)[1]

using Zygote, FiniteDifferences
using Plots: plot!
get_n(ω) = solve_n(ω,ε⁻¹,Δx,Δy,Δz)[1]
get_ng(ω) = solve_n(ω,ε⁻¹,Δx,Δy,Δz)[2]
∂n_∂ω_FD(ω) = central_fdm(3, 1)(get_n,ω)
∂n_∂ω_AD(ω) = Zygote.gradient(get_n,ω)[1]
∂ng_∂ω_FD(ω) = central_fdm(3, 1)(get_ng,ω)
∂ng_∂ω_AD(ω) = real(Zygote.gradient(get_ng,ω)[1])
grad_n_AD = ∂n_∂ω_AD.(ωs)
grad_n_FD = ∂n_∂ω_FD.(ωs)
grad_ng_AD = ∂ng_∂ω_AD.(ωs)
grad_ng_FD = ∂ng_∂ω_FD.(ωs)

plot(ωs,grad_n_AD,color=:Blue)
plot!(ωs,grad_n_FD,color=:Black)

plot(ωs,grad_ng_AD,color=:Blue)
plot!(ωs,grad_ng_FD,color=:Black)
plot!(ωs,grad_ng_AD/3.,color=:Red)

#plot!(p,oms,ngs_mpb,label="ng_mpb")
#plot!(p,oms,ns,label="n")
#plot!(p,oms,ngs,label="ng")

##
using IterativeSolvers
k_guess = ω * sqrt(1/minimum([minimum(ε⁻¹[a,a,:,:,:]) for a=1:3]))
ds.k = k_mpb #k_guess
kpg_mag, kpg_mn = calc_kpg(k,Δx,Δy,Δz,Nx,Ny,Nz)
ds.kpg_mag .= kpg_mag
ds.mn .=  kpg_mn

Mop = M̂!(ε⁻¹,ds)
Pop = P=P̂!(ε⁻¹,ds)

Mop * H

Hc = copy(ds.H)

h = ds.𝓕 * t2c(ds.H,kpg_mn)
h = ds.𝓕 * t2c(H_mpb2,kpg_mn)
compare_fields(h_mpb,h,ds)


using FFTW
# res = IterativeSolvers.lobpcg(M̂(ε⁻¹,ds),false,ds.H⃗;P=P̂(ε⁻¹,ds),maxiter,tol)
res = IterativeSolvers.lobpcg(M̂!(ε⁻¹,ds),false,ds.H⃗;P=P̂!(ε⁻¹,ds),maxiter=3000,tol=1e-8)
H =  res.X
Ha = reshape(H,(2,Nx,Ny,Nz))
d = fft( kx_t2c(Ha,kpg_mn,kpg_mag), (2:4))
d = ds.𝓕 * kcross_t2c(Ha,k_mpb,gx,gy,gz)
compare_fields(d_mpb,d,ds)
compare_fields(d_mpb,ds.d,ds)

using FFTW
gx =  collect(fftfreq(Nx,Nx/Δx))
gy =  collect(fftfreq(Ny,Ny/Δy))
gz =  collect(fftfreq(Nz,Nz/Δz))

function KplusG(kz,gx,gy,gz)
	# scale = ds.kpG[i,j,k].mag
	kpg = [-gx; -gy; kz-gz]
	mag = norm(kpg) #sqrt(sum(abs2.(kpg)))
	if mag==0
		n = [0.; 1.; 0.] # SVector(0.,1.,0.)
		m = [0.; 0.; 1.] # SVector(0.,0.,1.)
	else
		if kpg[1]==0. && kpg[2]==0.    # put n in the y direction if k+G is in z
			n = [0.; 1.; 0.] #SVector(0.,1.,0.)
		else                                # otherwise, let n = z x (k+G), normalized
			ntemp = [0.; 0.; 1.] × kpg  #SVector(0.,0.,1.) × kpg
			n = ntemp / norm(ntemp) # sqrt(sum(abs2.(ntemp)))
		end
	end
	# m = n x (k+G), normalized
	mtemp = n × kpg
	m = mtemp / norm(mtemp) # sqrt(sum(abs2.(mtemp))) #sqrt( mtemp[1]^2 + mtemp[2]^2 + mtemp[3]^2 )
	return kpg, mag, m, n
end

function kcross_t2c(Hin,kz,gx,gy,gz)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = similar(Hin,3,Nx,Ny,Nz) # Zygote.Buffer(Hin,3,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = KplusG(kz,gx[i],gy[j],gz[k])
        Hout[1,i,j,k] = ( Hin[1,i,j,k] * n[1] - Hin[2,i,j,k] * m[1] ) * -mag
        Hout[2,i,j,k] = ( Hin[1,i,j,k] * n[2] - Hin[2,i,j,k] * m[2] ) * -mag
        Hout[3,i,j,k] = ( Hin[1,i,j,k] * n[3] - Hin[2,i,j,k] * m[3] ) * -mag
    end
    return Hout
end
d = fft(kcross_t2c(Ha,k_mpb,gx,gy,gz),(2:4))
compare_fields(d_mpb,d,ds)

# kpg, mag, m, n = KplusG(kz,gx[i],gy[j],gz[k])
# kpg_array = [KplusG(kz,gx[i],gy[j],gz[k]) for i=1:Nx,j=1:Ny,k=1:Nz]
using StaticArrays, FFTW

function k_mn(k::SArray{Tuple{3},Float64,1,3})
    mag = sqrt(k[1]^2 + k[2]^2 + k[3]^2)
    if mag==0
        n = SVector(0.,1.,0.)
        m = SVector(0.,0.,1.)
    else
        if k[1]==0. && k[2]==0.    # put n in the y direction if k+G is in z
            n = SVector(0.,1.,0.)
        else                                # otherwise, let n = z x (k+G), normalized
            temp = SVector(0.,0.,1.) × k
            n = temp / sqrt( temp[1]^2 + temp[2]^2 + temp[3]^2 )
        end
    end

    # m = n x (k+G), normalized
    mtemp = n × k
    m = mtemp / sqrt( mtemp[1]^2 + mtemp[2]^2 + mtemp[3]^2 )
    return m,n
end

struct KVec
    k::SVector{3,Float64}   # 3-vector
    mag::Float64            # vector magnitude
    m::SVector{3,Float64}   # k vector normal with even parity in y
    n::SVector{3,Float64}   # k vector normal with even parity in x
end

KVec(k::SVector{3,Float64}) = KVec(
    k,
    sqrt(sum(abs2(ki) for ki in k)), # sqrt(sum(ki^2 for ki in k)),
    k_mn(k)...,
)

function kpG(k::SVector{3,Float64},g::MaxwellGrid)::Array{KVec,3}
    [KVec(k-gx-gy-gz) for gx=g.gx, gy=g.gy, gz=g.gz]
    # [KVec(k+gx+gy+gz) for gx=g.gx, gy=g.gy, gz=g.gz]
end

function kpG(kz::Float64,g::MaxwellGrid)::Array{KVec,3}
    # kpG(SVector(0.,0.,kz),g::MaxwellGrid)::Array{KVec,3}
    [KVec(SVector(0.,0.,kz)-gx-gy-gz) for gx=g.gx, gy=g.gy, gz=g.gz]
end


gx = [SVector(ggx, 0., 0.) for ggx in fftfreq(Nx,Nx/Δx)]     # gx
gy = [SVector(0., ggy, 0.) for ggy in fftfreq(Ny,Ny/Δy)]
gz = [SVector(0., 0., ggz) for ggz in fftfreq(Nz,Nz/1.0)]
kpg_array = [KVec(SVector(0.,0.,kz)-ggx-ggy-ggz) for ggx in gx, ggy in gy, ggz in gz]
kpg_old = [kpg_array[i,j,k].k[a] for a=1:3,i=1:Nx,j=1:Ny,k=1:Nz]
mag_old = [kpg_array[i,j,k].mag for i=1:Nx,j=1:Ny,k=1:Nz]
m_old = [kpg_array[i,j,k].m[a] for a=1:3,i=1:Nx,j=1:Ny,k=1:Nz]
n_old = [kpg_array[i,j,k].n[a] for a=1:3,i=1:Nx,j=1:Ny,k=1:Nz]


ds.mn[:,1,:,:,:] ≈ m_old
ds.mn[:,2,:,:,:] ≈ n_old
ds.kpg_mag ≈ mag_old







 #[:,eigind]
eigind = 1                  # eigenmode wavefn. magnetic fields in transverse pol. basis
ω² =  (real(res.λ[eigind]))
ω = sqrt(ω²)
ω
ω_mpb
using Roots
neigs = 1
eigind = 1
ds = MaxwellData(k_guess,g);
Roots.find_zero(k -> _solve_Δω²(k,ω,ε⁻¹,ds;neigs,eigind,maxiter=10000,tol=1e-10), ds.k, Roots.Newton(),verbose=true,rtol=1e-16,maxevals=1000)
kz = Roots.find_zero(k -> _solve_Δω²(k,ω,ε⁻¹,ds;neigs,eigind,maxiter=10000,tol=1e-10), ds.k, Roots.Newton(),atol=1e-16,maxevals=1000)

ng = ω / H_Mₖ_H(Ha,ε⁻¹,kpg_mn,kpg_mag)
abs(k_mpb - kz) / k_mpb
abs(ng_mpb - ng) / ng_mpb
abs(neff_mpb - ds.k/ω) / neff_mpb
H =  res.X #[:,eigind]                      # eigenmode wavefn. magnetic fields in transverse pol. basis
ω² =  (real(res.λ[eigind]))                # eigenmode temporal freq.,  neff = kz / ωₖ, kz = k[3]
Δω² = ω² - ωₜ^2
# ω²ₖ =   2 * real( ( H[:,eigind]' * M̂ₖ(ε⁻¹,ds) * H[:,eigind] )[1])       # ωₖ/∂kz = group velocity = c / ng, c = 1 here
Ha = reshape(H,(2,size(ε⁻¹)[end-2:end]...))
ω²ₖ = 2 * H_Mₖ_H(Ha,ε⁻¹,kpg_mn,kpg_mag,ds.𝓕) # ωₖ/∂kz = group velocity = c / ng, c = 1 here
ds.H⃗ .= H
ds.ω² = ω²
ds.ω²ₖ = ω²ₖ


##
eigind=1
kz = k
P = LinearMap(x -> H[:,eigind] * dot(H[:,eigind],x),length(H[:,eigind]),ishermitian=true)
A = M̂(ε⁻¹,kz,gx,gy,gz) - ω^2 * I
b = ( I  -  P ) * H̄[:,eigind]
λ⃗₀ = IterativeSolvers.bicgstabl(A,b,3)
ωₖ = real( ( H[:,eigind]' * Mₖ(H[:,eigind], ε⁻¹,kz,gx,gy,gz) )[1]) / ω # ds.ω²ₖ / ( 2 * ω )
# Hₖ =  ( I  -  P ) * * ( Mₖ(H[:,eigind], ε⁻¹,ds) / ω )
ω̄  =  ωₖ * real(k̄)
λ⃗₀ -= P*λ⃗₀ - ω̄  * H
Ha = reshape(H,(2,Nx,Ny,Nz))
Ha_F =  fft(kcross_t2c(Ha,kz,gx,gy,gz),(2:4))
λ₀ = reshape(λ⃗₀,(2,Nx,Ny,Nz))
λ₀_F  = fft(kcross_t2c(λ₀,kz,gx,gy,gz),(2:4))
# ε̄ ⁻¹ = ( 𝓕 * kcross_t2c(λ₀,ds) ) .* ( 𝓕 * kcross_t2c(Ha,ds) )
ε⁻¹_bar = [ Diagonal( real.(λ₀_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) ) for i=1:Nx,j=1:Ny,kk=1:Nz]

ε⁻¹_bar = [ Diagonal( real.(λ₀_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) )[a,b] for a=1:3,b=1:3,i=1:Nx,j=1:Ny,kk=1:Nz]
