using Revise
include("scripts/mpb_compare.jl")

## MPB find_k data from a ridge waveguide for reference
λ = p_def[1]
Δx = 6.0
Δy = 4.0
Δz = 1.0
Nx = 128
Ny = 128
Nz = 1
ω = 1/λ
n_mpb,ng_mpb = nng_rwg_mpb(p_def)
k_mpb = n_mpb * ω
ei_mpb = ε⁻¹_rwg_mpb(p_def)
p̄_mpb_FD = ∇nng_rwg_mpb_FD(p_def)
##
H_OM, k_OM = solve_k(ω,ei_mpb,Δx,Δy,Δz)
n_OM, ng_OM = solve_n(ω,ei_mpb,Δx,Δy,Δz)

##
using LinearAlgebra, LinearMaps, IterativeSolvers
neigs=1
eigind=1
maxiter=3000
tol=1e-8
ε⁻¹ = copy(ei_mpb)
g = make_MG(Δx,Δy,Δz,size(ε⁻¹)[end-2:end]...)
ds = make_MD(k_mpb,g)
# mag,mn = calc_kpg(k_mpb,g.g⃗)
##


##
eltype(M̂)
eltype(P̂)
size(M̂)
M̂(Hout,Hin)
M̂(vec(Hout),vec(Hin))
*(M̂,vec(Hin))




##
using Revise
using OptiMode
using ArrayInterface, StaticArrays, HybridArrays, BenchmarkTools, LinearAlgebra, FFTW,ChainRules, Zygote, FiniteDifferences
using StaticArrays: Dynamic, SVector
using Zygote: @showgrad, dropgrad
using IterativeSolvers: lobpcg!, lobpcg


p = [
    1.45,               #   propagation constant    `kz`            [μm⁻¹]
    1.7,                #   top ridge width         `w_top`         [μm]
    0.7,                #   ridge thickness         `t_core`        [μm]
    π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
    2.4,                #   core index              `n_core`        [1]
    1.4,                #   substrate index         `n_subs`        [1]
    0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
]
Δx = 6.0
Δy = 4.0
Δz = 1.0
Nx = 128
Ny = 128
Nz = 1
kz = p[1]

ms = ModeSolver(p[1], ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy), Δx, Δy, Δz, Nx, Ny, Nz)

# ε⁻¹ = permutedims(ei_mpb,(3,4,5,1,2))
# g = MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
# ds = MaxwellData(kz,g)
# ei = make_εₛ⁻¹(ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),g)
# # ε⁻¹ = HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3,3},Float64,5,5,Array{Float64,5}}(permutedims(ei,(3,4,5,1,2)))
# ε⁻¹ = HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},Float64,5,5,Array{Float64,5}}(ei)
# k⃗ = SVector(0.,0.,kz)
# ms = ModeSolver(k⃗, ε⁻¹, Δx, Δy, Δz, Nx, Ny, Nz)
##

# solve_ω²(ms,1.5,ε⁻¹)
# (ω²,H⃗), ω²H⃗_pb = Zygote.pullback(1.5,ε⁻¹) do x,y
# 	solve_ω²(ms,x,y)
# end



function calc_ng(p)
	# ε⁻¹ = make_εₛ⁻¹(ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),g)
	ε⁻¹ = HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},Float64,5,5,Array{Float64,5}}(make_εₛ⁻¹(ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),ms))
	solve_nω(ms,p[1],ε⁻¹;eigind=1)[2]
end

function calc_ng(ms,p)
	ng,ng_pb = Zygote.pullback(p) do p
		solve_nω(ms,p[1],ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],ms.M̂.Δx,ms.M̂.Δy);eigind=1)[2]
	end
	return (ng, real(ng_pb(1)[1]))
end


function calc_ng2(p)
	solve_n(ms,p[1],ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],ms.M̂.Δx,ms.M̂.Δy);eigind=1)[2]
end

function calc_ng2_pb(ms,p)
	ng,ng_pb = Zygote.pullback(p) do p
		solve_n(ms,p[1],ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],ms.M̂.Δx,ms.M̂.Δy);eigind=1)[2]
	end
	return (ng, real(ng_pb(1)[1]))
end

pω = [0.64, p[2:end]...]

calc_ng(ms,p)
calc_ng2(pω)
calc_ng2_pb(ms,pω)
calc_ng(p)
ng, ng_pb = Zygote.pullback(calc_ng,p); ng, real(ng_pb(1)[1])


eigind=1
∂ω²∂k, ∂ω²∂k_pb = Zygote.pullback(ms.H⃗[:,eigind],ms.M̂.ε⁻¹) do H⃗,ε⁻¹
	H_Mₖ_H(H⃗,ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n)
end

sizeof(∂ω²∂k_pb) # 984

function dom2dk(ms)
	2 * H_Mₖ_H(ms.H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n)
end

function dom2dk_pb(ms)
	# Zygote.pullback(ms.H⃗[:,eigind],ms.M̂.ε⁻¹) do H⃗,ε⁻¹
	# 	H_Mₖ_H(H⃗,ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n)
	# end
	Zygote.pullback(ms.H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n) do H,ei,mag,m,n
		2 * H_Mₖ_H(H,ei,mag,m,n)
	end
end


dom2dk(ms)
∂ω²∂k, ∂ω²∂k_pb = dom2dk_pb(ms)

@btime dom2dk($ms) # 4.167 ms (170 allocations: 5.26 MiB)
@btime dom2dk_pb($ms) # 4.442 ms (409 allocations: 5.29 MiB)

real(ng_pb(1.)[1])

p2 = [
    1.5,               #   propagation constant    `kz`            [μm⁻¹]
    1.7,                #   top ridge width         `w_top`         [μm]
    0.7,                #   ridge thickness         `t_core`        [μm]
    π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
    2.4,                #   core index              `n_core`        [1]
    1.4,                #   substrate index         `n_subs`        [1]
    0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
]

@time calc_ng(ms,p2)

FiniteDifferences.jacobian(central_fdm(3,1),x->calc_ng(x),p)[1][1,:]

FiniteDifferences.jacobian(central_fdm(5,1),x->calc_ng2(x),pω)[1][1,:]


#  do p
# 	calc_ng(p)
# end
ng2, ng2_pb = Zygote.pullback(p) do p
	calc_ng2(p)
end









real(ng2_pb(1.)[1])








ei1 = HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},Float64,5,5,Array{Float64,5}}(make_εₛ⁻¹(ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),g))
ei2 = make_εₛ⁻¹(ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),g)
strides(ei1)
strides(ei2)


using FiniteDifferences
@show p̄_FD = FiniteDifferences.jacobian(central_fdm(3,1),x->calc_ng(x),p)[1][1,:]
p̄_FD  # = p̄_FD[1][1,:]








function nngω_rwg_OM(p::Vector{Float64} = p0;
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
n_rwg_OM(p) = nngω_rwg_OM(p)[1]
ng_rwg_OM(p) = nngω_rwg_OM(p)[2]
# @show n_OM, n_OM_pb = Zygote.pullback(n_rwg_OM,p0)

@show ng_OM, ng_OM_pb = Zygote.pullback(ng_rwg_OM,p)
@show ng_OM_err = abs(ng - ng_OM) / ng
@show p̄_OM = real(ng_OM_pb(1)[1])
@show p̄_OM_err = abs.(p̄_FD .- p̄_OM) ./ abs.(p̄_FD)

##




nng_pb((0,1))

ωₜ = sqrt(0.425)
ω²,H⃗ = solve_ω²(ms,1.45)
√ω²[1]
Δω² = ω²[1] - ωₜ^2
H_Mₖ_H(H⃗[:,1],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n)
√ω²[1] / H_Mₖ_H(H⃗[:,1],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n)
_solve_Δω²(ms,1.45,ωₜ)
solve_k(ms,1.5)
[solve_k(ms,x) for x in 0.6:0.01:0.65]
solve_n(ms,0.65)
[solve_n(ms,x) for x in 0.6:0.01:0.65]

using LinearAlgebra
A = ms.M̂-ω²[1]*I
λ⃗₀ = IterativeSolvers.bicgstabl(
								A, # A
								H̄[:,eigind] - H⃗[:,eigind] * dot(H⃗[:,eigind],H̄[:,eigind]), # b,
								3,  # "l"
								)

using
(mag,m,n), mag_m_n_pb  = Zygote.pullback(1.53) do kz
	mag_m_n(kz,ms.M̂.g⃗)
end

H = reshape(H⃗[:,1],(2,Nx,Ny,Nz))
𝓕 = plan_fft(randn(ComplexF64, (2,Nx,Ny,Nz)),(2:4))
HF, F_pb = Zygote.pullback(H) do x
	𝓕 * x
end

F_pb(H)

ms2 = ModeSolver(1.49, ε⁻¹, Δx, Δy, Δz)
Nx2,Ny2,Nz2 = size(ε⁻¹)[1:3]
ModeSolver{T}(k, ε⁻¹, Δx, Δy, Δz, Nx, Ny, Nz; nev, tol, maxiter)
solve_ω²(1.49,ε⁻¹) # ;ms

##
Hin = HybridArray{Tuple{2,Dynamic(),Dynamic(),Dynamic()},ComplexF64}(randn(ComplexF64, (2,Nx,Ny,Nz)))
e = HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},ComplexF64}(randn(ComplexF64, (3,Nx,Ny,Nz)))
d = HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},ComplexF64}(randn(ComplexF64, (3,Nx,Ny,Nz)))
Hout = HybridArray{Tuple{2,Dynamic(),Dynamic(),Dynamic()},ComplexF64}(randn(ComplexF64, (2,Nx,Ny,Nz)))



HybridArray{Tuple{2,Dynamic(),Dynamic(),Dynamic()},ComplexF64}(randn(SVector{2,ComplexF64}, (Nx,Ny,Nz)))

kx_tc!(d,Hin,ms.M̂.m⃗,ms.M̂.n⃗,ms.M̂.mag)

ms.M̂(Hout,Hin)

##
=ms)

##


res = lobpcg(ms.M̂,false,1;maxiter=3000,tol=1e-6)

res = lobpcg!(ms.iterator;log=true,maxiter=3000,not_zeros=false,tol=1e-8)
res_norms = vcat([st.residual_norms for st in res.trace]...)
ritz_vals = vcat([st.ritz_values for st in res.trace]...)
plot(res_norms,yscale=:log10)
# plot(real(ritz_vals),yscale=:log10)
# plot(real(ritz_vals))

update_k(ms,1.49)

res = lobpcg!(ms.iterator;log=true,maxiter=3000,not_zeros=false,tol=1e-8)



M̂.k⃗[3]
res = IterativeSolvers.lobpcg!(solver;log=false,maxiter=3000,not_zeros=false,tol=1e-8)
H =  res.X[:,1]
ω² =  real(res.λ[1])
ω = sqrt(ω²)
neff = p[1] / ω
ng = ω / H_Mₖ_H(H,M̂.ε⁻¹,M̂.mag,M̂.m⃗,M̂.n⃗)

##



##
kx_tc!(d,Hin,m,n,mag)
kx_tc!(d,Hin,m⃗,n⃗,mag)
mul!(d.data,𝓕,d.data)
eid!(e,ε⁻¹,d)
mul!(e.data,𝓕⁻¹,e.data)
kx_ct!(Hout,e,m⃗,n⃗,mag,Ninv)
_M!(Hout,Hin,e, d, ε⁻¹, m, n, mag, 𝓕, 𝓕⁻¹, Ninv)

@btime kx_tc!($d,$Hin,$m⃗,$n⃗,$mag)
@btime mul!($d.data,$𝓕,$d.data)
@btime eid!($e,$ε⁻¹,$d)
@btime mul!($e.data,$𝓕⁻¹,$e.data)
@btime kx_ct!($Hout,$e,$m⃗,$n⃗,$mag,$Ninv)
@btime _M!($Hout,$Hin,$e, $d, $ε⁻¹, $m, $n, $mag, $𝓕, $𝓕⁻¹, $Ninv)
##

*(M̂,vec(Hin))

##


solver.A

res = lobpcg!(solver;log=false,maxiter=2000,not_zeros=false,tol=1e-8)
real(res.λ[1])








update_k(solver.A,1.55)

res = lobpcg!(solver;log=false,maxiter=2000,not_zeros=false,tol=1e-8)
real(res.λ[1])


for kk in 1.4:0.01:1.5
    @show kk
    @show nn = kk / sqrt(real(lobpcg!(solver; log=false, maxiter=1000, not_zeros=false,tol).λ[1]))
end
res.λ[eigind])

@show solver.iteration




X = solver.XBlocks.block
solver.XBlocks



##


res = IterativeSolvers.lobpcg(M̂!(ε⁻¹,ds),false,ds.H⃗;P=P̂!(ε⁻¹,ds),maxiter,tol)
H =  res.X[:,eigind]                       # eigenmode wavefn. magnetic fields in transverse pol. basis
ω² =  real(res.λ[eigind])                     # eigenmode temporal freq.,  neff = kz / ω, kz = k[3]
ω = sqrt(ω²)

res2 = lobpcg(M̂!(ei,ds),false,ds.H⃗;P=P̂!(ei,ds),maxiter=3000,tol=1e-8)
H2=  res2.X[:,1]
ω²2 =  real(res2.λ[1])
ω2 = sqrt(ω²2)
neff2 = p[1] / ω2

function b2f(v::AbstractVector;Nx=128,Ny=128,Nz=1)
	vec( permutedims( reshape(v,(Nx,Ny,Nz,2)), (4,1,2,3) ) )
end

function f2b(v::AbstractVector;Nx=128,Ny=128,Nz=1)
	vec( permutedims( reshape(v,(2,Nx,Ny,Nz)), (2,3,4,1) ) )
end
H2c1 = copy(H2)
H2c2 = copy(H2)

H2c1 ./ (ms.M̂*H2c2)
H2c1 ./ (M̂!(ei,ds)*H2c2)
(ms.M̂*H2c1) ./ (M̂!(ei,ds)*H2c2)

Mop = M̂!(ei,ds)
Pop = P̂!(ei,ds)

(ms.M̂*copy(H2)) ./ (Mop*copy(H2))

(Mop * H2) ./ (b2f(M̂*f2b(H2)))

PH2_1 = ldiv!(similar(H2),Pop,copy(H2))
PH2_2 = b2f(ldiv!(similar(H2),HelmholtzPreconditioner(M̂),f2b(copy(H2))))
PH2_3 = (b2f(P̂(similar(H2),f2b(copy(H2)))))
PH2_1 ./ PH2_2

(H2) ./ (b2f(f2b(H2)))

P̂*f2b(H2)

ds.mn[:,:,20,30,1]




ds.kpg_mag[20,30,1]

ms = ModeSolver(k⃗, ε⁻¹, Δx, Δy, Δz, Nx, Ny, Nz)

ms.M̂.m⃗[20,30,1]




ms.M̂.n⃗[20,30,1]




ms.M̂.mag[20,30,1]


ms.M̂.m⃗.data.parent

m = copy(ms.M̂.m⃗)
n = copy(ms.M̂.n⃗)
mag = copy(ms.M̂.mag)
gg = ms.M̂.g⃗
k⃗ = ms.M̂.k⃗

m[20,30,1]




mag_m_n!(mag,m,n,k⃗,gg)


m[20,30,1]




gg[20,30,1]





ds.g⃗[20,30,1]




g2 = _g⃗(Δx,Δy,Δz,Nx,Ny,Nz)
g2[20,30,1]

ms.M̂.g⃗ = _g⃗(Δx,Δy,Δz,Nx,Ny,Nz)

# H2 ./ b2f(M̂*f2b(H2))
#
# Mop = M̂!(ei,ds)
# Pop = P̂!(ei,ds)
#
# (Mop * H2) ./ (b2f(M̂*f2b(H2)))
#
# PH2_1 = ldiv!(similar(H2),Pop,copy(H2))
# PH2_2 = b2f(ldiv!(similar(H2),HelmholtzPreconditioner(M̂),f2b(copy(H2))))
# PH2_3 = (b2f(P̂(similar(H2),f2b(copy(H2)))))
# PH2_1 ./ PH2_2
#
# (H2) ./ (b2f(f2b(H2)))
#
# P̂*f2b(H2)


##
using LinearAlgebra, StaticArrays, ArrayInterface, ChainRules, Zygote, ReverseDiff, ForwardDiff, FiniteDifferences

ChainRulesCore.rrule(T::Type{<:SVector}, xs::Number...) = ( T(xs...), dv -> (nothing, dv...) )
ChainRulesCore.rrule(T::Type{<:SVector}, x::AbstractVector) = ( T(x), dv -> (nothing, dv) )
# ChainRules.rrule(T::Type{<:Base.ReinterpretArray}, x::AbstractVector) = ( T(x), dv -> (nothing, dv) )
ChainRulesCore.rrule(T::Type{<:Base.ReinterpretArray{T,NA,SVector{NS,T}}} where {T,NA,NS})(x::AbstractArray) = T(x), dv -> (nothing, reinterpret(reshape,SVector{NS,T},dv))
# @adjoint (T::Type{<:Base.ReinterpretArray{T,NA,SVector{NS,T}, Array{SVector{NS,T},NB}, true }} where {T,NA,NS,NB})(x::AbstractArray) = T(x), dv -> (nothing, reinterpret(reshape,SVector{NS,T},dv))

function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{T1},A::AbstractArray{SVector{N1,T1},N2}) where {T1,N1,N2}
	return ( reinterpret(reshape,T1,A), Δ->( NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret( reshape,SVector{N1,T1}, Δ ) ) )
end

function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{<:SVector{N1,T1}},A::AbstractArray{T1}) where {T1,N1}
	return ( reinterpret(reshape,type,A), Δ->( NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret( reshape, eltype(A), Δ ) ) )
end

# function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{<:SVector{N1,T1}},A::AbstractArray{T1}) where {T1,N1}
# 	Ω = reinterpret(reshape,type,A)
# 	function reint_SV_pb(Δ)
# 		# @show Δ
# 		# @show _,_,Ā = Δ
# 		# @show Ār = reinterpret( reshape, eltype(A), Δ )
# 		return  NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret( reshape, eltype(A), Δ )
# 	end
# 	return (Ω , reint_SV_pb )
# end

import Base.one
function one(::Type{SVector{N,T}}) where {N,T}
	 SVector(ones(T,N)...)
end

ChainRules.refresh_rules()
Zygote.refresh()

function foo(x)
	# sum(abs2,reinterpret(reshape,Float64,x))
	sum(reinterpret(reshape,Float64,x))
end

function goo(x)
	# sum(abs2,reinterpret(reshape,Float64,x))
	sum(abs2,sum(reinterpret(reshape,SVector{3,Float64},x)))
end

x1 = [ SVector(randn(Float64,3)...) for i=1:2,j=1:2,k=1:2]
x2 = randn(Float64,3,2,2,2)
foo(x1)
goo(x2)
Zygote.gradient(foo,x1)[1]
Zygote.gradient(goo,x2)[1]
ReverseDiff.gradient(foo,x1)
ForwardDiff.gradient(foo,x1)

ReverseDiff.gradient(goo,x2)
ForwardDiff.gradient(goo,x2)


FiniteDifferences.jacobian(central_fdm(2,1),foo,x1)

x2r = reinterpret(reshape,SVector{3,Float64},x2)
x2rc = copy(x2r)
x2rcr = reinterpret(reshape,Float64,x2rc)

using ReverseDiff, ForwardDiff, Nabla
