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
struct HelmholtzMap{T} <: LinearMap{T}
    k⃗::AbstractVector{T}
    ε⁻¹::AbstractArray{T}
end

function Base.:(*)(A::HelmholtzMap, x::AbstractVector)
    length(x) == A.N || throw(DimensionMismatch())
    if ismutating(A)
        y = similar(x, promote_type(eltype(A), eltype(x)), A.M)
        A.f(y, x)
    else
        y = A.f(x)
    end
    return y
end



solver = LOBPCGIterator(M̂!(ε⁻¹,ds),false,ds.H⃗,P̂!(ε⁻¹,ds),nothing)

solver.A

solver.A = M̂!(ε⁻¹,make_MD(kk,g))

for kk in 1.4:0.01:1.5
    @show kk
    @show nn = kk / sqrt(real(lobpcg!(solver; log=false, maxiter=1000, not_zeros=false,tol).λ[1]))
end
res.λ[eigind])

@show solver.iteration




X = solver.XBlocks.block
solver.XBlocks


res = IterativeSolvers.lobpcg(M̂!(ε⁻¹,ds),false,ds.H⃗;P=P̂!(ε⁻¹,ds),maxiter,tol)
H =  res.X[:,eigind]                       # eigenmode wavefn. magnetic fields in transverse pol. basis
ω² =  real(res.λ[eigind])                     # eigenmode temporal freq.,  neff = kz / ω, kz = k[3]
ω = sqrt(ω²)
