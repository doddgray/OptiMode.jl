using  IterativeSolvers, Roots # , KrylovKit
export solve_ω, _solve_Δω, solve_k, solve_ω_pc


function solve_ω(k,ε⁻¹::Array{SHM3,2},g::MaxwellGrid;neigs=1,eigind=1,maxiter=10000,tol=1e-8)
    ds = MaxwellData(k,g)
    res = IterativeSolvers.lobpcg(M̂ₖ(ε⁻¹,ds),false,neigs;maxiter,tol,P=P̂ₖ(SHM3.(inv.(ε⁻¹)),ds))
    Hₖ = evecs=res.X[:,eigind]                      # eigenmode wavefn. magnetic fields in transverse pol. basis
    ωₖ = 2π * √(real(res.λ[eigind]))                # eigenmode temporal freq.,  neff = kz / ωₖ, kz = k[3] 
    ∂ₖωₖ = (2π)^2 * real((Hₖ' * ∂ₖM̂ₖ(ε⁻¹,ds) * Hₖ)[1]) / ωₖ    # ∂ωₖ/∂kz = group velocity = c / ng, c = 1 here
    return Hₖ, ωₖ, ∂ₖωₖ
end

"""
modified solve_ω version for Newton solver, which wants a fn. x -> f(x), f(x)/f'(x) 
"""
function _solve_Δω(k,ω,ε⁻¹::Array{SHM3,2},g::MaxwellGrid;neigs=1,eigind=1,maxiter=10000,tol=1e-8)
    ds = MaxwellData(k,g)
    res = IterativeSolvers.lobpcg(M̂ₖ(ε⁻¹,ds),false,neigs;P=P̂ₖ(SHM3.(inv.(ε⁻¹)),ds),maxiter,tol)
    Hₖ = evecs=res.X[:,eigind]                      # eigenmode wavefn. magnetic fields in transverse pol. basis
    ωₖ = 2π * √(real(res.λ[eigind]))                # eigenmode temporal freq.,  neff = kz / ωₖ, kz = k[3] 
    ∂ₖωₖ = (2π)^2 * real((Hₖ' * ∂ₖM̂ₖ(ε⁻¹,ds) * Hₖ)[1]) / ωₖ    # ∂ωₖ/∂kz = group velocity = c / ng, c = 1 here
    return ωₖ - ω , (ωₖ - ω) / ∂ₖωₖ
end

# ε::Array{SArray{Tuple{3,3},Float64,2,9},2}
function solve_k(ω::Float64,k₀::Float64,ε⁻¹::Array{SHM3,2},g::MaxwellGrid;neigs=1,eigind=1,maxiter=10000,tol=1e-8)
    return Roots.find_zero(k -> _solve_Δω(k,ω,ε⁻¹,g;neigs,eigind,maxiter,tol), k₀, Roots.Newton())
end

# ::Array{SArray{Tuple{3,3},Complex{Float64},2,9},2}
function solve_k(ω::Float64,ε⁻¹::Array{SHM3,2},g::MaxwellGrid;k₀::Float64=ω*√(maximum(reinterpret(Float64,SHM3.(inv.(ε⁻¹))))),neigs=1,eigind=1,maxiter=10000,tol=1e-8)
    return solve_k(ω,k₀,ε⁻¹,g;neigs,eigind,maxiter,tol)
end

function solve_k(ω::Float64,shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D,Δx::Real,Δy::Real,Nx::Int,Ny::Int;k₀::Float64=ω*√εₘₐₓ(shapes),neigs=1,eigind=1,maxiter=10000,tol=1e-8) 
    g = MaxwellGrid(Δx,Δy,Nx,Ny)
    return solve_k(ω,k₀,εₛ⁻¹(shapes,g),g;neigs,eigind,maxiter,tol)
end



