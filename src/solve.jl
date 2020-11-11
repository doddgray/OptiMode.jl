using  IterativeSolvers, Roots # , KrylovKit
export solve_ω, _solve_Δω², solve_k


function solve_ω(k::SVector{3},ε⁻¹::Array{SHM3,3},g::MaxwellGrid;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    ds = MaxwellData(k,g)
    res = IterativeSolvers.lobpcg(M̂(ε⁻¹,ds),false,neigs;P=P̂(ε⁻¹,ds),maxiter,tol)
    H =  res.X #[:,eigind]                       # eigenmode wavefn. magnetic fields in transverse pol. basis
    ω =  √(real(res.λ[eigind]))                     # eigenmode temporal freq.,  neff = kz / ω, kz = k[3] 
    # ωₖ =   real( ( H' * M̂ₖ(ε⁻¹,ds) * H )[1]) / ω       # ωₖ/∂kz = group velocity = c / ng, c = 1 here
    ωₖ =   real( ( H[:,eigind]' * M̂ₖ(ε⁻¹,ds) * H[:,eigind] )[1]) / ω       # ωₖ/∂kz = group velocity = c / ng, c = 1 here
    return H, ω, ωₖ
end

"""
modified solve_ω version for Newton solver, which wants (x -> f(x), f(x)/f'(x)) as input to solve f(x) = 0
"""
function _solve_Δω²(k,ωₜ,ε⁻¹::Array{SHM3,3},ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    ds.k = SVector(0.,0.,k)
    ds.kpG .= kpG(SVector(0.,0.,k),ds.grid)
    # res = IterativeSolvers.lobpcg(M̂(ε⁻¹,ds),false,ds.H⃗;P=P̂(ε⁻¹,ds),maxiter,tol)
    res = IterativeSolvers.lobpcg(M̂!(ε⁻¹,ds),false,ds.H⃗;P=P̂!(ε⁻¹,ds),maxiter,tol)
    H =  res.X #[:,eigind]                      # eigenmode wavefn. magnetic fields in transverse pol. basis
    ω² =  (real(res.λ[eigind]))                # eigenmode temporal freq.,  neff = kz / ωₖ, kz = k[3] 
    Δω² = ω² - ωₜ^2
    ω²ₖ =   2 * real( ( H[:,eigind]' * M̂ₖ(ε⁻¹,ds) * H[:,eigind] )[1])       # ωₖ/∂kz = group velocity = c / ng, c = 1 here
    ds.H⃗ .= H
    ds.ω² = ω²
    ds.ω²ₖ = ω²ₖ
    return Δω² , Δω² / ω²ₖ
end

function solve_k(ω::Float64,k₀::Float64,ε⁻¹::Array{SHM3,3},g::MaxwellGrid;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    ds = MaxwellData(k₀,g)
    kz = Roots.find_zero(k -> _solve_Δω²(k,ω,ε⁻¹,ds;neigs,eigind,maxiter,tol), k₀, Roots.Newton())
    ds.ω = √ds.ω²
    ds.ωₖ = ds.ω²ₖ / ( 2 * ds.ω )
    return kz, ds
end

function solve_k(ω::Float64,ε⁻¹::Array{SHM3,3},ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    kz = Roots.find_zero(k -> _solve_Δω²(k,ω,ε⁻¹,ds;neigs,eigind,maxiter,tol), ds.k[3], Roots.Newton())
    ds.ω = √ds.ω²
    ds.ωₖ = ds.ω²ₖ / ( 2 * ds.ω )
    return kz
end



# function solve_ω(k,ε⁻¹::Array{SHM3,3},g::MaxwellGrid;neigs=1,eigind=1,maxiter=10000,tol=1e-8)
#     ds = MaxwellData(k,g)
#     res = IterativeSolvers.lobpcg(M̂ₖ(ε⁻¹,ds),false,neigs;maxiter,tol,P=P̂ₖ(SHM3.(inv.(ε⁻¹)),ds))
#     Hₖ = evecs=res.X[:,eigind]                      # eigenmode wavefn. magnetic fields in transverse pol. basis
#     ωₖ = 2π * √(real(res.λ[eigind]))                # eigenmode temporal freq.,  neff = kz / ωₖ, kz = k[3] 
#     ∂ₖωₖ = (2π)^2 * real((Hₖ' * ∂ₖM̂ₖ(ε⁻¹,ds) * Hₖ)[1]) / ωₖ    # ∂ωₖ/∂kz = group velocity = c / ng, c = 1 here
#     return Hₖ, ωₖ, ∂ₖωₖ
# end

# function solve_ω!(k,ε⁻¹::Array{SHM3,3},g::MaxwellGrid,Hw::Array{SVector{3,ComplexF64}};neigs=1,eigind=1,maxiter=10000,tol=1e-8)
#     ds = MaxwellData(k,g)
#     res = IterativeSolvers.lobpcg(M̂ₖ!(ε⁻¹,ds,Hw),false,neigs;maxiter,tol,P=P̂ₖ!(SHM3.(inv.(ε⁻¹)),ds,Hw))
#     Hₖ = evecs=res.X[:,eigind]                      # eigenmode wavefn. magnetic fields in transverse pol. basis
#     ωₖ = 2π * √(real(res.λ[eigind]))                # eigenmode temporal freq.,  neff = kz / ωₖ, kz = k[3] 
#     ∂ₖωₖ = (2π)^2 * real((Hₖ' * ∂ₖM̂ₖ(ε⁻¹,ds) * Hₖ)[1]) / ωₖ    # ∂ωₖ/∂kz = group velocity = c / ng, c = 1 here
#     return Hₖ, ωₖ, ∂ₖωₖ
# end


# """
# modified solve_ω version for Newton solver, which wants a fn. x -> f(x), f(x)/f'(x) 
# """
# function _solve_Δω(k,ω,ε⁻¹::Array{SHM3,3},g::MaxwellGrid;neigs=1,eigind=1,maxiter=10000,tol=1e-8)
#     ds = MaxwellData(k,g)
#     res = IterativeSolvers.lobpcg(M̂ₖ(ε⁻¹,ds),false,neigs;P=P̂ₖ(SHM3.(inv.(ε⁻¹)),ds),maxiter,tol)
#     Hₖ = evecs=res.X[:,eigind]                      # eigenmode wavefn. magnetic fields in transverse pol. basis
#     ωₖ = 2π * √(real(res.λ[eigind]))                # eigenmode temporal freq.,  neff = kz / ωₖ, kz = k[3] 
#     ∂ₖωₖ = (2π)^2 * real((Hₖ' * ∂ₖM̂ₖ(ε⁻¹,ds) * Hₖ)[1]) / ωₖ    # ∂ωₖ/∂kz = group velocity = c / ng, c = 1 here
#     return ωₖ - ω , (ωₖ - ω) / ∂ₖωₖ
# end

# # ε::Array{SArray{Tuple{3,3},Float64,2,9},3}
# function solve_k(ω::Float64,k₀::Float64,ε⁻¹::Array{SHM3,3},g::MaxwellGrid;neigs=1,eigind=1,maxiter=10000,tol=1e-8)
#     return Roots.find_zero(k -> _solve_Δω(k,ω,ε⁻¹,g;neigs,eigind,maxiter,tol), k₀, Roots.Newton())
# end

# # ::Array{SArray{Tuple{3,3},Complex{Float64},2,9},3}
# function solve_k(ω::Float64,ε⁻¹::Array{SHM3,3},g::MaxwellGrid;k₀::Float64=ω*√(maximum(reinterpret(Float64,SHM3.(inv.(ε⁻¹))))),neigs=1,eigind=1,maxiter=10000,tol=1e-8)
#     return solve_k(ω,k₀,ε⁻¹,g;neigs,eigind,maxiter,tol)
# end

# function solve_k(ω::Float64,shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D,Δx::Real,Δy::Real,Nx::Int,Ny::Int;k₀::Float64=ω*√εₘₐₓ(shapes),neigs=1,eigind=1,maxiter=10000,tol=1e-8) 
#     g = MaxwellGrid(Δx,Δy,Nx,Ny)
#     return solve_k(ω,k₀,εₛ⁻¹(shapes,g),g;neigs,eigind,maxiter,tol)
# end



