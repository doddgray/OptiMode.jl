using LinearAlgebra, StaticArrays, IterativeSolvers, ChainRules, Plots
using FiniteDifferences, ForwardDiff, Zygote, IterativeSolvers # ReverseDiff
# using ChainRulesTestUtils, Test

#include("eigen_grad.jl")

function rrule(::typeof(eigen), X::AbstractMatrix{<:Real};k=1)
    F = eigen(X)
    function eigen_pullback(Ȳ::Composite{<:Eigen})
        ∂X = @thunk(eigen_rev(F, Ȳ.values, Ȳ.vectors,k))
        return (NO_FIELDS, ∂X)
    end
    return F, eigen_pullback
end

function rrule(::typeof(getproperty), F::T, x::Symbol) where T <: Eigen
    function getproperty_eigen_pullback(Ȳ)
        C = Composite{T}
        ∂F = if x === :values
            C(values=Ȳ,)
        elseif x === :vectors
            C(vectors=Ȳ,)
        end
        return NO_FIELDS, ∂F, DoesNotExist()
    end
    return getproperty(F, x), getproperty_eigen_pullback
end

function eigen_rev(ΛV::Eigen,Λ̄,V̄,k)

    Λ = ΛV.values
    V = ΛV.vectors
    A = V*diagm(Λ)*V'

    Ā = zeros(size(A))
    tempĀ = zeros(size(A))
    # eigen(A).values are in descending order, this current implementation
    # assumes that the input matrix positive definite
    for j = length(Λ):-1:1
        tempĀ = (I-V[:,j]*V[:,j]') ./ norm(A*V[:,j])
        for i = 1:k-1
            tempĀ += A^i * (I-V[:,j]*V[:,j]') ./ (norm(A*V[:,j])^(i+1))
        end
        tempĀ *= V̄[:,j]*V[:,j]'
        # x = A ./ norm(A*V[:,j])
        # p = fill(I, (k - 1))
        # evalpoly(x, p)
        # v = V[:,j]
        # w = (v̄ .-v .* dot(v, v̄)) ./ norm(A*v)
        # tempĀ = (tempĀ * w) * v'
        Ā += tempĀ
        A = A - A*V[:,j]*V[:,j]'
    end
    return Ā
end



# using BLAS

function frule(
    (_, ΔA),
    ::typeof(eigen!),
    A::LinearAlgebra.RealHermSymComplexHerm{<:BLAS.BlasReal,<:StridedMatrix};
    sortby::Union{Function,Nothing}=nothing,
)
    F = eigen!(A; sortby=sortby)
    ΔA isa AbstractZero && return F, ΔA
    λ, U = F.values, F.vectors
    tmp = U' * ΔA
    ∂K = mul!(ΔA.data, tmp, U)
    ∂Kdiag = @view ∂K[diagind(∂K)]
    ∂λ = real.(∂Kdiag)
    ∂K ./= λ' .- λ
    fill!(∂Kdiag, 0)
    ∂U = mul!(tmp, U, ∂K)
    _eigen_norm_phase_fwd!(∂U, A, U)
    ∂F = Composite{typeof(F)}(values = ∂λ, vectors = ∂U)
    return F, ∂F
end

function rrule(
    ::typeof(eigen),
    A::LinearAlgebra.RealHermSymComplexHerm;
    sortby::Union{Function,Nothing}=nothing,
)
    F = eigen(A; sortby=sortby)
    function eigen_pullback(ΔF::Composite{<:Eigen})
        λ, U = F.values, F.vectors
        Δλ, ΔU = ΔF.values, ΔF.vectors
        ∂A = eigen_rev(A, λ, U, Δλ, ΔU)
        return NO_FIELDS, ∂A
    end
    eigen_pullback(ΔF::AbstractZero) = (NO_FIELDS, ΔF)
    return F, eigen_pullback
end

function eigen_rev(A::LinearAlgebra.RealHermSymComplexHerm, λ, U, ∂λ, ∂U)
    if ∂U isa AbstractZero
        ∂λ isa AbstractZero && return (NO_FIELDS, ∂λ + ∂U)
        ∂K = Diagonal(∂λ)
        ∂A = U * ∂K * U'
    else
        ∂U = copyto!(similar(∂U), ∂U)
        _eigen_norm_phase_rev!(∂U, A, U)
        ∂K = U' * ∂U
        ∂K ./= λ' .- λ
        ∂K[diagind(∂K)] = ∂λ
        ∂A = mul!(∂K, U * ∂K, U')
    end
    return _hermitrize!(∂A, A)
end

_eigen_norm_phase_fwd!(∂V, ::LinearAlgebra.RealHermSym, V) = ∂V
function _eigen_norm_phase_fwd!(∂V, A::Hermitian, V)
    k = A.uplo === 'U' ? size(A, 1) : 1
    @inbounds for i in axes(V, 2)
        v = @view V[:, i]
        vₖ, ∂vₖ = real(v[k]), ∂V[k, i]
        ∂v .-= v .* (imag(∂vₖ) / ifelse(iszero(vₖ), one(vₖ), vₖ))
    end
    return ∂V
end

_eigen_norm_phase_rev!(∂V, ::LinearAlgebra.RealHermSym, V) = ∂V
function _eigen_norm_phase_rev!(∂V, A::Hermitian, V)
    k = A.uplo === 'U' ? size(A, 1) : 1
    @inbounds for i in axes(V, 2)
        v, ∂v = @views V[:, i], ∂V[:, i]
        vₖ = real(v[k])
        ∂c = dot(v, ∂v)
        ∂v[k] -= im * (imag(∂c) / ifelse(iszero(vₖ), one(vₖ), vₖ))
    end
    return ∂V
end

#####
##### `eigvals!`/`eigvals`
#####

function frule(
    (_, ΔA),
    ::typeof(eigvals!),
    A::LinearAlgebra.RealHermSymComplexHerm{<:BLAS.BlasReal,<:StridedMatrix};
    sortby::Union{Function,Nothing}=nothing,
)
    ΔA isa AbstractZero && return eigvals!(A; sortby=sortby), ΔA
    F = eigen!(A; sortby=sortby)
    λ, U = F.values, F.vectors
    tmp = ΔA * U
    # diag(U' * tmp) without computing matrix product
    ∂λ = similar(λ)
    @inbounds for i in eachindex(λ)
        ∂λ[i] = @views real(dot(U[:, i], tmp[:, i]))
    end
    return λ, ∂λ
end

function rrule(
    ::typeof(eigvals),
    A::LinearAlgebra.RealHermSymComplexHerm;
    sortby::Union{Function,Nothing}=nothing,
)
    F = eigen(A; sortby=sortby)
    λ = F.values
    function eigvals_pullback(Δλ)
        U = F.vectors
        ∂A = _hermitrize!(U * Diagonal(Δλ) * U', A)
        return NO_FIELDS, ∂A
    end
    eigvals_pullback(Δλ::AbstractZero) = (NO_FIELDS, Δλ)
    return λ, eigvals_pullback
end

# in-place hermitrize matrix, optionally wrapping like A
function _hermitrize!(A)
    A .= (A .+ A') ./ 2
    return A
end
function _hermitrize!(∂A, A)
    _hermitrize!(∂A)
    return _symhermtype(A)(∂A, Symbol(A.uplo))
end


Zygote.@adjoint enumerate(xs) = enumerate(xs), diys -> (map(last, diys),)
_ndims(::Base.HasShape{d}) where {d} = d
_ndims(x) = Base.IteratorSize(x) isa Base.HasShape ? _ndims(Base.IteratorSize(x)) : 1
Zygote.@adjoint function Iterators.product(xs...)
                    d = 1
                    Iterators.product(xs...), dy -> ntuple(length(xs)) do n
                        nd = _ndims(xs[n])
                        dims = ntuple(i -> i<d ? i : i+nd, ndims(dy)-nd)
                        d += nd
                        func = sum(y->y[n], dy; dims=dims)
                        ax = axes(xs[n])
                        reshape(func, ax)
                    end
                end


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

ChainRulesCore.refresh_rules()
Zygote.refresh()


## Test AD sensitivity analysis of random matrices using rrules defined above

function proc_eigs(Xone,αone)
    sum2(x->abs2(x)^2,Xone) #* αone^4
end

function A_from_p(p)
    N = Int(sqrt(length(p)))
    A0 = Zygote.Buffer(zeros(ComplexF64,(N,N)))
    for i = 1:N
        for j=1:i
            A0[j,i] = p[(sum(1:(i-1))+1)+j-1]
            # println("ind: $((sum(1:(i-1))+1)+j -1)")
        end
    end
    for i = 2:N
        for j=1:i-1
            A0[j,i] += p[ ((sum(1:N)) + (sum(1:(i-2))+1) + j -1) ]*im
            # println("ind: $((sum(1:N)) + (sum(1:(i-1))+1) + j -1)")
        end
    end
    A = Hermitian(copy(A0))
end

function foo(p)
    # A = Hermitian(diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:N]...) + im*diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:(N-1)]...))
    A = A_from_p(p)
    F = eigen(A)
    α = F.values
    X = F.vectors
    Xone = Zygote.@showgrad(X[:,1])
    αone = Zygote.@showgrad(α[1])
    # sum(x->abs2(x)^6,Xone) * αone^2
    proc_eigs(Xone,αone)
end

function ∂foo_SJ(p,α,X,ᾱ,X̄;i=1)
    # A = Hermitian(diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:N]...) + im*diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:(N-1)]...))
    # A = A_from_p(p)
    A, Ap_pb = Zygote.pullback(A_from_p,p)
    α,X = eigen(A)
    X̄,ᾱ = Zygote.gradient(proc_eigs,X[:,1],α[1])
    P = I - X[:,i] * X[:,i]'
    b = P * X̄ #[i]
    λ₀ = IterativeSolvers.bicgstabl(A-α[i]*I,b,3)
    if isnothing(ᾱ)
        ᾱ = 0.
    end
    λ = λ₀ - ᾱ * X[:,i]
    Ā = -λ * X[:,i]'
    Ap_pb(Ā)
end

N = 30
p = randn(Float64,N^2)
# A = Hermitian(diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:N]...) + im*diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:(N-1)]...))
# A0 = zeros(ComplexF64,(N,N))
# for i = 1:N
#     for j=1:i
#         A0[j,i] = p[(sum(1:(i-1))+1)+j-1]
#         # println("ind: $((sum(1:(i-1))+1)+j -1)")
#     end
# end
# for i = 2:N
#     for j=1:i-1
#         A0[j,i] += p[ ((sum(1:N)) + (sum(1:(i-2))+1) + j -1) ]*im
#         # println("ind: $((sum(1:N)) + (sum(1:(i-1))+1) + j -1)")
#     end
# end
A = A_from_p(p)
α,X = eigen(A)
foo(p)
n_FD = 2
# plot(α,label="eigvals(A)",legend=:bottomright);scatter!(α,label=nothing)
proc_eigs(X[:,1],α[1])
foo(p)
X̄,ᾱ = Zygote.gradient(proc_eigs,X[:,1],α[1])
p̄_AD = real.(Zygote.gradient(foo,p)[1])
p̄_FD = FiniteDifferences.grad(central_fdm(n_FD, 1),foo,p)[1]
p̄_SJ = real.(∂foo_SJ(p,α,X,ᾱ,X̄)[1])
using Plots: plot, plot!, scatter, scatter!
pp = plot([-maximum(abs.(p̄_AD)),maximum(abs.(p̄_AD))],[-maximum(abs.(p̄_AD)),maximum(abs.(p̄_AD))],c=:black,label="y=x",legend=:bottomright)
scatter!(p̄_AD,p̄_FD,label="AD/FD")
scatter!(p̄_AD,p̄_SJ,label="AD/SJ")
# Āᵢⱼ_ADr = vec(real.(Ā_AD))
# Āᵢⱼ_ADi = vec(imag.(Ā_AD))
# Āᵢⱼ_FDr = vec(real.(Ā_FD))
# Āᵢⱼ_FDi = vec(imag.(Ā_FD))
# Āᵢⱼ_SJr = vec(real.(Ā_SJ))
# Āᵢⱼ_SJi = vec(imag.(Ā_SJ))
#
# using Plots: plot, plot!, scatter, scatter!
# p = plot([-1,1],[-1,1],c=:black,label="y=x",legend=:bottomright)
# scatter!(Āᵢⱼ_ADr,Āᵢⱼ_FDr,label="AD/FD_r")
# scatter!(Āᵢⱼ_ADi,Āᵢⱼ_FDi,label="AD/FD_i")
# scatter!(Āᵢⱼ_ADr,Āᵢⱼ_SJr,label="SJ/AD_r")
# scatter!(Āᵢⱼ_ADi,Āᵢⱼ_SJi,label="SJ/AD_i")

# [ ( A*X⃗[:,i] - α[i] * X⃗[:,i]) for i = 1:N]

## Now test eigen rrule fns with Helmholtz Operator matrices
using Revise
using OptiMode, BenchmarkTools, DataFrames, CSV, FFTW
using LinearMaps
## parameters
w           =   1.7
t_core      =   0.7
edge_gap    =   0.5               # μm
n_core      =   2.4
n_subs      =   1.4
λ           =   1.55                  # μm
Δx          =   6.                    # μm
Δy          =   4.                    # μm
Δz          =   1.
Nx          =   16
Ny          =   16
Nz          =   1
kz          =   1.45
p = [kz,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz]
# ω           =   1 / λ
##

g = MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
ds = MaxwellData(kz,g)
s1 = ridge_wg(w,t_core,edge_gap,n_core,n_subs,Δx,Δy)
ei = make_εₛ⁻¹(s1,g)
eii = similar(ei); [ (eii[a,b,i,j,k] = inv(ei[:,:,i,j,k])[a,b]) for a=1:3,b=1:3,i=1:Nx,j=1:Ny,k=1:Nz ] # eii = epsilon tensor field (eii for epsilon_inverse_inverse, yea it's dumb)
Mop = M̂!(ei,ds)
M = Matrix(Mop)
# M̂(ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> M(H,ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)::AbstractArray{ComplexF64,1},*(2,size(ε⁻¹)[end-2:end]...),ishermitian=true,ismutating=false)
# function M(H,ε⁻¹,mn,kpg_mag,𝓕::FFTW.cFFTWPlan,𝓕⁻¹)
#     kx_c2t( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * kx_t2c(H,mn,kpg_mag), ε⁻¹), mn,kpg_mag)
# end
kxt2c_op = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_t2c( reshape(H,(2,ds.Nx,ds.Ny,ds.Nz)), ds.mn, ds.kpg_mag ) )::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),*(2,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
kxt2c = Matrix(kxt2c_op)
F_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ds.𝓕*reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)))::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
# F_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
F = Matrix(F_op)
einv_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec( ε⁻¹_dot( reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)), ei ) )::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
einv = Matrix(einv_op)
Finv_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ds.𝓕⁻¹*reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)))::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
# Finv_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(bfft(reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
Finv = Matrix(Finv_op)
kxc2t_op = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_c2t( reshape(H,(3,ds.Nx,ds.Ny,ds.Nz)), ds.mn, ds.kpg_mag ) )::AbstractArray{ComplexF64,1},*(2,ds.Nx,ds.Ny,ds.Nz),*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
kxc2t = Matrix(kxc2t_op)
@assert -kxc2t * Finv * einv * F * kxt2c ≈ M

zxt2c_op = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( zx_t2c( reshape(H,(2,ds.Nx,ds.Ny,ds.Nz)), ds.mn ) )::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),*(2,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
zxt2c = Matrix(zxt2c_op)

heatmap(real(kxt2c))
heatmap(imag(kxt2c))
heatmap(real(F))
heatmap(imag(F))
heatmap(real(einv))
heatmap(imag(einv))
heatmap(real(Finv))
heatmap(imag(Finv))
heatmap(real(kxc2t))
heatmap(imag(kxc2t))
heatmap(real(M))
heatmap(imag(M))
kxt2c_op * ds.H⃗[:,1]

Hv = ds.H⃗[:,1]
dv = kxt2c * Hv
Finv * (F * dv) ≈ dv
mag_dv = sum(abs2.(dv))
mag_Fdv = sum(abs2.(F*dv))
mag_Finvdv = sum(abs2.(Finv*dv))
mag_Fdv / mag_dv
mag_dv / mag_Finvdv
# ein̄v = (-kxc2t * Finv)' * M̄ * (F * kxt2c)'
# heatmap(real(ein̄v))
# heatmap(imag(ein̄v))

function ei_dot_rwg(w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
    grid = OptiMode.make_MG(Δx, Δy, Δz, Nx, Ny, Nz)
    shapes = ridge_wg(w,t_core,edge_gap,n_core,n_subs,Δx,Δy)
    ei_field = make_εₛ⁻¹(shapes,grid)
    # ei_matrix_buf = Zygote.bufferfrom(zeros(Float64,(3*Nx*Ny*Nz),(3*Nx*Ny*Nz)))
    ei_matrix_buf = Zygote.bufferfrom(zeros(ComplexF64,(3*Nx*Ny*Nz),(3*Nx*Ny*Nz)))
    for i=1:Nx,j=1:Ny,a=1:3,b=1:3
        q = (Ny * (j-1) + i)
        ei_matrix_buf[(3*q-2)+a-1,(3*q-2)+b-1] = ei_field[a,b,i,j,1]
    end
    # return copy(ei_matrix_buf)
    return Hermitian(copy(ei_matrix_buf))
end

ei_dot_rwg(w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz) ≈ einv #real.(einv)

function M_components(kz,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,NxF,NyF,NzF)
    Nx,Ny,Nz = Zygote.ignore() do
        (Int(round(NxF)),Int(round(NyF)),Int(round(NzF)))
    end
    mag, mn = calc_kpg(kz, Δx, Δy, Δz, Nx, Ny, Nz)
    kcr_t2c = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_t2c( reshape(H,(2,Nx,Ny,Nz)), mn, mag ) )::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),*(2,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕 = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end

    𝓕⁻¹ = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ifft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    kcr_c2t = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_c2t( reshape(H,(3,Nx,Ny,Nz)), mn, mag ) )::AbstractArray{ComplexF64,1},*(2,Nx,Ny,Nz),*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    eeii = ei_dot_rwg(w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
    return ( kcr_c2t, 𝓕⁻¹, eeii, 𝓕, kcr_t2c )
end

function make_M(kz,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,NxF,NyF,NzF)
    Nx,Ny,Nz = Zygote.ignore() do
        (Int(round(NxF)),Int(round(NyF)),Int(round(NzF)))
    end
    mag, mn = calc_kpg(kz, Δx, Δy, Δz, Nx, Ny, Nz)
    kcr_t2c = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_t2c( reshape(H,(2,Nx,Ny,Nz)), mn, mag ) )::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),*(2,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕 = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕⁻¹ = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ifft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    kcr_c2t = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_c2t( reshape(H,(3,Nx,Ny,Nz)), mn, mag ) )::AbstractArray{ComplexF64,1},*(2,Nx,Ny,Nz),*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    eeii = ei_dot_rwg(w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
    M = -kcr_c2t * 𝓕⁻¹ * eeii * 𝓕 * kcr_t2c
    # @assert M' ≈ M
    return Hermitian(M)
end
@assert make_M(p...) ≈ M

function make_M_eidot(kz,eidot,Δx,Δy,Δz,NxF,NyF,NzF)
    Nx,Ny,Nz = Zygote.ignore() do
        (Int(round(NxF)),Int(round(NyF)),Int(round(NzF)))
    end
    mag, mn = calc_kpg(kz, Δx, Δy, Δz, Nx, Ny, Nz)
    kcr_t2c = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_t2c( reshape(H,(2,Nx,Ny,Nz)), mn, mag ) )::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),*(2,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕 = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕⁻¹ = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ifft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    kcr_c2t = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_c2t( reshape(H,(3,Nx,Ny,Nz)), mn, mag ) )::AbstractArray{ComplexF64,1},*(2,Nx,Ny,Nz),*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    # eeii = ei_dot_rwg(w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
    M = -kcr_c2t * 𝓕⁻¹ * eidot * 𝓕 * kcr_t2c
    # @assert M' ≈ M
    return Hermitian(M)
end
eid = ei_dot_rwg(w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
@assert make_M_eidot(kz,eid,Δx,Δy,Δz,Nx,Ny,Nz) ≈ M


function proc_eigs(Xone,αone)
    sum2(x->abs2(x)^2,Xone) * abs2(αone)^2
end

function solve_dense(params)
    # MM = make_M(params...)
    # Eigs = eigen(MM)
    # α = Eigs.values
    # X = Eigs.vectors
    Eigs = eigen(make_M(params...))
    Xone = Eigs.vectors[:,1]
    αone = Eigs.values[1]
    proc_eigs(Xone,αone)
end

function solve_dense_eidot(kz,eidot::Hermitian{ComplexF64, Matrix{ComplexF64}},Δx,Δy,Δz,Nx,Ny,Nz)
    # MM = make_M(params...)
    # Eigs = eigen(MM)
    # α = Eigs.values
    # X = Eigs.vectors
    Eigs = eigen(make_M_eidot(kz,eidot,Δx,Δy,Δz,Nx,Ny,Nz))
    Xone = Eigs.vectors[:,1]
    αone = Eigs.values[1]
    proc_eigs(Xone,αone)
end

function ∂solve_dense_SJ(p,α,X,ᾱ,X̄;i=1)
    # w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz,mn,mag = p
    # Mk, Mk_pb = Zygote.pullback(M_dense,w,t_core,edge_gap,n_core,n_subs)
    Mk, Mk_pb = Zygote.pullback(make_M,p...)
    α,X = eigen(Mk)
    X̄,ᾱ = Zygote.gradient(proc_eigs,X[:,1],α[1])
    P = I - X[:,i] * X[:,i]'
    b = P * X̄ #[i]
    λ₀ = IterativeSolvers.bicgstabl(Mk-α[i]*I,b,3)
    if isnothing(ᾱ)
        ᾱ = 0.
    end
    λ = λ₀ - ᾱ * X[:,i]
    M̄k = -λ * X[:,i]'
    Mk_pb(M̄k)
end

M = make_M(p...)
αX = eigen(M)
# @btime eigen(make_M($p...))
# 41.165 ms (32813 allocations: 9.78 MiB) for Nx=Ny=8, size(M)=(128,128)
# @btime eigen($M)
# 19.841 s (26 allocations: 132.17 MiB) for Nx=Ny=32, size(M)=(2048,2048)
# 864.363 ms (24 allocations: 9.04 MiB) for Nx=Ny=16, size(M)=(512,512)
# 0.021 s (24 allocations: 0.78 MiB) for Nx=Ny=8, size(M)=(128,128)
α = αX.values
X = αX.vectors
proc_eigs(X[:,1],α[1])
solve_dense(p)
X̄,ᾱ = Zygote.gradient(proc_eigs,X[:,1],α[1])
P̂ = I - X[:,1] * X[:,1]'
b = P̂ * X̄ #[1]
λ₀ = IterativeSolvers.bicgstabl(M-α[1]*I,b,3)
if isnothing(ᾱ)
    ᾱ = 0.
end
λ = λ₀ - ᾱ * X[:,1]
M̄ = -λ * X[:,1]'
heatmap(real(M̄))
heatmap(imag(M̄))
kcr_c2t, 𝓕⁻¹, eeii, 𝓕, kcr_t2c = M_components(p...)
eīdot1 = (-kcr_c2t * 𝓕⁻¹)' * M̄ * (𝓕 * kcr_t2c)'
eīdot3 = (-kcr_c2t * 𝓕⁻¹)' * M̄ * (𝓕 * kcr_t2c)'#((-kcr_c2t * 𝓕⁻¹)' * M̄' * (𝓕 * kcr_t2c)')'
eid = ei_dot_rwg(w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
@assert solve_dense_eidot(kz,eid,Δx,Δy,Δz,Nx,Ny,Nz) ≈ solve_dense(p)
eīdot2 = Zygote.gradient(solve_dense_eidot,kz,eid,Δx,Δy,Δz,Nx,Ny,Nz)[2]
eīdot1 ≈ eīdot2
real(diag(eīdot1,0)) ≈ real(diag(eīdot2,0))
real(diag(eīdot1,1)) ≈ real(diag(eīdot2,1))

real(diag(eīdot3,0)) ≈ real(diag(eīdot2,0))
real(diag(eīdot3,1)) ≈ real(diag(eīdot2,1))
real(diag(eīdot3,1)) ≈ real(diag(eīdot1,1))

real(diag(eīdot3,2)) ≈ real(diag(eīdot3,-2))
real(diag(eīdot2,2)) ≈ real(diag(eīdot2,-2))
real(diag(eīdot2,1)) ≈ real(diag(eīdot2,-1))
# heatmap(real(ein̄v))
# heatmap(imag(ein̄v))
##
function compare_eīdot(diagind;figsize=(800,800),xlims=(300,525))
        plt_comp_r = plot(
                                real(diag(eīdot2,diagind)),
                                xlim=xlims,
                                c=:red,
                                linewidth=3,
                                label="eī1rd$diagind",
                        )
        plot!(real(diag(eīdot1,-diagind)),
                                xlim=xlims,
                                c=:purple,
                                linewidth=3,
                                label="eī1rd-$diagind",
                                )
        plot!(real(diag(eīdot3,diagind)),
                                xlim=xlims,
                                c=:black,
                                linewidth=1,
                                linestyle=:dash,
                                label="eī3rd$diagind",
                                )
        plot!(real(diag(eīdot3,-diagind)),
                                xlim=xlims,
                                c=:orange,
                                linewidth=1,
                                linestyle=:dash,
                                label="eī3rd-$diagind",
                                )

        plt_comp_i = plot(imag(diag(eīdot2,diagind)),
                                xlim=xlims,
                                c=:blue,
                                linewidth=3,
                                label="eī1id$diagind",
                                )
        plot!(imag(diag(eīdot1,-diagind)),
                                xlim=xlims,
                                c=:green,
                                linewidth=3,
                                label="eī1id-$diagind",
                                )
        plot!(imag(diag(eīdot3,diagind)),
                                xlim=xlims,
                                linewidth=1,
                                c=:black,
                                linestyle=:dash,
                                label="eī3id$diagind",
                                )
        plot!(imag(diag(eīdot3,-diagind)),
                                xlim=xlims,
                                linewidth=1,
                                c=:orange,
                                linestyle=:dash,
                                label="eī3id-$diagind",
                                )
    l = @layout [   a
                    b   ]
    plot(plt_comp_r,
        plt_comp_i,
        layout=l,
        size=figsize,
        )
end

compare_eīdot(1;figsize=(800,800),xlims=(340,380))

##

p̄_AD = [Zygote.gradient(solve_dense,p)[1][begin:end-4]...]
p̄_FD = FiniteDifferences.grad(central_fdm(2, 1),solve_dense,p)[1][begin:end-4]
p̄_SJ = [∂solve_dense_SJ(p,α,X,ᾱ,X̄)[begin:end-4]...]
p̄_AD[1] = 1.0e-12
p̄_SJ[1] = 1.0e-12
using Plots: plot, plot!, scatter, scatter!
pp = plot([-maximum(abs.(p̄_AD)),maximum(abs.(p̄_AD))],[-maximum(abs.(p̄_AD)),maximum(abs.(p̄_AD))],c=:black,label="y=x",legend=:bottomright)
scatter!(p̄_AD,p̄_FD,label="AD/FD")
scatter!(p̄_AD,p̄_SJ,label="AD/SJ")

ω² = real.(α)
plot(ω²,label="ω²",legend=:topleft)
ω = sqrt.(ω²)
neff = kz ./ ω
eig_ind = 1
H = reshape(X[:,eig_ind],(size(X,1),1))
plt_neff = plot(neff,label="neff",legend=:topright)
scatter!(plt_neff,neff[1:10],label="neff",legend=:topright)
grid = OptiMode.make_MG(Δx, Δy, Δz, Nx, Ny, Nz)
shapes = ridge_wg(w,t_core,edge_gap,n_core,n_subs,Δx,Δy)
ei_field = make_εₛ⁻¹(shapes,grid)
plot_ε(ei_field,grid.x,grid.y) #;cmap=cgrad(:viridis))
plot_d⃗(H,kz,grid)


df_p_8x8 = DataFrame(   p = p,
                        p̄_AD = p̄_AD,
                        p̄_FD = p̄_FD,
                        p̄_SJ = p̄_SJ,
                    )
name="M_entry_grads_rwg_8x8"
path="/home/dodd/github/OptiMode/test/"
CSV.write(*(path,name,".csv"), df_p_8x8)

##




##

# function A_from_p(p)
#     N = Int(sqrt(length(p)))
#     A0 = Zygote.Buffer(zeros(ComplexF64,(N,N)))
#     for i = 1:N
#         for j=1:i
#             A0[j,i] = p[(sum(1:(i-1))+1)+j-1]
#             # println("ind: $((sum(1:(i-1))+1)+j -1)")
#         end
#     end
#     for i = 2:N
#         for j=1:i-1
#             A0[j,i] += p[ ((sum(1:N)) + (sum(1:(i-2))+1) + j -1) ]*im
#             # println("ind: $((sum(1:N)) + (sum(1:(i-1))+1) + j -1)")
#         end
#     end
#     A = Hermitian(copy(A0))
# end

# function A_from_p_RD(p)
#     N = Int(sqrt(length(p)))
#     A0 = zeros(ComplexF64,(N,N)) # Zygote.Buffer(zeros(ComplexF64,(N,N)))
#     for i = 1:N
#         for j=1:i
#             A0[j,i] = p[(sum(1:(i-1))+1)+j-1]
#             # println("ind: $((sum(1:(i-1))+1)+j -1)")
#         end
#     end
#     for i = 2:N
#         for j=1:i-1
#             A0[j,i] += p[ ((sum(1:N)) + (sum(1:(i-2))+1) + j -1) ]*im
#             # println("ind: $((sum(1:N)) + (sum(1:(i-1))+1) + j -1)")
#         end
#     end
#     A = Hermitian(A0)
# end
#
# function A_from_p_real(p)
#     N = Int(sqrt(length(p)))
#     Ar = zeros(eltype(p),(N,N))
#     for i = 1:N
#         for j=1:i
#             Ar[j,i] = p[(sum(1:(i-1))+1)+j-1]
#             # println("ind: $((sum(1:(i-1))+1)+j -1)")
#         end
#     end
#     Ar
# end
#
# function A_from_p_imag(p)
#     N = Int(sqrt(length(p)))
#     Ai = zeros(eltype(p),(N,N))
#     for i = 2:N
#         for j=1:i-1
#             Ai[j,i] += p[ ((sum(1:N)) + (sum(1:(i-2))+1) + j -1) ]
#             # println("ind: $((sum(1:N)) + (sum(1:(i-1))+1) + j -1)")
#         end
#     end
#     Ai
# end
#
# function foo(p)
#     # A = Hermitian(diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:N]...) + im*diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:(N-1)]...))
#     A = Hermitian(reshape(p,(2048,2048)))#A_from_p(p)
#     F = eigen(A)
#     α = F.values
#     X = F.vectors
#     Xone = Zygote.@showgrad(X[:,1])
#     αone = Zygote.@showgrad(α[1])
#     # sum(x->abs2(x)^6,Xone) * αone^2
#     proc_eigs(Xone,αone)
# end
#
#
# function ∂foo_SJ(p,α,X,ᾱ,X̄;i=1)
#     # A = Hermitian(diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:N]...) + im*diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:(N-1)]...))
#     # A = A_from_p(p)
#     # A, Ap_pb = Zygote.pullback(A_from_p,p)
#     A, Ap_pb = Zygote.pullback(x->Hermitian(reshape(x,(2048,2048))),p)
#     α,X = eigen(A)
#     X̄,ᾱ = Zygote.gradient(proc_eigs,X[:,1],α[1])
#     P = I - X[:,i] * X[:,i]'
#     b = P * X̄ #[i]
#     λ₀ = IterativeSolvers.bicgstabl(A-α[i]*I,b,3)
#     if isnothing(ᾱ)
#         ᾱ = 0.
#     end
#     λ = λ₀ - ᾱ * X[:,i]
#     Ā = -λ * X[:,i]'
#     Ap_pb(Ā)
# end



##

using ReverseDiff #: JacobianTape, JacobianConfig, jacobian, jacobian!, compile
# using LinearAlgebra: mul!

#########
# setup #
#########

# some objective functions to work with
f(a, b) = (a + b) * (a * b)'
g!(out, a, b) = mul!(out, a + b, a * b)

# pre-record JacobianTapes for `f` and `g` using inputs of shape 10x10 with Float64 elements
const f_tape = ReverseDiff.JacobianTape(f, (rand(10, 10), rand(10, 10)))
const g_tape = ReverseDiff.JacobianTape(g!, rand(10, 10), (rand(10, 10), rand(10, 10)))

# compile `f_tape` and `g_tape` into more optimized representations
const compiled_f_tape = ReverseDiff.compile(f_tape)
const compiled_g_tape = ReverseDiff.compile(g_tape)

# some inputs and work buffers to play around with
a, b = rand(10, 10), rand(10, 10)
inputs = (a, b)
output = rand(10, 10)
results = (similar(a, 100, 100), similar(b, 100, 100))
fcfg = ReverseDiff.JacobianConfig(inputs)
gcfg = ReverseDiff.JacobianConfig(output, inputs)

####################
# taking Jacobians #
####################

# with pre-recorded/compiled tapes (generated in the setup above) #
#-----------------------------------------------------------------#

# these should be the fastest methods, and non-allocating
ReverseDiff.jacobian!(results, compiled_f_tape, inputs)
ReverseDiff.jacobian!(results, compiled_g_tape, inputs)
ReverseDiff.jacobian!(results, f_tape, inputs)
ReverseDiff.jacobian!(results, g_tape, inputs)

const A_from_p_tape = ReverseDiff.JacobianTape(A_from_p, p)
const A_from_p_RD_tape = ReverseDiff.JacobianTape(A_from_p_RD, p)
const A_from_p_real_tape = ReverseDiff.JacobianTape(A_from_p_real, p)
const A_from_p_imag_tape = ReverseDiff.JacobianTape(A_from_p_imag, p)
A_from_p_real_cfg = ReverseDiff.JacobianConfig(p)

inputs = (p,)
results = (similar(p))
ReverseDiff.jacobian!(results, A_from_p_real_tape, p)
ReverseDiff.jacobian(A_from_p_RD, p)
ReverseDiff.jacobian(A_from_p_real, p)

f(rand(10, 10), rand(10, 10))

##
function f(p)
    a = [      1 + 0.1*sin(p[1])                 0.05
               -0.05                       2 + 0.1*cos(p[1])^2        ]
    A = Hermitian(a * a')
    α, X⃗ = eigen(A)
    return α[1]*sum(abs2.(X⃗))
end

x = 0:0.05:3π
p = [ [xx 3.0 ] for xx in x ]

# @show val = f.(p)
# @show jac = [ ReverseDiff.jacobian(f, pp) for pp in p]


##
xs = 0:0.05:3π
grads = [ Zygote.gradient(x->f(x),xx)[1] for xx in xs]

plot(xs,grads)
##

v1 = [v[1] for v in val]
j11 = [j[1,1] for j in jac]

# v2 = [v[2] for v in val]
# j22 = [j[2,2] for j in jac]

l1 = plot(x,v1,label="v1")
plot!(l1,x,j11,label="j11")
# plot!(l1,x,v2,label="v2")
# plot!(l1,x,j22,label="j22")

##

function f(a)
    A = a * a'
    α, X⃗ = eigen(A)
    return α
end

inputs = rand(3,3)
@show jac = ReverseDiff.jacobian(f, inputs);
@show typeof(jac);
@show size(jac[1]);
@show size(jac[2]);

##
W = rand(2, 3); x = rand(3);
Zygote.gradient(W -> sum(W*x), W)[1]

##

function g(a)
    A = a * a'
    α, X⃗ = eigen(A)
    return sum(α)
end

x = rand(3,3)
Zygote.gradient(x -> g(x), x)[1]


## Functions

function foo(p::AbstractArray)
    A0 = Matrix(reshape(p,(3,3)))
    A = A0 * A0'
    # A = SHermitianCompact{10,Float64,sum(1:10)}(p)
    α, X⃗ = eigen(A)
    return sum(X⃗) + sum(α) + sum(p)
end

function goo(p::AbstractArray)
    A0 = Matrix(reshape(p,(3,3)))
    A = A0 * A0'
    # A = SHermitianCompact{10,Float64,sum(1:10)}(p)
    α, X⃗ = eigen(A)
    return α
end



p = randn(9)
foo(p)
goo(p)

## Finite Difference Gradients

# for p = randn(N²)
# tested with N = 3

#### Find dfoo/dx via FiniteDifferences.jl
using FiniteDifferences
@show FiniteDifferences.grad(central_fdm(3,1),foo,p)        # tuple w/ size=(N²,) Array of gradient of foo w.r.t. p⃗ components
@show FiniteDifferences.jacobian(central_fdm(3,1),goo,p)    # tuple w/ size=(N²,) Array of gradient of foo w.r.t. p⃗ components

##

#### Find dfoo/dx via ForwardDiff.jl
using ForwardDiff
@show dfoo_fmad = ForwardDiff.gradient(foo, p)
@show dgoo_fmad = ForwardDiff.jacobian(goo, p)

##

#### Find dfoo/dx via Zygote.jl
using Zygote
Zygote.gradient(foo, p)

##

foo'(randn(100))

##

foo'(randn(Float64,sum(1:10)))


##

dx = 0.03
x = collect(-π:dx:π)
f = tan
ylim = (-10,10)


p = plot(x,f.(x);ylim,label="f")
plot!(p,x,f'.(x);ylim,label="f'")


## General function of eigenproblem results -> compare with Sec. 4 of https://math.mit.edu/~stevenj/18.336/adjoint.pdf

N = 12
T = Float64 #ComplexF64

p = randn(T,sum(1:N))
A = SHermitianCompact{N,T,sum(1:N)}(p)
A = A * A'

# p = randn(T,N,N)
# A = Matrix(randn(p)
# A = A * A'

# α, X⃗ = eigen(A)


# function g(X⃗,α,p)
#     sum(X⃗) + sum(α) + sum(p)
# end

##
