using LinearAlgebra, StaticArrays, IterativeSolvers, ChainRules, Plots
using FiniteDifferences, ForwardDiff, Zygote, IterativeSolvers # ReverseDiff

#include("eigen_grad.jl")
##
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

##



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
##





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
##
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

##
using ChainRulesTestUtils, Test




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
