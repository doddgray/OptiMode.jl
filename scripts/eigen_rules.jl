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
