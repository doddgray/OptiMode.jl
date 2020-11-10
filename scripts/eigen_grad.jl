using ChainRulesCore

ChainRulesCore.debug_mode() = true

#####
##### `eigen`
#####



@Zygote.adjoint function LinearAlgebra.eigen(A::LinearAlgebra.RealHermSymComplexHerm)
    dU = eigen(A)
    return dU, function (Δ)
      d, U = dU
      d̄, Ū = Δ
      if Ū === nothing
        P = Diagonal(d̄)
      else
        F = inv.(d' .- d)
        P = F .* (U' * Ū)
        if d̄ === nothing
          P[diagind(P)] .= 0
        else
          P[diagind(P)] = d̄
        end
      end
      return (U * P * U',)
    end
  end

Zygote.refresh()

# function rrule(::typeof(eigen), X::AbstractMatrix{<:Real})
#     F = eigen(X)
#     function eigen_pullback(Ȳ::Composite)
#         # `getproperty` on `Composite`s ensures we have no thunks.
#         ∂X = eigen_rev(F, Ȳ.values, Ȳ.vectors)
#         return (NO_FIELDS, ∂X)
#     end
#     return F, eigen_pullback
# end

# function rrule(::typeof(getproperty), F::T, x::Symbol) where T <: Eigen
#     function getproperty_eigen_pullback(Ȳ)
#         C = Composite{T}
#         ∂F = if x === :values
#             C(values=Ȳ,)
#         elseif x === :vectors
#             C(vectors=Ȳ,)
#         # elseif x === :V
#         #     C(V=Ȳ,)
#         # elseif x === :Vt
#         #     # TODO: https://github.com/JuliaDiff/ChainRules.jl/issues/106
#         #     throw(ArgumentError("Vt is unsupported; use V and transpose the result"))
#         end
#         return NO_FIELDS, ∂F, DoesNotExist()
#     end
#     return getproperty(F, x), getproperty_eigen_pullback
# end

# # When not `Zero`s expect `λ̄ ::AbstractVector, Ū::AbstractMatrix`
# function eigen_rev(res::Eigen, λ̄ , Ū)
#     λ = res.values
#     U = res.vectors

#     k = length(λ)
#     T = eltype(λ)
#     F = T[i == j ? 0 : inv(@inbounds λ[j] - λ[i]) for i = 1:k, j = 1:k]
#     D = Diagonal(λ)
#     D̄ = λ̄  isa AbstractZero ? λ̄  : Diagonal(λ̄ )
#     Ut = U'
#     Ā = conj.(U) * ( D̄ + ( F .* ( Ut * Ū ) ) ) * Ut
#     return Ā
# end

######################################################################################

# Manifolds.jl Forward mode eigen AD by mateuszbaran
# https://github.com/JuliaManifolds/Manifolds.jl/pull/27

##
using LinearAlgebra
using ForwardDiff

import LinearAlgebra.eigen
function eigen(A::StridedMatrix{<:ForwardDiff.Dual})
    A_values = map(d -> d.value, A)
    A_values_eig = eigen(A_values)
    UinvAU = A_values_eig.vectors \ A * A_values_eig.vectors
    vals_diff = diag(UinvAU)
    F = similar(A_values)
    for i ∈ axes(A_values, 1), j ∈ axes(A_values, 2)
        if i == j
            F[i, j] = 0
        else
            F[i, j] = inv(A_values_eig.values[j] - A_values_eig.values[i])
        end
    end
    vectors_diff = A_values_eig.vectors * (F .* UinvAU)
    for i ∈ eachindex(vectors_diff)
        vectors_diff[i] = ForwardDiff.Dual{ForwardDiff.tagtype(vectors_diff[i])}(A_values_eig.vectors[i], vectors_diff[i].partials)
    end
    return Eigen(vals_diff, vectors_diff)
end

function f(A)
    A_eig = eigen(A)
    return A_eig.vectors * Diagonal(A_eig.values) / A_eig.vectors
end
##
ForwardDiff.derivative(t -> f([t+1 t; t t+1]), 1.0)
##
ForwardDiff.jacobian(x -> f(x), [1.0 2.0; 4.0 1.0])

##

######################################################################################

### Legume code

# def vjp_maker_eigsh(ans, x, **kwargs):
#     """Gradient for eigenvalues and vectors of a hermitian matrix."""
#     numeig = kwargs['k']
#     N = x.shape[-1]
#     w, v = ans              # Eigenvalues, eigenvectors.
#     vc = np.conj(v)
    
#     def vjp(g):
#         wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
#         w_repeated = np.repeat(w[..., np.newaxis], numeig, axis=-1)

#         # Eigenvalue part
#         vjp_temp = np.dot(vc * wg[..., np.newaxis, :], T(v)) 

#         # Add eigenvector part only if non-zero backward signal is present.
#         # This can avoid NaN results for degenerate cases if the function 
#         # depends on the eigenvalues only.
#         if np.any(vg):
#             off_diag = np.ones((numeig, numeig)) - np.eye(numeig)
#             F = off_diag / (T(w_repeated) - w_repeated + np.eye(numeig))
#             vjp_temp += np.dot(np.dot(vc, F * np.dot(T(v), vg)), T(v))

#         return vjp_temp

#     return vjp

# def vjp_maker_eigsh(ans, mat, **kwargs):
#     """Steven Johnson method extended to a Hermitian matrix
#     https://math.mit.edu/~stevenj/18.336/adjoint.pdf
#     """
#     numeig = kwargs['k']
#     N = mat.shape[0]
    
#     def vjp(g):
#         vjp_temp = np.zeros_like(mat)
#         for iv in range(numeig):
#             a = ans[0][iv]
#             v = ans[1][:, iv]
#             vc = np.conj(v)
#             ag = g[0][iv]
#             vg = g[1][:, iv]

#             # Eigenvalue part
#             vjp_temp += ag*np.outer(vc, v)

#             # Add eigenvector part only if non-zero backward signal is present.
#             # This can avoid NaN results for degenerate cases if the function 
#             # depends on the eigenvalues only.
#             if np.any(vg):
#                 # Projection operator on space orthogonal to v
#                 P = np.eye(N, N) - np.outer(vc, v)
#                 Amat = T(mat - a*np.eye(N, N))
#                 b = P.dot(vg)

#                 # Initial guess orthogonal to v
#                 v0 = P.dot(np.random.randn(N))

#                 # Find a solution lambda_0 using conjugate gradient 
#                 (l0, _) = sp.linalg.cg(Amat, b, x0=v0, atol=0)
#                 # Project to correct for round-off errors
#                 l0 = P.dot(l0)

#                 vjp_temp -= np.outer(l0, v)
 
#         return vjp_temp
