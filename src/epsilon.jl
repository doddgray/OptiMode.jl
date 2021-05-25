export ε_tensor, εᵥ, flat
export mult, Δₘ

# vacuum relative dielectric permittivity tensor
# NB: I would use I/UniformScaling, but seems worse for type stability, vectorization, AD...
const εᵥ = @SMatrix	[	1. 	0. 	0.
	                	0. 	1. 	0.
	                	0. 	0. 	1.  ]

function ε_tensor(n::T)::SMatrix{3,3,T,9} where T<:Real
    n² = n^2
	SMatrix{3,3,T,9}( 	n², 	0., 	0.,
						0., 	n², 	0.,
						0., 	0., 	n²	)
end

function ε_tensor(n₁::Real,n₂::Real,n₃::Real)::SMatrix{3,3}
	SMatrix{3,3}( 	n₁^2, 	0., 	0.,
					0., 	n₂^2, 	0.,
					0., 	0., 	n₃^2	)
end

function ε_tensor(ns::NTuple{3,Real})::SMatrix{3,3}
	SMatrix{3,3}( 	ns[1]^2, 	0., 	0.,
					0., 	ns[2]^2, 	0.,
					0., 	0., 	ns[3]^2	)
end

function ε_tensor(ns::AbstractVector{<:Real})::SMatrix{3,3}
	SMatrix{3,3}( 	ns[1]^2, 	0., 	0.,
					0., 	ns[2]^2, 	0.,
					0., 	0., 	ns[3]^2	)
end

ε_tensor(ε::AbstractMatrix)::SMatrix{3,3} = SMatrix{3,3}(ε)
# ε_tensor(fₙ::Function) = λ -> ε_tensor(fₙ(λ))
# ε_tensor(fₙ₁::Function,fₙ₂::Function,fₙ₃::Function) = λ -> ε_tensor(fₙ₁(λ),fₙ₂(λ),fₙ₃(λ))

function flat(ε::AbstractArray{TA}) where TA<:SMatrix{3,3,T} where T<:Real
    reshape(reinterpret(reshape,T,ε),(3,3,size(ε)...))
end

function flat(ε::AbstractArray{TA}) where TA<:SMatrix{3,3,T,N} where {T<:Real,N}
    reshape(reinterpret(reshape,T,ε),(3,3,size(ε)...))
end

flat(A::Matrix{SMatrix{3, 3, T, 9} where T}) = reshape(reinterpret(reshape,T,A),(3,3,size(A)...))

"""
################################################################################
#																			   #
#						 Nonlinear Susceptibilities					   		   #
#																			   #
################################################################################
"""


# function mult(χ::AbstractArray{T,3},v₁::AbstractVector,v₂::AbstractVector) where T<:Real
# 	@tullio v₃[i] := χ[i,j,k] * v₁[j] * v₂[k]
# end
#
# function mult(χ::AbstractArray{T,4},v₁::AbstractVector,v₂::AbstractVector,v₃::AbstractVector) where T<:Real
# 	@tullio v₄[i] := χ[i,j,k,l] * v₁[j] * v₂[k] * v₃[l]
# end
#
# """
# 	Δₘ(λs::AbstractVector, χᵣ::AbstractArray{T,3}, λᵣs::AbstractVector)
#
# Miller's Delta scaling for dispersive nonlinear susceptibilities.
#
# Inputs:
# 	-	λs		Frequencies/wavelengths at which to calculate nonlinear tensor
# 	-	fε       Function to calculate linear susceptiblity, for scaling χ(λ)
# 	-	χᵣ		Reference value of nonlinear tensor
# 	-	λᵣs		Reference frequencies/wavelengths corresponding to χᵣ
# """
# function Δₘ(λs::AbstractVector, fε::Function, χᵣ::AbstractArray{T,3}, λᵣs::AbstractVector) where T
# 	dm = flat(map( (lm,lmr) -> (diag(fε(lm)).-1.) ./ (diag(fε(lmr)).-1.), λs, λᵣs ))
# 	@tullio χ[i,j,k] := χᵣ[i,j,k] * dm[i,1] * dm[j,2] * dm[k,3] fastmath=true
# end


"""
################################################################################
#																			   #
#							   Plotting methods					   			   #
#																			   #
################################################################################
"""
