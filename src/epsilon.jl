export ε_tensor, εᵥ, flat
export Δₘ

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

function flat(ε::AbstractArray{TA}) where TA<:SMatrix{3,3,T} where T<:Real
    reshape(reinterpret(reshape,T,ε),(3,3,size(ε)...))
end

function flat(ε::AbstractArray{TA}) where TA<:SMatrix{3,3,T,N} where {T<:Real,N}
    reshape(reinterpret(reshape,T,ε),(3,3,size(ε)...))
end

flat(A::Matrix{SMatrix{3, 3, T, 9} where T}) = reshape(reinterpret(reshape,T,A),(3,3,size(A)...))
