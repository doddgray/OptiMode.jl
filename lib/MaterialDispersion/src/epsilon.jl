export ε_tensor, εᵥ, flat
export Δₘ

# NB: I would use I/UniformScaling, but seems worse for type stability, vectorization, AD...
"""
    εᵥ

The vacuum relative-permittivity tensor: the 3×3 identity (as a static matrix).
"""
const εᵥ = @SMatrix	[	1. 	0. 	0.
	                	0. 	1. 	0.
	                	0. 	0. 	1.  ]

"""
    ε_tensor(n::Real)           -> SMatrix{3,3}
    ε_tensor(n₁, n₂, n₃)        -> SMatrix{3,3}
    ε_tensor(ns)                -> SMatrix{3,3}
    ε_tensor(ε::AbstractMatrix) -> SMatrix{3,3}

Construct a (diagonal) relative-permittivity tensor from refractive indices,
``ε_{ii} = n_i^2`` — isotropic from a single index, or birefringent from three
principal indices (tuple/vector). A 3×3 matrix input is passed through as a static
matrix.
"""
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

"""
    flat(ε::AbstractArray{<:SMatrix{3,3}}) -> AbstractArray

Reinterpret an array of 3×3 static matrices (one tensor per grid point) as a "flat"
`(3, 3, size(ε)...)` numeric array — the tensor-field memory layout used throughout
the eigensolver pipeline.
"""
function flat(ε::AbstractArray{TA}) where TA<:SMatrix{3,3,T} where T<:Real
    reshape(reinterpret(reshape,T,ε),(3,3,size(ε)...))
end

function flat(ε::AbstractArray{TA}) where TA<:SMatrix{3,3,T,N} where {T<:Real,N}
    reshape(reinterpret(reshape,T,ε),(3,3,size(ε)...))
end

flat(A::Matrix{SMatrix{3, 3, T, 9} where T}) = reshape(reinterpret(reshape,T,A),(3,3,size(A)...))
