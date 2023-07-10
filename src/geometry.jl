export Geometry, fεs, fεs!, fnn̂gs, nn̂gs, fnĝvds, nĝvds, matinds
export εₘₐₓ, nₘₐₓ, materials

matinds(geom::Vector{<:Shape}) = vcat((matinds0 = map(s->findfirst(m->isequal(get_model(Material(s.data),:ε,:λ),get_model(m,:ε,:λ)), materials(geom)),geom); matinds0),maximum(matinds0)+1)
matinds(shapes,mats) = vcat(map(s->findfirst(m->isequal(get_model(Material(s.data),:ε,:λ),get_model(m,:ε,:λ)),mats),shapes),length(mats)+1)


materials(shapes) = unique(Material.(getfield.(shapes,:data)))


fεs(mats::AbstractVector{<:NumMat})	   =  vcat( getproperty.(mats,(:fε,)) 	 , x->εᵥ	   )
fnn̂gs(mats::AbstractVector{<:NumMat})	=  vcat( getproperty.(mats,(:fnng,))  , x->εᵥ	 	)
fnĝvds(mats::AbstractVector{<:NumMat}) =  vcat( getproperty.(mats,(:fngvd,)) , x->zero(εᵥ) )
fχ⁽²⁾s(mats::AbstractVector{<:NumMat}) =  vcat( getproperty.(mats,(:fχ⁽²⁾,)) , (x1,x2,x3)->zeros(eltype(εᵥ),3,3,3) )#χ⁽²⁾_vac )


# Code below may be deprecated? 
# July 10, 2023
struct Geometry
	shapes::Vector #{<:Shape{N}}
	materials::Vector #{AbstractMaterial}
	material_inds::Vector #{Int}
	fεs #{Function}
	fnn̂gs #{Function}
	fnĝvds #{Function}
	fχ⁽²⁾s #{Function}
end


function Geometry(shapes)  #where S<:Shape{N} where N
	mats =  materials(shapes)
	fes = fεs(mats)
	fnngs = fnn̂gs(mats)
	fngvds = fnĝvds(mats)
	fchi2s = fχ⁽²⁾s(mats)
	return Geometry(
		shapes,
		mats,
		matinds(shapes),
		fes,
		fnngs,
		fngvds,
		fchi2s,
	)
end


"""
################################################################################
#																			   #
#							    Utility methods					   			   #
#																			   #
################################################################################
"""

@inline function εₘₐₓ(ω::T,shapes::AbstractVector{<:GeometryPrimitives.Shape}) where T<:Real
    maximum(real(vcat(diag.(εs(shapes,inv(ω)))...)))
end
@inline function εₘₐₓ(ω::T,geom::Geometry) where T<:Real
    maximum(real(vcat(diag.(εs(geom.shapes,inv(ω)))...)))
end

@inline function nₘₐₓ(ω::T,shapes::AbstractVector{<:GeometryPrimitives.Shape}) where T<:Real
    real(sqrt( εₘₐₓ(ω,shapes) ))
end
@inline function nₘₐₓ(ω::T,geom::Geometry) where T<:Real
    real(sqrt( εₘₐₓ(ω,geom.shapes) ))
end

k_guess(ω,geom) = nₘₐₓ(ω,geom) * ω
k_guess(ω,ε⁻¹::AbstractArray{<:Real,4}) = first(ω) * sqrt(1/minimum([minimum(ε⁻¹[a,a,:,:]) for a=1:3]))
k_guess(ω,ε⁻¹::AbstractArray{<:Real,5}) = first(ω) * sqrt(1/minimum([minimum(ε⁻¹[a,a,:,:,:]) for a=1:3]))

