"""
################################################################################
#																			   #
#		Methods for conversion between field types, interpolation, etc.		   #
#																			   #
################################################################################
"""

export unflat, _d2ẽ!, _H2d!, _H2e!, E⃗, E⃗x, E⃗y, E⃗z, H⃗, H⃗x, H⃗y, H⃗z, S⃗, S⃗x, S⃗y, S⃗z
export normE!, Ex_norm, Ey_norm, val_magmax, ax_magmax, idx_magmax,
		 canonicalize_phase, canonicalize_phase!


"""
In-place/mutating methods
"""

#########################################
# d,e <: Array versions of _H2d! & _d2ẽ!
##########################################

function _H2d!(d::AbstractArray{Complex{T},N}, Hin::AbstractArray{Complex{T},N},
	mn::AbstractArray{T}, mag::AbstractArray{T},
	𝓕!::FFTW.cFFTWPlan)::AbstractArray{Complex{T},N} where {T<:Real,N}
    kx_tc!(d,Hin,mn,mag);
    mul!(d,𝓕!,d);
	return d
end

function _d2ẽ!(e::AbstractArray{Complex{T},N}, d::AbstractArray{Complex{T},N},
	ε⁻¹,m::AbstractArray{T,N}, n::AbstractArray{T,N}, mag::AbstractArray{T},
	𝓕⁻¹!::FFTW.cFFTWPlan)::AbstractArray{Complex{T},N} where {T<:Real,N}
    eid!(e,ε⁻¹,d);
    mul!(e,𝓕⁻¹!,e);
	return e
end
##
function _H2d!(d::AbstractArray{Complex{T},4}, Hin::AbstractArray{Complex{T},4},
	M̂::HelmholtzMap{3,T})::AbstractArray{Complex{T},4} where T<:Real
    kx_tc!(d,Hin,M̂.mn,M̂.mag);
    mul!(d,M̂.𝓕!,d);
	return d
end

function _d2ẽ!(e::AbstractArray{Complex{T},4}, d::AbstractArray{Complex{T},4},
	M̂::HelmholtzMap{3,T})::AbstractArray{Complex{T},4} where T<:Real
    eid!(e,M̂.ε⁻¹,d);
    mul!(e,M̂.𝓕⁻¹!,e);
	return e
end

function _H2d!(d::AbstractArray{Complex{T},3}, Hin::AbstractArray{Complex{T},3},
	M̂::HelmholtzMap{2,T})::AbstractArray{Complex{T},3} where T<:Real
    kx_tc!(d,Hin,M̂.mn,M̂.mag);
    mul!(d,M̂.𝓕!,d);
	return d
end

function _d2ẽ!(e::AbstractArray{Complex{T},3}, d::AbstractArray{Complex{T},3},
	M̂::HelmholtzMap{2,T})::AbstractArray{Complex{T},3} where T<:Real
    eid!(e,M̂.ε⁻¹,d);
    mul!(e,M̂.𝓕⁻¹!,e);
	return e
end

##
function _H2d!(d::AbstractArray{Complex{T},4}, Hin::AbstractArray{Complex{T},4},
	ms::ModeSolver{3,T})::AbstractArray{Complex{T},4} where T<:Real
    kx_tc!(d,Hin,ms.M̂.mn,ms.M̂.mag);
    mul!(d,ms.M̂.𝓕!,d);
	return d
end

function _d2ẽ!(e::AbstractArray{Complex{T},4}, d::AbstractArray{Complex{T},4},
	ms::ModeSolver{3,T})::AbstractArray{Complex{T},4} where T<:Real
    eid!(e,ms.M̂.ε⁻¹,d);
    mul!(e,ms.M̂.𝓕⁻¹!,e);
	return e
end

function _H2d!(d::AbstractArray{Complex{T},3}, Hin::AbstractArray{Complex{T},3},
	ms::ModeSolver{2,T})::AbstractArray{Complex{T},3} where T<:Real
    kx_tc!(d,Hin,ms.M̂.mn,ms.M̂.mag);
    mul!(d,ms.M̂.𝓕!,d);
	return d
end

function _d2ẽ!(e::AbstractArray{Complex{T},3}, d::AbstractArray{Complex{T},3},
	ms::ModeSolver{2,T})::AbstractArray{Complex{T},3} where T<:Real
    eid!(e,ms.M̂.ε⁻¹,d);
    mul!(e,ms.M̂.𝓕⁻¹!,e);
	return e
end

#########################################
# end: d,e <: Array versions of _H2d! & _d2ẽ!
##########################################


@inline function flat(f::AbstractArray{SVector{3,T}}) where T
	reinterpret(reshape,T,f)
end

@inline function flat(f::AbstractArray{T}) where {T<:Number}
	return f #reinterpret(reshape,T,f)
end

@inline function unflat(f,nvec::Int,Ns)
	reshape(f,(nvec,Ns...))
end

@inline function unflat(f; ms::ModeSolver)
	Ns = size(ms.grid)
	nev = size(f,2)
	ratio = length(f) // ( nev * N(ms.grid) ) |> Int # (length of vector) / (number of grid points)
	[reshape(f[:,i],(ratio,Ns...)) for i=1:nev]
end

@inline function unflat(f,grid::Grid)
	Ns = size(grid)
	ratio = length(f) //  N(grid) |> Int # (length of vector) / (number of grid points)
	reshape(f,(ratio,Ns...))
end

function H⃗(ms::ModeSolver{ND,T}; svecs=true) where {ND,T<:Real}
	Harr = [ fft( tc(unflat(ms.H⃗;ms)[eigind],ms.M̂.mn), (2:1+ND) ) for eigind=1:size(ms.H⃗,2) ]#.* ms.M̂.Ninv
	return Harr
end


function H⃗(k,Hv::AbstractArray{Complex{T}},ω::T,ε⁻¹,nng,grid::Grid{ND}; normalized=true, nnginv=false)::Array{Complex{T},ND+1} where {ND,T<:Real}
	Ns = size(grid)
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	Hₜ = reshape(Hv,(2,Ns...))
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
	if normalized
		E0 = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
		imagmax = argmax(abs2.(E0))
		E0magmax = copy(E0[imagmax])
		E1 = E0 / E0[imagmax]
		if nnginv
			E1norm = dot(E1,_dot(sliceinv_3x3(copy(nng)),E1)) * δV(grid)
		else
			E1norm = dot(E1,_dot(nng,E1)) * δV(grid)
		end
		E = E1 / sqrt(E1norm)
		H1 = fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) ) / ω
	else
		H1 = fft( tc( Hₜ,mns ), (2:1+ND) )
	end

	return H1
end

H⃗x(ms::ModeSolver) = H⃗(ms;svecs=false)[1,eachindex(ms.grid)...]
H⃗y(ms::ModeSolver) = H⃗(ms;svecs=false)[2,eachindex(ms.grid)...]
H⃗z(ms::ModeSolver) = H⃗(ms;svecs=false)[3,eachindex(ms.grid)...]

function E⃗(ms::ModeSolver{ND,T}) where {ND,T<:Real}
	Earr = [ 1im * ε⁻¹_dot( fft( kx_tc( unflat(ms.H⃗; ms)[eigind],ms.M̂.mn,ms.M̂.mag), (2:1+ND) ), copy(flat( ms.M̂.ε⁻¹ ))) for eigind=1:size(ms.H⃗,2) ]
	return Earr
end

function E⃗(ms::ModeSolver{ND,T}, eigind::Int) where {ND,T<:Real}
	E = 1im * ε⁻¹_dot( fft( kx_tc( unflat(ms.H⃗; ms)[eigind],ms.M̂.mn,ms.M̂.mag), (2:1+ND) ), copy(flat( ms.M̂.ε⁻¹ )))
	return E
end

function E⃗(evec::AbstractVector{Complex{T}}, ms::ModeSolver{ND,T}) where {ND,T<:Real}
	return 1im * ε⁻¹_dot( fft( kx_tc( reshape(evec, (2,size(ms.grid)...)), ms.M̂.mn, ms.M̂.mag), (2:1+ND) ), copy(flat( ms.M̂.ε⁻¹ )))
end

function E⃗(evecs::AbstractVector{TV}, ms::ModeSolver{ND,T}) where {ND,T<:Real,TV<:AbstractVector{Complex{T}}}
	# return E⃗.(evecs,(ms,))
	return [E⃗(ev,ms) for ev in evecs]
end


"""
    E⃗(k, evec, ε⁻¹, ∂ε_∂ω, grid; canonicalize=true, normalized=true) -> Array

Real-space electric field of a mode solution, reconstructed from the transverse
eigenvector via ``\\vec{D} ∝ (\\vec{k}+\\vec{G})×\\vec{H}`` (spectral curl `kx_tc` +
FFT) and ``\\vec{E} = ε^{-1}\\vec{D}``. Returns a complex `(3, size(grid)...)` array.

- `canonicalize=true` multiplies by a global phase making the largest-magnitude
  component purely real (convenient for plotting and perturbation integrals).
- `normalized=true` rescales so that ``\\int \\vec{E}^*·(∂ε/∂ω)·\\vec{E}\\,dV = 1``
  (the dispersive energy normalization used by the group-index machinery; pass the
  smoothed `∂ε_∂ω` field).

See also [`H⃗`](@ref), [`S⃗`](@ref), `ModeAnalysis.poynting_z`.
"""
function E⃗(k,evec::AbstractArray{Complex{T}},ε⁻¹,∂ε_∂ω,grid::Grid{ND}; canonicalize=true, normalized=true)::Array{Complex{T},ND+1} where {ND,T<:Real}
	evec_gridshape = reshape(evec,(2,size(grid)...))
	magmn = mag_mn(k,grid)
	E0 = ε⁻¹_dot( fft( kx_tc( evec_gridshape,last(magmn),first(magmn)), (2:1+ND) ), ε⁻¹)
	
	if canonicalize
		phase_factor = cis(-angle(val_magmax(E0)))
	else
		phase_factor = one(T)
	end
	if normalized
		norm_factor = inv( sqrt( dot(E0,_dot(∂ε_∂ω,E0)) * δV(grid) )  ) 
	else
		norm_factor = one(T)
	end
	return norm_factor * (E0 * phase_factor)
end

E⃗x(ms::ModeSolver) = [ E[1,eachindex(ms.grid)...] for E in E⃗(ms;svecs=false) ]
E⃗y(ms::ModeSolver) = [ E[2,eachindex(ms.grid)...] for E in E⃗(ms;svecs=false) ]
E⃗z(ms::ModeSolver) = [ E[3,eachindex(ms.grid)...] for E in E⃗(ms;svecs=false) ]

import Base: abs2
abs2(v::SVector) = real(dot(v,v))

"""
    S⃗(ms::ModeSolver)

Time-averaged Poynting-vector fields ``\\vec{S} = \\mathrm{Re}(\\vec{E}^* × \\vec{H})``
of the modes currently stored in `ms` (arbitrary overall normalization). `S⃗x`/`S⃗y`/
`S⃗z` return single Cartesian components; for power-normalized intensity profiles see
`ModeAnalysis.mode_intensity`.
"""
function S⃗(ms::ModeSolver{ND,T}; svecs=true) where {ND,T<:Real}
	# Ssvs = real.( cross.( conj.(E⃗(ms)), H⃗(ms) ) )
	Ssvs = map((E,H)->real.( cross.( conj.(E), H) ), E⃗(ms), H⃗(ms) )
	svecs ? Ssvs : [ reshape( reinterpret( Complex{T},  Ssvs[i]), (3,size(ms.grid)...)) for i in length(Ssvs) ]
end

S⃗x(ms::ModeSolver) = map((E,H)->real( getindex.( cross.( conj.(E), H), 1)), E⃗(ms), H⃗(ms) ) #real.( getindex.( cross.( conj.(E⃗(ms)), H⃗(ms) ), 1) )
S⃗y(ms::ModeSolver) = map((E,H)->real( getindex.( cross.( conj.(E), H), 2)), E⃗(ms), H⃗(ms) ) #real.( getindex.( cross.( conj.(E⃗(ms)), H⃗(ms) ), 2) )
S⃗z(ms::ModeSolver) = map((E,H)->real( getindex.( cross.( conj.(E), H), 3)), E⃗(ms), H⃗(ms) ) #real.( getindex.( cross.( conj.(E⃗(ms)), H⃗(ms) ), 3) )

"""
	val_magmax(F::AbstractArray)

Return the largest-magnitude component of an array.

This is useful for canonicalizing phase of a complex vector field.
"""
val_magmax(F::AbstractArray{T} where {T<:Number}) =  @inbounds F[argmax(abs2.(F))]
@inline idx_magmax(F::AbstractArray{T} where {T<:Number}) =  argmax(abs2.(F))
@inline ax_magmax(F::AbstractArray{T} where {T<:Number}) =  argmax(abs2.(F))[1]


"""

Canonicalize the phase of one or all eigenmodes in a ModeSolver struct.

This shifts the phase of each mode field such that the largest magnitude
component of the corresponding Electric field `E⃗` is purely real.
"""


function canonicalize_phase(evec::AbstractArray{Complex{T}},k::T,ε⁻¹,grid::Grid{ND}) where {ND,T<:Real}
	return evec * cis(-angle(val_magmax(E⃗(k,evec,ε⁻¹,ε⁻¹,grid;canonicalize=false,normalized=false))))     
end

function canonicalize_phase!(evec::AbstractArray{Complex{T}},k::T,ε⁻¹,grid::Grid{ND}) where {ND,T<:Real}
	evec *= cis(-angle(val_magmax(E⃗(k,evec,ε⁻¹,ε⁻¹,grid;canonicalize=false,normalized=false))))
	return nothing
end



function normE!(ms)
	E = E⃗(ms;svecs=false)
	Eperp = view(E,1:2,eachindex(ms.grid)...)
	imagmax = argmax(abs2.(Eperp))
	# Enorm = E / Eperp(imagmax)
	ms.H⃗ *= inv(Eperp[imagmax])
end

function normE(E)
	Eperp = view(E,1:2,eachindex(ms.grid)...)
	imagmax = argmax(abs2.(Eperp))
	# Enorm = E / Eperp(imagmax)
	ms.H⃗ *= inv(Eperp[imagmax])
end

function Ex_norm(ms)
	E = E⃗(ms;svecs=false)
	Eperp = view(E,1:2,eachindex(ms.grid)...)
	imagmax = argmax(abs2.(Eperp))
	# Enorm = E / Eperp(imagmax)
	view(E,1,eachindex(ms.grid)...) * inv(Eperp[imagmax])
end


function Ey_norm(ms)
	E = E⃗(ms;svecs=false)
	Eperp = view(E,1:2,eachindex(ms.grid)...)
	imagmax = argmax(abs2.(Eperp))
	# Enorm = E / Eperp(imagmax)
	view(E,2,eachindex(ms.grid)...) * inv(Eperp[imagmax])
end

