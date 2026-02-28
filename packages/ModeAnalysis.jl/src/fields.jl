"""
################################################################################
#																			   #
#		Methods for conversion between field types, interpolation, etc.		   #
#																			   #
################################################################################
"""

export unflat, _d2ẽ!, _H2d!, _H2e!, E⃗, E⃗x, E⃗y, E⃗z, H⃗, H⃗x, H⃗y, H⃗z, S⃗, S⃗x, S⃗y, S⃗z
export normE!, Ex_norm, Ey_norm, val_magmax, ax_magmax, idx_magmax, group_index,
		 canonicalize_phase, canonicalize_phase!

export E_relpower_xyz, Eslices, count_E_nodes, mode_viable, mode_idx 


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

#########################3################
# d,e <: HybridArray versions of _H2d! & _d2ẽ!
##########################################

function _H2d!(d::TA1, Hin::AbstractArray{Complex{T},N},
	mn::AbstractArray{T}, mag::AbstractArray{T},
	𝓕!::FFTW.cFFTWPlan)::AbstractArray{Complex{T},N} where {TA1<:HybridArray,T<:Real,N}
    kx_tc!(d,Hin,mn,mag);
    mul!(d.data,𝓕!,d.data);
	return d
end

function _d2ẽ!(e::TA2, d::TA1,
	ε⁻¹,m::AbstractArray{T,N}, n::AbstractArray{T,N}, mag::AbstractArray{T},
	𝓕⁻¹!::FFTW.cFFTWPlan)::AbstractArray{Complex{T},N} where {TA1<:HybridArray,TA2<:HybridArray,T<:Real,N}
    eid!(e,ε⁻¹,d);
    mul!(e.data,𝓕⁻¹!,e.data);
	return e
end
##
function _H2d!(d::TA1, Hin::AbstractArray{Complex{T},4},
	M̂::HelmholtzMap{3,T})::AbstractArray{Complex{T},4} where {TA1<:HybridArray,T<:Real}
    kx_tc!(d,Hin,M̂.mn,M̂.mag);
    mul!(d.data,M̂.𝓕!,d.data);
	return d
end

function _d2ẽ!(e::TA2, d::TA1,
	M̂::HelmholtzMap{3,T})::AbstractArray{Complex{T},4} where {TA1<:HybridArray,TA2<:HybridArray,T<:Real}
    eid!(e,M̂.ε⁻¹,d);
    mul!(e.data,M̂.𝓕⁻¹!,e.data);
	return e
end

function _H2d!(d::TA1, Hin::AbstractArray{Complex{T},3},
	M̂::HelmholtzMap{2,T})::AbstractArray{Complex{T},3} where {TA1<:HybridArray,T<:Real}
    kx_tc!(d,Hin,M̂.mn,M̂.mag);
    mul!(d.data,M̂.𝓕!,d.data);
	return d
end

function _d2ẽ!(e::TA2, d::TA1,
	M̂::HelmholtzMap{2,T})::AbstractArray{Complex{T},3} where {TA1<:HybridArray,TA2<:HybridArray,T<:Real}
    eid!(e,M̂.ε⁻¹,d);
    mul!(e.data,M̂.𝓕⁻¹!,e.data);
	return e
end

##
function _H2d!(d::TA1, Hin::AbstractArray{Complex{T},4},
	ms::ModeSolver{3,T})::AbstractArray{Complex{T},4} where {TA1<:HybridArray,T<:Real}
    kx_tc!(d,Hin,ms.M̂.mn,ms.M̂.mag);
    mul!(d.data,ms.M̂.𝓕!,d.data);
	return d
end

function _d2ẽ!(e::TA2, d::TA1,
	ms::ModeSolver{3,T})::AbstractArray{Complex{T},4} where {TA1<:HybridArray,TA2<:HybridArray,T<:Real}
    eid!(e,ms.M̂.ε⁻¹,d);
    mul!(e.data,ms.M̂.𝓕⁻¹!,e.data);
	return e
end

function _H2d!(d::TA1, Hin::AbstractArray{Complex{T},3},
	ms::ModeSolver{2,T})::AbstractArray{Complex{T},3} where {TA1<:HybridArray,T<:Real}
    kx_tc!(d,Hin,ms.M̂.mn,ms.M̂.mag);
    mul!(d.data,ms.M̂.𝓕!,d.data);
	return d
end

function _d2ẽ!(e::TA2, d::TA1,
	ms::ModeSolver{2,T})::AbstractArray{Complex{T},3} where {TA1<:HybridArray,TA2<:HybridArray,T<:Real}
    eid!(e,ms.M̂.ε⁻¹,d);
    mul!(e.data,ms.M̂.𝓕⁻¹!,e.data);
	return e
end

#####################################################
# end: d,e <: HybridArray versions of _H2d! & _d2ẽ!
#####################################################

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

function H⃗(k,H⃗::AbstractArray{Complex{T}},ω::T,geom::Geometry,grid::Grid{ND}; svecs=true, normalized=true) where {ND,T<:Real}
	Ns = size(grid)
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	Hₜ = reshape(H⃗,(2,Ns...))
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
	ε⁻¹, nng⁻¹, ngvd⁻¹ = εₛ⁻¹_nngₛ⁻¹_ngvdₛ⁻¹(ω,geom,grid)
	H = fft( tc( Hₜ,mns ), (2:1+ND) )
	return H
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
			E1norm = dot(E1,_dot(slice_inv(copy(nng.data)),E1)) * δ(grid)
		else
			E1norm = dot(E1,_dot(nng,E1)) * δ(grid)
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

"""
######################################################################################
#
#		methods for calculating the group index from a mode field
#
######################################################################################
"""

"""
	`group_index(k::Real,evec,ω::Real,ε⁻¹,∂ε_∂ω,grid)`

Calculate the modal group index `ng = d|k|/dω` from the 
wavevector magnitude `k`, Helmholtz eigenvector `evec`, frequency `ω`, smoothed
inverse dielectric tensor and first-order dispersion `ε⁻¹` and `∂ε_∂ω`, 
and the corresponding spatial `grid<:Grid`.

This function should be compatible with reverse-mode auto-differentiation.
"""
function group_index(k::Real,evec,ω,ε⁻¹,∂ε_∂ω,grid)
    mag,mn = mag_mn(k,grid) 
	return (ω + HMH(vec(evec), _dot( ε⁻¹, ∂ε_∂ω, ε⁻¹ ),mag,mn)/2) / HMₖH(vec(evec),ε⁻¹,mag,mn)
	# note that this formula assumes (HMH(...), HMₖH(...))>0 (positive eigenvalues)
end

"""
################################################################################
#																			   #
#							Mode Filtering Functions					   	   #
#																			   #
################################################################################
"""

"""
Calculate the relative E-field power along each grid axis (integrated over space) for a mode field, and return these values as a 3-Tuple.
This is useful for categorizing/distinguishing mode polarization, for example quasi-TE modes might give values like (0.95,0.04,0.01) while
quasi-TM modes give values more like (0.01,0.98,0.01).
"""
function E_relpower_xyz(eps::AbstractArray{T,4},E::AbstractArray{Complex{T},3}) where T<:Real
    return normalize( real( @tullio E_abspower_xyz[i] := conj(E)[i,ix,iy] * eps[i,j,ix,iy] * E[j,ix,iy] ))
end

function E_relpower_xyz(eps::AbstractArray{T,5},E::AbstractArray{Complex{T},4}) where T<:Real
    return normalize( real( @tullio E_abspower_xyz[i] := conj(E)[i,ix,iy,iz] * eps[i,j,ix,iy,iz] * E[j,ix,iy,iz] ))
end

"""
Return slices (views) of E-field component `component_idx` along each axis intersecting the point at Cartesian index `pos_idx`
"""
function Eslices(E,component_idx,pos_idx::CartesianIndex{2})
    return ( view(E,component_idx,:,pos_idx[2]), view(E,component_idx,pos_idx[1],:) ) 
end

"""
Return slices (views) of E-field component `component_idx` along each axis intersecting the point at Cartesian index `pos_idx`
"""
function Eslices(E,component_idx,pos_idx::CartesianIndex{3})
    return ( view(E,component_idx,:,pos_idx[2],pos_idx[3]), view(E,component_idx,pos_idx[1],:,pos_idx[3]), view(E,component_idx,pos_idx[1],pos_idx[2],:) ) 
end

"""
Count the number of E-field zero crossings along each grid axis intersecting the peak E-field intensity maximum for a given field component `component_idx`.
This assumes that the input E-field amplitude is normalized to and real at (phase=0) the peak amplitude component-wise amplitude. 
Slices of the specified E-field component are cropped at the points where the component magnitude drops below `rel_amp_min` to avoid counting 
zero-crossings in numerical noise where the E-field magnitude is negligible.

This is useful when filtering for a specific mode order (eg. TE₀₀ or TM₂₁) in the presence of mode-crossings.
"""
function count_E_nodes(E,eps,component_idx;rel_amp_min=0.1) 
    peak_idx = argmax(real(_3dot(E,eps,E)[component_idx,..]))
    E_slcs = Eslices(real(E),component_idx,peak_idx)
    node_counts = map(E_slcs) do E_slice
        min_idx = findfirst(x->(x>rel_amp_min),abs.(E_slice))
        max_idx = findlast(x->(x>rel_amp_min),abs.(E_slice))
        n_zero_xing = sum(abs.(diff(sign.(E_slice[min_idx:max_idx]))))
        return n_zero_xing
    end
    return node_counts
end

"""
Return `true` if the following conditions are met:
    (1) The mode E-field `E` is dominantly polarized along axis index `pol_idx`
    (2) The Hermite-Gaussian mode order computed by `count_E_nodes` is equal to `mode_order`
Otherwise, return `false`.  
"""
function mode_viable(E,eps;pol_idx=1,mode_order=(0,0),rel_amp_min=0.4)
    Epwr = E_relpower_xyz(eps,E)
    Epol_axind = argmax(Epwr)
    E_mode_order = count_E_nodes(E,eps,Epol_axind;rel_amp_min)
    return ( isequal(argmax(Epwr),pol_idx) && isequal(E_mode_order,mode_order) )
end

"""
Identify the mode of a given order/polarization in an vector of
eigenmode E-fields `Es`, length `nbands`, found by solving Helmholtz Equation at a single frequency.
"""
function mode_idx(Es::AbstractVector,eps;pol_idx=1,mode_order=(0,0),rel_amp_min=0.4)
    return findfirst(EE->mode_viable(EE,eps;pol_idx,mode_order,rel_amp_min),Es)
end

