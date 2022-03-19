"""
################################################################################
#																			   #
#		Methods for conversion between field types, interpolation, etc.		   #
#																			   #
################################################################################
"""

export unflat, _d2ẽ!, _H2d!, _H2e!, E⃗, E⃗x, E⃗y, E⃗z, H⃗, H⃗x, H⃗y, H⃗z, S⃗, S⃗x, S⃗y, S⃗z
export normE!, Ex_norm, Ey_norm, val_magmax, ax_magmax, idx_magmax,
		 canonicalize_phase, canonicalize_phase! # , mn
# export _cross, _cross_x, _cross_y, _cross_z, _sum_cross, _sum_cross_x, _sum_cross_y, _sum_cross_z
# export _outer
 #, slice_inv 	# stuff to get rid of soon


"""
In-place/mutating methods
"""
function _H2d!(d::AbstractArray{Complex{T},N}, Hin::AbstractArray{Complex{T},N},
	m::AbstractArray{T,N}, n::AbstractArray{T,N}, mag::AbstractArray{T},
	𝓕!::FFTW.cFFTWPlan)::AbstractArray{Complex{T},N} where {T<:Real,N}
    kx_tc!(d,Hin,m,n,mag);
    mul!(d.data,𝓕!,d.data);
	return d
end

function _d2ẽ!(e::AbstractArray{Complex{T},N}, d::AbstractArray{Complex{T},N},
	ε⁻¹,m::AbstractArray{T,N}, n::AbstractArray{T,N}, mag::AbstractArray{T},
	𝓕⁻¹!::FFTW.cFFTWPlan)::AbstractArray{Complex{T},N} where {T<:Real,N}
    eid!(e,ε⁻¹,d);
    mul!(e.data,𝓕⁻¹!,e.data);
	return e
end
##
function _H2d!(d::AbstractArray{Complex{T},4}, Hin::AbstractArray{Complex{T},4},
	M̂::HelmholtzMap{3,T})::AbstractArray{Complex{T},4} where T<:Real
    kx_tc!(d,Hin,M̂.m,M̂.n,M̂.mag);
    mul!(d.data,M̂.𝓕!,d.data);
	return d
end

function _d2ẽ!(e::AbstractArray{Complex{T},4}, d::AbstractArray{Complex{T},4},
	M̂::HelmholtzMap{3,T})::AbstractArray{Complex{T},4} where T<:Real
    eid!(e,M̂.ε⁻¹,d);
    mul!(e.data,M̂.𝓕⁻¹!,e.data);
	return e
end

function _H2d!(d::AbstractArray{Complex{T},3}, Hin::AbstractArray{Complex{T},3},
	M̂::HelmholtzMap{2,T})::AbstractArray{Complex{T},3} where T<:Real
    kx_tc!(d,Hin,M̂.m,M̂.n,M̂.mag);
    mul!(d.data,M̂.𝓕!,d.data);
	return d
end

# function _H2e!(e, Hin, M̂::HelmholtzMap{ND,T}) where {ND,T<:Real}
#     _H2d!(e,Hin,M̂);
# 	eid!(e,M̂.ε⁻¹,e);
# 	return e
# end

function _d2ẽ!(e::AbstractArray{Complex{T},3}, d::AbstractArray{Complex{T},3},
	M̂::HelmholtzMap{2,T})::AbstractArray{Complex{T},3} where T<:Real
    eid!(e,M̂.ε⁻¹,d);
    mul!(e.data,M̂.𝓕⁻¹!,e.data);
	return e
end

##
function _H2d!(d::AbstractArray{Complex{T},4}, Hin::AbstractArray{Complex{T},4},
	ms::ModeSolver{3,T})::AbstractArray{Complex{T},4} where T<:Real
    kx_tc!(d,Hin,ms.M̂.m,ms.M̂.n,ms.M̂.mag);
    mul!(d.data,ms.M̂.𝓕!,d.data);
	return d
end

function _d2ẽ!(e::AbstractArray{Complex{T},4}, d::AbstractArray{Complex{T},4},
	ms::ModeSolver{3,T})::AbstractArray{Complex{T},4} where T<:Real
    eid!(e,ms.M̂.ε⁻¹,d);
    mul!(e.data,ms.M̂.𝓕⁻¹!,e.data);
	return e
end

function _H2d!(d::AbstractArray{Complex{T},3}, Hin::AbstractArray{Complex{T},3},
	ms::ModeSolver{2,T})::AbstractArray{Complex{T},3} where T<:Real
    kx_tc!(d,Hin,ms.M̂.m,ms.M̂.n,ms.M̂.mag);
    mul!(d.data,ms.M̂.𝓕!,d.data);
	return d
end

function _d2ẽ!(e::AbstractArray{Complex{T},3}, d::AbstractArray{Complex{T},3},
	ms::ModeSolver{2,T})::AbstractArray{Complex{T},3} where T<:Real
    eid!(e,ms.M̂.ε⁻¹,d);
    mul!(e.data,ms.M̂.𝓕⁻¹!,e.data);
	return e
end

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
	# nev = 1 # size(f,2)
	# ratio = length(f) // ( nev * N(grid) ) |> Int # (length of vector) / (number of grid points)
	# [reshape(f[:,i],(ratio,Ns...)) for i=1:nev]
	ratio = length(f) //  N(grid) |> Int # (length of vector) / (number of grid points)
	reshape(f,(ratio,Ns...))
end

function H⃗(ms::ModeSolver{ND,T}; svecs=true) where {ND,T<:Real}
	Harr = [ fft( tc(unflat(ms.H⃗;ms)[eigind],ms.M̂.mn), (2:1+ND) ) for eigind=1:size(ms.H⃗,2) ]#.* ms.M̂.Ninv
	# svecs ? [ reinterpret(reshape, SVector{3,Complex{T}},  Harr[eigind]) for eigind=1:size(ms.H⃗,2) ] : Harr
	return Harr
end

function H⃗(k,H⃗::AbstractArray{Complex{T}},ω::T,geom::Geometry,grid::Grid{ND}; svecs=true, normalized=true) where {ND,T<:Real}
	Ns = size(grid)
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	Hₜ = reshape(H⃗,(2,Ns...))
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
	# ε⁻¹ = εₛ⁻¹(ω,geom,grid)
	ε⁻¹, nng⁻¹, ngvd⁻¹ = εₛ⁻¹_nngₛ⁻¹_ngvdₛ⁻¹(ω,geom,grid)
	H = fft( tc( Hₜ,mns ), (2:1+ND) )
	# if normalized
	# 	imagmax = argmax(abs2.(E0))
	# 	E1 = E0 / E0[imagmax]
	# 	E1s = reinterpret(reshape, SVector{3,Complex{T}},  E1)
	# 	E1norm = sum(dot.(E1s, inv.(nng⁻¹) .* E1s )) * δ(grid)
	# 	E = E1 / sqrt(E1norm)
	# else
	# 	E = E0
	# end
	# svecs ?  reinterpret(reshape, SVector{3,Complex{T}},  H) : H
	return H
end

function H⃗(k,Hv::AbstractArray{Complex{T}},ω::T,ε⁻¹,nng,grid::Grid{ND}; normalized=true, nnginv=false)::Array{Complex{T},ND+1} where {ND,T<:Real}
	Ns = size(grid)
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	Hₜ = reshape(Hv,(2,Ns...))
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
	# H0 = fft( tc( Hₜ,mns ), (2:1+ND) )
	if normalized
		E0 = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
		imagmax = argmax(abs2.(E0))
		E0magmax = copy(E0[imagmax])
		E1 = E0 / E0[imagmax]
		# E1s = reinterpret(reshape, SVector{3,Complex{T}},  E1)
		if nnginv
			E1norm = dot(E1,_dot(slice_inv(copy(nng.data)),E1)) * δ(grid)
		else
			E1norm = dot(E1,_dot(nng,E1)) * δ(grid)
		end
		E = E1 / sqrt(E1norm)
		# fieldnorm = E0magmax * sqrt(E1norm)
		H1 = fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) ) / ω
		# H1 = -1.0 * fft( tc( Hₜ,mns ), (2:1+ND) ) / E0magmax / sqrt(E1norm) / ω
	else
		# E = E0
		# fieldnorm = 1.0
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

# function E⃗(k,H⃗::AbstractArray{Complex{T}},ω::T,geom::Geometry,grid::Grid{ND}; normalized=true)::Array{Complex{T},ND+1} where {ND,T<:Real}
# 	Ns = size(grid)
# 	mag,m⃗,n⃗ = mag_m_n(k,grid)
# 	H = reshape(H⃗,(2,Ns...))
# 	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
# 	# ε⁻¹ = εₛ⁻¹(ω,geom,grid)
# 	ε⁻¹, nng⁻¹, ngvd⁻¹ = εₛ⁻¹_nngₛ⁻¹_ngvdₛ⁻¹(ω,geom,grid)
# 	E0 = 1im * ε⁻¹_dot( fft( kx_tc( H,mns,mag), (2:1+ND) ), flat(ε⁻¹))
# 	if normalized
# 		imagmax = argmax(abs2.(E0))
# 		E1 = E0 / E0[imagmax]
# 		E1s = reinterpret(reshape, SVector{3,Complex{T}},  E1)
# 		E1norm = sum(dot.(E1s, inv.(nng⁻¹) .* E1s )) * δ(grid)
# 		E = E1 / sqrt(E1norm)
# 	else
# 		E = E0
# 	end
# 	#svecs ?  reinterpret(reshape, SVector{3,Complex{T}},  E) : E
# 	return E
# end

function E⃗(k,evec::AbstractArray{Complex{T}},ε⁻¹,∂ε_∂ω,grid::Grid{ND}; canonicalize=true, normalized=true)::Array{Complex{T},ND+1} where {ND,T<:Real}
	evec_gridshape = reshape(evec,(2,size(grid)...))
	magmn = mag_mn(k,grid)
	E0 = 1im * ε⁻¹_dot( fft( kx_tc( evec_gridshape,last(magmn),first(magmn)), (2:1+ND) ), ε⁻¹)
	
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
	canonicalize_phase!(ms::ModeSolver[, eig_idx::Int])

Canonicalize the phase of one or all eigenmodes in a ModeSolver struct.

This shifts the phase of each mode field such that the largest magnitude
component of the corresponding Electric field `E⃗` is purely real.
"""
# function canonicalize_phase!(ms::ModeSolver,eig_idx::Int)
#     ms.H⃗[:,eig_idx] = cis(-angle(val_magmax(E⃗(ms,eig_idx)))) * ms.H⃗[:,eig_idx]
#     return nothing
# end
# function canonicalize_phase!(ms::ModeSolver)
#     for eig_idx=1:size(ms.H⃗,2)
#         canonicalize_phase!(ms,eig_idx)
#     end
#     return nothing
# end
# function canonicalize_phase!(evec::AbstractVector{Complex{T}}, ms::ModeSolver) where {T<:Real}
#     evec *= cis(-angle(val_magmax(E⃗(evec,ms))))
#     return nothing
# end
# function canonicalize_phase!(evecs::AbstractVector{TV}, ms::ModeSolver) where {T<:Real,TV<:AbstractVector{Complex{T}}}
#     # canonicalize_phase!.(evecs,(ms,))
# 	# foreach(ev->canonicalize_phase!(ev,ms),evecs)
# 	for ev in evecs
# 		ev *= cis(-angle(val_magmax(E⃗(ev,ms))))
# 	end
#     return nothing
# end
# function canonicalize_phase(evec::AbstractVector{Complex{T}}, ms::ModeSolver) where {T<:Real}
#     return evec * cis(-angle(val_magmax(E⃗(evec,ms))))
# end
# function canonicalize_phase(evecs::AbstractVector{TV}, ms::ModeSolver) where {T<:Real,TV<:AbstractVector{Complex{T}}}
#     # return canonicalize_phase.(evecs,(ms,))
# 	return map(ev->canonicalize_phase(ev,ms),evecs)
# end


function canonicalize_phase(evec::AbstractArray{Complex{T}},k::T,ε⁻¹,grid::Grid{ND}) where {ND,T<:Real}
	return evec * cis(-angle(val_magmax(E⃗(k,evec,ε⁻¹,ε⁻¹,grid;canonicalize=false,normalized=false))))     
end

function canonicalize_phase!(evec::AbstractArray{Complex{T}},k::T,ε⁻¹,grid::Grid{ND}) where {ND,T<:Real}
	evec *= cis(-angle(val_magmax(E⃗(k,evec,ε⁻¹,ε⁻¹,grid;canonicalize=false,normalized=false))))
	return nothing
end

# function canonicalize_phase(evecs::AbstractVector{TV}, ms::ModeSolver) where {T<:Real,TV<:AbstractVector{Complex{T}}}
#     # return canonicalize_phase.(evecs,(ms,))
# 	return map(ev->canonicalize_phase(ev,ms),evecs)
# end



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

# function ngdisp1(ω,H,geom,mag,m,n)
# 	ω = sqrt(real(ms.ω²[eigind]))
# 	ω / HMₖH(ms.H⃗[:,eigind],nngₛ(ω,geom;ms),ms.M̂.mag,ms.M̂.m,ms.M̂.n)
# end
#
# function ngdisp1(ms,eigind)
# 	ω = sqrt(real(ms.ω²[eigind]))
# 	ω / HMₖH(ms.H⃗[:,eigind],nngₛ(ω,geom;ms),ms.M̂.mag,ms.M̂.m,ms.M̂.n)
# end
# ( sum( dot.( ( inv.(ms.M̂.ε⁻¹) .* E⃗(ms)[1] ), E⃗(ms)[1]))*δ(ms.grid) ) ./ ( sum.(S⃗z(ms))*δ(ms.grid) )

# sum( real.( getindex.( cross.( conj.(E⃗(ms)), H⃗(ms) ), 3) ) )

# ∫(intgnd;ms::ModeSolver) = sum(ingnd)*δ(ms.grid)

# function 𝓘(ms::ModeSolver{ND,T}) where {ND,T<:Real}
#
# end






"""
################################################################################
#																			   #
#							   Plotting methods					   			   #
#																			   #
################################################################################
"""

# using Makie: heatmap
# export plot_field, plot_field!

# function plot_field!(pos,F,grid;cmap=:diverging_bkr_55_10_c35_n256,label_base=["x","y"],label="E",xlim=nothing,ylim=nothing,axind=1)
# 	xs = x(grid)
# 	ys = y(grid)
# 	xlim = isnothing(xlim) ? Tuple(extrema(xs)) : xlim
# 	ylim = isnothing(ylim) ? Tuple(extrema(ys)) : ylim

# 	# ax = [Axis(pos[1,j]) for j=1:2]
# 	# ax = [Axis(pos[j]) for j=1:2]
# 	labels = label.*label_base
# 	Fs = [view(F,j,:,:) for j=1:2]
# 	magmax = maximum(abs,F)
# 	hm = heatmap!(pos, xs, ys, real(Fs[axind]),colormap=cmap,label=labels[1],colorrange=(-magmax,magmax))
# 	# ax1 = pos[1]
# 	# hms = [heatmap!(pos[j], xs, ys, real(Fs[j]),colormap=cmap,label=labels[j],colorrange=(-magmax,magmax)) for j=1:2]
# 	# hm1 = heatmap!(ax1, xs, ys, real(Fs[1]),colormap=cmap,label=labels[1],colorrange=(-magmax,magmax))
# 	# ax2 = pos[2]
# 	# hm2 = heatmap!(ax2, xs, ys, real(Fs[2]),colormap=cmap,label=labels[2],colorrange=(-magmax,magmax))
# 	# hms = [hm1,hm2]
# 	# cbar = Colorbar(pos[1,3], heatmaps[2],  width=20 )
# 	# wfs_E = [wireframe!(ax_E[j], xs, ys, Es[j], colormap=cmap_E,linewidth=0.02,color=:white) for j=1:2]
# 	# map( (axx,ll)->text!(axx,ll,position=(-1.4,1.1),textsize=0.7,color=:white), ax, labels )
# 	# hideydecorations!.(ax[2])
# 	# [ax[1].ylabel= "y [μm]" for axx in ax[1:1]]
# 	# for axx in ax
# 	# 	axx.xlabel= "x [μm]"
# 	# 	xlims!(axx,xlim)
# 	# 	ylims!(axx,ylim)
# 	# 	axx.aspect=DataAspect()
# 	# end
# 	# linkaxes!(ax...)
# 	return hm
# end

# function plot_field(F,grid;cmap=:diverging_bkr_55_10_c35_n256,label_base=["x","y"],label="E",xlim=nothing,ylim=nothing)
# 	fig=Figure()
# 	ax = fig[1,1] = Axis(fig)
# 	hms = plot_field!(ax,F,grid;cmap,label_base,label,xlim,ylim)
# 	fig
# end
