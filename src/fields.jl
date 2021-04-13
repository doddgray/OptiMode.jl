"""
################################################################################
#																			   #
#		Methods for conversion between field types, interpolation, etc.		   #
#																			   #
################################################################################
"""

export unflat, _d2ẽ!, _H2d!, E⃗, E⃗x, E⃗y, E⃗z, H⃗, H⃗x, H⃗y, H⃗z, S⃗, S⃗x, S⃗y, S⃗z, normE!, Ex_norm, Ey_norm

export mn 	# stuff to get rid of soon

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
	nev = size(f,2)
	ratio = length(f) // ( nev * N(grid) ) |> Int # (length of vector) / (number of grid points)
	[reshape(f[:,i],(ratio,Ns...)) for i=1:nev]
end

mn(ms::ModeSolver) = vcat(reshape(ms.M̂.m,(1,3,size(ms.grid)...)),reshape(ms.M̂.n,(1,3,size(ms.grid)...)))



function H⃗(ms::ModeSolver{ND,T}; svecs=true) where {ND,T<:Real}
	Harr = [ fft( tc(unflat(ms.H⃗;ms)[eigind],mn(ms)), (2:1+ND) ) for eigind=1:size(ms.H⃗,2) ]#.* ms.M̂.Ninv
	svecs ? [ reinterpret(reshape, SVector{3,Complex{T}},  Harr[eigind]) for eigind=1:size(ms.H⃗,2) ] : Harr
end
H⃗x(ms::ModeSolver) = H⃗(ms;svecs=false)[1,eachindex(ms.grid)...]
H⃗y(ms::ModeSolver) = H⃗(ms;svecs=false)[2,eachindex(ms.grid)...]
H⃗z(ms::ModeSolver) = H⃗(ms;svecs=false)[3,eachindex(ms.grid)...]

function E⃗(ms::ModeSolver{ND,T}; svecs=true) where {ND,T<:Real}
	# eif = flat(ε⁻¹)
	Earr = [ 1im * ε⁻¹_dot( fft( kx_tc( unflat(ms.H⃗; ms)[eigind],mn(ms),ms.M̂.mag), (2:1+ND) ), copy(flat( ms.M̂.ε⁻¹ ))) for eigind=1:size(ms.H⃗,2) ]
	svecs ? [ reinterpret(reshape, SVector{3,Complex{T}},  Earr[eigind]) for eigind=1:size(ms.H⃗,2) ] : Earr
end

function E⃗(k,H⃗::AbstractArray{Complex{T}},ω::T,geom::AbstractVector{<:Shape},grid::Grid{ND}; svecs=true, normalized=true) where {ND,T<:Real}
	Ns = size(grid)
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	H = reshape(H⃗,(2,Ns...))
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
	# ε⁻¹ = εₛ⁻¹(ω,geom,grid)
	ε⁻¹, nng⁻¹, ngvd⁻¹ = εₛ⁻¹_nngₛ⁻¹_ngvdₛ⁻¹(ω,geom,grid)
	E0 = 1im * ε⁻¹_dot( fft( kx_tc( H,mns,mag), (2:1+ND) ), flat(ε⁻¹))
	if normalized
		imagmax = argmax(abs2.(E0))
		E1 = E0 / E0[imagmax]
		E1s = reinterpret(reshape, SVector{3,Complex{T}},  E1)
		E1norm = sum(dot.(E1s, inv.(nng⁻¹) .* E1s )) * δ(grid)
		E = E1 / sqrt(E1norm)
	else
		E = E0
	end
	svecs ?  reinterpret(reshape, SVector{3,Complex{T}},  E) : E
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
# 	ω / H_Mₖ_H(ms.H⃗[:,eigind],nngₛ(ω,geom;ms),ms.M̂.mag,ms.M̂.m,ms.M̂.n)
# end
#
# function ngdisp1(ms,eigind)
# 	ω = sqrt(real(ms.ω²[eigind]))
# 	ω / H_Mₖ_H(ms.H⃗[:,eigind],nngₛ(ω,geom;ms),ms.M̂.mag,ms.M̂.m,ms.M̂.n)
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
