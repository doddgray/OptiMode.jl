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

@inline function unflat(f,nvec::Int,Ns)
	reshape(f,(nvec,Ns...))
end

@inline function unflat(f; ms::ModeSolver)
	Ns = size(ms.grid)
	ratio = length(f) //  N(ms.grid) |> Int # (length of vector) / (number of grid points)
	reshape(f,(ratio,Ns...))
end

mn(ms::ModeSolver) = vcat(reshape(ms.M̂.m,(1,3,size(ms.grid)...)),reshape(ms.M̂.n,(1,3,size(ms.grid)...)))



function H⃗(ms::ModeSolver{ND,T}; svecs=true) where {ND,T<:Real}
	Harr = fft( tc(unflat(ms.H⃗;ms),mn(ms)), (2:1+ND) ) #.* ms.M̂.Ninv
	svecs ? reinterpret(reshape, SVector{3,Complex{T}},  Harr) : Harr
end
H⃗x(ms::ModeSolver) = H⃗(ms;svecs=false)[1,eachindex(ms.grid)...]
H⃗y(ms::ModeSolver) = H⃗(ms;svecs=false)[2,eachindex(ms.grid)...]
H⃗z(ms::ModeSolver) = H⃗(ms;svecs=false)[3,eachindex(ms.grid)...]

function E⃗(ms::ModeSolver{ND,T}; svecs=true) where {ND,T<:Real}
	Earr = ε⁻¹_dot( fft( kx_tc( unflat(ms.H⃗; ms),mn(ms),ms.M̂.mag), (2:1+ND) ), ms.M̂.ε⁻¹ )
	svecs ? reinterpret(reshape, SVector{3,Complex{T}},  Earr) : Earr
end
E⃗x(ms::ModeSolver) = E⃗(ms;svecs=false)[1,eachindex(ms.grid)...]
E⃗y(ms::ModeSolver) = E⃗(ms;svecs=false)[2,eachindex(ms.grid)...]
E⃗z(ms::ModeSolver) = E⃗(ms;svecs=false)[3,eachindex(ms.grid)...]

import Base: abs2
abs2(v::SVector) = real(dot(v,v))

function S⃗(ms::ModeSolver{ND,T}; svecs=true) where {ND,T<:Real}
	Ssvs = real.( cross.( conj.(E⃗(ms)), H⃗(ms) ) )
	svecs ? Ssvs : reshape( reinterpret( Complex{T},  Ssvs), (3,size(ms.grid)...))
end

S⃗x(ms::ModeSolver) = real.( getindex.( cross.( conj.(E⃗(ms)), H⃗(ms) ), 1) )
S⃗y(ms::ModeSolver) = real.( getindex.( cross.( conj.(E⃗(ms)), H⃗(ms) ), 2) )
S⃗z(ms::ModeSolver) = real.( getindex.( cross.( conj.(E⃗(ms)), H⃗(ms) ), 3) )

function normE!(ms)
	E = E⃗(ms;svecs=false)
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
