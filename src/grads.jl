using Zygote: @adjoint, Numeric, literal_getproperty, accum
export sum2, jacobian

# @adjoint (T::Type{<:SArray})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
# @adjoint (T::Type{<:SArray})(x::AbstractArray) = T(x), dv -> (nothing, dv)
# @adjoint (T::Type{<:SMatrix})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
# @adjoint (T::Type{<:SMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
# @adjoint (T::Type{<:SVector})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
# @adjoint (T::Type{<:SVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)
# @adjoint (T::Type{<:HybridArray})(x::AbstractArray) = T(x), dv -> (nothing, dv)
#

# AD rules for array Constructors
ChainRulesCore.rrule(T::Type{<:SArray}, xs::Number...) = ( T(xs...), dv -> (nothing, dv...) )
ChainRulesCore.rrule(T::Type{<:SArray}, x::AbstractArray) = ( T(x), dv -> (nothing, dv) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, xs::Number...) = ( T(xs...), dv -> (nothing, dv...) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, x::AbstractMatrix) = ( T(x), dv -> (nothing, dv) )
ChainRulesCore.rrule(T::Type{<:SVector}, xs::Number...) = ( T(xs...), dv -> (nothing, dv...) )
ChainRulesCore.rrule(T::Type{<:SVector}, x::AbstractVector) = ( T(x), dv -> (nothing, dv) )
ChainRulesCore.rrule(T::Type{<:HybridArray}, x::AbstractArray) = ( T(x), dv -> (nothing, dv) )

# AD rules for reinterpreting back and forth between N-D arrays of SVectors and (N+1)-D arrays
function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{T1},A::AbstractArray{SVector{N1,T1},N2}) where {T1,N1,N2}
	return ( reinterpret(reshape,T1,A), Δ->( NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret( reshape,SVector{N1,T1}, Δ ) ) )
end
function ChainRulesCore.rrule(::typeof(reinterpret),reshape,type::Type{<:SVector{N1,T1}},A::AbstractArray{T1}) where {T1,N1}
	return ( reinterpret(reshape,type,A), Δ->( NO_FIELDS, ChainRulesCore.Zero(), ChainRulesCore.Zero(), reinterpret( reshape, eltype(A), Δ ) ) )
end

# Examples of how to assert type stability for broadcasting custom types (see https://github.com/FluxML/Zygote.jl/issues/318 )
# Base.similar(bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{V}}, ::Type{T}) where {T<:Real, V<:Real3Vector} = Real3Vector(Vector{T}(undef,3))
# Base.similar(bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{V}}, ::Type{T}) where {T, V<:Real3Vector} = Array{T}(undef, size(bc))

@adjoint enumerate(xs) = enumerate(xs), diys -> (map(last, diys),)
_ndims(::Base.HasShape{d}) where {d} = d
_ndims(x) = Base.IteratorSize(x) isa Base.HasShape ? _ndims(Base.IteratorSize(x)) : 1
@adjoint function Iterators.product(xs...)
                    d = 1
                    Iterators.product(xs...), dy -> ntuple(length(xs)) do n
                        nd = _ndims(xs[n])
                        dims = ntuple(i -> i<d ? i : i+nd, ndims(dy)-nd)
                        d += nd
                        func = sum(y->y[n], dy; dims=dims)
                        ax = axes(xs[n])
                        reshape(func, ax)
                    end
                end


function sum2(op,arr)
    return sum(op,arr)
end

function sum2adj( Δ, op, arr )
    n = length(arr)
    g = x->Δ*Zygote.gradient(op,x)[1]
    return ( nothing, map(g,arr))
end

@adjoint function sum2(op,arr)
    return sum2(op,arr),Δ->sum2adj(Δ,op,arr)
end

# now-removed Zygote trick to improve stability of `norm` pullback
# found referenced here: https://github.com/JuliaDiff/ChainRules.jl/issues/338
function Zygote._pullback(cx::Zygote.AContext, ::typeof(norm), x::AbstractArray, p::Real = 2)
  fallback = (x, p) -> sum(abs.(x).^p .+ eps(0f0)) ^ (one(eltype(x)) / p) # avoid d(sqrt(x))/dx == Inf at 0
  Zygote._pullback(cx, fallback, x, p)
end

"""
jacobian(f,x) : stolen from https://github.com/FluxML/Zygote.jl/pull/747/files

Construct the Jacobian of `f` where `x` is a real-valued array
and `f(x)` is also a real-valued array.
"""
function jacobian(f,x)
    y,back  = Zygote.pullback(f,x)
    k  = length(y)
    n  = length(x)
    J  = Matrix{eltype(y)}(undef,k,n)
    e_i = fill!(similar(y), 0)
    @inbounds for i = 1:k
        e_i[i] = oneunit(eltype(x))
        J[i,:] = back(e_i)[1]
        e_i[i] = zero(eltype(x))
    end
    (J,)
end

# @adjoint function solve_k(ω::Number,ε⁻¹::AbstractArray)
# 	Ω = solve_k(ω,ε⁻¹)
# 	#H̄::AbstractArray, k̄::Number
# 	Ω, Δ -> begin
# 		H, kz = Ω
# 		H̄, k̄ = Δ
# 		# hacky handling of non-differentiated parameters for now
# 		eigind = 1
# 		Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
# 		gx = collect(fftfreq(Nx,Nx/6.0))
# 		gy = collect(fftfreq(Ny,Ny/4.0))
# 		gz = collect(fftfreq(Nz,Nz/1.0))
# 		## end hacky parameter handling
#
# 		P = LinearMap(x -> H[:,eigind] * dot(H[:,eigind],x),length(H[:,eigind]),ishermitian=true)
# 		A = M̂(ε⁻¹,kz,gx,gy,gz) - ω^2 * I
# 		b = ( I  -  P ) * H̄[:,eigind]
# 		λ⃗₀ = IterativeSolvers.bicgstabl(A,b,3)
# 		ωₖ = real( ( H[:,eigind]' * Mₖ(H[:,eigind], ε⁻¹,kz,gx,gy,gz) )[1]) / ω # ds.ω²ₖ / ( 2 * ω )
# 		# Hₖ =  ( I  -  P ) * * ( Mₖ(H[:,eigind], ε⁻¹,ds) / ω )
# 		ω̄  =  ωₖ * real(k̄)
# 		λ⃗₀ -= P*λ⃗₀ - ω̄  * H
# 		Ha = reshape(H,(2,Nx,Ny,Nz))
# 		Ha_F =  fft(kcross_t2c(Ha,kz,gx,gy,gz),(2:4))
# 		λ₀ = reshape(λ⃗₀,(2,Nx,Ny,Nz))
# 		λ₀_F  = fft(kcross_t2c(λ₀,kz,gx,gy,gz),(2:4))
# 		# ε̄ ⁻¹ = ( 𝓕 * kcross_t2c(λ₀,ds) ) .* ( 𝓕 * kcross_t2c(Ha,ds) )
# 		# ε⁻¹_bar = [ Diagonal( real.(λ₀_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) ) for i=1:Nx,j=1:Ny,kk=1:Nz]
# 		ε⁻¹_bar = [ Diagonal( real.(λ₀_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) )[a,b] for a=1:3,b=1:3,i=1:Nx,j=1:Ny,kk=1:Nz]
# 		return ω̄ , ε⁻¹_bar
# 	end
# end

### Zygote StructArrays rules from https://github.com/cossio/ZygoteStructArrays.jl
@adjoint function (::Type{SA})(t::Tuple) where {SA<:StructArray}
    sa = SA(t)
    back(Δ::NamedTuple) = (values(Δ),)
    function back(Δ::AbstractArray{<:NamedTuple})
        nt = (; (p => [getproperty(dx, p) for dx in Δ] for p in propertynames(sa))...)
        return back(nt)
    end
    return sa, back
end

@adjoint function (::Type{SA})(t::NamedTuple) where {SA<:StructArray}
    sa = SA(t)
    back(Δ::NamedTuple) = (NamedTuple{propertynames(sa)}(Δ),)
    function back(Δ::AbstractArray)
        back((; (p => [getproperty(dx, p) for dx in Δ] for p in propertynames(sa))...))
    end
    return sa, back
end

@adjoint function (::Type{SA})(a::A) where {T,SA<:StructArray,A<:AbstractArray{T}}
    sa = SA(a)
    function back(Δsa)
        Δa = [(; (p => Δsa[p][i] for p in propertynames(Δsa))...) for i in eachindex(a)]
        return (Δa,)
    end
    return sa, back
end

# Must special-case for Complex (#1)
@adjoint function (::Type{SA})(a::A) where {T<:Complex,SA<:StructArray,A<:AbstractArray{T}}
    sa = SA(a)
    function back(Δsa) # dsa -> da
        Δa = [Complex(Δsa.re[i], Δsa.im[i]) for i in eachindex(a)]
        (Δa,)
    end
    return sa, back
end

@adjoint function literal_getproperty(sa::StructArray, ::Val{key}) where {key}
    key::Symbol
    result = getproperty(sa, key)
    function back(Δ::AbstractArray)
        nt = (; (k => zero(v) for (k,v) in pairs(fieldarrays(sa)))...)
        return (Base.setindex(nt, Δ, key), nothing)
    end
    return result, back
end

@adjoint Base.getindex(sa::StructArray, i...) = sa[i...], Δ -> ∇getindex(sa,i,Δ)
@adjoint Base.view(sa::StructArray, i...) = view(sa, i...), Δ -> ∇getindex(sa,i,Δ)
function ∇getindex(sa::StructArray, i, Δ::NamedTuple)
    dsa = (; (k => ∇getindex(v,i,Δ[k]) for (k,v) in pairs(fieldarrays(sa)))...)
    di = map(_ -> nothing, i)
    return (dsa, map(_ -> nothing, i)...)
end
# based on
# https://github.com/FluxML/Zygote.jl/blob/64c02dccc698292c548c334a15ce2100a11403e2/src/lib/array.jl#L41
∇getindex(a::AbstractArray, i, Δ::Nothing) = nothing
function ∇getindex(a::AbstractArray, i, Δ)
    if i isa NTuple{<:Any, Integer}
        da = Zygote._zero(a, typeof(Δ))
        da[i...] = Δ
    else
        da = Zygote._zero(a, eltype(Δ))
        dav = view(da, i...)
        dav .= Zygote.accum.(dav, Zygote._droplike(Δ, dav))
    end
    return da
end

@adjoint function (::Type{NT})(t::Tuple) where {K,NT<:NamedTuple{K}}
    nt = NT(t)
    back(Δ::NamedTuple) = (values(NT(Δ)),)
    return nt, back
end

# # https://github.com/FluxML/Zygote.jl/issues/680
# @adjoint function (T::Type{<:Complex})(re, im)
# 	back(Δ::Complex) = (nothing, real(Δ), imag(Δ))
# 	back(Δ::NamedTuple) = (nothing, Δ.re, Δ.im)
# 	T(re, im), back
# end



#### ChainRules

# @non_differentiable MaxwellGrid(Δx::Real,Δy::Real,Δz::Real,Nx::Int,Ny::Int,Nz::Int)
# @non_differentiable MaxwellData(kz::Real,g::MaxwellGrid)

# function ChainRulesCore.rrule(::typeof(solve_k), ω::Number, ε⁻¹::Array{<:Union{Real,Complex}})
# function ChainRulesCore.rrule(::typeof(solve_k), ω::Number, ε⁻¹::Array{Float64,5},Δx::Real,Δy::Real,Δz::Real;kguess=k_guess(ω,ε⁻¹),neigs=1,eigind=1,maxiter=3000,tol=1e-8,l=2)
#     Ω = solve_k(ω,ε⁻¹,Δx,Δy,Δz;neigs,eigind,maxiter,tol)
#     function solve_k_pullback(ΔΩ) # ω̄ ₖ)
#         H, kz = Ω
# 		H̄, k̄ = ΔΩ
# 		Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
# 		Ha = reshape(H,(2,Nx,Ny,Nz))
# 		kpg_mag, mn = calc_kpg(kz,Δx,Δy,Δz,Nx,Ny,Nz)
# 		ωₖ = H_Mₖ_H(Ha,ε⁻¹,kpg_mag,mn) / ω
# 		ω̄  = k̄ / ωₖ  #  ωₖ * k̄
# 		ω̄sq =  ω̄  / (2ω)
# 		𝓕 = plan_fft(randn(ComplexF64, (3,Nx,Ny,Nz)))
# 		𝓕⁻¹ = plan_ifft(randn(ComplexF64, (3,Nx,Ny,Nz)))
# 		if typeof(H̄)==ChainRulesCore.Zero
# 			# return (NO_FIELDS, ω̄ , ChainRulesCore.Zero(),ChainRulesCore.Zero(),ChainRulesCore.Zero(),ChainRulesCore.Zero())
# 			λ⃗ =  ω̄sq * H[:,eigind]  # ( ( 2 * k̄ / ωₖ ) * H )
# 		else
# 			H̄a = reshape(H̄,(2,Nx,Ny,Nz))
# 			P = LinearMap(x -> H[:,eigind] * dot(H[:,eigind],x),length(H[:,eigind]),ishermitian=true)
# 			A = M̂(ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹) - ω^2 * I
# 			b = ( I  -  P ) * H̄[:,eigind]
# 			λ⃗₀ = IterativeSolvers.bicgstabl(A,b,l)
# 			λ⃗ = λ⃗₀ - ω̄sq * H[:,eigind]
# 			λ⃗a = reshape(λ⃗,(2,Nx,Ny,Nz))
# 			# ωₖ = real( ( H[:,eigind]' * Mₖ(H[:,eigind], ε⁻¹,kz,gx,gy,gz) )[1]) / ω # ds.ω²ₖ / ( 2 * ω )
# 			# H̄_dot_Hₖ =  dot(H̄[:,eigind],( I  -  P ) * vec(Mₖ(Ha, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)))
# 			# H̄_dot_Hₖ =  dot(( I  -  P ) * H̄[:,eigind], vec(Mₖ(Ha, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)))
# 			# H̄_dot_Hₖ =  dot(H̄[:,eigind],vec(Mₖ(Ha, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)))
# 			# H̄_dot_Hₖ =  dot(λ⃗₀,vec(Mₖ(Ha, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹))) / ω^2
# 			# H̄_dot_Hₖ =  dot(H,vec(Mₖ(λ⃗₀a, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹))) /  ω #ωₖ
# 			 # H̄_dot_Hₖ =  dot(λ⃗₀,vec(Mₖ(H̄a, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)))
# 			# ω̄  += H̄_dot_Hₖ
# 			# ω̄  +=  ωₖ / H̄_dot_Hₖ
# 			# λ⃗₀ -= P*λ⃗₀ - ( ( 2 * k̄ / ωₖ ) * H )
# 		end
# 		Ha_F = 𝓕 * kx_t2c(Ha,mn,kpg_mag) #fft(kcross_t2c(Ha,kz,gx,gy,gz),(2:4))
# 		λ₀ = reshape(λ⃗₀,(2,Nx,Ny,Nz))
# 		λ₀_F  = 𝓕 * kx_t2c(λ₀,mn,kpg_mag) #fft(kcross_t2c(λ₀,kz,gx,gy,gz),(2:4))
# 		# ε̄ ⁻¹ = ( 𝓕 * kcross_t2c(λ₀,ds) ) .* ( 𝓕 * kcross_t2c(Ha,ds) )
# 		ε⁻¹_bar = [ Diagonal( real.(λ₀_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) )[a,b] for a=1:3,b=1:3,i=1:Nx,j=1:Ny,kk=1:Nz]
# 		return (NO_FIELDS, ω̄ , ε⁻¹_bar,ChainRulesCore.Zero(),ChainRulesCore.Zero(),ChainRulesCore.Zero())
#     end
#     return (Ω, solve_k_pullback)
# end

function ChainRulesCore.rrule(::typeof(solve_ω²), k::T, ε⁻¹::Array{T,5},Δx::T,Δy::T,Δz::T;neigs=1,eigind=1,maxiter=3000,tol=1e-8) where T<:Real
    Ω = solve_ω²(k,ε⁻¹,Δx,Δy,Δz;neigs,eigind,maxiter,tol)
    function solve_ω²_pullback(ΔΩ) # ω̄ ₖ)
        H⃗, ω² = Ω
		H̄, ω̄sq = ΔΩ
		Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
		H = reshape(H⃗[:,eigind],(2,Nx,Ny,Nz))
		(mag, mn), magmn_pb = Zygote.pullback(k) do k
		    # calc_kpg(k,make_MG(Δx, Δy, Δz, Nx, Ny, Nz).g⃗)
			calc_kpg(k,Δx,Δy,Δz,Nx,Ny,Nz)
		end
	    if typeof(ω̄sq)==ChainRulesCore.Zero
			ω̄sq = 0.
		end
		𝓕 = plan_fft(randn(ComplexF64, (3,Nx,Ny,Nz)),(2:4))
		𝓕⁻¹ = plan_ifft(randn(ComplexF64, (3,Nx,Ny,Nz)),(2:4))
		if typeof(H̄)==ChainRulesCore.Zero
			λ⃗ =  -ω̄sq * H⃗[:,eigind]
		else
			λ⃗₀ = IterativeSolvers.bicgstabl(
											M̂_old(ε⁻¹,mn,mag,𝓕,𝓕⁻¹)-ω²[eigind]*I, # A
											H̄[:,eigind] - H⃗[:,eigind] * dot(H⃗[:,eigind],H̄[:,eigind]), # b,
											3,  # "l"
											)
			λ⃗ = λ⃗₀ - (ω̄sq + dot(H⃗[:,eigind],λ⃗₀)) * H⃗[:,eigind]  # (P * λ⃗₀) + ω̄sq * H⃗[:,eigind] # λ⃗₀ + ω̄sq * H⃗[:,eigind]
		end
		λ = reshape(λ⃗,(2,Nx,Ny,Nz))
		d =  𝓕 * kx_t2c( H , mn, mag )  / (Nx * Ny * Nz) # fft( kx_t2c( H , mn, mag ) ,(2:4))  / (Nx * Ny * Nz)
		λd = 𝓕 * kx_t2c( λ, mn, mag ) # fft( kx_t2c(λ, mn, mag ),(2:4))
		d⃗ = vec( d )
		λ⃗d = vec( λd )
		# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
		λẽ = vec( 𝓕⁻¹ * ε⁻¹_dot(λd,ε⁻¹) )
		ẽ = vec( 𝓕⁻¹ * ε⁻¹_dot(d,ε⁻¹) * (Nx * Ny * Nz) ) # pre-scales needed to compensate fft/ifft normalization asymmetry. If bfft is used, this will need to be adjusted
		λẽ_3v = reinterpret(SVector{3,ComplexF64},λẽ)
		ẽ_3v = reinterpret(SVector{3,ComplexF64},ẽ)
		λ_2v = reinterpret(SVector{2,ComplexF64},λ⃗)
		H_2v = reinterpret(SVector{2,ComplexF64},H⃗[:,eigind])
		kx̄ = reshape( reinterpret(Float64, -real.( λẽ_3v .* adjoint.(conj.(H_2v)) + ẽ_3v .* adjoint.(conj.(λ_2v)) ) ), (3,2,Nx,Ny,Nz) )
		@tullio māg[ix,iy,iz] := mn[a,2,ix,iy,iz] * kx̄[a,1,ix,iy,iz] - mn[a,1,ix,iy,iz] * kx̄[a,2,ix,iy,iz]
		mn̄_signs = [-1 ; 1]
		@tullio mn̄[a,b,ix,iy,iz] := kx̄[a,3-b,ix,iy,iz] * mag[ix,iy,iz] * mn̄_signs[b] nograd=mn̄_signs
		k̄ = magmn_pb((māg,mn̄))[1]
		# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
		# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field
		ε⁻¹_bar = zeros(Float64,(3,3,Nx,Ny,Nz))
		@avx for iz=1:Nz,iy=1:Ny,ix=1:Nx
	        q = (Nz * (iz-1) + Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
	        for a=1:3 # loop over diagonal elements: {11, 22, 33}
	            ε⁻¹_bar[a,a,ix,iy,iz] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
	        end
	        for a2=1:2 # loop over first off diagonal
	            ε⁻¹_bar[a2,a2+1,ix,iy,iz] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
	        end
	        # a = 1, set 1,3 and 3,1, second off-diagonal
	        ε⁻¹_bar[1,3,ix,iy,iz] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
	    end
		return (NO_FIELDS, k̄, ε⁻¹_bar,ChainRulesCore.Zero(),ChainRulesCore.Zero(),ChainRulesCore.Zero())
    end
    return (Ω, solve_ω²_pullback)
end

function update_k_pb(M̂::HelmholtzMap{T},k⃗::SVector{3,T}) where T<:Real
	(mag, m, n), mag_m_n_pb = Zygote.pullback(k⃗) do x
		mag_m_n(x,dropgrad(M̂.g⃗))
	end
	M̂.mag = mag
	M̂.inv_mag = [inv(mm) for mm in mag]
	M̂.m⃗ = m #HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(m.parent))
	M̂.n⃗ = n #HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(n.parent))
	M̂.m = HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.m⃗))
	M̂.n = HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,T,M̂.n⃗))
	M̂.k⃗ = k⃗
	return (mag, m, n), mag_m_n_pb
end

update_k_pb(M̂::HelmholtzMap{T},kz::T) where T<:Real = update_k_pb(M̂,SVector{3,T}(0.,0.,kz))

function ChainRulesCore.rrule(::typeof(solve_ω²), ms::ModeSolver{T},k::Union{T,SVector{3,T}},ε⁻¹::AbstractArray{T,5};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
	(mag, m⃗, n⃗), mag_m_n_pb = update_k_pb(ms.M̂,k)
	Ω = solve_ω²(ms,ε⁻¹; nev, eigind, maxiter, tol, log)
    function solve_ω²_pullback(ΔΩ) # ω̄ ₖ)
        ω², H⃗ = Ω
		ω̄sq, H̄ = ΔΩ
		Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
		H = reshape(H⃗,(2,Nx,Ny,Nz))
		mn2 = vcat(reshape(ms.M̂.m,(1,3,Nx,Ny,Nz)),reshape(ms.M̂.n,(1,3,Nx,Ny,Nz)))
	    if typeof(ω̄sq)==ChainRulesCore.Zero
			ω̄sq = 0.
		end
		if typeof(H̄)==ChainRulesCore.Zero
			λ⃗ =  -ω̄sq * H⃗
		else
			λ⃗₀ = IterativeSolvers.bicgstabl(
											ms.M̂-ω²*I, # A
											H̄ - H⃗ * dot(H⃗,H̄), # b,
											3,  # "l"
											)
			λ⃗ = λ⃗₀ - (ω̄sq + dot(H⃗,λ⃗₀)) * H⃗  # (P * λ⃗₀) + ω̄sq * H⃗ # λ⃗₀ + ω̄sq * H⃗
		end
		λ = reshape(λ⃗,(2,Nx,Ny,Nz))
		d =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  / (Nx * Ny * Nz) # fft( kx_t2c( H , mn, mag ) ,(2:4))  / (Nx * Ny * Nz)
		λd = ms.M̂.𝓕 * kx_tc( λ, mn2, mag ) # fft( kx_t2c(λ, mn, mag ),(2:4))
		d⃗ = vec( d )
		λ⃗d = vec( λd )
		# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
		λẽ = reinterpret(reshape,SVector{3,Complex{T}}, (ms.M̂.𝓕⁻¹ * ε⁻¹_dot(λd,ε⁻¹) / (Nx * Ny * Nz)) )
		ẽ = reinterpret(reshape,SVector{3,Complex{T}}, (ms.M̂.𝓕⁻¹ * ε⁻¹_dot(d,ε⁻¹)) ) # pre-scales needed to compensate fft/
		kx̄_m⃗ = real.( λẽ .* conj.(view(H,2,:,:,:)) .+ ẽ .* conj.(view(λ,2,:,:,:)) )
		kx̄_n⃗ =  -real.( λẽ .* conj.(view(H,1,:,:,:)) .+ ẽ .* conj.(view(λ,1,:,:,:)) )
		māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
		# m̄ = kx̄_m⃗ .* mag
		# n̄ = kx̄_n⃗ .* mag
		k̄ = mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1]
		if !(typeof(k)<:SVector)
			k̄ = k̄[3]
		end
		# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
		# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field
		ε⁻¹_bar = HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},Float64,5,5,Array{Float64,5}}(zeros(Float64,(3,3,Nx,Ny,Nz)))
		@avx for iz=1:Nz,iy=1:Ny,ix=1:Nx
	        q = (Nz * (iz-1) + Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
	        for a=1:3 # loop over diagonal elements: {11, 22, 33}
	            ε⁻¹_bar[a,a,ix,iy,iz] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
	        end
	        for a2=1:2 # loop over first off diagonal
	            ε⁻¹_bar[a2,a2+1,ix,iy,iz] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
	        end
	        # a = 1, set 1,3 and 3,1, second off-diagonal
	        ε⁻¹_bar[1,3,ix,iy,iz] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
	    end
		return (NO_FIELDS, ChainRulesCore.Zero(), k̄, ε⁻¹_bar)
    end
    return (Ω, solve_ω²_pullback)
end

function ChainRulesCore.rrule(::typeof(solve_k), ms::ModeSolver{T},ω::T,ε⁻¹::AbstractArray{T,5};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real
	k, H⃗ = solve_k(ms,ω,ε⁻¹; nev, eigind, maxiter, tol, log ,ω²_tol)
	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
		mag_m_n(x,dropgrad(ms.M̂.g⃗))
	end
    function solve_k_pullback(ΔΩ) # ω̄ ₖ)
		k̄, H̄ = ΔΩ
		Nx,Ny,Nz = ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz
		H = reshape(H⃗,(2,Nx,Ny,Nz))
		mn2 = vcat(reshape(ms.M̂.m,(1,3,Nx,Ny,Nz)),reshape(ms.M̂.n,(1,3,Nx,Ny,Nz)))
	    if typeof(k̄)==ChainRulesCore.Zero
			k̄ = 0.
		end
		@show k̄
		@show k̄ / ms.∂ω²∂k[eigind]
		@show ω̄sq_eff = -k̄ / ms.∂ω²∂k[eigind] - 1. * ms.∂ω²∂k[eigind]
		if typeof(H̄)==ChainRulesCore.Zero
			λ⃗ =  ω̄sq_eff * H⃗ #-( k̄ / ms.∂ω²∂k[eigind] ) * H⃗
			# λ⃗ =  0 * H⃗
		else
			λ⃗₀ = IterativeSolvers.bicgstabl(
											ms.M̂-(ω^2)*I, # A
											H̄ - H⃗ * dot(H⃗,H̄), # b,
											3,  # "l"
											)
			λ⃗ = λ⃗₀ - ( ω̄sq_eff  + dot(H⃗,λ⃗₀)) * H⃗  # (P * λ⃗₀) + ω̄sq * H⃗ # λ⃗₀ + ω̄sq * H⃗
			# λ⃗ = λ⃗₀ -  dot(H⃗,λ⃗₀) * H⃗  # (P * λ⃗₀) + ω̄sq * H⃗ # λ⃗₀ + ω̄sq * H⃗
		end
		λ = reshape(λ⃗,(2,Nx,Ny,Nz))
		d =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  / (Nx * Ny * Nz) # fft( kx_t2c( H , mn, mag ) ,(2:4))  / (Nx * Ny * Nz)
		λd = ms.M̂.𝓕 * kx_tc( λ, mn2, mag ) # fft( kx_t2c(λ, mn, mag ),(2:4))
		d⃗ = vec( d )
		λ⃗d = vec( λd )
		# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
		# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field
		ε⁻¹_bar = HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},Float64,5,5,Array{Float64,5}}(zeros(Float64,(3,3,Nx,Ny,Nz)))
		@avx for iz=1:Nz,iy=1:Ny,ix=1:Nx
	        q = (Nz * (iz-1) + Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
	        for a=1:3 # loop over diagonal elements: {11, 22, 33}
	            ε⁻¹_bar[a,a,ix,iy,iz] = real( -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1]) )
	        end
	        for a2=1:2 # loop over first off diagonal
	            ε⁻¹_bar[a2,a2+1,ix,iy,iz] = real( -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2]) )
	        end
	        # a = 1, set 1,3 and 3,1, second off-diagonal
	        ε⁻¹_bar[1,3,ix,iy,iz] = real( -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q]) )
	    end


		if typeof(H̄)==ChainRulesCore.Zero
			λ⃗ =  -k̄ / ms.∂ω²∂k[eigind] * H⃗ #-( k̄ / ms.∂ω²∂k[eigind] ) * H⃗
			# λ⃗ =  0 * H⃗
		else
			λ⃗₀ = IterativeSolvers.bicgstabl(
											ms.M̂-(ω^2)*I, # A
											H̄ - H⃗ * dot(H⃗,H̄), # b,
											3,  # "l"
											)
			λ⃗ = λ⃗₀ - ( k̄ / ms.∂ω²∂k[eigind]  + dot(H⃗,λ⃗₀)) * H⃗
			# λ⃗ = λ⃗₀ -  dot(H⃗,λ⃗₀) * H⃗  # (P * λ⃗₀) + ω̄sq * H⃗ # λ⃗₀ + ω̄sq * H⃗
		end

		λ = reshape(λ⃗,(2,Nx,Ny,Nz))
		# d =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  / (Nx * Ny * Nz) # fft( kx_t2c( H , mn, mag ) ,(2:4))  / (Nx * Ny * Nz)
		λd = ms.M̂.𝓕 * kx_tc( λ, mn2, mag ) # fft( kx_t2c(λ, mn, mag ),(2:4))
		# d⃗ = vec( d )
		λ⃗d = vec( λd )

		# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
		λẽ = reinterpret(reshape,SVector{3,Complex{T}}, (ms.M̂.𝓕⁻¹ * ε⁻¹_dot(λd,ε⁻¹) / (Nx * Ny * Nz)) )
		ẽ = reinterpret(reshape,SVector{3,Complex{T}}, (ms.M̂.𝓕⁻¹ * ε⁻¹_dot(d,ε⁻¹)) ) # pre-scales needed to compensate fft/
		kx̄_m⃗ = real.( λẽ .* conj.(view(H,2,:,:,:)) .+ ẽ .* conj.(view(λ,2,:,:,:)) )
		kx̄_n⃗ =  -real.( λẽ .* conj.(view(H,1,:,:,:)) .+ ẽ .* conj.(view(λ,1,:,:,:)) )
		māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
		# m̄ = kx̄_m⃗ .* mag
		# n̄ = kx̄_n⃗ .* mag
		@show k̄_kx = mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1]
		# if !(typeof(k)<:SVector)
		# 	k̄_kx = k̄_kx[3]
		# end
		@show k̄_kx / ms.∂ω²∂k[eigind]
		@show ω̄sq = k̄_kx  / ms.∂ω²∂k[eigind] #
		@show ω̄  = ω̄sq * 2 * ω
		# @show ω̄sq = ω̄sq_eff
		# @show ω̄sq = ( -k̄_kx + k̄ ) / ms.∂ω²∂k[eigind] #

		return (NO_FIELDS, ChainRulesCore.Zero(), ω̄  , ε⁻¹_bar)
    end
    return ((k, H⃗), solve_k_pullback)
end
