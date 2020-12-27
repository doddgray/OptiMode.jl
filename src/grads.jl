using Zygote: @adjoint, Numeric, literal_getproperty, accum

export sum2, jacobian

@adjoint (T::Type{<:SArray})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
@adjoint (T::Type{<:SArray})(x::AbstractArray) = T(x), dv -> (nothing, dv)
@adjoint (T::Type{<:SMatrix})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
@adjoint (T::Type{<:SMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
@adjoint (T::Type{<:SVector})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
@adjoint (T::Type{<:SVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)


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
function ChainRulesCore.rrule(::typeof(solve_k), ω::Number, ε⁻¹::Array{Float64,5},Δx::Real,Δy::Real,Δz::Real;kguess=k_guess(ω,ε⁻¹),neigs=1,eigind=1,maxiter=3000,tol=1e-8,l=2)
    Ω = solve_k(ω,ε⁻¹,Δx,Δy,Δz;neigs,eigind,maxiter,tol)
    function solve_k_pullback(ΔΩ) # ω̄ ₖ)
        H, kz = Ω
		H̄, k̄ = ΔΩ
		Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
		Ha = reshape(H,(2,Nx,Ny,Nz))
		kpg_mag, mn = calc_kpg(kz,Δx,Δy,Δz,Nx,Ny,Nz)
		ωₖ = H_Mₖ_H(Ha,ε⁻¹,kpg_mag,mn) / ω
		ω̄  = k̄ / ωₖ  #  ωₖ * k̄
		ω̄sq =  ω̄  / (2ω)
		𝓕 = plan_fft(randn(ComplexF64, (3,Nx,Ny,Nz)))
		𝓕⁻¹ = plan_ifft(randn(ComplexF64, (3,Nx,Ny,Nz)))
		if typeof(H̄)==ChainRulesCore.Zero
			# return (NO_FIELDS, ω̄ , ChainRulesCore.Zero(),ChainRulesCore.Zero(),ChainRulesCore.Zero(),ChainRulesCore.Zero())
			λ⃗ =  ω̄sq * H[:,eigind]  # ( ( 2 * k̄ / ωₖ ) * H )
		else
			H̄a = reshape(H̄,(2,Nx,Ny,Nz))
			P = LinearMap(x -> H[:,eigind] * dot(H[:,eigind],x),length(H[:,eigind]),ishermitian=true)
			A = M̂(ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹) - ω^2 * I
			b = ( I  -  P ) * H̄[:,eigind]
			λ⃗₀ = IterativeSolvers.bicgstabl(A,b,l)
			λ⃗ = λ⃗₀ - ω̄sq * H[:,eigind]
			λ⃗a = reshape(λ⃗,(2,Nx,Ny,Nz))
			# ωₖ = real( ( H[:,eigind]' * Mₖ(H[:,eigind], ε⁻¹,kz,gx,gy,gz) )[1]) / ω # ds.ω²ₖ / ( 2 * ω )
			# H̄_dot_Hₖ =  dot(H̄[:,eigind],( I  -  P ) * vec(Mₖ(Ha, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)))
			# H̄_dot_Hₖ =  dot(( I  -  P ) * H̄[:,eigind], vec(Mₖ(Ha, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)))
			# H̄_dot_Hₖ =  dot(H̄[:,eigind],vec(Mₖ(Ha, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)))
			# H̄_dot_Hₖ =  dot(λ⃗₀,vec(Mₖ(Ha, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹))) / ω^2
			# H̄_dot_Hₖ =  dot(H,vec(Mₖ(λ⃗₀a, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹))) /  ω #ωₖ
			 # H̄_dot_Hₖ =  dot(λ⃗₀,vec(Mₖ(H̄a, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)))
			# ω̄  += H̄_dot_Hₖ
			# ω̄  +=  ωₖ / H̄_dot_Hₖ
			# λ⃗₀ -= P*λ⃗₀ - ( ( 2 * k̄ / ωₖ ) * H )
		end
		Ha_F = 𝓕 * kx_t2c(Ha,mn,kpg_mag) #fft(kcross_t2c(Ha,kz,gx,gy,gz),(2:4))
		λ₀ = reshape(λ⃗₀,(2,Nx,Ny,Nz))
		λ₀_F  = 𝓕 * kx_t2c(λ₀,mn,kpg_mag) #fft(kcross_t2c(λ₀,kz,gx,gy,gz),(2:4))
		# ε̄ ⁻¹ = ( 𝓕 * kcross_t2c(λ₀,ds) ) .* ( 𝓕 * kcross_t2c(Ha,ds) )
		ε⁻¹_bar = [ Diagonal( real.(λ₀_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) )[a,b] for a=1:3,b=1:3,i=1:Nx,j=1:Ny,kk=1:Nz]
		return (NO_FIELDS, ω̄ , ε⁻¹_bar,ChainRulesCore.Zero(),ChainRulesCore.Zero(),ChainRulesCore.Zero())
    end
    return (Ω, solve_k_pullback)
end


function ChainRulesCore.rrule(::typeof(solve_ω²), k::Number, ε⁻¹::Array{Float64,5},Δx::Real,Δy::Real,Δz::Real;neigs=1,eigind=1,maxiter=3000,tol=1e-8,l=2)
    Ω = solve_ω²(k,ε⁻¹,Δx,Δy,Δz;neigs,eigind,maxiter,tol)
    function solve_ω²_pullback(ΔΩ) # ω̄ ₖ)
        H, ω² = Ω
		H̄, ω̄sq = ΔΩ
		Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
		Ha = reshape(H,(2,Nx,Ny,Nz))
		kpg_mag, mn = calc_kpg(k,Δx,Δy,Δz,Nx,Ny,Nz)
		# ωₖ = H_Mₖ_H(Ha,ε⁻¹,kpg_mag,mn) / ω
		# ω̄  = k̄ / ωₖ  #  ωₖ * k̄
		# ω̄sq =  ω̄  / (2ω)
		ω = √(ω²)
		k̄ = 2 * H_Mₖ_H(Ha,ε⁻¹,kpg_mag,mn) * ω̄sq
		𝓕 = plan_fft(randn(ComplexF64, (3,Nx,Ny,Nz)))
		𝓕⁻¹ = plan_ifft(randn(ComplexF64, (3,Nx,Ny,Nz)))
		if typeof(H̄)==ChainRulesCore.Zero
			# return (NO_FIELDS, ω̄ , ChainRulesCore.Zero(),ChainRulesCore.Zero(),ChainRulesCore.Zero(),ChainRulesCore.Zero())
			λ⃗ =  ω̄sq * H[:,eigind]  # ( ( 2 * k̄ / ωₖ ) * H )
		else
			H̄a = reshape(H̄,(2,Nx,Ny,Nz))
			P = LinearMap(x -> H[:,eigind] * dot(H[:,eigind],x),length(H[:,eigind]),ishermitian=true)
			A = M̂(ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹) - (ω² * I)
			b = ( I  -  P ) * H̄[:,eigind]
			λ⃗₀ = IterativeSolvers.bicgstabl(A,b,l)
			λ⃗ = λ⃗₀ - ω̄sq * H[:,eigind]
			# λ⃗a = reshape(λ⃗,(2,Nx,Ny,Nz))
			# ωₖ = real( ( H[:,eigind]' * Mₖ(H[:,eigind], ε⁻¹,kz,gx,gy,gz) )[1]) / ω # ds.ω²ₖ / ( 2 * ω )
			# H̄_dot_Hₖ =  dot(H̄[:,eigind],( I  -  P ) * vec(Mₖ(Ha, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)))
			# H̄_dot_Hₖ =  dot(( I  -  P ) * H̄[:,eigind], vec(Mₖ(Ha, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)))
			# H̄_dot_Hₖ =  dot(H̄[:,eigind],vec(Mₖ(Ha, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)))
			# H̄_dot_Hₖ =  dot(λ⃗₀,vec(Mₖ(Ha, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹))) / ω^2
			# H̄_dot_Hₖ =  dot(H,vec(Mₖ(λ⃗₀a, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹))) /  ω #ωₖ
			 # H̄_dot_Hₖ =  dot(λ⃗₀,vec(Mₖ(H̄a, ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)))
			# ω̄  += H̄_dot_Hₖ
			# ω̄  +=  ωₖ / H̄_dot_Hₖ
			# λ⃗₀ -= P*λ⃗₀ - ( ( 2 * k̄ / ωₖ ) * H )
		end
		Ha_F = 𝓕 * kx_t2c(Ha,mn,kpg_mag) #fft(kcross_t2c(Ha,kz,gx,gy,gz),(2:4))
		λa = reshape(λ⃗,(2,Nx,Ny,Nz))
		λa_F  = 𝓕 * kx_t2c(λa,mn,kpg_mag) #fft(kcross_t2c(λ₀,kz,gx,gy,gz),(2:4))
		# ε̄ ⁻¹ = ( 𝓕 * kcross_t2c(λ₀,ds) ) .* ( 𝓕 * kcross_t2c(Ha,ds) )
		ε⁻¹_bar = [ Diagonal( real.(λa_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) )[a,b] for a=1:3,b=1:3,i=1:Nx,j=1:Ny,kk=1:Nz]
		return (NO_FIELDS, k̄, ε⁻¹_bar,ChainRulesCore.Zero(),ChainRulesCore.Zero(),ChainRulesCore.Zero())
    end
    return (Ω, solve_ω²_pullback)
end
