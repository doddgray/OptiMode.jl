### ForwardDiff interop with ChainRules `frule`s ###
"""
# generic patttern for a ForwardDiff.Dual-compatible function method from Mohamed Tarek on Slack #autodiff

function f(x::Vector{<:ForwardDiff.Dual{T}}) where {T}
    xv, Δx = ForwardDiff.value.(x), reduce(vcat, transpose.(ForwardDiff.partials.(x)))
    out, Δf = ChainRulesCore.frule((NoTangent(), Δx), f, xv)
    if out isa Real
      return ForwardDiff.Dual{T}(out, ForwardDiff.Partials(Tuple(Δf)))
    elseif out isa Vector
      return ForwardDiff.Dual{T}.(out, ForwardDiff.Partials.(Tuple.(eachrow(Δf))))
    else
        throw("Unsupported output.")
    end
end

# and below, a macro based on this pattern
"""
using ChainRulesCore, ForwardDiff

macro ForwardDiff_frule(f)
	quote
		function $(esc(f))(x::Vector{<:ForwardDiff.Dual{T}}) where {T}
		  xv, Δx = ForwardDiff.value.(x), reduce(vcat, transpose.(ForwardDiff.partials.(x)))
		  out, Δf = ChainRulesCore.frule((NoTangent(), Δx), $(esc(f)), xv)
		  if out isa Real
			return ForwardDiff.Dual{T}(out, ForwardDiff.Partials(Tuple(Δf)))
		  elseif out isa Vector
			return ForwardDiff.Dual{T}.(out, ForwardDiff.Partials.(Tuple.(eachrow(Δf))))
		  else
		  	throw("Unsupported output.")
		  end
		end
	end
end

"""
f1(x) = sum(x)
function ChainRulesCore.frule((_, Δx), ::typeof(f1), x::AbstractVector{<:Number})
  println("frule was used")
  return f1(x), sum(Δx, dims = 1)
end

f2(x) = x
function ChainRulesCore.frule((_, Δx), ::typeof(f2), x::AbstractVector{<:Number})
  println("frule was used")
  return f2(x), Δx
end

@ForwardDiff_frule f1
ForwardDiff.gradient(f1, rand(3))
# frule was used
# 3-element Vector{Float64}:
#  1.0
#  1.0
#  1.0

@ForwardDiff_frule f2
ForwardDiff.jacobian(f2, rand(3))
# frule was used
# 3×3 Matrix{Float64}:
#  1.0  0.0  0.0
#  0.0  1.0  0.0
#  0.0  0.0  1.0
"""


"""
# Example code for defining custom ForwardDiff rules, copied from YingboMa's gist:
# https://gist.github.com/YingboMa/c22dcf8239a62e01b27ac679dfe5d4c5
using ForwardDiff
goo((x, y, z),) = [x^2*z, x*y*z, abs(z)-y]
foo((x, y, z),) = [x^2*z, x*y*z, abs(z)-y]
function foo(u::Vector{ForwardDiff.Dual{T,V,P}}) where {T,V,P}
    # unpack: AoS -> SoA
    vs = ForwardDiff.value.(u)
    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ForwardDiff.partials, hcat, u)
    # get f(vs)
    val = foo(vs)
    # get J(f, vs) * ps (cheating). Write your custom rule here
    jvp = ForwardDiff.jacobian(goo, vs) * ps
    # pack: SoA -> AoS
    return map(val, eachrow(jvp)) do v, p
        ForwardDiff.Dual{T}(v, p...) # T is the tag
    end
end
ForwardDiff.gradient(u->sum(cumsum(foo(u))), [1, 2, 3]) == ForwardDiff.gradient(u->sum(cumsum(goo(u))), [1, 2, 3])
"""




"""
ForwardDiff Complex number support
ref: https://github.com/JuliaLang/julia/pull/36030
https://github.com/JuliaDiff/ForwardDiff.jl/pull/455
"""
Base.float(d::ForwardDiff.Dual{T}) where T = ForwardDiff.Dual{T}(float(d.value), d.partials)
Base.prevfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = ForwardDiff.Dual{T}(prevfloat(float(d.value)), d.partials)
Base.nextfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = ForwardDiff.Dual{T}(nextfloat(float(d.value)), d.partials)
function Base.ldexp(x::T, e::Integer) where T<:ForwardDiff.Dual
    if e >=0
        x * (1<<e)
    else
        x / (1<<-e)
    end
end

"""
ForwardDiff FFT support
ref: https://github.com/JuliaDiff/ForwardDiff.jl/pull/495/files
https://discourse.julialang.org/t/forwarddiff-and-zygote-cannot-automatically-differentiate-ad-function-from-c-n-to-r-that-uses-fft/52440/18
"""
ForwardDiff.value(x::Complex{<:ForwardDiff.Dual}) = Complex(x.re.value, x.im.value)
ForwardDiff.partials(x::Complex{<:ForwardDiff.Dual}, n::Int) = Complex(ForwardDiff.partials(x.re, n), ForwardDiff.partials(x.im, n))
ForwardDiff.npartials(x::Complex{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N} = N
ForwardDiff.npartials(::Type{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N} = N
# AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = float.(x .+ 0im)
AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d) + 0im
AbstractFFTs.realfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d)

for plan in [:plan_fft, :plan_ifft, :plan_bfft]
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x)) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, region=1:ndims(x)) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), region)

    end
end

# rfft only accepts real arrays
AbstractFFTs.plan_rfft(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x)) =
    AbstractFFTs.plan_rfft(ForwardDiff.value.(x), region)

for plan in [:plan_irfft, :plan_brfft]  # these take an extra argument, only when complex?
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x)) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, d::Integer, region=1:ndims(x)) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), d, region)

    end
end

for P in [:Plan, :ScaledPlan]  # need ScaledPlan to avoid ambiguities
    @eval begin

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:ForwardDiff.Dual}) =
            _apply_plan(p, x)

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}) =
            _apply_plan(p, x)

    end
end

function _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray)
    xtil = p * ForwardDiff.value.(x)
    dxtils = ntuple(ForwardDiff.npartials(eltype(x))) do n
        p * ForwardDiff.partials.(x, n)
    end
    map(xtil, dxtils...) do val, parts...
        Complex(
            ForwardDiff.Dual(real(val), map(real, parts)),
            ForwardDiff.Dual(imag(val), map(imag, parts)),
        )
    end
end

# used with the ForwardDiff+FFTW code above, this Zygote.extract method
# enables Zygote.hessian to work on real->real functions that internally use
# FFTs (and thus complex numbers)
using Zygote
import Zygote: extract
function Zygote.extract(xs::AbstractArray{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N}
  J = similar(xs, complex(V), N, length(xs))
  for i = 1:length(xs), j = 1:N
    J[j, i] = xs[i].re.partials.values[j] + im * xs[i].im.partials.values[j]
  end
  x0 = ForwardDiff.value.(xs)
  return x0, J
end