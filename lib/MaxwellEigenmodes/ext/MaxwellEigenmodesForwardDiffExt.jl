# Forward-mode (ForwardDiff) support for the FFTs inside the Maxwell operators.
#
# The eigenmode quadratic forms (`HMH`, `HMₖH`, `ε⁻¹_dot`) apply `fft`/`ifft` to the complex
# field. When those operators are differentiated with ForwardDiff — directly (e.g. ∂/∂k of
# `group_index`) or through the `group_index`/perturbation `frule`s, which compute their JVP
# with `ForwardDiff.derivative` and so carry a `Dual` seed `t` that promotes the field to
# `Complex{Dual}` — the field reaching `fft` has element type `Complex{Dual{T,V,N}}`, for
# which `AbstractFFTs.plan_fft` has no method (AbstractFFTs ships no ForwardDiff extension):
#
#   MethodError: no method matching plan_fft(::Array{Complex{ForwardDiff.Dual{…}}}, ::UnitRange)
#
# The DFT is C-linear, so it acts independently on the primal value and on each partial:
# fft(a + Σ εᵢ bᵢ) = fft(a) + Σ εᵢ fft(bᵢ). We therefore transform the primal-value array and
# each of the N partial arrays with the ordinary (Float) FFT and recombine the `Dual`s — never
# differentiating the FFTW plan itself. This makes Enzyme *forward* mode (which bridges these
# `frule`s) and plain ForwardDiff both work through the eigensolve post-processing.

module MaxwellEigenmodesForwardDiffExt

using MaxwellEigenmodes
using AbstractFFTs
using ForwardDiff
using ForwardDiff: Dual, Partials

# Primal-value / i-th-partial of a Complex{Dual} as an ordinary Complex{<:AbstractFloat}.
@inline _cval(z::Complex{<:Dual}) = Complex(ForwardDiff.value(real(z)), ForwardDiff.value(imag(z)))
@inline _cpart(z::Complex{<:Dual}, i) = Complex(ForwardDiff.partials(real(z), i), ForwardDiff.partials(imag(z), i))

# Apply the ordinary FFT `op` to the value and each partial, then recombine into Complex{Dual}.
function _dual_fft(op, x::AbstractArray{Complex{Dual{T,V,N}}}, region) where {T,V,N}
    yv = op(map(_cval, x), region)                                  # Array{Complex{V}}
    yp = ntuple(i -> op(map(z -> _cpart(z, i), x), region), Val(N)) # NTuple{N, Array{Complex{V}}}
    out = similar(x)                                                # Array{Complex{Dual{T,V,N}}}
    @inbounds for j in eachindex(out, yv)
        vj = yv[j]
        re = Dual{T}(real(vj), Partials(ntuple(i -> real(yp[i][j]), Val(N))))
        im = Dual{T}(imag(vj), Partials(ntuple(i -> imag(yp[i][j]), Val(N))))
        out[j] = Complex(re, im)
    end
    return out
end

# Method extensions on AbstractFFTs (only the `Complex{Dual}` element type, with and without an
# explicit transform region — the Maxwell operators always pass a region tuple).
AbstractFFTs.fft(x::AbstractArray{<:Complex{<:Dual}}, region) = _dual_fft(AbstractFFTs.fft, x, region)
AbstractFFTs.ifft(x::AbstractArray{<:Complex{<:Dual}}, region) = _dual_fft(AbstractFFTs.ifft, x, region)
AbstractFFTs.bfft(x::AbstractArray{<:Complex{<:Dual}}, region) = _dual_fft(AbstractFFTs.bfft, x, region)
AbstractFFTs.fft(x::AbstractArray{<:Complex{<:Dual}}) = _dual_fft(AbstractFFTs.fft, x, 1:ndims(x))
AbstractFFTs.ifft(x::AbstractArray{<:Complex{<:Dual}}) = _dual_fft(AbstractFFTs.ifft, x, 1:ndims(x))
AbstractFFTs.bfft(x::AbstractArray{<:Complex{<:Dual}}) = _dual_fft(AbstractFFTs.bfft, x, 1:ndims(x))

end # module
