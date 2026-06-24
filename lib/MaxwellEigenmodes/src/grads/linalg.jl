#### Forward/reverse-mode rules for the pointwise 3×3 tensor-field inverse `sliceinv_3x3`.
#
# `sliceinv_3x3` converts between the smoothed permittivity `ε` and the inverse
# permittivity `ε⁻¹` consumed by the eigensolver, so it sits on the
# geometry/material → smoothing → solve path. Its kernel `_sliceinv_3x3_cols` uses a
# `Threads.@threads` loop, which native Enzyme reverse mode cannot trace. These
# closed-form ChainRules rules supply the exact per-pixel matrix-inverse differential
#
#     Y = A⁻¹,   dY = −A⁻¹ (dA) A⁻¹,   Ā = −Aᵀ⁻¹ Ȳ Aᵀ⁻¹ = −Yᵀ Ȳ Yᵀ ,
#
# evaluated with static 3×3 matrices. They are bridged to Enzyme (forward & reverse)
# via `@import_frule`/`@import_rrule` in the package's Enzyme extension, so every AD
# backend differentiates the ε ⇄ ε⁻¹ conversion cleanly and without the threaded loop.

# Per-pixel `-Lᵒ * X * Rᵒ` over (3,3,Ns...) tensor fields, where `Lᵒ`/`Rᵒ` are `L`/`R`
# (optionally transposed per `tL`/`tR`). Single-threaded; the rule body is opaque to the
# outer AD engine, so the in-place fill is safe.
function _sliceinv_diff(L::AbstractArray, X::AbstractArray, R::AbstractArray; tL::Bool=false, tR::Bool=false)
    TL, TX, TR = eltype(L), eltype(X), eltype(R)
    T = promote_type(TL, TX, TR)
    Npix = length(X) ÷ 9
    Lf = reshape(L, 9, Npix); Xf = reshape(X, 9, Npix); Rf = reshape(R, 9, Npix)
    out = Matrix{T}(undef, 9, Npix)
    @inbounds for p in 1:Npix
        Lp = SMatrix{3,3,TL,9}(ntuple(i -> Lf[i, p], Val(9)))
        Xp = SMatrix{3,3,TX,9}(ntuple(i -> Xf[i, p], Val(9)))
        Rp = SMatrix{3,3,TR,9}(ntuple(i -> Rf[i, p], Val(9)))
        out[:, p] = vec(-((tL ? transpose(Lp) : Lp) * Xp * (tR ? transpose(Rp) : Rp)))
    end
    return reshape(out, size(X))
end

for ND in (4, 5)
    @eval function ChainRulesCore.frule((_, Ȧ), ::typeof(sliceinv_3x3), A::AbstractArray{T,$ND}) where {T<:Number}
        Y = sliceinv_3x3(A)
        (Ȧ isa AbstractZero) && return Y, zero(Y)
        return Y, _sliceinv_diff(Y, Ȧ, Y)                       # dY = −Y Ȧ Y
    end

    @eval function ChainRulesCore.rrule(::typeof(sliceinv_3x3), A::AbstractArray{T,$ND}) where {T<:Number}
        Y = sliceinv_3x3(A)
        function sliceinv_3x3_pullback(Ȳ)
            Ȳu = ChainRulesCore.unthunk(Ȳ)
            Ā = _sliceinv_diff(Y, Ȳu, Y; tL=true, tR=true)      # Ā = −Yᵀ Ȳ Yᵀ
            return (ChainRulesCore.NoTangent(), Ā)
        end
        return Y, sliceinv_3x3_pullback
    end
end
