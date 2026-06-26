# Complex-capable tensor-field contractions.
#
# `MaxwellEigenmodes._dot`/`HMH`/`HMₖH` deliberately take `real(ε⁻¹)` internally (the
# unperturbed dielectric is real). First-order *loss* perturbations need a complex
# dielectric-tensor perturbation Δε (Im Δε ≠ 0), so the perturbation numerator
# ⟨E|Δε|E⟩ is assembled here with contractions that preserve complex entries. These mirror
# the index conventions of `MaxwellEigenmodes._dot`/`_outer` but never drop the imaginary
# part. They differentiate natively (Tullio) under ForwardDiff/Zygote/Enzyme/Mooncake.

"""
    _tv(T, v)

Pointwise 3×3-tensor-field · 3-vector-field contraction: `out[a] = Σ_b T[a,b]·v[b]` at
every grid point. `T` is `(3,3,dims...)`, `v` is `(3,dims...)`; complex-capable.
"""
function _tv(T::AbstractArray{<:Number,4}, v::AbstractArray{<:Number,3})
    @tullio out[a, ix, iy] := T[a, b, ix, iy] * v[b, ix, iy]
end
function _tv(T::AbstractArray{<:Number,5}, v::AbstractArray{<:Number,4})
    @tullio out[a, ix, iy, iz] := T[a, b, ix, iy, iz] * v[b, ix, iy, iz]
end

"""
    _tt(A, B)

Pointwise 3×3-tensor-field · 3×3-tensor-field product: `out[a,c] = Σ_b A[a,b]·B[b,c]` at
every grid point. Used to build `ε⁻¹·Δε·ε⁻¹`. Complex-capable.
"""
function _tt(A::AbstractArray{<:Number,4}, B::AbstractArray{<:Number,4})
    @tullio out[a, c, ix, iy] := A[a, b, ix, iy] * B[b, c, ix, iy]
end
function _tt(A::AbstractArray{<:Number,5}, B::AbstractArray{<:Number,5})
    @tullio out[a, c, ix, iy, iz] := A[a, b, ix, iy, iz] * B[b, c, ix, iy, iz]
end

"""
    _quad(v, T)

Pointwise Hermitian quadratic form summed over the grid: `Σ_grid Σ_{a,b} conj(v[a])·T[a,b]·v[b]`.
Complex-capable; returns a scalar (`ComplexF64` in general).
"""
function _quad(v::AbstractArray{<:Number,3}, T::AbstractArray{<:Number,4})
    @tullio s := conj(v[a, ix, iy]) * T[a, b, ix, iy] * v[b, ix, iy]
end
function _quad(v::AbstractArray{<:Number,4}, T::AbstractArray{<:Number,5})
    @tullio s := conj(v[a, ix, iy, iz]) * T[a, b, ix, iy, iz] * v[b, ix, iy, iz]
end
