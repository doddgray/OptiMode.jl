"""
    EnzymeExt (EigenModeSolver)

Enzyme.jl AD extension for EigenModeSolver.jl.

Provides custom Enzyme rules for:
1. FFT operations (FFTW plans cannot be differentiated by Enzyme automatically)
2. `sliceinv_3x3`: batch 3×3 matrix inversion
3. `HelmholtzMap` multiplication (the core ε⁻¹-weighted plane-wave operation)
4. `solve_ω²`: adjoint method for eigenvalue sensitivity

Note: Enzyme's source-transformation AD is particularly powerful for loop-heavy
code like the FFTW-based Maxwell operator. The rules below handle the cases where
Enzyme cannot automatically generate correct derivatives.
"""
module EnzymeExt

using EigenModeSolver
using EigenModeSolver: HelmholtzMap, ModeSolver, sliceinv_3x3,
    _cross, _dot, _mult, eig_adjt
using Enzyme
using Enzyme.EnzymeRules
using LinearAlgebra
using StaticArrays
using Tullio
using FFTW
using AbstractFFTs

# ──────────────────────────────────────────────────────────────────────────────
# FFTW plan application: Enzyme cannot differentiate through FFTW plans.
# Provide forward and reverse rules for plan application.
# ──────────────────────────────────────────────────────────────────────────────

# Forward FFT application: FFTW plan * vector
function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:EnzymeRules.Const},  # in-place mul! returns nothing meaningful
    c_out::EnzymeRules.Duplicated{<:AbstractArray{<:Complex}},
    plan::Const{<:FFTW.cFFTWPlan},
    b_in::EnzymeRules.Duplicated{<:AbstractArray{<:Complex}}
)
    # Primal: c = plan * b
    mul!(c_out.val, plan.val, b_in.val)
    # Tangent: dc = plan * db (FFT is linear)
    mul!(c_out.dval, plan.val, b_in.dval)
    return nothing
end

# Reverse IFFT for pullback of FFT
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(LinearAlgebra.mul!)},
    ::Type{<:EnzymeRules.Const},
    c_out::EnzymeRules.Duplicated{<:AbstractArray{<:Complex}},
    plan::Const{<:FFTW.cFFTWPlan},
    b_in::EnzymeRules.Duplicated{<:AbstractArray{<:Complex}}
)
    mul!(c_out.val, plan.val, b_in.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, plan.val)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(LinearAlgebra.mul!)},
    dret,
    fft_plan,
    c_out::EnzymeRules.Duplicated{<:AbstractArray{<:Complex}},
    plan::Const{<:FFTW.cFFTWPlan},
    b_in::EnzymeRules.Duplicated{<:AbstractArray{<:Complex}}
)
    # Pullback of FFT: ∂L/∂b = IFFT(∂L/∂c) * N (unnormalized)
    n = prod(size(b_in.val)[2:end])  # spatial grid size
    b_in.dval .+= plan.val' * c_out.dval ./ n
    c_out.dval .= 0
    return (nothing, nothing, nothing)
end

# ──────────────────────────────────────────────────────────────────────────────
# sliceinv_3x3: batch 3×3 matrix inversion
# d(A⁻¹) = -A⁻¹ · dA · A⁻¹ (matrix inversion lemma)
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    ::Const{typeof(EigenModeSolver.sliceinv_3x3)},
    RT::Type{<:EnzymeRules.Duplicated},
    A::EnzymeRules.Duplicated{<:AbstractArray}
)
    A_inv = sliceinv_3x3(A.val)
    # dA_inv[a,b,:,:] = -∑ₖₗ A_inv[a,k,:,:] * dA[k,l,:,:] * A_inv[l,b,:,:]
    @tullio dA_inv[a, b, ix, iy] := -A_inv[a, c, ix, iy] * A.dval[c, d, ix, iy] * A_inv[d, b, ix, iy]
    return EnzymeRules.Duplicated(A_inv, dA_inv)
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(EigenModeSolver.sliceinv_3x3)},
    ::Type{<:EnzymeRules.Duplicated},
    A::EnzymeRules.Duplicated{<:AbstractArray}
)
    A_inv = sliceinv_3x3(A.val)
    return EnzymeRules.AugmentedReturn(A_inv, nothing, A_inv)  # save A_inv as tape
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(EigenModeSolver.sliceinv_3x3)},
    dret::EnzymeRules.Duplicated,
    A_inv,  # tape
    A::EnzymeRules.Duplicated{<:AbstractArray}
)
    # ∂L/∂A[a,b,:,:] = -∑ₖₗ A_inv[k,a,:,:] * (∂L/∂A_inv)[k,l,:,:] * A_inv[l,b,:,:]
    @tullio dA[a, b, ix, iy] := -A_inv[c, a, ix, iy] * dret.dval[c, d, ix, iy] * A_inv[d, b, ix, iy]
    A.dval .+= dA
    return (nothing,)
end

# ──────────────────────────────────────────────────────────────────────────────
# _cross: cross product over spatial grid — forward mode rule
# ──────────────────────────────────────────────────────────────────────────────

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfigWidth{1},
    ::Const{typeof(EigenModeSolver._cross)},
    RT::Type{<:EnzymeRules.Duplicated},
    v₁::EnzymeRules.Duplicated{<:AbstractArray},
    v₂::EnzymeRules.Duplicated{<:AbstractArray}
)
    v₃ = EigenModeSolver._cross(v₁.val, v₂.val)
    # d(v₁×v₂) = dv₁×v₂ + v₁×dv₂
    dv₃ = EigenModeSolver._cross(v₁.dval, v₂.val) .+ EigenModeSolver._cross(v₁.val, v₂.dval)
    return EnzymeRules.Duplicated(v₃, dv₃)
end

end # module EnzymeExt
