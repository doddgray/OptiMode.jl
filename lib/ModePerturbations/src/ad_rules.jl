# AD-rule infrastructure for ModePerturbations.
#
# Most exported quantities are AD-differentiable with no custom rules:
#   • `payne_lacey_slab_loss`, `substrate_leakage_loss`, `cascaded_chi2_n2_eff`,
#     `kerr_gamma`, … are pure scalar arithmetic → native in ForwardDiff, Zygote,
#     Enzyme (forward & reverse) and Mooncake.
#   • `perturbation_Δk` and everything built on it (`Δneff_perturbation`,
#     `thermo_optic_Δneff`, `kerr_spm_Δneff`, `shg_*`, …) compose FFTs, the `HMₖH`
#     quadratic form (which carries its own ChainRules `rrule`) and Tullio contractions:
#     these differentiate through Zygote (reverse) and ForwardDiff (forward) directly.
#
# Enzyme's forward mode cannot create FFTW plans and its reverse mode needs the FFT rule
# imported; the scalar real-valued projection `_perturbation_scalar` below is given a
# Zygote-backed `rrule` and a ForwardDiff-backed `frule` (exactly as `ModeAnalysis`
# does for `group_index`), which the Enzyme package extension imports. `perturbation_Δk`
# is expressed through it so the complex result is covered by differentiating its real and
# imaginary parts.

export _perturbation_re, _perturbation_im

# Real-valued scalar kernels the AD rules target (kept separate, with no trailing discrete
# argument, so Enzyme's `@import_rrule`/`@import_frule` accept the signature — exactly the
# shape `ModeAnalysis` uses for `group_index`). `Re(Δk)` → index shift, `Im(Δk)` → loss.
_perturbation_re(k::Real, evec, ε⁻¹, Δε, grid::Grid) = real(perturbation_Δk(k, evec, ε⁻¹, Δε, grid))
_perturbation_im(k::Real, evec, ε⁻¹, Δε, grid::Grid) = imag(perturbation_Δk(k, evec, ε⁻¹, Δε, grid))

# concretely-typed tangent helpers (Enzyme's imported rules need concrete floats for the
# Active scalar cotangent and array cotangents / NoTangent for the inactive ones)
_re_tangent(x)::Float64 = Float64(real(x))
_re_tangent(::Nothing)::Float64 = 0.0
_arr_tangent(x::AbstractArray) = x
_arr_tangent(::Nothing) = NoTangent()

# Reverse- and forward-mode rules for both kernels. The forward pass composes FFTs, the
# `HMₖH` quadratic form and Tullio contractions; Zygote differentiates that program for the
# `rrule` (reverse) and ForwardDiff for the `frule` (forward — Enzyme cannot create FFTW
# plans), exactly as `ModeAnalysis` does for `group_index`. The Mooncake (`@from_rrule`) and
# Enzyme (`@import_rrule`/`@import_frule`) extensions reuse these.
# The rules differentiate the *underlying* `perturbation_Δk` program (FFTs + `HMₖH` +
# Tullio), NOT the kernel that carries the rule — otherwise Zygote would re-enter this very
# `rrule` and recurse forever (cf. `ModeAnalysis`'s `_group_index_kernel`). `perturbation_Δk`
# has no rrule of its own, so Zygote/ForwardDiff differentiate its body directly.
for (kernel, proj) in ((:_perturbation_re, :real), (:_perturbation_im, :imag))
    @eval begin
        function ChainRulesCore.rrule(::typeof($kernel), k::Real, evec, ε⁻¹, Δε, grid::Grid)
            y, zpb = Zygote.pullback(
                (k_, ev_, ei_, de_) -> $proj(perturbation_Δk(k_, ev_, ei_, de_, grid)),
                k, evec, ε⁻¹, Δε)
            function _pb(ȳ)
                k̄, ēv, eī, dē = zpb(ȳ)
                return (NoTangent(), _re_tangent(k̄), _arr_tangent(ēv),
                    _arr_tangent(eī), _arr_tangent(dē), NoTangent())
            end
            return y, _pb
        end
        function ChainRulesCore.frule((_, Δk_, Δev, Δei, Δde, _), ::typeof($kernel),
                k::Real, evec, ε⁻¹, Δε, grid::Grid)
            y = $proj(perturbation_Δk(k, evec, ε⁻¹, Δε, grid))
            _v(Δ, x) = Δ isa AbstractZero ? zero(x) : Δ
            dk = _v(Δk_, k)
            dev, dei, dde = _v(Δev, evec), _v(Δei, ε⁻¹), _v(Δde, Δε)
            ẏ = ForwardDiff.derivative(
                t -> $proj(perturbation_Δk(k + t * dk, evec .+ t .* dev, ε⁻¹ .+ t .* dei,
                             Δε .+ t .* dde, grid)), 0.0)
            return y, ẏ
        end
    end
end
