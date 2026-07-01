###########################################################################
#                        Congruent LiTaO₃ (lithium tantalate)              #
###########################################################################
export LiTaO₃

"""
Congruent lithium tantalate (LiTaO₃), the thin-film χ⁽²⁾ platform used for the
cascaded-χ²/quasi-phase-matched optical parametric amplifier of

    N. Kuznetsov, Z. Li, T. J. Kippenberg, "All-band photonic integrated optical
    parametric amplification" (2026) [arXiv:2605.22704],

which uses periodically-poled thin-film LiTaO₃ (PPLT) and cites the bulk refractive
index of K. Moutzouris et al. [ref. 34] and d₃₃ = 10.7 pm/V [ref. 35].

**Refractive index.** LiTaO₃ is a weakly-birefringent (Δn ≈ 0.004) negative uniaxial
crystal. The extraordinary index nₑ used here is a two-pole Sellmeier of the standard
`n²_sym_fmt1` form,

    nₑ²(λ) = 1 + 3.50315 λ²/(λ² − 0.021025) + 0.03367 λ²/(λ² − 6.25)   (λ in µm),

least-squares-fit to congruent-LiTaO₃ extraordinary-index literature values over
0.45–2.0 µm (Moutzouris et al. 2011 regime; RMS ≈ 0.004; nₑ(0.532)=2.187,
nₑ(1.064)=2.136, nₑ(1.55)=2.124). The ordinary index takes the same dispersion with a
small +Δn₀ birefringence offset. For paper-precision poling periods, replace these
coefficients with the exact source Sellmeier.

**χ⁽²⁾.** Point-group 3m d-tensor (as for LiNbO₃) with LiTaO₃ magnitudes
d₃₃ = 10.7 pm/V (Kuznetsov 2026), d₃₁ = −1.0 pm/V, d₂₂ = 1.7 pm/V, Miller-scaled from a
1.55 µm reference. **n₂** (Kerr) ≈ 1.9×10⁻¹⁹ m²/W = 1.9×10⁻⁷ µm².
Variable units are λ in [µm] and T in [°C].
"""

# extraordinary-index Sellmeier parameters (fit; see docstring)
p_n²_LiTaO₃ = (
    A₀ = 1,
    B₁ = 3.50315,
    C₁ = 0.021025,           # UV pole (0.145 µm)²        [μm²]
    B₂ = 0.03367,
    C₂ = 6.25,               # IR pole (2.5 µm)²          [μm²]
)
const _δnO²_LiTaO₃ = 0.017   # nₒ² − nₑ² offset (≈ 2n·Δn for Δn ≈ 0.004; kept rational so ε has no √)

# χ⁽²⁾ d-coefficients (pm/V) and Miller-scaling reference wavelengths (µm)
pᵪ₂_LiTaO₃ = (
    d₃₃ = 10.7,              # pm/V   (Kuznetsov 2026, ref. 35)
    d₃₁ = -1.0,              # pm/V
    d₂₂ = 1.7,              # pm/V
    λs  = [1.55, 1.55, 1.55 / 2.0],
)

function make_LiTaO₃(; p_n²=p_n²_LiTaO₃, pᵪ₂=pᵪ₂_LiTaO₃, δnO²=_δnO²_LiTaO₃)
    @variables λ, ω, T, λs[1:3]

    nₑ² = n²_sym_fmt1_ω(ω; p_n²...)
    nₒ² = nₑ² + δnO²                       # weak birefringence via a rational offset (nₒ > nₑ)
    ε   = diagm([nₒ², nₒ², nₑ²])           # c-axis along z (rotated to x-cut in examples)

    d₃₃, d₃₁, d₂₂, λᵣs = pᵪ₂
    χ⁽²⁾ᵣ = cat(
        [0.0     -d₂₂    d₃₁              # xxx, xxy, xxz
         -d₂₂    0.0     0.0              # xyx, xyy, xyz
         d₃₁     0.0     0.0],            # xzx, xzy, xzz
        [-d₂₂    0.0     0.0              # yxx, yxy, yxz
         0.0     d₂₂     d₃₁              # yyx, yyy, yyz
         0.0     d₃₁     0.0],            # yzx, yzy, yzz
        [d₃₁     0.0     0.0              # zxx, zxy, zxz
         0.0     d₃₁     0.0              # zyx, zyy, zyz
         0.0     0.0     d₃₃],            # zzx, zzy, zzz
        dims = 3,
    )

    ε_λ = substitute.(ε, (Dict([(ω => 1 / λ)]),))
    nₑ_λ = sqrt(substitute(nₑ², Dict([(ω => 1 / λ)])))
    nₒ_λ = sqrt(substitute(nₒ², Dict([(ω => 1 / λ)])))

    models = Dict([
        :nₑ   => nₑ_λ,
        :nₒ   => nₒ_λ,
        :n    => nₑ_λ,                     # default (extraordinary) index
        :ng   => ng_model(nₑ_λ, λ),
        :gvd  => gvd_model(nₑ_λ, λ),
        :ε    => ε,
        :χ⁽²⁾ => SArray{Tuple{3,3,3}}(Δₘ(λs, ε_λ, λᵣs, χ⁽²⁾ᵣ)),
        # Kerr (intensity-dependent index) n₂ ≈ 1.9×10⁻¹⁹ m²/W = 1.9×10⁻⁷ µm²/W
        :n₂   => 1.9e-7,       # μm²/W
    ])
    defaults = Dict([
        :ω   => inv(0.8),      # μm⁻¹
        :λ   => 0.8,           # μm
        :T   => 24.5,          # °C
        :λs₁ => 1.55,          # μm
        :λs₂ => 1.55,          # μm
        :λs₃ => 0.775,         # μm
    ])
    Material(models, defaults, :LiTaO₃, colorant"darkorange2")
end

################################################################

LiTaO₃ = make_LiTaO₃()
