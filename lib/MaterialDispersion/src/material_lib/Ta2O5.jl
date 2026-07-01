################################################################################
#                             Tantala  (Ta₂O₅)                                  #
################################################################################
export Ta₂O₅

"""
Amorphous tantalum pentoxide (Ta₂O₅, "tantala"), an emerging CMOS-compatible χ⁽³⁾
integrated-photonics platform (broadband ultralow loss, high Kerr nonlinearity).

The refractive-index model is the third-order Sellmeier fit of

    J. A. Black, R. Streater, K. F. Lamee, D. R. Carlson, S.-P. Yu, S. B. Papp,
    "Group-velocity dispersion engineering of tantala integrated photonics,"
    Optics Letters 46, 817 (2021) [arXiv:2009.14190], Eqn. (1):

      n²(λ) = 1 + 0.033 λ²/(λ² − 0.368²)
                + 3.212 λ²/(λ² − 0.1639²)
                + 3.747 λ²/(λ² − 14.5²)          (λ in µm),

fit to spectroscopic-ellipsometry data over 500 nm – 5 µm (RMS error 0.00057).

Kerr coefficient n₂ = 6.2×10⁻¹⁹ m²/W = 6.2×10⁻⁷ µm²/W (Black 2021, ~2.6× that of
Si₃N₄). Thermo-optic coefficient dn/dT = 8.8×10⁻⁶ K⁻¹ (Black 2021).
Variable units are λ in [µm] and T in [°C].
"""

p_n²_Ta₂O₅ = (
    A₀ = 1,
    B₁ = 0.033,
    C₁ = (0.368)^2,          #                           [μm²]
    B₂ = 3.212,
    C₂ = (0.1639)^2,         #                           [μm²]
    B₃ = 3.747,
    C₃ = (14.5)^2,           #                           [μm²]
    dn_dT = 8.8e-6,          # thermo-optic coefficient  [K⁻¹]
    T₀ = 20.0,               # reference temperature     [°C]
    dn²_dT = 2 * sqrt(n²_sym_fmt1(1.55; A₀=1, B₁=0.033, C₁=(0.368)^2, B₂=3.212, C₂=(0.1639)^2, B₃=3.747, C₃=(14.5)^2)) * 8.8e-6,
)

n²_Ta₂O₅(λ, T)   = n²_sym_fmt1(λ; p_n²_Ta₂O₅...)   + p_n²_Ta₂O₅.dn²_dT * (T - p_n²_Ta₂O₅.T₀)
n²_Ta₂O₅_ω(ω, T) = n²_sym_fmt1_ω(ω; p_n²_Ta₂O₅...) + p_n²_Ta₂O₅.dn²_dT * (T - p_n²_Ta₂O₅.T₀)

function make_Ta₂O₅(; p_n²=p_n²_Ta₂O₅)
    @variables ω, λ, T
    n² = n²_Ta₂O₅_ω(ω, T)
    n_λ = sqrt(substitute(n², Dict([(ω => 1 / λ)])))
    ng = ng_model(n_λ, λ)
    gvd = gvd_model(n_λ, λ)
    models = Dict([
        :n   => n_λ,
        :ng  => ng,
        :gvd => gvd,
        :ε   => diagm([n², n², n²]),
        # Kerr (intensity-dependent index) n₂ ≈ 6.2×10⁻¹⁹ m²/W = 6.2×10⁻⁷ µm²/W (Black 2021)
        :n₂  => 6.2e-7,        # μm²/W
    ])
    defaults = Dict([
        :ω => inv(0.8),        # μm⁻¹
        :λ => 0.8,             # μm
        :T => p_n².T₀,         # °C
    ])
    Material(models, defaults, :Ta₂O₅, colorant"goldenrod1")
end

################################################################

Ta₂O₅ = make_Ta₂O₅()
