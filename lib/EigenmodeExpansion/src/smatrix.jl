# ──────────────────────────────────────────────────────────────────────────────
#  EME scattering matrices: interface (mode matching), propagation, and cascade.
#
#  This reproduces MEOW's overlap-based interface solve and diagonal propagation
#  matrices, and SAX's circuit cascade (a Redheffer star product over the linear
#  chain  p₀ ⋆ i₀₁ ⋆ p₁ ⋆ i₁₂ ⋆ … ⋆ p_{n-1}). All operations are matrix
#  multiplies and (regularized) linear solves, so the whole cascade is
#  differentiable in both forward and reverse mode.
# ──────────────────────────────────────────────────────────────────────────────

export SMat, interface_smatrix, propagation_smatrix, star, cascade, transmission, reflection

using LinearAlgebra

"""
    SMat

A multimode scattering matrix in block form. The first `nl` ports are on the left
and the next `nr` on the right; `S` maps incoming amplitudes `[a_left; a_right]`
to outgoing amplitudes `[b_left; b_right]` with block structure
`[[R_LL, T_RL], [T_LR, R_RR]]`. Mirrors SAX's `(S, port_map)` for a two-sided
component.
"""
struct SMat{C<:Complex}
    S::Matrix{C}
    nl::Int
    nr::Int
end

s11(m::SMat) = @view m.S[1:m.nl, 1:m.nl]
s12(m::SMat) = @view m.S[1:m.nl, m.nl+1:end]
s21(m::SMat) = @view m.S[m.nl+1:end, 1:m.nl]
s22(m::SMat) = @view m.S[m.nl+1:end, m.nl+1:end]

"left→right transmission block of a device S-matrix"
transmission(m::SMat) = m.S[m.nl+1:end, 1:m.nl]
"left reflection block of a device S-matrix"
reflection(m::SMat) = m.S[1:m.nl, 1:m.nl]

# ── AD-friendly regularized pseudo-inverse solve ─────────────────────────────

"""
    reg_solve(A, B; reg=1e-9) -> X ≈ pinv(A) * B

Tikhonov-regularized least-squares solve `X = (AᴴA + μI)⁻¹ AᴴB`, with `μ` scaled
by the mean eigenvalue of `AᴴA`. This is the differentiable analogue of MEOW's
truncated-SVD `tsvd_solve`: it uses only matmuls and one Hermitian solve, so it
carries clean forward- and reverse-mode rules (unlike a hard SVD truncation).
"""
function reg_solve(A::AbstractMatrix, B::AbstractMatrix; reg::Real=1e-9)
    AhA = A' * A
    n = size(AhA, 1)
    μ = reg * (real(tr(AhA)) / max(n, 1) + eps())
    return (AhA + μ * I) \ (A' * B)
end

# ── interface S-matrix (MEOW `compute_interface_s_matrix`) ───────────────────

"""
    enforce_passivity(σ; method=:invert) -> σ′

Project the singular values of an interface S-matrix onto a passive interval
(≤ 1), exactly as MEOW's `enforce_passivity`: `:none` leaves them unchanged,
`:clip` caps at 1, `:invert` maps `σ > 1` to `1/σ`, `:subtract` maps `σ > 1` to
`max(0, 2 − σ)`. A truncated modal basis can give a slightly non-passive
(`σ > 1`) raw interface; this is the standard model correction.
"""
function enforce_passivity(σ::AbstractVector; method::Symbol=:invert)
    method === :none && return σ
    method === :clip && return min.(σ, one(eltype(σ)))
    method === :invert && return ifelse.(σ .> 1, inv.(σ), σ)
    method === :subtract && return max.(ifelse.(σ .> 1, 2 .- σ, σ), zero(eltype(σ)))
    throw(ArgumentError("unknown passivity method $method"))
end

"""
    interface_smatrix(modes_l, modes_r; conjugate=false, reg=1e-9,
                      reciprocity=true, passivity=:invert) -> SMat

Mode-matching interface S-matrix between the left and right modal bases of a step
discontinuity, following MEOW. The transmission blocks come from the
overlap-matrix solve

    A_LR = O_LR + O_RLᵀ ;  T_LR = pinv(A_LR) (2 I_L)
    A_RL = O_RL + O_LRᵀ ;  T_RL = pinv(A_RL) (2 I_R)

(`ᵀ`→`ᴴ` when `conjugate=true`), and the reflection blocks are reconstructed from
both continuity equations and averaged:

    R_LL = ½((O_RLᵀ T_LR − I) + (I − O_LR T_LR))
    R_RR = ½((O_LRᵀ T_RL − I) + (I − O_RL T_RL)).

The modes must be power-normalized in the same inner product (the `G = I` form);
`build_mode` does this. `passivity` enforces `|S| ≤ 1` via [`enforce_passivity`](@ref)
(pass `:none` for a clean reverse-mode adjoint); `reciprocity` symmetrizes the
result as `½(S + Sᵀ)`.
"""
function interface_smatrix(modes_l, modes_r; conjugate::Bool=false, reg::Real=1e-9,
                           reciprocity::Bool=true, passivity::Symbol=:invert)
    NL, NR = length(modes_l), length(modes_r)
    O_LR = overlap_matrix(modes_l, modes_r; conjugate)   # (NL, NR)
    O_RL = overlap_matrix(modes_r, modes_l; conjugate)   # (NR, NL)
    O_RL_adj = conjugate ? O_RL' : transpose(O_RL)       # (NL, NR)
    O_LR_adj = conjugate ? O_LR' : transpose(O_LR)       # (NR, NL)
    I_L = Matrix{ComplexF64}(I, NL, NL)
    I_R = Matrix{ComplexF64}(I, NR, NR)

    A_LR = O_LR .+ O_RL_adj                              # (NL, NR)
    T_LR = reg_solve(Matrix(A_LR), 2.0 .* I_L; reg)      # (NR, NL)
    A_RL = O_RL .+ O_LR_adj                              # (NR, NL)
    T_RL = reg_solve(Matrix(A_RL), 2.0 .* I_R; reg)      # (NL, NR)

    R_LL = 0.5 .* ((O_RL_adj * T_LR .- I_L) .+ (I_L .- O_LR * T_LR))
    R_RR = 0.5 .* ((O_LR_adj * T_RL .- I_R) .+ (I_R .- O_RL * T_RL))

    S = [R_LL T_RL; T_LR R_RR]
    # passivity enforcement via SVD (MEOW). Skipped for `:none` — the SVD is not
    # reverse-mode-AD friendly, so pass `passivity=:none` when differentiating.
    if passivity !== :none
        F = svd(Matrix(S))
        S = (F.U .* transpose(enforce_passivity(F.S; method=passivity))) * F.Vt
    end
    if reciprocity
        S = 0.5 .* (S .+ transpose(S))
    end
    return SMat(Matrix(S), NL, NR)
end

# ── propagation S-matrix (MEOW `compute_propagation_s_matrix`) ───────────────

"""
    propagation_smatrix(modes, length) -> SMat

Diagonal propagation S-matrix for one cell: each mode acquires the phase
`exp(2πi · k · L)` (with `k` the propagation constant β in cycles/μm and `L` the
cell length), transmitting left↔right with no reflection.
"""
function propagation_smatrix(modes, L::Real)
    N = length(modes)
    ph = [cis(2π * modes[i].k * L) for i in 1:N]
    Z = zeros(eltype(ph), N, N)
    D = Matrix(Diagonal(ph))
    S = [Z D; D Z]                                       # S21 = S12 = diag(phase)
    return SMat(Matrix(S), N, N)
end

# ── Redheffer star product / cascade (SAX circuit equivalent) ────────────────

"""
    star(A, B) -> SMat

Redheffer star product connecting `A`'s right ports to `B`'s left ports, giving a
component with `A`'s left ports and `B`'s right ports. This is the per-connection
operation that SAX's circuit backend performs internally when cascading the EME
stack.
"""
function star(A::SMat, B::SMat)
    A.nr == B.nl || throw(DimensionMismatch("port count mismatch in cascade: $(A.nr) vs $(B.nl)"))
    A11, A12, A21, A22 = s11(A), s12(A), s21(A), s22(A)
    B11, B12, B21, B22 = s11(B), s12(B), s21(B), s22(B)
    n = A.nr
    Ic = Matrix{ComplexF64}(I, n, n)
    M1 = (Ic .- B11 * A22) \ Matrix(B11 * A21)           # (I - B11 A22)⁻¹ B11 A21
    M2 = (Ic .- B11 * A22) \ Matrix(B12)                 # (I - B11 A22)⁻¹ B12
    M3 = (Ic .- A22 * B11) \ Matrix(A21)                 # (I - A22 B11)⁻¹ A21
    M4 = (Ic .- A22 * B11) \ Matrix(A22 * B12)           # (I - A22 B11)⁻¹ A22 B12
    C11 = A11 .+ A12 * M1
    C12 = A12 * M2
    C21 = B21 * M3
    C22 = B22 .+ B21 * M4
    S = [C11 C12; C21 C22]
    return SMat(Matrix(S), A.nl, B.nr)
end

"""
    cascade(smats) -> SMat

Cascade an ordered list of S-matrices (left to right) via repeated [`star`](@ref)
products. For an EME stack this is `[prop₀, iface₀₁, prop₁, iface₁₂, …, prop_{n-1}]`,
yielding the full device S-matrix between the input facet's modes and the output
facet's modes.
"""
function cascade(smats::AbstractVector{<:SMat})
    isempty(smats) && throw(ArgumentError("cannot cascade an empty stack"))
    acc = smats[1]
    for k in 2:length(smats)
        acc = star(acc, smats[k])
    end
    return acc
end
