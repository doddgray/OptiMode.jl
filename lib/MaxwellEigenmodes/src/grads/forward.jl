#### Forward-mode (frule) sensitivities for the Helmholtz eigensolves.
#
# Enzyme forward mode (and Zygote/Diffractor forward) cannot natively differentiate the
# FFTW-planned, KrylovKit-based `solve_k`/`solve_k_periodic` (they try to differentiate the
# `ModeSolver`/FFT-plan construction). These `ChainRulesCore.frule`s supply the exact
# forward tangents — the forward-mode companions of the adjoint `rrule`s in
# `grads/solve.jl` and `grads/period.jl` — and are bridged to Enzyme via
# `Enzyme.@import_frule` in the package's Enzyme extension.

# Apply the Helmholtz operator M̂[A] (tensor field `A` in place of ε⁻¹) at polarization basis
# (mag,mn) to a grid-shaped transverse field `Hg`. Since ⟨H|_M_apply(H,ε⁻¹,mag,mn)⟩ =
# HMH(H,ε⁻¹,mag,mn) (= ω² for a normalized eigenvector), this returns M̂[A]·Hg as a
# (2, size(grid)...) transverse field. Linear in `A`, so M̂[ε̇⁻¹]·Hg is exact.
function _M_apply(Hg::AbstractArray{Complex{T}}, A, mag, mn, fftax) where {T<:Real}
    kx_ct(ifft(ε⁻¹_dot(fft(kx_tc(Hg, mn, mag), fftax), A), fftax), mn, mag)
end

# Forward tangents (k̇, ėv) of one guided eigenpair (k,ev) of M̂(k,ε⁻¹) with eigenvalue ω²,
# for input tangents (ω̇, ε̇⁻¹) and a per-plane-wave operator k-derivative weight (1 for the
# Bloch wavevector; g_z/Λ for the period — see `solve_k_periodic`). `kg_weight===nothing`
# means the standard kz derivative (uniform weight 1).
function _solve_k_fwd_pair(ms, ω::T, k::T, ev, ω̇::T, ε̇, has_ε̇::Bool, grid::Grid{ND,T},
        fftax, Ns) where {ND,T<:Real}
    mag, mn = ms.M̂.mag, ms.M̂.mn
    evg = reshape(ev, (2, Ns...))
    ∂ω²∂k = 2 * HMₖH(ev, ms.M̂.ε⁻¹, mag, mn)
    Q = has_ε̇ ? HMH(ev, ε̇, mag, mn) : zero(T)            # ⟨ev|M̂[ε̇⁻¹]|ev⟩
    k̇ = (2ω*ω̇ - Q) / ∂ω²∂k
    # dM̂·ev = k̇·∂ₖ(M̂·ev) + M̂[ε̇⁻¹]·ev   (ε part exact; k part a tight central FD in k)
    hk = T(1e-6) * max(abs(k), one(T))
    magp, mnp = mag_mn(k + hk, grid)
    magm, mnm = mag_mn(k - hk, grid)
    dMk_ev = (_M_apply(evg, ms.M̂.ε⁻¹, magp, mnp, fftax) .-
              _M_apply(evg, ms.M̂.ε⁻¹, magm, mnm, fftax)) ./ (2hk)
    dM_ev = k̇ .* dMk_ev
    if has_ε̇
        dM_ev = dM_ev .+ _M_apply(evg, ε̇, mag, mn, fftax)
    end
    rhs = (2ω*ω̇) .* evg .- dM_ev
    ėv = eig_adjt(ms.M̂, ω^2, ev, 0.0, vec(rhs); P̂=ms.P̂)   # deflated, ⟨ev|ėv⟩ = 0 gauge
    return k̇, ėv
end

function ChainRulesCore.frule((_, Δω, Δε⁻¹, _, _), ::typeof(solve_k),
        ω::T, ε⁻¹::AbstractArray{T}, grid::Grid{ND,T}, solver::AbstractEigensolver;
        nev=1, kwargs...) where {ND,T<:Real}
    kmags, evecs = solve_k(ω, ε⁻¹, grid, solver; nev, kwargs...)
    ω̇ = Δω isa AbstractZero ? zero(T) : T(Δω)
    has_ε̇ = !(Δε⁻¹ isa AbstractZero)
    ε̇ = has_ε̇ ? Δε⁻¹ : ε⁻¹
    fftax = _fftaxes(grid); Ns = size(grid)
    k̇s = Vector{T}(undef, nev)
    ėvecs = Vector{Vector{Complex{T}}}(undef, nev)
    for (i, (k, ev)) in enumerate(zip(kmags, evecs))
        ms = ModeSolver(k, ε⁻¹, grid; nev)
        k̇s[i], ėvecs[i] = _solve_k_fwd_pair(ms, ω, k, ev, ω̇, ε̇, has_ε̇, grid, fftax, Ns)
    end
    # Return the output tangent as a plain tuple matching the primal `(kmags, evecs)`
    # structure (Enzyme's `@import_frule`/`Duplicated` does not accept a ChainRules
    # `Tangent`); a structural tuple is a valid Tuple differential for ChainRules too.
    return (kmags, evecs), (k̇s, ėvecs)
end

"""
Forward-mode rule for [`solve_k_periodic`](@ref). Like the `solve_k` frule but with the
extra absolute-period tangent: the fixed-ω constraint eigval(k,ε⁻¹,Λ)=ω² adds the period
term to k̇, and the operator directional derivative `dM̂·ev` is taken jointly in (k,Λ)
(the period enters M̂ through the reciprocal-lattice z-components g_z = m/Λ).
"""
function ChainRulesCore.frule((_, Δω, Δε⁻¹, ΔΛ, _, _), ::typeof(solve_k_periodic),
        ω::T, ε⁻¹::AbstractArray{T}, Λ::Real, grid::Grid{3,T}, solver::AbstractEigensolver;
        nev=1, kwargs...) where {T<:Real}
    g = _grid_with_period(grid, Λ)
    kmags, evecs = solve_k(ω, ε⁻¹, g, solver; nev, kwargs...)
    ω̇ = Δω isa AbstractZero ? zero(T) : T(Δω)
    Λ̇ = ΔΛ isa AbstractZero ? zero(T) : T(ΔΛ)
    has_ε̇ = !(Δε⁻¹ isa AbstractZero); ε̇ = has_ε̇ ? Δε⁻¹ : ε⁻¹
    fftax = _fftaxes(g); Ns = size(g)
    k̇s = Vector{T}(undef, nev)
    ėvecs = Vector{Vector{Complex{T}}}(undef, nev)
    for (i, (k, ev)) in enumerate(zip(kmags, evecs))
        ms = ModeSolver(k, ε⁻¹, g; nev)
        mag, mn = ms.M̂.mag, ms.M̂.mn
        evg = reshape(ev, (2, Ns...))
        ∂ω²∂k = 2 * HMₖH(ev, ms.M̂.ε⁻¹, mag, mn)
        ∂ω²∂Λ = ∂ω²_∂Λ(ev, ms.M̂.ε⁻¹, mag, mn, g)
        Q = has_ε̇ ? HMH(ev, ε̇, mag, mn) : zero(T)
        k̇ = (2ω*ω̇ - Q - Λ̇*∂ω²∂Λ) / ∂ω²∂k
        k̇s[i] = k̇
        # joint (k,Λ) operator directional derivative by central FD; ε part exact
        h = T(1e-6) * max(abs(k), one(T))
        gp = _grid_with_period(grid, Λ + h*Λ̇); gm = _grid_with_period(grid, Λ - h*Λ̇)
        magp, mnp = mag_mn(k + h*k̇, gp); magm, mnm = mag_mn(k - h*k̇, gm)
        dM_ev = (_M_apply(evg, ms.M̂.ε⁻¹, magp, mnp, fftax) .-
                 _M_apply(evg, ms.M̂.ε⁻¹, magm, mnm, fftax)) ./ (2h)
        if has_ε̇
            dM_ev = dM_ev .+ _M_apply(evg, ε̇, mag, mn, fftax)
        end
        rhs = (2ω*ω̇) .* evg .- dM_ev
        ėvecs[i] = eig_adjt(ms.M̂, ω^2, ev, 0.0, vec(rhs); P̂=ms.P̂)
    end
    return (kmags, evecs), (k̇s, ėvecs)
end
