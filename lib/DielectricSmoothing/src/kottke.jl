export normcart, τ_trans, τ⁻¹_trans, avg_param, avg_param_rot,
    εₑ_∂ωεₑ, εₑ_∂ωεₑ_∂²ωεₑ, εₑᵣ_∂ωεₑᵣ, εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ

"""
Create and return a local Cartesian coordinate system `S` (an ortho-normal 3x3 matrix) from a 3-vector `n0`.
`n0` inputs will be outward-pointing surface-normal vectors from shapes in a geometry, and `S` matrix outputs
will be used to rotate dielectric tensors into a coordinate system with two transverse axes and one perpendicular
axis w.r.t a (locally) planar dielectric interface. This allows transverse and perpendicular tensor components
to be smoothed differently, see Kottke Phys. Rev. E paper.

The input 3-vector `n` is assumed to be normalized such that `sum(abs2,n) == 1`
"""
function normcart(n)
    n3 = SVector{3}(n[1], n[2], n[3])
    h_temp = SVector(-n3[2], n3[1], zero(n3[1]))   # ẑ × n; ignores "gimbal lock" edge case where n ≈ [0,0,1]
    h = h_temp / (h_temp[1]^2 + h_temp[2]^2)^(1//2)
    v = n3 × h   # the third unit vector `v` is just the cross of `n` and `h`
    S = hcat(n3, h, v)  # S is a unitary 3x3 matrix (static, type-stable for AD)
    return S
end

"""
    τ_trans(ε)

Kottke's τ-transform of a dielectric tensor expressed in interface coordinates (axis 1
normal to the interface). Writing ``ε`` in blocks with ``ε_{11}`` the
normal-normal component,

```math
τ(ε) = \\begin{pmatrix}
 -1/ε_{11}        &  ε_{12}/ε_{11}                       & ε_{13}/ε_{11} \\\\
 ε_{21}/ε_{11}    &  ε_{22} - ε_{21}ε_{12}/ε_{11}        & ε_{23} - ε_{21}ε_{13}/ε_{11} \\\\
 ε_{31}/ε_{11}    &  ε_{32} - ε_{31}ε_{12}/ε_{11}        & ε_{33} - ε_{31}ε_{13}/ε_{11}
\\end{pmatrix}.
```

The τ variables are chosen so that the fields they multiply (``D_⊥`` and ``E_∥``) are
*continuous* across the interface, making a plain volume-weighted average of `τ`
physically correct; the smoothed tensor is recovered with [`τ⁻¹_trans`](@ref). For a
diagonal ε and an axis-aligned interface this reproduces the classic rule: harmonic
mean for the normal component, arithmetic mean for tangential components. See Kottke,
Farjadpour & Johnson, Phys. Rev. E **77**, 036611 (2008).
"""
@inline function τ_trans(ε)
    i = inv(ε[1,1])
    # `SMatrix` (column-major) keeps this type-stable and non-allocating for any element
    # type — Real, the `J2` Taylor jet used for dispersion propagation, or AD `Dual`s.
    @inbounds SMatrix{3,3}(
        -i,                                 ε[2,1]*i,                     ε[3,1]*i,            # col 1
         ε[1,2]*i,        ε[2,2] - ε[2,1]*ε[1,2]*i,   ε[3,2] - ε[3,1]*ε[1,2]*i,                # col 2
         ε[1,3]*i,        ε[2,3] - ε[2,1]*ε[1,3]*i,   ε[3,3] - ε[3,1]*ε[1,3]*i,                # col 3
    )
end

"""
    τ⁻¹_trans(τ)

Inverse of [`τ_trans`](@ref): recover a dielectric tensor from averaged τ variables,
``τ^{-1}(τ(ε)) = ε``.
"""
@inline function τ⁻¹_trans(τ)
    i = inv(τ[1,1])
    @inbounds SMatrix{3,3}(
        -i,                                -τ[2,1]*i,                    -τ[3,1]*i,           # col 1
        -τ[1,2]*i,        τ[2,2] - τ[2,1]*τ[1,2]*i,   τ[3,2] - τ[3,1]*τ[1,2]*i,                # col 2
        -τ[1,3]*i,        τ[2,3] - τ[2,1]*τ[1,3]*i,   τ[3,3] - τ[3,1]*τ[1,3]*i,                # col 3
    )
end

"""
    avg_param(ε₁, ε₂, S, r₁)

Kottke sub-pixel average of two dielectric tensors `ε₁`, `ε₂` meeting at a (locally)
planar interface inside one pixel:

```math
\\tilde{ε} = S\\, τ^{-1}\\!\\Big( r_1\\, τ(S^Tε_1S) + (1-r_1)\\, τ(S^Tε_2S) \\Big) S^T,
```

where `S = normcart(n̂₁₂)` rotates into interface coordinates (first axis along the
unit normal `n̂₁₂` pointing from material 1 into material 2), `r₁ ∈ [0,1]` is the
volume fraction of the pixel occupied by material 1, and `τ`/`τ⁻¹` are the
continuity-respecting transforms [`τ_trans`](@ref)/[`τ⁻¹_trans`](@ref). This
anisotropic smoothing eliminates the first-order staircasing error of a discretized
material interface (Kottke, Farjadpour & Johnson, Phys. Rev. E **77**, 036611 (2008)).

[`avg_param_rot`](@ref) is the same average for tensors already expressed in interface
coordinates (no `S` rotation).
"""
function avg_param(ε₁, ε₂, S, r₁)
    τ1 = τ_trans( transpose(S) * ε₁ * S ) # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans( transpose(S) * ε₂ * S )
    τavg = ( r₁ * τ1 ) + ( (1-r₁) * τ2 ) # volume-weighted average
    return S * τ⁻¹_trans(τavg) * transpose(S)
end

function avg_param_rot(ε₁ᵣ, ε₂ᵣ, r₁)
    τavg = ( r₁ * τ_trans( ε₁ᵣ ) ) + ( (1-r₁) * τ_trans( ε₂ᵣ ) ) # volume-weighted average
    return τ⁻¹_trans(τavg)
end

####### Kottke-smoothed dielectric tensor with exact dispersion (∂ω, ∂²ω) propagation #######
#
# The Kottke average εₑ = τ⁻¹( r₁ τ(ε₁) + (1−r₁) τ(ε₂) ) is a closed-form algebraic map of
# the material tensors, so the dispersion of the *smoothed* tensor follows exactly by the
# chain rule. Rather than materialize the full symbolic Jacobian/Hessian of that map per
# pixel (a very large generated function — slow to compile, hostile to Enzyme), we carry a
# second-order Taylor jet `J2 = (value, ∂ω, ∂²ω)` through the *same* closed-form transforms
# (`τ_trans`/`τ⁻¹_trans`/`avg_param_rot`). The jet computes only the directional ω-deriva-
# tives actually needed, is type-stable and non-allocating, compiles fast, and — being
# explicit dual arithmetic rather than nested AD — stays single-level differentiable, so
# ForwardDiff/Zygote/Mooncake/Enzyme all traverse `smooth_ε` (forward and reverse).

# Truncated 2nd-order Taylor number in the (single) frequency variable: `v + d·δω + ½dd·δω²`
# stored as `(v, d, dd) = (f, f′, f″)`. Closed under +, −, ×, /, inv.
struct J2{T<:Real} <: Number
    v::T
    d::T
    dd::T
end
@inline J2{T}(x::Real) where {T} = J2{T}(T(x), zero(T), zero(T))
Base.zero(::Type{J2{T}}) where {T} = J2{T}(zero(T), zero(T), zero(T))
Base.one(::Type{J2{T}})  where {T} = J2{T}(one(T),  zero(T), zero(T))
Base.convert(::Type{J2{T}}, x::Real)   where {T} = J2{T}(x)
Base.convert(::Type{J2{T}}, x::J2{T})  where {T} = x
Base.convert(::Type{J2{T}}, x::J2)     where {T} = J2{T}(T(x.v), T(x.d), T(x.dd))
Base.promote_rule(::Type{J2{T}}, ::Type{S})     where {T,S<:Real} = J2{promote_type(T,S)}
Base.promote_rule(::Type{J2{T}}, ::Type{J2{S}}) where {T,S}       = J2{promote_type(T,S)}
@inline Base.:+(a::J2, b::J2) = J2(a.v+b.v, a.d+b.d, a.dd+b.dd)
@inline Base.:-(a::J2, b::J2) = J2(a.v-b.v, a.d-b.d, a.dd-b.dd)
@inline Base.:-(a::J2)        = J2(-a.v, -a.d, -a.dd)
@inline Base.:*(a::J2, b::J2) = J2(a.v*b.v, a.d*b.v + a.v*b.d, a.dd*b.v + 2*a.d*b.d + a.v*b.dd)
@inline function Base.inv(a::J2)                       # g=1/f ⇒ g′=−f′/f², g″=−f″/f²+2f′²/f³
    i = inv(a.v)
    J2(i, -a.d*i*i, -a.dd*i*i + 2*a.d*a.d*i*i*i)
end
@inline Base.:/(a::J2, b::J2) = a * inv(b)

# Build a static 3×3 jet tensor from per-entry (value, ∂ω, ∂²ω) data (column-major / `vec`
# order). The 1st-order method (no ∂²ω) zeros the 2nd-derivative slot.
@inline _jetmat(ε, dε) =
    SMatrix{3,3}(ntuple(k -> (@inbounds J2(ε[k], dε[k], zero(ε[k]))), Val(9)))
@inline _jetmat(ε, dε, ddε) =
    SMatrix{3,3}(ntuple(k -> (@inbounds J2(ε[k], dε[k], ddε[k])), Val(9)))
@inline _vals(E) = SVector(ntuple(k -> (@inbounds E[k].v),  Val(9)))
@inline _ders(E) = SVector(ntuple(k -> (@inbounds E[k].d),  Val(9)))
@inline _dds(E)  = SVector(ntuple(k -> (@inbounds E[k].dd), Val(9)))

# Just the ∂ω part of the smoothed (interface-coordinate) tensor.
function ∂ωεₑᵣ(r₁, ε₁, ε₂, ∂ω_ε₁, ∂ω_ε₂)
    Eeff = avg_param_rot(_jetmat(ε₁, ∂ω_ε₁), _jetmat(ε₂, ∂ω_ε₂), r₁)
    return reshape(_ders(Eeff), (3, 3))
end
@inline ∂ωεₑᵣ(r₁, ε₁_∂ωε₁, ε₂_∂ωε₂) = @inbounds ∂ωεₑᵣ(
    r₁,
    reshape(ε₁_∂ωε₁[1:9], (3, 3)),  reshape(ε₂_∂ωε₂[1:9], (3, 3)),
    reshape(ε₁_∂ωε₁[10:18], (3, 3)), reshape(ε₂_∂ωε₂[10:18], (3, 3)),
)

"""
    εₑᵣ_∂ωεₑᵣ(r₁, ε₁, ε₂, ∂ω_ε₁, ∂ω_ε₂) -> SVector (length 18)

Smoothed dielectric tensor *and its exact frequency derivative* in interface ("rotated")
coordinates. Because the Kottke average ``ε_e = f(r_1, ε_1, ε_2)`` is a closed-form
algebraic map, the chain rule gives the dispersion of the smoothed tensor exactly:

```math
\\frac{∂ε_e}{∂ω} = \\frac{∂f}{∂ε_1}\\frac{∂ε_1}{∂ω} + \\frac{∂f}{∂ε_2}\\frac{∂ε_2}{∂ω},
```

evaluated by propagating a first-order Taylor jet through [`avg_param_rot`](@ref). Returns
`vcat(vec(εₑ), vec(∂ωεₑ))`. Vector-input methods take the per-material data packed as
`vcat(vec(ε), vec(∂ωε))`.
"""
function εₑᵣ_∂ωεₑᵣ(r₁::Real, ε₁::AbstractMatrix{<:Real}, ε₂::AbstractMatrix{<:Real}, ∂ω_ε₁::AbstractMatrix{<:Real}, ∂ω_ε₂::AbstractMatrix{<:Real})
    Eeff = avg_param_rot(_jetmat(ε₁, ∂ω_ε₁), _jetmat(ε₂, ∂ω_ε₂), r₁)
    return vcat(_vals(Eeff), _ders(Eeff))
end

function εₑᵣ_∂ωεₑᵣ(r₁::Real, ε₁_∂ωε₁::AbstractVector{<:Real}, ε₂_∂ωε₂::AbstractVector{<:Real})
    return @inbounds εₑᵣ_∂ωεₑᵣ(
        r₁,
        reshape(ε₁_∂ωε₁[1:9], (3, 3)),  reshape(ε₂_∂ωε₂[1:9], (3, 3)),
        reshape(ε₁_∂ωε₁[10:18], (3, 3)), reshape(ε₂_∂ωε₂[10:18], (3, 3)),
    )
end

"""
    εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(r₁, ε₁, ε₂, ∂ω_ε₁, ∂ω_ε₂, ∂²ω_ε₁, ∂²ω_ε₂) -> SVector (length 27)

Like [`εₑᵣ_∂ωεₑᵣ`](@ref) but also propagating the *second* frequency derivative through
the smoothing map:

```math
\\frac{∂^2ε_e}{∂ω^2} = \\sum_m \\frac{∂f}{∂ε_m}\\frac{∂^2ε_m}{∂ω^2}
 + \\sum_{m,n} \\frac{∂^2 f}{∂ε_m∂ε_n}\\frac{∂ε_m}{∂ω}\\frac{∂ε_n}{∂ω}.
```

This is obtained exactly by carrying a *second-order* Taylor jet (`J2`) through
[`avg_param_rot`](@ref) — the jet's `*`/`inv` rules encode precisely the first and second
total-ω-derivatives above, with no symbolic Jacobian/Hessian. Returns
`vcat(vec(εₑ), vec(∂ωεₑ), vec(∂²ωεₑ))`; the kernel behind the third slice of
[`smooth_ε`](@ref)'s output (group-velocity-dispersion).
"""
function εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(r₁::T1, ε₁::AbstractMatrix{T2}, ε₂::AbstractMatrix{T3}, ∂ω_ε₁::AbstractMatrix{<:Real}, ∂ω_ε₂::AbstractMatrix{<:Real}, ∂²ω_ε₁::AbstractMatrix{<:Real}, ∂²ω_ε₂::AbstractMatrix{<:Real}) where {T1<:Real,T2<:Real,T3<:Real}
    Eeff = avg_param_rot(_jetmat(ε₁, ∂ω_ε₁, ∂²ω_ε₁), _jetmat(ε₂, ∂ω_ε₂, ∂²ω_ε₂), r₁)
    return vcat(_vals(Eeff), _ders(Eeff), _dds(Eeff))
end

function εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(r₁::Real, ε₁_∂ωε₁_∂²ωε₁::AbstractVector{<:Real}, ε₂_∂ωε₂_∂²ωε₂::AbstractVector{<:Real})
    return @inbounds εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(
        r₁,
        reshape(ε₁_∂ωε₁_∂²ωε₁[1:9], (3, 3)),   reshape(ε₂_∂ωε₂_∂²ωε₂[1:9], (3, 3)),
        reshape(ε₁_∂ωε₁_∂²ωε₁[10:18], (3, 3)), reshape(ε₂_∂ωε₂_∂²ωε₂[10:18], (3, 3)),
        reshape(ε₁_∂ωε₁_∂²ωε₁[19:27], (3, 3)), reshape(ε₂_∂ωε₂_∂²ωε₂[19:27], (3, 3)),
    )
end

function _rotate(S::AbstractMatrix{<:Real},ε::AbstractMatrix{<:Real})
    transpose(S) * (ε * S)
end

"""
    εₑ_∂ωεₑ(r₁, S, ε₁, ε₂, ∂ω_ε₁, ∂ω_ε₂) -> Vector (length 18)

Lab-frame version of [`εₑᵣ_∂ωεₑᵣ`](@ref): rotates the input tensors into interface
coordinates with `S = normcart(n̂)`, applies the dispersion-propagating Kottke kernel,
and rotates the results back, `εₑ = S εₑᵣ Sᵀ`.
"""
function εₑ_∂ωεₑ(r₁::Real,S::AbstractMatrix{<:Real},ε₁::AbstractMatrix{<:Real},ε₂::AbstractMatrix{<:Real},∂ω_ε₁::AbstractMatrix{<:Real},∂ω_ε₂::AbstractMatrix{<:Real})
    res_rot = εₑᵣ_∂ωεₑᵣ(r₁,_rotate(S,ε₁),_rotate(S,ε₂),_rotate(S,∂ω_ε₁),_rotate(S,∂ω_ε₂))
    eps = @inbounds vec(_rotate(transpose(S),reshape(res_rot[1:9],(3,3))))
    deps = @inbounds vec(_rotate(transpose(S),reshape(res_rot[10:18],(3,3))))
    return vcat(eps,deps)
end

function εₑ_∂ωεₑ(r₁::Real,S::AbstractMatrix{<:Real},ε₁_∂ωε₁::AbstractVector{<:Real},ε₂_∂ωε₂::AbstractVector{<:Real})
    res_rot = @inbounds εₑᵣ_∂ωεₑᵣ(
        r₁,
        _rotate(S,reshape(ε₁_∂ωε₁[1:9],(3,3))),
        _rotate(S,reshape(ε₂_∂ωε₂[1:9],(3,3))),
        _rotate(S,reshape(ε₁_∂ωε₁[10:18],(3,3))),
        _rotate(S,reshape(ε₂_∂ωε₂[10:18],(3,3))),
    )
    eps = @inbounds vec(_rotate(transpose(S),reshape(res_rot[1:9],(3,3))))
    deps = @inbounds vec(_rotate(transpose(S),reshape(res_rot[10:18],(3,3))))
    return vcat(eps,deps)
end

"""
    εₑ_∂ωεₑ_∂²ωεₑ(r₁, S, ε₁, ε₂, ∂ω_ε₁, ∂ω_ε₂, ∂²ω_ε₁, ∂²ω_ε₂) -> Vector (length 27)

Lab-frame version of [`εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ`](@ref): Kottke smoothing with exact
propagation of first and second frequency derivatives, rotated into/out of interface
coordinates by `S = normcart(n̂)`. The vector-input method takes per-material data
packed as `vcat(vec(ε), vec(∂ωε), vec(∂²ωε))` — the per-material column layout produced
by `MaterialDispersion._f_ε_mats`.
"""
function εₑ_∂ωεₑ_∂²ωεₑ(r₁::Real,S::AbstractMatrix{<:Real},ε₁::AbstractMatrix{<:Real},ε₂::AbstractMatrix{<:Real},∂ω_ε₁::AbstractMatrix{<:Real},∂ω_ε₂::AbstractMatrix{<:Real},∂²ω_ε₁::AbstractMatrix{<:Real},∂²ω_ε₂::AbstractMatrix{<:Real})
    res_rot = εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(r₁,_rotate(S,ε₁),_rotate(S,ε₂),_rotate(S,∂ω_ε₁),_rotate(S,∂ω_ε₂),_rotate(S,∂²ω_ε₁),_rotate(S,∂²ω_ε₂))
    eps = @inbounds vec(_rotate(transpose(S),reshape(res_rot[1:9],(3,3))))
    deps = @inbounds vec(_rotate(transpose(S),reshape(res_rot[10:18],(3,3))))
    ddeps = @inbounds vec(_rotate(transpose(S),reshape(res_rot[19:27],(3,3))))
    return vcat(eps,deps,ddeps)
end

function εₑ_∂ωεₑ_∂²ωεₑ(r₁::Real,S::AbstractMatrix{<:Real},ε₁_∂ωε₁_∂²ωε₁::AbstractVector{<:Real},ε₂_∂ωε₂_∂²ωε₂::AbstractVector{<:Real})
    res_rot = @inbounds εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(
        r₁,
        _rotate(S,reshape(ε₁_∂ωε₁_∂²ωε₁[1:9],(3,3))),
        _rotate(S,reshape(ε₂_∂ωε₂_∂²ωε₂[1:9],(3,3))),
        _rotate(S,reshape(ε₁_∂ωε₁_∂²ωε₁[10:18],(3,3))),
        _rotate(S,reshape(ε₂_∂ωε₂_∂²ωε₂[10:18],(3,3))),
        _rotate(S,reshape(ε₁_∂ωε₁_∂²ωε₁[19:27],(3,3))),
        _rotate(S,reshape(ε₂_∂ωε₂_∂²ωε₂[19:27],(3,3))),
    )
    eps = @inbounds vec(_rotate(transpose(S),reshape(res_rot[1:9],(3,3))))
    deps = @inbounds vec(_rotate(transpose(S),reshape(res_rot[10:18],(3,3))))
    ddeps = @inbounds vec(_rotate(transpose(S),reshape(res_rot[19:27],(3,3))))
    return vcat(eps,deps,ddeps)
end

# Reverse rule for the dispersion-propagating kernel, w.r.t. the per-material data columns
# (the differentiated inputs in `smooth_ε`). The jet propagation traces natively under
# ForwardDiff / Enzyme / Mooncake, but Zygote cannot differentiate the `J2`-struct /
# `SVector(ntuple(...))` construction; this `rrule` lets Zygote consume the kernel through
# a small ForwardDiff Jacobian instead. `r₁` (fill fraction) and `S` (interface frame) are
# geometry — `@non_differentiable` for ChainRules — so they take `NoTangent`.
function ChainRulesCore.rrule(::typeof(εₑ_∂ωεₑ_∂²ωεₑ), r₁::Real, S::AbstractMatrix{<:Real},
        col1::AbstractVector{<:Real}, col2::AbstractVector{<:Real})
    y = εₑ_∂ωεₑ_∂²ωεₑ(r₁, S, col1, col2)
    c1, c2 = collect(col1), collect(col2)
    function εₑ_∂ωεₑ_∂²ωεₑ_pullback(ȳ)
        ȳv = collect(ChainRulesCore.unthunk(ȳ))
        J1 = ForwardDiff.jacobian(c -> εₑ_∂ωεₑ_∂²ωεₑ(r₁, S, c, c2), c1)
        J2 = ForwardDiff.jacobian(c -> εₑ_∂ωεₑ_∂²ωεₑ(r₁, S, c1, c), c2)
        return (NoTangent(), NoTangent(), NoTangent(), transpose(J1) * ȳv, transpose(J2) * ȳv)
    end
    return y, εₑ_∂ωεₑ_∂²ωεₑ_pullback
end

@inline herm_vec(A::AbstractMatrix) = @inbounds [A[1,1], A[2,1], A[3,1], A[2,2], A[3,2], A[3,3] ]
@inline function herm_vec(A::SHermitianCompact{3,T,6}) where T<:Number
    return A.lowertriangle
end
