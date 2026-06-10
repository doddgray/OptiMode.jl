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
    h_temp =  [ 0, 0, 1 ] × n # for now ignore "gimbal lock" edge case where n ≈ [0,0,1]
    h = h_temp ./ (h_temp[1]^2 + h_temp[2]^2)^(1//2)
	v = n × h   # the third unit vector `v` is just the cross of `n` and `h`
    S = [ n h v ]  # S is a unitary 3x3 matrix
    return S
end

function τ_trans(ε)
    return               [     -inv(ε[1,1])            ε[1,2]/ε[1,1]                           ε[1,3]/ε[1,1]
                                ε[2,1]/ε[1,1]       ε[2,2] - ε[2,1]*ε[1,2]/ε[1,1]           ε[2,3] - ε[2,1]*ε[1,3]/ε[1,1]
                                ε[3,1]/ε[1,1]       ε[3,2] - ε[3,1]*ε[1,2]/ε[1,1]           ε[3,3] - ε[3,1]*ε[1,3]/ε[1,1]       ]
end

function τ⁻¹_trans(τ)
    return           [      -inv(τ[1,1])           -τ[1,2]/τ[1,1]                          -τ[1,3]/τ[1,1]
                            -τ[2,1]/τ[1,1]       τ[2,2] - τ[2,1]*τ[1,2]/τ[1,1]           τ[2,3] - τ[2,1]*τ[1,3]/τ[1,1]
                            -τ[3,1]/τ[1,1]       τ[3,2] - τ[3,1]*τ[1,2]/τ[1,1]           τ[3,3] - τ[3,1]*τ[1,3]/τ[1,1]          ]
end

"""
Kottke averaging of two dielectric tensors `ε₁`, `ε₂` across a planar material interface whose normal vector `n₁₂` pointing from material 1 to material 2.
`n₁₂` is the first column of the unitary rotation matrix `S` which rotates `ε₁` and `ε₂` into the "frame of the interface" so that dielectric tensor
elements interacting with E-fields parallel and perpendicular to the interface can be averaged differently.

`r₁` is the ratio of the averaging volume filled by material 1. The normal vector mentioned above should point out from material 1 to material 2.
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

####### Symbolic computation of Kottke-smoothed dielectric tensor `f_εₑ_sym(r₁, n[1:3], ε₁[1:3,1:3], ε₂[1:3,1:3])` #######
# as a function of symbolic variables:
#
#  Variable             Physical Meaning                                                    Domain/Properties
# ________________________________________________________________________________________________________________________________________________________________
#   r₁              |   fraction of smoothing pixel occupied by material 1              |   Real scalar,    (0,1)
#   n[1:3]          |   local normal vector of material 1/2 interface, pointing 1→2     |   Real 3-vector,   sum(abs2,n) = 1
#   ε₁[1:3,1:3]     |   material 1 dielectric tensor                                    |   3x3 matrix, Hermitian w/o loss, Hermitian & Real w/o loss or magnetism
#   ε₂[1:3,1:3]     |   material 2 dielectric tensor                                    |   3x3 matrix, Hermitian w/o loss, Hermitian & Real w/o loss or magnetism

### 3D Non-Hermitian case ###

function _f_εₑᵣ_sym()
    p = @variables r₁ ε₁ᵣ_11 ε₁ᵣ_12 ε₁ᵣ_13 ε₁ᵣ_21 ε₁ᵣ_22 ε₁ᵣ_23 ε₁ᵣ_31 ε₁ᵣ_32 ε₁ᵣ_33 ε₂ᵣ_11 ε₂ᵣ_12 ε₂ᵣ_13 ε₂ᵣ_21 ε₂ᵣ_22 ε₂ᵣ_23 ε₂ᵣ_31 ε₂ᵣ_32 ε₂ᵣ_33
    ε₁ᵣ = [ ε₁ᵣ_11  ε₁ᵣ_12  ε₁ᵣ_13 ;  ε₁ᵣ_21  ε₁ᵣ_22  ε₁ᵣ_23 ; ε₁ᵣ_31  ε₁ᵣ_32  ε₁ᵣ_33 ]
    ε₂ᵣ = [ ε₂ᵣ_11  ε₂ᵣ_12  ε₂ᵣ_13 ;  ε₂ᵣ_21  ε₂ᵣ_22  ε₂ᵣ_23 ; ε₂ᵣ_31  ε₂ᵣ_32  ε₂ᵣ_33 ]
    τavg = ( r₁ * τ_trans( ε₁ᵣ ) ) + ( (1-r₁) * τ_trans( ε₂ᵣ ) )
    f_εₑᵣ_sym = simplify_fractions.(vec(τ⁻¹_trans(τavg)))
    return f_εₑᵣ_sym, p
end

const _εₑᵣ_sym_cache = let
    f_εₑᵣ_sym, prot = _f_εₑᵣ_sym()
    fj_εₑᵣ_sym, fjh_εₑᵣ_sym = _fj_fjh_sym(f_εₑᵣ_sym, prot)
    (f_εₑᵣ_sym, fj_εₑᵣ_sym, fjh_εₑᵣ_sym, prot)
end

# Generate the smoothing kernels by evaluating the generated code into *this* module
# (evaluating into MaterialDispersion would break incremental precompilation).
# out-of-place (non-mutating) functions:
const f_εₑᵣ   = Core.eval(@__MODULE__, oop_fn_expr(_εₑᵣ_sym_cache[1],_εₑᵣ_sym_cache[4]));
const fj_εₑᵣ  = Core.eval(@__MODULE__, oop_fn_expr(_εₑᵣ_sym_cache[2],_εₑᵣ_sym_cache[4]));
const fjh_εₑᵣ = Core.eval(@__MODULE__, oop_fn_expr(_εₑᵣ_sym_cache[3],_εₑᵣ_sym_cache[4]));
# in-place (mutating) functions:
const f_εₑᵣ!   = Core.eval(@__MODULE__, ip_fn_expr(_εₑᵣ_sym_cache[1],_εₑᵣ_sym_cache[4]));
const fj_εₑᵣ!  = Core.eval(@__MODULE__, ip_fn_expr(_εₑᵣ_sym_cache[2],_εₑᵣ_sym_cache[4]));
const fjh_εₑᵣ! = Core.eval(@__MODULE__, ip_fn_expr(_εₑᵣ_sym_cache[3],_εₑᵣ_sym_cache[4]));

∂ωεₑᵣ(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)   = @views @inbounds reshape( fj_εₑᵣ(vcat(r₁,vec(ε₁),vec(ε₂)))[:,2:end]  * vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂)), (3,3) )

@inline ∂ωεₑᵣ(r₁,ε₁_∂ωε₁,ε₂_∂ωε₂) = @inbounds ∂ωεₑᵣ(
    r₁,
    reshape(ε₁_∂ωε₁[1:9],(3,3)),
    reshape(ε₂_∂ωε₂[1:9],(3,3)),
    reshape(ε₁_∂ωε₁[10:18],(3,3)),
    reshape(ε₂_∂ωε₂[10:18],(3,3)),
)

function εₑᵣ_∂ωεₑᵣ(r₁::Real,ε₁::AbstractMatrix{<:Real},ε₂::AbstractMatrix{<:Real},∂ω_ε₁::AbstractMatrix{<:Real},∂ω_ε₂::AbstractMatrix{<:Real})
    fj_εₑᵣ_12 = similar(ε₁,9,20)
    fj_εₑᵣ!(fj_εₑᵣ_12,vcat(r₁,vec(ε₁),vec(ε₂)));
    f_εₑᵣ_12, j_εₑᵣ_12 = @views @inbounds fj_εₑᵣ_12[:,1], fj_εₑᵣ_12[:,2:end];
    εₑᵣ_12 = reshape(f_εₑᵣ_12,(3,3))
    v_∂ω = vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂));
    ∂ω_εₑᵣ_12 = reshape( j_εₑᵣ_12 * v_∂ω, (3,3) );
    return vcat(vec(εₑᵣ_12), vec(∂ω_εₑᵣ_12))
end

function εₑᵣ_∂ωεₑᵣ(r₁::Real,ε₁_∂ωε₁::AbstractVector{<:Real},ε₂_∂ωε₂::AbstractVector{<:Real})
    return @inbounds εₑᵣ_∂ωεₑᵣ(
        r₁,
        reshape(ε₁_∂ωε₁[1:9],(3,3)),
        reshape(ε₂_∂ωε₂[1:9],(3,3)),
        reshape(ε₁_∂ωε₁[10:18],(3,3)),
        reshape(ε₂_∂ωε₂[10:18],(3,3)),
    )
end

function εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(r₁::T1,ε₁::AbstractMatrix{T2},ε₂::AbstractMatrix{T3},∂ω_ε₁::AbstractMatrix{<:Real},∂ω_ε₂::AbstractMatrix{<:Real},∂²ω_ε₁::AbstractMatrix{<:Real},∂²ω_ε₂::AbstractMatrix{<:Real}) where {T1<:Real,T2<:Real,T3<:Real}
    fjh_εₑᵣ_12 = fjh_εₑᵣ(vcat(r₁,vec(ε₁),vec(ε₂)));
    f_εₑᵣ_12, j_εₑᵣ_12, h_εₑᵣ_12 = @views @inbounds fjh_εₑᵣ_12[:,1], fjh_εₑᵣ_12[:,2:20], reshape(fjh_εₑᵣ_12[:,21:381],(9,19,19));
    εₑᵣ_12 = reshape(f_εₑᵣ_12,(3,3))
    v_∂ω, v_∂²ω = vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂)), vcat(0.0,vec(∂²ω_ε₁),vec(∂²ω_ε₂));
    ∂ω_εₑᵣ_12 = reshape( j_εₑᵣ_12 * v_∂ω, (3,3) );
    ∂ω²_εₑᵣ_12 = @views reshape( [dot(v_∂ω,h_εₑᵣ_12[i,:,:],v_∂ω) for i=1:9] + j_εₑᵣ_12*v_∂²ω , (3,3) );
    return vcat(vec(εₑᵣ_12), vec(∂ω_εₑᵣ_12), vec(∂ω²_εₑᵣ_12))
end

function εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(r₁::Real,ε₁_∂ωε₁_∂²ωε₁::AbstractVector{<:Real},ε₂_∂ωε₂_∂²ωε₂::AbstractVector{<:Real})
    return @inbounds εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(
        r₁,
        reshape(ε₁_∂ωε₁_∂²ωε₁[1:9],(3,3)),
        reshape(ε₂_∂ωε₂_∂²ωε₂[1:9],(3,3)),
        reshape(ε₁_∂ωε₁_∂²ωε₁[10:18],(3,3)),
        reshape(ε₂_∂ωε₂_∂²ωε₂[10:18],(3,3)),
        reshape(ε₁_∂ωε₁_∂²ωε₁[19:27],(3,3)),
        reshape(ε₂_∂ωε₂_∂²ωε₂[19:27],(3,3)),
    )
end

function _rotate(S::AbstractMatrix{<:Real},ε::AbstractMatrix{<:Real})
    transpose(S) * (ε * S)
end

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

@inline herm_vec(A::AbstractMatrix) = @inbounds [A[1,1], A[2,1], A[3,1], A[2,2], A[3,2], A[3,3] ]
@inline function herm_vec(A::SHermitianCompact{3,T,6}) where T<:Number
    return A.lowertriangle
end
