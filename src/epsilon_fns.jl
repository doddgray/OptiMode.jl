export normcart, τ_trans, τ⁻¹_trans, avg_param, avg_param_rot, _f_ε_mats, _fj_ε_mats, _fjh_ε_mats, 
    ε_views, εₑ_∂ωεₑ_∂²ωεₑ, εₑ_∂ωεₑ

rules_2D = Prewalk(PassThrough(@acrule sin(~x)^2 + cos(~x)^2 => 1 ))

####### Symbolic Jacobians (`j_sym`) and Hessians (`h_sym`) of symbolic array-valued functions `f_sym`,
####### fused into single arrays `fj_sym` or `fjh_sym` for performant function generation 

_fj_sym(f_sym,p) = hcat( vec(f_sym), Symbolics.jacobian(vec(scalarize(f_sym)),p) ) # fj_sym, size = length(f) x ( 1 + length(p) )

function _fj_fjh_sym(f_sym,p)
    j_sym = Symbolics.jacobian(vec(scalarize(f_sym)),p)
    h_sym = mapreduce(x->transpose(vec(Symbolics.hessian(x,p))),vcat,vec(f_sym));
    return hcat(vec(f_sym),j_sym), hcat(vec(f_sym),j_sym,h_sym)  # (fj_sym, fjh_sym), sizes length(f) x ( 1 + length(p) ),  length(f) x ( 1 + length(p) + length(p)^2 ) 
end

###### ε_mats Generation and Utility Functions ######

function _f_ε_mats_sym(mats,p_syms=(:ω,))
    @variables ω
    Dom = Differential(ω)
    p = [ω, (Num(Sym{Real}(p_sym)) for p_sym in p_syms[2:end])...]
    ε_∂ωε_∂²ωε_mats = mapreduce(hcat,mats) do mm
        eps_vec = vec(get_model(mm,:ε,p_syms...))
        dom_eps_vec = expand_derivatives.(Dom.(eps_vec))
        ddom_eps_vec = expand_derivatives.(Dom.(dom_eps_vec))
        return vcat(eps_vec,dom_eps_vec,ddom_eps_vec)
    end
    return 1.0*ε_∂ωε_∂²ωε_mats, p
end

"""
Example:
    ```
    mats = [MgO_LiNbO₃,Si₃N₄,SiO₂,LiB₃O₅];
    n_mats = length(mats);
    f_ε_mats, f_ε_mats! = _f_ε_mats(mats,(:ω,:T))
    @show ω0,T0 = ( 0.2*rand(Float64)+0.8, 20.0*rand(Float64)+20.0 ) # frequency and temperature, in μm⁻¹ and C respectively
    p0 = [ω0,T0];
    f0 = f_ε_mats(p0);
    ε, ∂ωε, ∂²ωε = ε_views(f,n_mats);
    ```
"""
function _f_ε_mats(mats,p_syms=(:ω,);expression=Val{false})
    f_ε_mats_sym, p = _f_ε_mats_sym(mats,p_syms)
    f_ε_∂ωε_∂²ωε = eval_fn_oop(f_ε_mats_sym, p)
    f_ε_∂ωε_∂²ωε! = eval_fn_ip(f_ε_mats_sym, p)
    return f_ε_∂ωε_∂²ωε, f_ε_∂ωε_∂²ωε!
end

"""
Example:
    ```
    mats = [MgO_LiNbO₃,Si₃N₄,SiO₂,LiB₃O₅];
    n_mats = length(mats);
    fj_ε_mats, fj_ε_mats! = _fj_ε_mats(mats,(:ω,:T))
    @show ω0,T0 = ( 0.2*rand(Float64)+0.8, 20.0*rand(Float64)+20.0 ) # frequency and temperature, in μm⁻¹ and C respectively
    p0 = [ω0,T0];
    fj0 = fj_ε_mats(p0);
    (ε, ∂ωε, ∂²ωε), (∂ω_ε, ∂ω_∂ωε, ∂ω_∂²ωε), (∂T_ε, ∂T_∂ωε, ∂T_∂²ωε) = ε_views.(eachcol(fj0),(n_mats,));
    ```
"""
function _fj_ε_mats(mats,p_syms=(:ω,);expression=Val{false})
    f_ε_mats_sym, p = _f_ε_mats_sym(mats,p_syms) # 27*length(mats) x 1 
    j_ε_mats_sym = Symbolics.jacobian(vec(f_ε_mats_sym),p); # 27*length(mats) x length(p)
    fj_ε_mats_sym = hcat(vec(f_ε_mats_sym),j_ε_mats_sym); # 27*length(mats) x (1 + length(p))
    fj_ε_∂ωε_∂²ωε = eval_fn_oop(fj_ε_mats_sym, p)
    fj_ε_∂ωε_∂²ωε! = eval_fn_ip(fj_ε_mats_sym, p)
    return fj_ε_∂ωε_∂²ωε, fj_ε_∂ωε_∂²ωε!
end

"""
Example:
    ```
    mats = [MgO_LiNbO₃,Si₃N₄,SiO₂,LiB₃O₅];
    n_mats = length(mats);
    fjh_ε_mats, fjh_ε_mats! = _fjh_ε_mats(mats,(:ω,:T))
    @show ω0,T0 = ( 0.2*rand(Float64)+0.8, 20.0*rand(Float64)+20.0 ) # frequency and temperature, in μm⁻¹ and C respectively
    p0 = [ω0,T0];
    fjh0 = fjh_ε_mats(p0);
    (ε, ∂ωε, ∂²ωε), (∂ω_ε, ∂ω_∂ωε, ∂ω_∂²ωε), (∂T_ε, ∂T_∂ωε, ∂T_∂²ωε) = ε_views.(eachcol(fjh0),(n_mats,));
    ```
"""
function _fjh_ε_mats(mats,p_syms=(:ω,);expression=Val{false})
    f_ε_mats_sym, p = _f_ε_mats_sym(mats,p_syms) # 27*length(mats) x 1
    j_ε_mats_sym = Symbolics.jacobian(vec(f_ε_mats_sym),p); # 27*length(mats) x length(p)
    h_ε_mats_sym = mapreduce(x->transpose(vec(Symbolics.hessian(x,p))),vcat,vec(f_ε_mats_sym)); # 27*length(mats) x length(p)^2
    fjh_ε_mats_sym = hcat(vec(f_ε_mats_sym),j_ε_mats_sym,h_ε_mats_sym); # 27*length(mats) x ( 1 + length(p)*(1+length(p)) )
    fjh_ε_∂ωε_∂²ωε = eval_fn_oop(fjh_ε_mats_sym, p)
    fjh_ε_∂ωε_∂²ωε! = eval_fn_ip(fjh_ε_mats_sym, p)
    return fjh_ε_∂ωε_∂²ωε, fjh_ε_∂ωε_∂²ωε!
end

ε_views(εv,n_mats;nε=3) = ([ reshape(view(view(εv,(1+(i-1)*9*n_mats):(i*9*n_mats)), (1+9*(mat_idx-1)):(9+9*(mat_idx-1))), (3,3)) for mat_idx=1:n_mats] for i=1:nε) 

#### end ε_mats Generation and Utility Functions ####

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
    τ2 = substitute(τ1, Dict([ε₁[1,1]=>ε₂[1,1], ε₁[1,2]=>ε₂[1,2], ε₁[1,3]=>ε₂[1,3], ε₁[2,1]=>ε₂[2,1], ε₁[2,2]=>ε₂[2,2], ε₁[2,3]=>ε₂[2,3], ε₁[3,1]=>ε₂[3,1], ε₁[3,2]=>ε₂[3,2], ε₁[3,3]=>ε₂[3,3], ]))
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

f_εₑᵣ_sym, prot = _f_εₑᵣ_sym();
fj_εₑᵣ_sym, fjh_εₑᵣ_sym = _fj_fjh_sym(f_εₑᵣ_sym, prot);
# generate out-of-place (non-mutating) functions
f_εₑᵣ   = eval_fn_oop(f_εₑᵣ_sym,prot);
fj_εₑᵣ  = eval_fn_oop(fj_εₑᵣ_sym,prot);
fjh_εₑᵣ  = eval_fn_oop(fjh_εₑᵣ_sym,prot);
# generate in-place (mutating) functions
f_εₑᵣ!   = eval_fn_ip(f_εₑᵣ_sym,prot);
fj_εₑᵣ!  = eval_fn_ip(fj_εₑᵣ_sym,prot);
fjh_εₑᵣ! = eval_fn_ip(fjh_εₑᵣ_sym,prot);
# # force compilation of these generated functions
# fout_rot = f_εₑᵣ(rand(19));
# fjout_rot = fj_εₑᵣ(rand(19));
# fjhout_rot = fjh_εₑᵣ(MVector{19}(rand(19)));
# f_εₑᵣ!(similar(fout_rot),rand(19));
# fj_εₑᵣ!(similar(fjout_rot),rand(19));
# fjh_εₑᵣ!(similar(fjout_rot,9,381),rand(19));

∂ωεₑᵣ(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)   = @views @inbounds reshape( fj_εₑᵣ(vcat(r₁,vec(ε₁),vec(ε₂)))[:,2:end]  * vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂)), (3,3) )

@inline ∂ωεₑᵣ(r₁,ε₁_∂ωε₁,ε₂_∂ωε₂) = @inbounds ∂ωεₑᵣ(
    r₁,
    reshape(ε₁_∂ωε₁[1:9],(3,3)),
    reshape(ε₂_∂ωε₂[1:9],(3,3)),
    reshape(ε₁_∂ωε₁[10:18],(3,3)),
    reshape(ε₂_∂ωε₂[10:18],(3,3)),
)

function εₑᵣ_∂ωεₑᵣ(r₁::Real,ε₁::AbstractMatrix{<:Real},ε₂::AbstractMatrix{<:Real},∂ω_ε₁::AbstractMatrix{<:Real},∂ω_ε₂::AbstractMatrix{<:Real})
    fj_εₑᵣ_12 = similar(ε₁,9,20) # fj_εₑᵣ(vcat(r₁,vec(ε₁),vec(ε₂)));
    fj_εₑᵣ!(fj_εₑᵣ_12,vcat(r₁,vec(ε₁),vec(ε₂)));
    f_εₑᵣ_12, j_εₑᵣ_12 = @views @inbounds fj_εₑᵣ_12[:,1], fj_εₑᵣ_12[:,2:end];
    εₑᵣ_12 = reshape(f_εₑᵣ_12,(3,3))
    v_∂ω = vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂));
    ∂ω_εₑᵣ_12 = reshape( j_εₑᵣ_12 * v_∂ω, (3,3) );
    # return εₑᵣ_12, ∂ω_εₑᵣ_12
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
    fjh_εₑᵣ_12::MMatrix{9, 381, promote_type(T1,T2,T3), 3429} = fjh_εₑᵣ(MVector{19}(vcat(r₁,vec(ε₁),vec(ε₂))));

    f_εₑᵣ_12, j_εₑᵣ_12, h_εₑᵣ_12 = @views @inbounds fjh_εₑᵣ_12[:,1], fjh_εₑᵣ_12[:,2:20], reshape(fjh_εₑᵣ_12[:,21:381],(9,19,19));
    εₑᵣ_12 = reshape(f_εₑᵣ_12,(3,3))
    v_∂ω, v_∂²ω = vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂)), vcat(0.0,vec(∂²ω_ε₁),vec(∂²ω_ε₂));
    ∂ω_εₑᵣ_12 = reshape( j_εₑᵣ_12 * v_∂ω, (3,3) );
    ∂ω²_εₑᵣ_12 = @views reshape( [dot(v_∂ω,h_εₑᵣ_12[i,:,:],v_∂ω) for i=1:9] + j_εₑᵣ_12*v_∂²ω , (3,3) );
    # return εₑᵣ_12, ∂ω_εₑᵣ_12, ∂ω²_εₑᵣ_12
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
