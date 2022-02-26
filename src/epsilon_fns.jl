export normcart, τ_trans, τ⁻¹_trans, avg_param, avg_param_rot, _f_ε_mats, _fj_ε_mats, _fjh_ε_mats, ε_views, εₑ_∂ωεₑ_∂²ωεₑ, εₑ_∂ωεₑ, εₑ_∂ωεₑ_∂²ωεₑ_herm, εₑ_∂ωεₑ_herm

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
	ε_mats = mapreduce(mm->vec(get_model(mm,:ε,p_syms...)),hcat,mats);
    @variables ω
	Dom = Differential(ω)
    p = [ω, (Num(Sym{Real}(p_sym)) for p_sym in p_syms[2:end])...]
    ∂ωε_mats = expand_derivatives.(Dom.(ε_mats));
    ∂²ωε_mats = expand_derivatives.(Dom.(∂ωε_mats));
    ε_∂ωε_∂²ωε_mats = 1.0*hcat(ε_mats,∂ωε_mats,∂²ωε_mats);
    return vec(ε_∂ωε_∂²ωε_mats), p
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
    # f_ε_∂ωε_∂²ωε_ex, f_ε_∂ωε_∂²ωε!_ex = build_function(f_ε_mats_sym, p; expression)
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
    # fj_ε_mats_ex, fj_ε_mats!_ex = build_function(fj_ε_mats_sym, p ; expression);
    # return fj_ε_mats_ex, fj_ε_mats!_ex;
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
    # fjh_ε_mats_ex, fjh_ε_mats!_ex = build_function(fjh_ε_mats_sym, p ; expression);
    # return fjh_ε_mats_ex, fjh_ε_mats!_ex;
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



# function τ_trans(ε::AbstractMatrix{T}) where T<:Real
#     return @inbounds SMatrix{3,3,T,9}(
#         -inv(ε[1,1]),      ε[2,1]/ε[1,1],                  ε[3,1]/ε[1,1],
#         ε[1,2]/ε[1,1],  ε[2,2] - ε[2,1]*ε[1,2]/ε[1,1],  ε[3,2] - ε[3,1]*ε[1,2]/ε[1,1],
#         ε[1,3]/ε[1,1],  ε[2,3] - ε[2,1]*ε[1,3]/ε[1,1],  ε[3,3] - ε[3,1]*ε[1,3]/ε[1,1]
#     )
# end

# function τ⁻¹_trans(τ::AbstractMatrix{T}) where T<:Real
#     return @inbounds SMatrix{3,3,T,9}(
#         -inv(τ[1,1]),          -τ[2,1]/τ[1,1],                 -τ[3,1]/τ[1,1],
#         -τ[1,2]/τ[1,1],     τ[2,2] - τ[2,1]*τ[1,2]/τ[1,1],  τ[3,2] - τ[3,1]*τ[1,2]/τ[1,1],
#         -τ[1,3]/τ[1,1],     τ[2,3] - τ[2,1]*τ[1,3]/τ[1,1],  τ[3,3]- τ[3,1]*τ[1,3]/τ[1,1]
#     )
# end

# function avg_param(ε_fg, ε_bg, S, rvol1)
#     τ1 = τ_trans(transpose(S) * ε_fg * S)  # express param1 in S coordinates, and apply τ transform
#     τ2 = τ_trans(transpose(S) * ε_bg * S)  # express param2 in S coordinates, and apply τ transform
#     τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)   # volume-weighted average
#     return SMatrix{3,3}(S * τ⁻¹_trans(τavg) * transpose(S))  # apply τ⁻¹ and transform back to global coordinates
# end


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
f_εₑᵣ   = eval_fn_oop(f_εₑᵣ_sym,prot);
fj_εₑᵣ  = eval_fn_oop(fj_εₑᵣ_sym,prot);
fjh_εₑᵣ  = eval_fn_oop(fjh_εₑᵣ_sym,prot);
f_εₑᵣ!   = eval_fn_ip(f_εₑᵣ_sym,prot);
fj_εₑᵣ!  = eval_fn_ip(fj_εₑᵣ_sym,prot);
fjh_εₑᵣ! = eval_fn_ip(fjh_εₑᵣ_sym,prot);
fout_rot = f_εₑᵣ(rand(19));
fjout_rot = fj_εₑᵣ(rand(19));
fjhout_rot = fjh_εₑᵣ(MVector{19}(rand(19)));
f_εₑᵣ!(similar(fout_rot),rand(19));
fj_εₑᵣ!(similar(fjout_rot),rand(19));
fjh_εₑᵣ!(similar(fjout_rot,9,381),rand(19));


∂ωεₑᵣ(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)   = @views @inbounds reshape( fj_εₑᵣ(vcat(r₁,vec(ε₁),vec(ε₂)))[:,2:end]  * vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂)), (3,3) )

function εₑᵣ_∂ωεₑᵣ(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)
    fj_εₑᵣ_12 = similar(ε₁,9,20) # fj_εₑᵣ(vcat(r₁,vec(ε₁),vec(ε₂)));
    fj_εₑᵣ!(fj_εₑᵣ_12,vcat(r₁,vec(ε₁),vec(ε₂)));
    f_εₑᵣ_12, j_εₑᵣ_12 = @views @inbounds fj_εₑᵣ_12[:,1], fj_εₑᵣ_12[:,2:end];
    εₑᵣ_12 = @views reshape(f_εₑᵣ_12,(3,3))
    v_∂ω = vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂));
    ∂ω_εₑᵣ_12 = @views reshape( j_εₑᵣ_12 * v_∂ω, (3,3) );
    return εₑᵣ_12, ∂ω_εₑᵣ_12
end

function εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂)
    fjh_εₑᵣ_12 = fjh_εₑᵣ(MVector{19}(vcat(r₁,vec(ε₁),vec(ε₂))));
    # fjh_εₑᵣ_12 = fjh_εₑᵣ(MVector{19}(r₁,ε₁...,ε₂...));
    # fjh_εₑᵣ_12 = similar(ε₁,9,381) # fjh_εₑᵣ(vcat(r₁,vec(ε₁),vec(ε₂)));
    # fjh_εₑᵣ!(fjh_εₑᵣ_12,vcat(r₁,vec(ε₁),vec(ε₂)));
    f_εₑᵣ_12, j_εₑᵣ_12, h_εₑᵣ_12 = @views @inbounds fjh_εₑᵣ_12[:,1], fjh_εₑᵣ_12[:,2:20], reshape(fjh_εₑᵣ_12[:,21:381],(9,19,19));
    εₑᵣ_12 = @views reshape(f_εₑᵣ_12,(3,3))
    v_∂ω, v_∂²ω = vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂)), vcat(0.0,vec(∂²ω_ε₁),vec(∂²ω_ε₂));
    ∂ω_εₑᵣ_12 = @views reshape( j_εₑᵣ_12 * v_∂ω, (3,3) );
    ∂ω²_εₑᵣ_12 = @views reshape( [dot(v_∂ω,h_εₑᵣ_12[i,:,:],v_∂ω) for i=1:9] + j_εₑᵣ_12*v_∂²ω , (3,3) );
    return εₑᵣ_12, ∂ω_εₑᵣ_12, ∂ω²_εₑᵣ_12
end

@inline _rotate(S,ε) = transpose(S) * (ε * S)
@inline εₑ_∂ωεₑ(r₁,S,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂) = _rotate.((transpose(S),),εₑᵣ_∂ωεₑᵣ(r₁,_rotate(S,ε₁),_rotate(S,ε₂),_rotate(S,∂ω_ε₁),_rotate(S,∂ω_ε₂)))
@inline εₑ_∂ωεₑ_∂²ωεₑ(r₁,S,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂) = _rotate.((transpose(S),),εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(r₁,_rotate(S,ε₁),_rotate(S,ε₂),_rotate(S,∂ω_ε₁),_rotate(S,∂ω_ε₂),_rotate(S,∂²ω_ε₁),_rotate(S,∂²ω_ε₂)))
@inline εₑ_∂ωεₑ_∂²ωεₑ(r₁,S,idx1,idx2,ε,∂ω_ε,∂²ω_ε) = @inbounds εₑ_∂ωεₑ_∂²ωεₑ(r₁,S,ε[idx1],ε[idx2],∂ω_ε[idx1],∂ω_ε[idx2],∂²ω_ε[idx1],∂²ω_ε[idx2])

@inline herm_vec(A::AbstractMatrix) = @inbounds [A[1,1], A[2,1], A[3,1], A[2,2], A[3,2], A[3,3] ]
@inline function herm_vec(A::SHermitianCompact{3,T,6}) where T<:Number
    return A.lowertriangle
end

function _f_εₑᵣ_herm_sym()
    p = @variables r₁ ε₁ᵣ_11 ε₁ᵣ_12 ε₁ᵣ_13 ε₁ᵣ_22 ε₁ᵣ_23 ε₁ᵣ_33 ε₂ᵣ_11 ε₂ᵣ_12 ε₂ᵣ_13 ε₂ᵣ_22 ε₂ᵣ_23 ε₂ᵣ_33
    ε₁ᵣ = [ ε₁ᵣ_11  ε₁ᵣ_12  ε₁ᵣ_13 ;  ε₁ᵣ_12  ε₁ᵣ_22  ε₁ᵣ_23 ; ε₁ᵣ_13  ε₁ᵣ_23  ε₁ᵣ_33 ];
    ε₂ᵣ = [ ε₂ᵣ_11  ε₂ᵣ_12  ε₂ᵣ_13 ;  ε₂ᵣ_12  ε₂ᵣ_22  ε₂ᵣ_23 ; ε₂ᵣ_13  ε₂ᵣ_23  ε₂ᵣ_33 ];
    τavg = ( r₁ * τ_trans( ε₁ᵣ ) ) + ( (1-r₁) * τ_trans( ε₂ᵣ ) )
    epse_rot = τ⁻¹_trans(τavg)
    # f_εₑᵣ_sym = simplify_fractions.(getindex.((epse_rot,),[1,2,3,5,6,9]))
    f_εₑᵣ_sym = simplify_fractions.(herm_vec(epse_rot))
    return f_εₑᵣ_sym, p
end

f_εₑᵣ_herm_sym, protH = _f_εₑᵣ_herm_sym();
fj_εₑᵣ_herm_sym, fjh_εₑᵣ_herm_sym = _fj_fjh_sym(f_εₑᵣ_herm_sym, protH);
f_εₑᵣ_herm   = eval_fn_oop(f_εₑᵣ_herm_sym,protH);
fj_εₑᵣ_herm  = eval_fn_oop(fj_εₑᵣ_herm_sym,protH);
fjh_εₑᵣ_herm  = eval_fn_oop(fjh_εₑᵣ_herm_sym,protH);
f_εₑᵣ_herm!   = eval_fn_ip(f_εₑᵣ_herm_sym,protH);
fj_εₑᵣ_herm!  = eval_fn_ip(fj_εₑᵣ_herm_sym,protH);
fjh_εₑᵣ_herm! = eval_fn_ip(fjh_εₑᵣ_herm_sym,protH);
fout_rot = f_εₑᵣ_herm(rand(13));
fjout_rot = fj_εₑᵣ_herm(rand(13));
fjhout_rot = fjh_εₑᵣ_herm(MVector{13}(rand(13)));
f_εₑᵣ_herm!(similar(fout_rot),rand(13));
fj_εₑᵣ_herm!(similar(fjout_rot),rand(13));
fjh_εₑᵣ_herm!(similar(fjhout_rot),rand(13));

∂ωεₑᵣ_herm(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)   = @views @inbounds reshape( fj_εₑᵣ_herm(vcat(r₁,vec(ε₁),vec(ε₂)))[:,2:end]  * vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂)), (3,3) )

function εₑᵣ_∂ωεₑᵣ_herm(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)
    fj_εₑᵣ_12 = similar(ε₁,6,14) # fj_εₑᵣ_herm(fj_εₑᵣ_12,vcat(r₁,herm_vec(ε₁),herm_vec(ε₂)));
    fj_εₑᵣ_herm!(fj_εₑᵣ_12,vcat(r₁,herm_vec(ε₁),herm_vec(ε₂)));
    f_εₑᵣ_12, j_εₑᵣ_12 = @views @inbounds fj_εₑᵣ_12[:,1], fj_εₑᵣ_12[:,2:end];
    εₑᵣ_12 = SHermitianCompact{3}( f_εₑᵣ_12 )
    v_∂ω = vcat(0.0,herm_vec(∂ω_ε₁),herm_vec(∂ω_ε₂));
    ∂ω_εₑᵣ_12 = SHermitianCompact{3}( j_εₑᵣ_12 * v_∂ω );
    return εₑᵣ_12, ∂ω_εₑᵣ_12
end

function εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ_herm(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂)
    fjh_εₑᵣ_12 = similar(ε₁,6,183) # fjh_εₑᵣ_herm(fjh_εₑᵣ_12,vcat(r₁,herm_vec(ε₁),herm_vec(ε₂)));
    fjh_εₑᵣ_herm!(fjh_εₑᵣ_12,vcat(r₁,herm_vec(ε₁),herm_vec(ε₂)));
    f_εₑᵣ_12, j_εₑᵣ_12, h_εₑᵣ_12 = @views @inbounds fjh_εₑᵣ_12[:,1], fjh_εₑᵣ_12[:,2:14], reshape(fjh_εₑᵣ_12[:,15:183],(6,13,13));
    εₑᵣ_12 = SHermitianCompact{3}(f_εₑᵣ_12,)
    v_∂ω, v_∂²ω = vcat(0.0,herm_vec(∂ω_ε₁),herm_vec(∂ω_ε₂)), vcat(0.0,herm_vec(∂²ω_ε₁),herm_vec(∂²ω_ε₂));
    ∂ω_εₑᵣ_12 = SHermitianCompact{3}( j_εₑᵣ_12 * v_∂ω,  );
    ∂ω²_εₑᵣ_12 = SHermitianCompact{3}( [dot(v_∂ω,h_εₑᵣ_12[i,:,:],v_∂ω) for i=1:6] + j_εₑᵣ_12*v_∂²ω  );
    return εₑᵣ_12, ∂ω_εₑᵣ_12, ∂ω²_εₑᵣ_12
end


@inline εₑ_∂ωεₑ_herm(r₁,S,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂) = _rotate.((transpose(S),),εₑᵣ_∂ωεₑᵣ_herm(r₁,_rotate(S,ε₁),_rotate(S,ε₂),_rotate(S,∂ω_ε₁),_rotate(S,∂ω_ε₂)))
@inline εₑ_∂ωεₑ_∂²ωεₑ_herm(r₁,S,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂) = _rotate.((transpose(S),),εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ_herm(r₁,_rotate(S,ε₁),_rotate(S,ε₂),_rotate(S,∂ω_ε₁),_rotate(S,∂ω_ε₂),_rotate(S,∂²ω_ε₁),_rotate(S,∂²ω_ε₂)))
@inline εₑ_∂ωεₑ_∂²ωεₑ_herm(r₁,S,idx1,idx2,ε,∂ω_ε,∂²ω_ε) = @inbounds εₑ_∂ωεₑ_∂²ωεₑ_herm(r₁,S,ε[idx1],ε[idx2],∂ω_ε[idx1],∂ω_ε[idx2],∂²ω_ε[idx1],∂²ω_ε[idx2])


###


function _f_epse3D_sym()
    p = @variables r₁ n_1 n_2 n_3 ε₁_11 ε₁_12 ε₁_13 ε₁_21 ε₁_22 ε₁_23 ε₁_31 ε₁_32 ε₁_33 ε₂_11 ε₂_12 ε₂_13 ε₂_21 ε₂_22 ε₂_23 ε₂_31 ε₂_32 ε₂_33 
    ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_21  ε₁_22  ε₁_23 ; ε₁_31  ε₁_32  ε₁_33 ] 
    ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_21  ε₂_22  ε₂_23 ; ε₂_31  ε₂_32  ε₂_33 ]
    n = [ n_1, n_2, n_3 ]
    S = simplify.(normcart(n))
    τ1 = τ_trans( S' * ε₁ * S )
    τ2 = τ_trans( S' * ε₂ * S )
    τavg =  r₁ * τ1  +  (1-r₁) * τ2 
    f_εₑ_sym = vec( S * τ⁻¹_trans(τavg) * S' )
    return f_εₑ_sym, p
end

function _f_epse3D_sym_Arr()
    # p = @variables r₁, n[1:3], ε₁[1:3,1:3], ε₂[1:3,1:3]
    p = @variables r₁, S[1:3,1:3], ε₁[1:3,1:3], ε₂[1:3,1:3]
    τ1 = τ_trans( scalarize(transpose(S) * ε₁ * S) )
    τ2 = τ_trans( scalarize(transpose(S) * ε₂ * S) )
    τavg =  r₁ * τ1  +  (1-r₁) * τ2 
    f_εₑ_sym = vec( scalarize(S * τ⁻¹_trans(τavg) ) * scalarize(transpose(S)) )
    pf = vcat(r₁,vec(scalarize(S)),vec(scalarize(ε₁)),vec(scalarize(ε₂)))
    return f_εₑ_sym, pf
end

### 3D Hermitian case ###

function _f_epse3DH_sym()
    p = @variables r₁ n_1 n_2 n_3 ε₁_11 ε₁_12 ε₁_13 ε₁_22 ε₁_23 ε₁_33 ε₂_11 ε₂_12 ε₂_13 ε₂_22 ε₂_23 ε₂_33
    ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_12  ε₁_22  ε₁_23 ; ε₁_13  ε₁_23  ε₁_33 ];
    ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_12  ε₂_22  ε₂_23 ; ε₂_13  ε₂_23  ε₂_33 ];
    n = [ n_1, n_2, n_3 ]
    S = simplify.(normcart(n))
    τ1 = τ_trans( transpose(S) * ε₁ * S )
    τ2 = τ_trans( transpose(S) * ε₂ * S )
    τavg =  r₁ * τ1  +  (1-r₁) * τ2 
    f_εₑ_sym = vec( S * τ⁻¹_trans(τavg) * transpose(S) )
    return f_εₑ_sym, p
end

### 2D Hermitian case ###

function _f_epse2DH_sym()
    p = @variables r₁ θ ε₁_11 ε₁_12 ε₁_13 ε₁_22 ε₁_23 ε₁_33 ε₂_11 ε₂_12 ε₂_13 ε₂_22 ε₂_23 ε₂_33
    ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_12  ε₁_22  ε₁_23 ; ε₁_13  ε₁_23  ε₁_33 ] 
    ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_12  ε₂_22  ε₂_23 ; ε₂_13  ε₂_23  ε₂_33 ]
    S = simplify.(simplify.(normcart([sin(θ), cos(θ), 0]);threaded=true); rewriter=rules_2D);
    τ1 = τ_trans( transpose(S) * ε₁ * S )
    τ2 = τ_trans( transpose(S) * ε₂ * S )
    τavg =  r₁ * τ1  +  (1-r₁) * τ2 
    f_εₑ_sym = vec( S * τ⁻¹_trans(τavg) * transpose(S) )
    return f_εₑ_sym, p
end



####### Functions to generate executable Julia and compiled C code from Symbolic Functions #######

# include("cse.jl")

# function _buildJ_serial(f_sym,p;fname=nothing,dir=pwd(),kw...)
#     f_ex, f!_ex = build_function(fj_εₑ_sym, p ; expression=Val{true});
# end


