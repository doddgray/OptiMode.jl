using LinearAlgebra
using Symbolics
using Symbolics: Sym, Num, scalarize
using SymbolicUtils: @rule, @acrule, @slots, RuleSet, numerators, denominators, flatten_pows
using SymbolicUtils.Rewriters: Chain, RestartedChain, PassThrough, Prewalk, Postwalk
using BenchmarkTools
# using RuntimeGeneratedFunctions
# RuntimeGeneratedFunctions.init(@__MODULE__)


rules_2D = Prewalk(PassThrough(@acrule sin(~x)^2 + cos(~x)^2 => 1 ))

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

####### Symbolic Jacobians (`j_sym`) and Hessians (`h_sym`) of symbolic array-valued functions `f_sym`,
####### fused into single arrays `fj_sym` or `fjh_sym` for performant function generation 

_fj_sym(f_sym,p) = hcat( vec(f_sym), Symbolics.jacobian(vec(scalarize(f_sym)),p) ) # fj_sym, size = length(f) x ( 1 + length(p) )

function _fj_fjh_sym(f_sym,p)
    j_sym = Symbolics.jacobian(vec(scalarize(f_sym)),p)
    h_sym = mapreduce(x->transpose(vec(Symbolics.hessian(x,p))),vcat,vec(f_sym));
    return hcat(vec(f_sym),j_sym), hcat(vec(f_sym),j_sym,h_sym)  # (fj_sym, fjh_sym), sizes length(f) x ( 1 + length(p) ),  length(f) x ( 1 + length(p) + length(p)^2 ) 
end

####### Functions to generate executable Julia and compiled C code from Symbolic Functions #######

include("cse.jl")

# function _buildJ_serial(f_sym,p;fname=nothing,dir=pwd(),kw...)
#     f_ex, f!_ex = build_function(fj_εₑ_sym, p ; expression=Val{true});
# end

####### Testing

f_epse2DH_sym, p2DH = _f_epse2DH_sym();
fj_epse2DH_sym, fjh_epse2DH_sym = _fj_fjh_sym(f_epse2DH_sym,p2DH);
fj_epse2DH = eval(fn_expr(fj_epse2DH_sym,p2DH));
fjh_epse2DH = eval(fn_expr(fjh_epse2DH_sym,p2DH));
fj_epse2DH(rand(14));
fjh_epse2DH(rand(14));

f_epse3DH_sym, p3DH = _f_epse3DH_sym();
fj_epse3DH_sym, fjh_epse3DH_sym = _fj_fjh_sym(f_epse3DH_sym,p3DH);
fj_epse3DH = eval(fn_expr(fj_epse3DH_sym,p3DH));
fjh_epse3DH = eval(fn_expr(fjh_epse3DH_sym,p3DH));
fj_epse3DH(rand(16));
fjh_epse3DH(rand(16));

f_epse3D_sym, p3D = _f_epse3D_sym();
fj_epse3D_sym, fjh_epse3D_sym = _fj_fjh_sym(f_epse3D_sym,p3D);
fj_epse3D = eval(fn_expr(fj_epse3D_sym,p3D));
fjh_epse3D = eval(fn_expr(fjh_epse3D_sym,p3D));
fj_epse3D(rand(22));
fjh_epse3D(rand(22));