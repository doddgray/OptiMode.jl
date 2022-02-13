using LinearAlgebra
# using OptiMode
using Symbolics
using Symbolics: Sym, Num
using SymbolicUtils: @rule, @acrule, @slots, RuleSet, numerators, denominators, flatten_pows
using SymbolicUtils.Rewriters: Chain, RestartedChain, PassThrough, Prewalk, Postwalk
using BenchmarkTools
# using RuntimeGeneratedFunctions
using IterTools: subsets
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
    ### assume input n is normalized, but if not normalize input vector `n0` to get `n` 
	# n = n0 / (n0[1]^2 + n0[2]^2 + n0[3]^2)^(1//2) # = norm(n0)   
	### Pick `h` to be a vector that is perpendicular to n, some arbitrary choices to make here
	# h = any(iszero.(n)) ? n × normalize(iszero.(n)) :  n × [1, 0 , 0]
    # h_temp = n × [ 0, 0, 1 ] # for now ignore edge case where n = [1,0,0]
    h_temp =  [ 0, 0, 1 ] × n # for now ignore edge case where n = [1,0,0]
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

function simp_εₑ(A, n_1, n_2, n_3)
    return simplify.( substitute(A, Dict([n_3^2 => 1 - n_1^2 - n_2^2 , ])) ; threaded=true ) #, expand=true)
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

# functions to compute  generate fast 

function _f_εₑ_sym()
	p = @variables r₁, n_1, n_2, n_3, ε₁_11, ε₁_12, ε₁_13, ε₁_21, ε₁_22, ε₁_23, ε₁_31, ε₁_32, ε₁_33, ε₂_11, ε₂_12, ε₂_13, ε₂_21, ε₂_22, ε₂_23, ε₂_31, ε₂_32, ε₂_33 
    # @variables r₁, n[1:3], ε₁[1:3,1:3], ε₂[1:3,1:3]
    ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_21  ε₁_22  ε₁_23 ; ε₁_31  ε₁_32  ε₁_33 ] 
    ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_21  ε₂_22  ε₂_23 ; ε₂_31  ε₂_32  ε₂_33 ]
    n = [ n_1, n_2, n_3 ] 
    f_εₑ_sym = avg_param(ε₁, ε₂, simplify.(normcart(n)), r₁); # 3 x 3
    return f_εₑ_sym, p
end

function _f_εₑ_sym_herm3()
	p = @variables r₁, n_1, n_2, n_3, ε₁_11, ε₁_12, ε₁_13, ε₁_22, ε₁_23, ε₁_33, ε₂_11, ε₂_12, ε₂_13, ε₂_22, ε₂_23, ε₂_33
    ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_12  ε₁_22  ε₁_23 ; ε₁_13  ε₁_23  ε₁_33 ] 
    ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_12  ε₂_22  ε₂_23 ; ε₂_13  ε₂_23  ε₂_33 ]
    f_εₑ_sym = avg_param(ε₁, ε₂, simplify.(normcart([ n_1, n_2, n_3 ])), r₁); # 3 x 3
    return f_εₑ_sym, p
end

function _f_εₑ_sym_herm2()
	p = @variables r₁, θ, ε₁_11, ε₁_12, ε₁_13, ε₁_22, ε₁_23, ε₁_33, ε₂_11, ε₂_12, ε₂_13, ε₂_22, ε₂_23, ε₂_33
    ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_12  ε₁_22  ε₁_23 ; ε₁_13  ε₁_23  ε₁_33 ] 
    ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_12  ε₂_22  ε₂_23 ; ε₂_13  ε₂_23  ε₂_33 ]
    S = simplify.(simplify.(normcart([sin(θ), cos(θ), 0]);threaded=true); rewriter=rules_2D);
    f_εₑ_sym = avg_param(ε₁, ε₂, S, r₁); # 3 x 3
    return f_εₑ_sym, p
end

function _f_εₑ()
    f_εₑ_sym, p = _f_εₑ_sym()
    f_εₑ_ex, f_εₑ!_ex = build_function(f_εₑ_sym, p ; expression=Val{true});
    return f_εₑ, f_εₑ! = eval(f_εₑ_ex), eval(f_εₑ!_ex);
end

function _fj_εₑ()
    f_εₑ_sym, p = _f_εₑ_sym()
    j_εₑ_sym = Symbolics.jacobian(vec(f_εₑ_sym),p); # 9 x 22
    fj_εₑ_sym = hcat(vec(f_εₑ_sym),j_εₑ_sym); # 10 x 22
    fj_εₑ_ex, fj_εₑ!_ex = build_function(fj_εₑ_sym, p ; expression=Val{true});
    fj_εₑ, fj_εₑ! = eval(fj_εₑ_ex), eval(fj_εₑ!_ex);
end

# fj_εₑ, fj_εₑ! = _fj_εₑ()

function _fjh_εₑ()
    f_εₑ_sym, p = _f_εₑ_sym()
    j_εₑ_sym = Symbolics.jacobian(vec(f_εₑ_sym),p); # 9 x 22
    h_εₑ_sym = mapreduce(x->transpose(vec(Symbolics.hessian(x,p))),vcat,vec(f_εₑ_sym)); # 9 x 22^2 = 484
    fjh_εₑ_sym = hcat(vec(f_εₑ_sym),j_εₑ_sym,h_εₑ_sym); # 9 x ( 1 + 22*(1+22) ) = 507
    fjh_εₑ_ex, fjh_εₑ!_ex = build_function(fjh_εₑ_sym, p ; expression=Val{true});
    fjh_εₑ, fjh_εₑ! = eval(fjh_εₑ_ex), eval(fjh_εₑ!_ex);
end

∂ω_εₑ(r₁,n,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)   = @views reshape( fj_εₑ(vcat(r₁,n,vec(ε₁),vec(ε₂)))[:,2:23]  * vcat(zeros(4),vec(∂ω_ε₁),vec(∂ω_ε₂)), (3,3) )


function _f_ε_mats_sym(mats,p_syms=(:ω,))
	ε_mats = mapreduce(mm->vec(get_model(mm,:ε,p_syms...)),hcat,mats);
    @variables ω
	Dom = Differential(ω)
    p = [ω, (Num(Sym{Real}(p_sym)) for p_sym in p_syms[2:end])...]
    ∂ωε_mats = expand_derivatives.(Dom.(ε_mats));
    ∂²ωε_mats = expand_derivatives.(Dom.(∂ωε_mats));
    ε_∂ωε_∂²ωε_mats = hcat(ε_mats,∂ωε_mats,∂²ωε_mats);
    return vec(ε_∂ωε_∂²ωε_mats), p
end

ε_views(εv,n_mats;nε=3) = ([ reshape(view(view(εv,(1+(i-1)*9*n_mats):(i*9*n_mats)), (1+9*(mat_idx-1)):(9+9*(mat_idx-1))), (3,3)) for mat_idx=1:n_mats] for i=1:nε) 

"""
Example:
    ```
    f_ε_mats, f_ε_mats! = _f_ε_mats(mats3,(:ω,:T))
    @show ω0,T0 = ( 0.2*rand(Float64)+0.8, 20.0*rand(Float64)+20.0 ) # frequency and temperature, in μm⁻¹ and C respectively
    p0 = [ω0,T0];
    f0 = f_ε_mats(p0);
    ε, ∂ωε, ∂²ωε = ε_views(f,n_mats);
    ```
"""
function _f_ε_mats(mats,p_syms=(:ω,))
    f_ε_mats_sym, p = _f_ε_mats_sym(mats,p_syms)
    f_ε_∂ωε_∂²ωε_ex, f_ε_∂ωε_∂²ωε!_ex = build_function(f_ε_mats_sym, p; expression=Val{true})
    return eval(f_ε_∂ωε_∂²ωε_ex), eval(f_ε_∂ωε_∂²ωε!_ex)
end

"""
Example:
    ```
    fj_ε_mats, fj_ε_mats! = _fj_ε_mats(mats3,(:ω,:T))
    @show ω0,T0 = ( 0.2*rand(Float64)+0.8, 20.0*rand(Float64)+20.0 ) # frequency and temperature, in μm⁻¹ and C respectively
    p0 = [ω0,T0];
    fj0 = fj_ε_mats(p0);
    (ε, ∂ωε, ∂²ωε), (∂ω_ε, ∂ω_∂ωε, ∂ω_∂²ωε), (∂T_ε, ∂T_∂ωε, ∂T_∂²ωε) = ε_views.(eachcol(fj0),(n_mats,));
    ```
"""
function _fj_ε_mats(mats,p_syms=(:ω,))
    f_ε_mats_sym, p = _f_ε_mats_sym(mats,p_syms) # 27*length(mats) x 1 
    j_ε_mats_sym = Symbolics.jacobian(vec(f_ε_mats_sym),p); # 27*length(mats) x length(p)
    fj_ε_mats_sym = hcat(vec(f_ε_mats_sym),j_ε_mats_sym); # 27*length(mats) x (1 + length(p))
    fj_ε_mats_ex, fj_ε_mats!_ex = build_function(fj_ε_mats_sym, p ; expression=Val{true});
    return eval(fj_ε_mats_ex), eval(fj_ε_mats!_ex);
end

"""
Example:
    ```
    fjh_ε_mats, fjh_ε_mats! = _fjh_ε_mats(mats3,(:ω,:T))
    @show ω0,T0 = ( 0.2*rand(Float64)+0.8, 20.0*rand(Float64)+20.0 ) # frequency and temperature, in μm⁻¹ and C respectively
    p0 = [ω0,T0];
    fjh0 = fjh_ε_mats(p0);
    (ε, ∂ωε, ∂²ωε), (∂ω_ε, ∂ω_∂ωε, ∂ω_∂²ωε), (∂T_ε, ∂T_∂ωε, ∂T_∂²ωε) = ε_views.(eachcol(fjh0),(n_mats,));
    ```
"""
function _fjh_ε_mats(mats,p_syms=(:ω,))
    f_ε_mats_sym, p = _f_ε_mats_sym(mats,p_syms) # 27*length(mats) x 1
    j_ε_mats_sym = Symbolics.jacobian(vec(f_ε_mats_sym),p); # 27*length(mats) x length(p)
    h_ε_mats_sym = mapreduce(x->transpose(vec(Symbolics.hessian(x,p))),vcat,vec(f_ε_mats_sym)); # 27*length(mats) x length(p)^2
    fjh_ε_mats_sym = hcat(vec(f_ε_mats_sym),j_ε_mats_sym,h_ε_mats_sym); # 27*length(mats) x ( 1 + length(p)*(1+length(p)) )
    fjh_ε_mats_ex, fjh_ε_mats!_ex = build_function(fjh_ε_mats_sym, p ; expression=Val{true});
    return eval(fjh_ε_mats_ex), eval(fjh_ε_mats!_ex);
end


rand_p_ω() = [ 0.2*rand(Float64)+0.8, ]
rand_p_ω_T() = vcat(rand_p_ω(),[20.0*rand(Float64)+20.0,])
rand_p_r_n() = [rand(), normalize(rand(3))...]
rand_p_εₑ() = [rand(), normalize(rand(3))..., (rand()+1.0)^2, 0.0, 0.0, 0.0, (rand()+1.0)^2, 0.0, 0.0, 0.0, (rand()+1.0)^2, (rand()+1.0)^2, 0.0, 0.0, 0.0, (rand()+1.0)^2, 0.0, 0.0, 0.0, (rand()+1.0)^2 ]


mats1 = [MgO_LiNbO₃,Si₃N₄];
mats2 = [MgO_LiNbO₃,Si₃N₄,SiO₂];
mats3 = [MgO_LiNbO₃,Si₃N₄,SiO₂,LiB₃O₅];

##

fj_εₑ, fj_εₑ! = _fj_εₑ();
fjh_εₑ, fjh_εₑ! = _fjh_εₑ();
##



f_εₑ_sym, p = _f_εₑ_sym();
n_1,n_2,n_3 = p[2:4];
f_εₑ_sym_sub = substitute(f_εₑ_sym, Dict([n_3^2 => 1 - n_1^2 - n_2^2 , ]));
f_εₑ_sym_sub_simp = simplify.(f_εₑ_sym_sub; threaded=true);
f_εₑ_sym_sub_simpex = simplify.(f_εₑ_sym_sub; threaded=true, expand=true);
f_εₑ_sym_sub_simpex2 = simplify.(f_εₑ_sym_sub; expand=true);
f_εₑ_sym_simpex2 = simplify.(f_εₑ_sym; expand=true);
##
fj_εₑ(rand_p_εₑ())
##

mats = [LN,SN,silica,LBO];
n_mats = length(mats);
fj_ε_mats, fj_ε_mats! = _fj_ε_mats(mats,(:ω,:T))
@show ω0,T0 = ( 0.2*rand(Float64)+0.8, 20.0*rand(Float64)+20.0 ) # frequency and temperature, in μm⁻¹ and C respectively
p0 = [ω0,T0];
fj = fj_ε_mats(p0);
(ε, ∂ωε, ∂²ωε), (∂ω_ε, ∂ω_∂ωε, ∂ω_∂²ωε), (∂T_ε, ∂T_∂ωε, ∂T_∂²ωε) = ε_views.(eachcol(fj),(n_mats,));

@show r1, n1, n2, n3 = rand_p_r_n()
fj_εₑ_12 = fj_εₑ(vcat([r1,n1,n2,n3],vec(ε[1]),vec(ε[2])))
@views εₑ_12 = reshape(fj_εₑ_12[:,1],(3,3))
@views j_εₑ_12 = fj_εₑ_12[:,2:23]
∂ω_εₑ_12 = ∂ω_εₑ(r1,[n1,n2,n3],ε[1],ε[2],∂ω_ε[1],∂ω_ε[2])  # = @views fj_εₑ(vcat(r₁,n,vec(ε₁),vec(ε₂)))[:,2:23] * vcat(zeros(4),vec(∂ω_ε₁),vec(∂ω_ε₂))

fjh_ε_mats, fjh_ε_mats! = _fjh_ε_mats(mats,(:ω,:T));
fjh = fjh_ε_mats(p0);
(ε, ∂ωε, ∂²ωε), (∂ω_ε, ∂ω_∂ωε, ∂ω_∂²ωε), (∂T_ε, ∂T_∂ωε, ∂T_∂²ωε),  (∂ω∂ω_ε, ∂ω∂ω_∂ωε, ∂ω∂ω_∂²ωε), (∂ω∂T_ε, ∂ω∂T_∂ωε, ∂ω∂T_∂²ωε), (∂T∂ω_ε, ∂T_∂ωε, ∂T∂ω_∂²ωε), (∂T∂T_ε, ∂T∂T_∂ωε, ∂T∂T_∂²ωε) = ε_views.(eachcol(fjh),(n_mats,));
@assert ∂ω∂ω_ε == ∂ω_∂ωε == ∂²ωε

fjh_εₑ_12 = fjh_εₑ(vcat([r1,n1,n2,n3],vec(ε[1]),vec(ε[2])));
f_εₑ_12, j_εₑ_12, h_εₑ_12 = @views fjh_εₑ_12[:,1], fjh_εₑ_12[:,2:23], reshape(fjh_εₑ_12[:,24:507],(9,22,22));

function ∂²ω_εₑ(r₁,n,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂)
    fjh_εₑ_12 = fjh_εₑ(vcat([r1,n1,n2,n3],vec(ε₁),vec(ε₂)));
    f_εₑ_12, j_εₑ_12, h_εₑ_12 = @views @inbounds fjh_εₑ_12[:,1], fjh_εₑ_12[:,2:23], reshape(fjh_εₑ_12[:,24:507],(9,22,22));
    ∂ω_εₑ_12 = @views reshape( j_εₑ_12 * vcat(zeros(4),vec(∂ω_ε₁),vec(∂ω_ε₂)), (3,3) );
    ∂ω²_εₑ_12 = @views reshape( fjh_εₑ(vcat(r₁,n,vec(ε₁),vec(ε₂)))[:,2:23] * vcat(zeros(4),vec(∂²ω_ε₁),vec(∂²ω_ε₂)), (3,3) )
end

∂²ω_εₑ_12 = ∂ω_εₑ(r1,[n1,n2,n3],ε[1],ε[2],∂ω_ε[1],∂ω_ε[2])
##


function smooth2(ε_mats,shapes,grid::Grid{ND}) where {ND}
	xyz, xyzc = x⃗(grid), x⃗c(grid)			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
	vxlmin,vxlmax = vxl_minmax(xyzc)
    ε_m, ∂ωε_m, ∂²ωε_m = ε_views(view(ε_mats,:,1),n_mats);
    sinds = proc_sinds(corner_sinds(shapes,xyzc))
    ε = zeros(Float64,9,size(grid)...)
    smoothed_vals_nested = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
        Tuple(smooth(sinds,shapes,geom.material_inds,mat_vals,xx,vn,vp))
    end


	arr_flatB = Zygote.Buffer(om_p,9,size(grid)...,n_fns)
	arr_flat = Zygote.forwarddiff(om_p) do om_p
		geom = f_geom(om_p[2:n_p+1])
		shapes = getfield(geom,:shapes)
		om_inv = inv(first(om_p))
		mat_vals = mapreduce(ss->[ map(f->(mat=SMatrix{3,3}(f(om_inv)); 0.5*(mat+mat')),getfield(geom,ss))... ], hcat, fnames)
		
		smoothed_vals_nested = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
			Tuple(smooth(sinds,shapes,geom.material_inds,mat_vals,xx,vn,vp))
		end
		smoothed_vals = hcat( [map(x->getindex(x,i),smoothed_vals_nested) for i=1:n_fns]...)
		smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
		return smoothed_vals_rr  # new spatially smoothed ε tensor array
	end
	copyto!(arr_flatB,copy(arr_flat))
	arr_flat_r = copy(arr_flatB)
	Nx = size(grid,1)
	Ny = size(grid,2)
	fn_arrs = [hybridize(view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,axes(grid)...,n),grid) for n=1:n_fns]
	return fn_arrs
end


## 3D non-Hermitian case
p = @variables r₁ n_1 n_2 n_3 ε₁_11 ε₁_12 ε₁_13 ε₁_21 ε₁_22 ε₁_23 ε₁_31 ε₁_32 ε₁_33 ε₂_11 ε₂_12 ε₂_13 ε₂_21 ε₂_22 ε₂_23 ε₂_31 ε₂_32 ε₂_33 
ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_21  ε₁_22  ε₁_23 ; ε₁_31  ε₁_32  ε₁_33 ] 
ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_21  ε₂_22  ε₂_23 ; ε₂_31  ε₂_32  ε₂_33 ]
n = [ n_1, n_2, n_3 ] 
# p = @variables r₁, n[1:3], ε₁[1:3,1:3], ε₂[1:3,1:3]
S = simplify.(normcart(n))
τ1 = τ_trans( transpose(S) * ε₁ * S )
# τ2 = substitute(τ1, Dict([ε₁[1,1]=>ε₂[1,1], ε₁[1,2]=>ε₂[1,2], ε₁[1,3]=>ε₂[1,3], ε₁[2,1]=>ε₂[2,1], ε₁[2,2]=>ε₂[2,2], ε₁[2,3]=>ε₂[2,3], ε₁[3,1]=>ε₂[3,1], ε₁[3,2]=>ε₂[3,2], ε₁[3,3]=>ε₂[3,3], ]))
τ2 = τ_trans( transpose(S) *ε₂ * S )  # substitute(τ1, Dict([ε₁=>ε₂,]))
τavg = ( r₁ * τ1 ) + ( (1-r₁) * τ2 )
f_εₑ_sym = S * τ⁻¹_trans(τavg) * transpose(S) 
j_εₑ_sym = Symbolics.jacobian(vec(f_εₑ_sym),p); # 9 x 22
h_εₑ_sym = mapreduce(x->transpose(vec(Symbolics.hessian(x,p))),vcat,vec(f_εₑ_sym));
fj_εₑ_sym = hcat(vec(f_εₑ_sym),j_εₑ_sym); # 9 x ( 1 + 22 ) = 23
fjh_εₑ_sym = hcat(vec(f_εₑ_sym),j_εₑ_sym,h_εₑ_sym); # 9 x ( 1 + 22*(1+22) ) = 507

fj_εₑ!_c = _build_function(CTarget, fj_εₑ_sym, p;
    columnmajor = true,
    conv        = toexpr,
    expression  = Val{false},
    fname       = :diffeqf,
    lhsname     = :du,
    libpath     = "fj_epse.so",
    compiler    = :gcc)
write(joinpath(pwd(),"fj_epse.c"),string(fj_εₑ!_c));

fjh_εₑ!_c = _build_function(CTarget, fjh_εₑ_sym, p;
    columnmajor = true,
    conv        = toexpr,
    expression  = Val{false},
    fname       = :diffeqf,
    lhsname     = :du,
    libpath     = "fjh_epse.so",
    compiler    = :gcc)
write(joinpath(pwd(),"fjh_epse.c"),string(fjh_εₑ!_c));


fj_εₑ_ex, fj_εₑ!_ex = build_function(fj_εₑ_sym, p ; expression=Val{true});
write(joinpath(pwd(),"fj_εₑ!.jl"),string(fj_εₑ!_ex))
write(joinpath(pwd(),"fj_εₑ.jl"), string(fj_εₑ_ex))

# fj_εₑ = eval(fj_εₑ_ex);
fj_εₑ! = eval(fj_εₑ!_ex);

fjh_εₑ_ex, fjh_εₑ!_ex = build_function(fjh_εₑ_sym, p ; expression=Val{true});
write(joinpath(pwd(),"fjh_εₑ!.jl"),string(fjh_εₑ!_ex))
write(joinpath(pwd(),"fjh_εₑ.jl"), string(fjh_εₑ_ex))

# fjh_εₑ = eval(fjh_εₑ_ex);
fjh_εₑ! = eval(fjh_εₑ!_ex);

p0 = rand(length(p));

fj0 = zeros(Float64,(9,23));
fj_εₑ!(fj0,p0);             #   this works for all cases

fjh0 = zeros(Float64,(9,507));
fjh_εₑ!(fjh0,p0);           #   this always returns without errors, 

## 3D Hermitian case

p = @variables r₁ n_1 n_2 n_3 ε₁_11 ε₁_12 ε₁_13 ε₁_22 ε₁_23 ε₁_33 ε₂_11 ε₂_12 ε₂_13 ε₂_22 ε₂_23 ε₂_33
ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_12  ε₁_22  ε₁_23 ; ε₁_13  ε₁_23  ε₁_33 ];
ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_12  ε₂_22  ε₂_23 ; ε₂_13  ε₂_23  ε₂_33 ];
S = simplify.(normcart([n_1, n_2, n_3]));
τ1 = τ_trans( transpose(S) * ε₁ * S );
τ2 = substitute(τ1, Dict([ε₁[1,1]=>ε₂[1,1], ε₁[1,2]=>ε₂[1,2], ε₁[1,3]=>ε₂[1,3], ε₁[2,1]=>ε₂[2,1], ε₁[2,2]=>ε₂[2,2], ε₁[2,3]=>ε₂[2,3], ε₁[3,1]=>ε₂[3,1], ε₁[3,2]=>ε₂[3,2], ε₁[3,3]=>ε₂[3,3], ]))
τavg = ( r₁ * τ1 ) + ( (1-r₁) * τ2 );
f_εₑ_sym = S * τ⁻¹_trans(τavg) * transpose(S);
j_εₑ_sym = Symbolics.jacobian(vec(f_εₑ_sym),p); # 6 x 16
h_εₑ_sym = mapreduce(x->transpose(vec(Symbolics.hessian(x,p))),vcat,vec(f_εₑ_sym));
fj_εₑ_sym = hcat(vec(f_εₑ_sym),j_εₑ_sym); # 6 x ( 1 + 16 ) = 17
fjh_εₑ_sym = hcat(vec(f_εₑ_sym),j_εₑ_sym,h_εₑ_sym); # 6 x ( 1 + 16*(1+16) ) = 273




fj_εₑ_ex, fj_εₑ!_ex = build_function(fj_εₑ_sym, p ; expression=Val{true});
write(joinpath(pwd(),"fj_εₑ!_Herm3D.jl"),string(fj_εₑ!_ex));
write(joinpath(pwd(),"fj_εₑ_Herm3D.jl"), string(fj_εₑ_ex));

# fj_εₑ = eval(fj_εₑ_ex);
fj_εₑ! = eval(fj_εₑ!_ex);

fjh_εₑ_ex, fjh_εₑ!_ex = build_function(fjh_εₑ_sym, p ; expression=Val{true});
write(joinpath(pwd(),"fjh_εₑ!_Herm3D.jl"),string(fjh_εₑ!_ex));
write(joinpath(pwd(),"fjh_εₑ_Herm3D.jl"), string(fjh_εₑ_ex));

# fjh_εₑ = eval(fjh_εₑ_ex);
fjh_εₑ! = eval(fjh_εₑ!_ex);

p0 = rand(length(p));

fj0 = zeros(Float64,(6,17));
fj_εₑ!(fj0,p0);             #   this works for all cases

fjh0 = zeros(Float64,(6,273));
fjh_εₑ!(fjh0,p0);           #   this always returns without errors, 

## 

## 2D Hermitian case
p = @variables r₁ θ ε₁_11 ε₁_12 ε₁_13 ε₁_22 ε₁_23 ε₁_33 ε₂_11 ε₂_12 ε₂_13 ε₂_22 ε₂_23 ε₂_33
ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_12  ε₁_22  ε₁_23 ; ε₁_13  ε₁_23  ε₁_33 ] 
ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_12  ε₂_22  ε₂_23 ; ε₂_13  ε₂_23  ε₂_33 ]
S = simplify.(simplify.(normcart([sin(θ), cos(θ), 0]);threaded=true); rewriter=rules_2D);
τ1 = τ_trans( transpose(S) * ε₁ * S )
τ2 = substitute(τ1, Dict([ε₁[1,1]=>ε₂[1,1], ε₁[1,2]=>ε₂[1,2], ε₁[1,3]=>ε₂[1,3], ε₁[2,1]=>ε₂[2,1], ε₁[2,2]=>ε₂[2,2], ε₁[2,3]=>ε₂[2,3], ε₁[3,1]=>ε₂[3,1], ε₁[3,2]=>ε₂[3,2], ε₁[3,3]=>ε₂[3,3], ]))
τavg = ( r₁ * τ1 ) + ( (1-r₁) * τ2 )
f_εₑ_sym = S * τ⁻¹_trans(τavg) * transpose(S)
p = [r₁, θ, ε₁_11, ε₁_12, ε₁_13, ε₁_22, ε₁_23, ε₁_33, ε₂_11, ε₂_12, ε₂_13, ε₂_22, ε₂_23, ε₂_33] 
j_εₑ_sym = Symbolics.jacobian(vec(f_εₑ_sym),p); # 6 x 22
h_εₑ_sym = mapreduce(x->transpose(vec(Symbolics.hessian(x,p))),vcat,vec(f_εₑ_sym));
fj_εₑ_sym = hcat(vec(f_εₑ_sym),j_εₑ_sym); # 6 x ( 1 + 14 ) = 15
fjh_εₑ_sym = hcat(vec(f_εₑ_sym),j_εₑ_sym,h_εₑ_sym); # 6 x ( 1 + 14*(1+14) ) = 211

fj_εₑ_ex, fj_εₑ!_ex = build_function(fj_εₑ_sym, p ; expression=Val{true});
write(joinpath(pwd(),"fj_εₑ!_Herm2D.jl"),string(fj_εₑ!_ex))
write(joinpath(pwd(),"fj_εₑ_Herm2D.jl"), string(fj_εₑ_ex))

# fj_εₑ = eval(fj_εₑ_ex);
fj_εₑ! = eval(fj_εₑ!_ex);

fjh_εₑ_ex, fjh_εₑ!_ex = build_function(fjh_εₑ_sym, p ; expression=Val{true});
write(joinpath(pwd(),"fjh_εₑ!_Herm2D.jl"),string(fjh_εₑ!_ex))
write(joinpath(pwd(),"fjh_εₑ_Herm2D.jl"), string(fjh_εₑ_ex))

# fjh_εₑ = eval(fjh_εₑ_ex);
fjh_εₑ! = eval(fjh_εₑ!_ex);

p0 = rand(length(p));

fj0 = zeros(Float64,(6,15));
fj_εₑ!(fj0,p0);             #   this works for all cases

fjh0 = zeros(Float64,(6,211));
fjh_εₑ!(fjh0,p0);           #   this always returns without errors, 


##

##
println("Build and compile functions for ε & ∂ωε vs. (ω, T, n̂, r):")
println("building f_ε_∂ωε_mats1:")
@time f_ε_∂ωε_mats1, f_ε_∂ωε_mats1! = _ε_fs(mats1)
println("f_ε_∂ωε_mats1 warmup run:")
@time res = res = f_ε_∂ωε_mats1(rand_p_ω());
res_size_f_ε_∂ωε_mats1 = size(res);
println("result size: $res_size_f_ε_∂ωε_mats1")
println("f_ε_∂ωε_mats1! warmup run:")
@time f_ε_∂ωε_mats1!(res,rand_p_ω());

println("building f_ε_∂ωε_mats1_T:")
@time f_ε_∂ωε_mats1_T, f_ε_∂ωε_mats1_T! = _ε_fs(mats1,(:T,))
println("f_ε_∂ωε_mats1_T warmup run:")
@time res = f_ε_∂ωε_mats1_T(rand_p_ω_T());
res_size_f_ε_∂ωε_mats1_T = size(res);
println("result size: $res_size_f_ε_∂ωε_mats1_T")
println("f_ε_∂ωε_mats1_T! warmup run:")
@time f_ε_∂ωε_mats1_T!(res,rand_p_ω_T());

println("building f_ε_∂ωε_mats2:")
@time f_ε_∂ωε_mats2, f_ε_∂ωε_mats2! = _ε_fs(mats2)
println("f_ε_∂ωε_mats2 warmup run:")
@time res = f_ε_∂ωε_mats2(rand_p_ω());
res_size_f_ε_∂ωε_mats2 = size(res);
println("result size: $res_size_f_ε_∂ωε_mats2")
println("f_ε_∂ωε_mats2! warmup run:")
@time f_ε_∂ωε_mats2!(res,rand_p_ω());

println("building f_ε_∂ωε_mats2_T:")
@time f_ε_∂ωε_mats2_T, f_ε_∂ωε_mats2_T! = _ε_fs(mats2,(:T,))
println("f_ε_∂ωε_mats2_T warmup run:")
@time res = f_ε_∂ωε_mats2_T(rand_p_ω_T());
res_size_f_ε_∂ωε_mats2_T = size(res);
println("result size: $res_size_f_ε_∂ωε_mats2_T")
println("f_ε_∂ωε_mats2_T! warmup run:")
@time f_ε_∂ωε_mats2_T!(res,rand_p_ω_T());

println("building f_ε_∂ωε_mats3:")
@time f_ε_∂ωε_mats3, f_ε_∂ωε_mats3! = _ε_fs(mats3)
println("f_ε_∂ωε_mats3 warmup run:")
@time res = f_ε_∂ωε_mats3(rand_p_ω());
res_size_f_ε_∂ωε_mats3 = size(res);
println("result size: $res_size_f_ε_∂ωε_mats3")
println("f_ε_∂ωε_mats3! warmup run:")
@time f_ε_∂ωε_mats3!(res,rand_p_ω());

println("building f_ε_∂ωε_mats3_T:")
@time f_ε_∂ωε_mats3_T, f_ε_∂ωε_mats3_T! = _ε_fs(mats3,(:T,))
println("f_ε_∂ωε_mats3_T warmup run:")
@time res = f_ε_∂ωε_mats3_T(rand_p_ω_T());
res_size_f_ε_∂ωε_mats3_T = size(res);
println("result size: $res_size_f_ε_∂ωε_mats3_T")
println("f_ε_∂ωε_mats3_T! warmup run:")
@time f_ε_∂ωε_mats3_T!(res,rand_p_ω_T());


println("Build and compile functions for ε, ∂ωε & ∂²ωε vs. (ω, T, n̂, r):")
println("building f_ε_∂ωε_∂²ωε_mats1:")
@time f_ε_∂ωε_∂²ωε_mats1, f_ε_∂ωε_∂²ωε_mats1! = _ε_fs2(mats1)
println("f_ε_∂ωε_∂²ωε_mats1 warmup run:")
@time res = f_ε_∂ωε_∂²ωε_mats1(rand_p_ω());
res_size_f_ε_∂ωε_∂²ωε_mats1 = size(res);
println("result size: $res_size_f_ε_∂ωε_∂²ωε_mats1")
println("f_ε_∂ωε_∂²ωε_mats1! warmup run:")
@time f_ε_∂ωε_∂²ωε_mats1!(res,rand_p_ω());

println("building f_ε_∂ωε_∂²ωε_mats1_T:")
@time f_ε_∂ωε_∂²ωε_mats1_T, f_ε_∂ωε_∂²ωε_mats1_T! = _ε_fs2(mats1,(:T,))
println("f_ε_∂ωε_∂²ωε_mats1_T warmup run:")
@time res = f_ε_∂ωε_∂²ωε_mats1_T(rand_p_ω_T());
res_size_f_ε_∂ωε_∂²ωε_mats1_T = size(res);
println("result size: $res_size_f_ε_∂ωε_∂²ωε_mats1_T")
println("f_ε_∂ωε_∂²ωε_mats1_T! warmup run:")
@time f_ε_∂ωε_∂²ωε_mats1_T!(res,rand_p_ω_T());

println("building f_ε_∂ωε_∂²ωε_mats2:")
@time f_ε_∂ωε_∂²ωε_mats2, f_ε_∂ωε_∂²ωε_mats2! = _ε_fs2(mats2)
println("f_ε_∂ωε_∂²ωε_mats2 warmup run:")
@time res = f_ε_∂ωε_∂²ωε_mats2(rand_p_ω());
res_size_f_ε_∂ωε_∂²ωε_mats2 = size(res);
println("result size: $res_size_f_ε_∂ωε_∂²ωε_mats2")
println("f_ε_∂ωε_∂²ωε_mats2! warmup run:")
@time f_ε_∂ωε_∂²ωε_mats2!(res,rand_p_ω());

println("building f_ε_∂ωε_∂²ωε_mats2_T:")
@time f_ε_∂ωε_∂²ωε_mats2_T, f_ε_∂ωε_∂²ωε_mats2_T! = _ε_fs2(mats2,(:T,))
println("f_ε_∂ωε_∂²ωε_mats2_T warmup run:")
@time res = f_ε_∂ωε_∂²ωε_mats2_T(rand_p_ω_T());
res_size_f_ε_∂ωε_∂²ωε_mats2_T = size(res);
println("result size: $res_size_f_ε_∂ωε_∂²ωε_mats2_T")
println("f_ε_∂ωε_∂²ωε_mats2_T! warmup run:")
@time f_ε_∂ωε_∂²ωε_mats2_T!(res,rand_p_ω_T());

println("building f_ε_∂ωε_∂²ωε_mats3:")
@time f_ε_∂ωε_∂²ωε_mats3, f_ε_∂ωε_∂²ωε_mats3! = _ε_fs2(mats3)
println("f_ε_∂ωε_∂²ωε_mats3 warmup run:")
@time res = f_ε_∂ωε_∂²ωε_mats3(rand_p_ω());
res_size_f_ε_∂ωε_∂²ωε_mats3 = size(res);
println("result size: $res_size_f_ε_∂ωε_∂²ωε_mats3")
println("f_ε_∂ωε_∂²ωε_mats3! warmup run:")
@time f_ε_∂ωε_∂²ωε_mats3!(res,rand_p_ω());

println("building f_ε_∂ωε_∂²ωε_mats3_T:")
@time f_ε_∂ωε_∂²ωε_mats3_T, f_ε_∂ωε_∂²ωε_mats3_T! = _ε_fs2(mats3,(:T,))
println("f_ε_∂ωε_∂²ωε_mats3_T warmup run:")
@time res = f_ε_∂ωε_∂²ωε_mats3_T(rand_p_ω_T());
res_size_f_ε_∂ωε_∂²ωε_mats3_T = size(res);
println("result size: $res_size_f_ε_∂ωε_∂²ωε_mats3_T")
println("f_ε_∂ωε_∂²ωε_mats3_T! warmup run:")
@time f_ε_∂ωε_∂²ωε_mats3_T!(res,rand_p_ω_T());

function test_ε_fs()
    p_ω = rand_p_ω()
    p_ω_T = rand_p_ω_T()

    println("Compute ε & ∂ωε vs. (ω, T, n̂, r):")
    println("f_ε_∂ωε_mats1")
    @btime f_ε_∂ωε_mats1($p_ω);
    println("f_ε_∂ωε_mats1!")
    res_f_ε_∂ωε_mats1 = rand(Float64,res_size_f_ε_∂ωε_mats1)
    @btime f_ε_∂ωε_mats1!($res_f_ε_∂ωε_mats1,$p_ω);
    println("f_ε_∂ωε_mats1_T")
    @btime f_ε_∂ωε_mats1_T($p_ω_T);
    println("f_ε_∂ωε_mats1_T!")
    res_f_ε_∂ωε_mats1_T = rand(Float64,res_size_f_ε_∂ωε_mats1_T)
    @btime f_ε_∂ωε_mats1_T!($res_f_ε_∂ωε_mats1_T,$p_ω_T);
    println("f_ε_∂ωε_mats2")
    @btime f_ε_∂ωε_mats2($p_ω);
    println("f_ε_∂ωε_mats2!")
    res_f_ε_∂ωε_mats2 = rand(Float64,res_size_f_ε_∂ωε_mats2)
    @btime f_ε_∂ωε_mats2!($res_f_ε_∂ωε_mats2,$p_ω);
    println("f_ε_∂ωε_mats2_T")
    @btime f_ε_∂ωε_mats2_T($p_ω_T);
    println("f_ε_∂ωε_mats2_T!")
    res_f_ε_∂ωε_mats2_T = rand(Float64,res_size_f_ε_∂ωε_mats2_T)
    @btime f_ε_∂ωε_mats2_T!($res_f_ε_∂ωε_mats2_T,$p_ω_T);
    println("f_ε_∂ωε_mats3")
    @btime f_ε_∂ωε_mats3($p_ω);
    println("f_ε_∂ωε_mats3!")
    res_f_ε_∂ωε_mats3 = rand(Float64,res_size_f_ε_∂ωε_mats3)
    @btime f_ε_∂ωε_mats3!($res_f_ε_∂ωε_mats3,$p_ω);
    println("f_ε_∂ωε_mats3_T")
    @btime f_ε_∂ωε_mats3_T($p_ω_T);
    println("f_ε_∂ωε_mats3_T!")
    res_f_ε_∂ωε_mats3_T = rand(Float64,res_size_f_ε_∂ωε_mats3_T)
    @btime f_ε_∂ωε_mats3_T!($res_f_ε_∂ωε_mats3_T,$p_ω_T);

    println("Compute ε, ∂ωε & ∂²ωε vs. (ω, T, n̂, r):")
    println("f_ε_∂ωε_∂²ωε_mats1")
    @btime f_ε_∂ωε_∂²ωε_mats1($p_ω);
    println("f_ε_∂ωε_∂²ωε_mats1!")
    res_f_ε_∂ωε_∂²ωε_mats1 = rand(Float64,res_size_f_ε_∂ωε_∂²ωε_mats1)
    @btime f_ε_∂ωε_∂²ωε_mats1!($res_f_ε_∂ωε_∂²ωε_mats1,$p_ω);
    println("f_ε_∂ωε_∂²ωε_mats1_T")
    @btime f_ε_∂ωε_∂²ωε_mats1_T($p_ω_T);
    println("f_ε_∂ωε_∂²ωε_mats1_T!")
    res_f_ε_∂ωε_∂²ωε_mats1_T = rand(Float64,res_size_f_ε_∂ωε_∂²ωε_mats1_T)
    @btime f_ε_∂ωε_∂²ωε_mats1_T!($res_f_ε_∂ωε_∂²ωε_mats1_T,$p_ω_T);
    println("f_ε_∂ωε_∂²ωε_mats2")
    @btime f_ε_∂ωε_∂²ωε_mats2($p_ω);
    println("f_ε_∂ωε_∂²ωε_mats2!")
    res_f_ε_∂ωε_∂²ωε_mats2 = rand(Float64,res_size_f_ε_∂ωε_∂²ωε_mats2)
    @btime f_ε_∂ωε_∂²ωε_mats2!($res_f_ε_∂ωε_∂²ωε_mats2,$p_ω);
    println("f_ε_∂ωε_∂²ωε_mats2_T")
    @btime f_ε_∂ωε_∂²ωε_mats2_T($p_ω_T);
    println("f_ε_∂ωε_∂²ωε_mats2_T!")
    res_f_ε_∂ωε_∂²ωε_mats2_T = rand(Float64,res_size_f_ε_∂ωε_∂²ωε_mats2_T)
    @btime f_ε_∂ωε_∂²ωε_mats2_T!($res_f_ε_∂ωε_∂²ωε_mats2_T,$p_ω_T);
    println("f_ε_∂ωε_∂²ωε_mats3")
    @btime f_ε_∂ωε_∂²ωε_mats3($p_ω);
    println("f_ε_∂ωε_∂²ωε_mats3!")
    res_f_ε_∂ωε_∂²ωε_mats3 = rand(Float64,res_size_f_ε_∂ωε_∂²ωε_mats3)
    @btime f_ε_∂ωε_∂²ωε_mats3!($res_f_ε_∂ωε_∂²ωε_mats3,$p_ω);
    println("f_ε_∂ωε_∂²ωε_mats3_T")
    @btime f_ε_∂ωε_∂²ωε_mats3_T($p_ω_T);
    println("f_ε_∂ωε_∂²ωε_mats3_T!")
    res_f_ε_∂ωε_∂²ωε_mats3_T = rand(Float64,res_size_f_ε_∂ωε_∂²ωε_mats3_T)
    @btime f_ε_∂ωε_∂²ωε_mats3_T!($res_f_ε_∂ωε_∂²ωε_mats3_T,$p_ω_T);

    # fj_ε_∂ωε_mats1, fj_ε_∂ωε_mats1! = _ε_fjs(mats1)
    # ε_∂ωε_mats1 = fj_ε_∂ωε_mats1(rand_p_ω())
    # fj_ε_∂ωε_mats1_T, fj_ε_∂ωε_mats1_T! = _ε_fjs(mats1,(:T,))
    # ε_∂ωε_mats1_T = fj_ε_∂ωε_mats1_T(rand_p_ω_T())
    # fj_ε_∂ωε_mats2, fj_ε_∂ωε_mats2! = _ε_fjs(mats2)
    # ε_∂ωε_mats2 = fj_ε_∂ωε_mats2(rand_p_ω())
    # fj_ε_∂ωε_mats2_T, fj_ε_∂ωε_mats2_T! = _ε_fjs(mats2,(:T,))
    # ε_∂ωε_mats2_T = fj_ε_∂ωε_mats2_T(rand_p_ω_T())
    # fj_ε_∂ωε_mats3, fj_ε_∂ωε_mats3! = _ε_fjs(mats3)
    # ε_∂ωε_mats3 = fj_ε_∂ωε_mats3(rand_p_ω())
    # fj_ε_∂ωε_mats3_T, fj_ε_∂ωε_mats3_T! = _ε_fjs(mats3,(:T,))
    # ε_∂ωε_mats3_T = fj_ε_∂ωε_mats3_T(rand_p_ω_T())

    # res96 = zeros(9,6);
    # res912 = zeros(9,12);
    # res920 = zeros(9,20);
    # println("Compute ε, ∂ωε & their jacobians vs. (ω, T, n̂, r):")
    # println("fj_ε_∂ωε_mats1")
    # @btime fj_ε_∂ωε_mats11($p_ω);
    # println("fj_ε_∂ωε_mats1!")
    # @btime fj_ε_∂ωε_mats11!($res96,$p_ω);
    # println("fj_ε_∂ωε_mats1_T")
    # @btime fj_ε_∂ωε_mats1_T($p_ω_T);
    # println("fj_ε_∂ωε_mats1_T!")
    # @btime fj_ε_∂ωε_mats1_T!($res96,$p_ω_T);
    # println("fj_ε_∂ωε_mats2")
    # @btime fj_ε_∂ωε_mats2($p_ω);
    # println("fj_ε_∂ωε_mats2!")
    # @btime fj_ε_∂ωε_mats2!($res912,$p_ω);
    # println("fj_ε_∂ωε_mats2_T")
    # @btime fj_ε_∂ωε_mats2_T($p_ω_T);
    # println("fj_ε_∂ωε_mats2_T!")
    # @btime fj_ε_∂ωε_mats2_T!($res912,$p_ω_T);
    # println("fj_ε_∂ωε_mats3")
    # @btime fj_ε_∂ωε_mats3($p_ω);
    # println("fj_ε_∂ωε_mats3!")
    # @btime fj_ε_∂ωε_mats3!($res920,$p_ω);
    # println("fj_ε_∂ωε_mats3_T")
    # @btime fj_ε_∂ωε_mats3_T($p_ω_T);
    # println("fj_ε_∂ωε_mats3_T!")
    # @btime fj_ε_∂ωε_mats3_T!($res920,$p_ω_T);


    # fj_ε_∂ωε_∂²ωε_mats1, fj_ε_∂ωε_∂²ωε_mats1! = _ε_fjs2(mats1)
    # ε_∂ωε_∂²ωε_mats1 = fj_ε_∂ωε_∂²ωε_mats1(rand_p_ω())
    # fj_ε_∂ωε_∂²ωε_mats1_T, fj_ε_∂ωε_∂²ωε_mats1_T! = _ε_fjs2(mats1,(:T,))
    # ε_∂ωε_∂²ωε_mats1_T = fj_ε_∂ωε_∂²ωε_mats1_T(rand_p_ω_T())
    # fj_ε_∂ωε_∂²ωε_mats2, fj_ε_∂ωε_∂²ωε_mats2! = _ε_fjs2(mats2)
    # ε_∂ωε_∂²ωε_mats2 = fj_ε_∂ωε_∂²ωε_mats2(rand_p_ω())
    # fj_ε_∂ωε_∂²ωε_mats2_T, fj_ε_∂ωε_∂²ωε_mats2_T! = _ε_fjs2(mats2,(:T,))
    # ε_∂ωε_∂²ωε_mats2_T = fj_ε_∂ωε_∂²ωε_mats2_T(rand_p_ω_T())
    # fj_ε_∂ωε_∂²ωε_mats3, fj_ε_∂ωε_∂²ωε_mats3! = _ε_fjs2(mats3)
    # ε_∂ωε_∂²ωε_mats3 = fj_ε_∂ωε_∂²ωε_mats3(rand_p_ω())
    # fj_ε_∂ωε_∂²ωε_mats3_T, fj_ε_∂ωε_∂²ωε_mats3_T! = _ε_fjs2(mats3,(:T,))
    # ε_∂ωε_∂²ωε_mats3_T = fj_ε_∂ωε_∂²ωε_mats3_T(rand_p_ω_T())

    # res96 = zeros(9,6);
    # res912 = zeros(9,12);
    # res920 = zeros(9,20);
    # println("Compute ε, ∂ωε, ∂²ωε & their jacobians vs. (ω, T, n̂, r):")
    # println("fj_ε_∂ωε_∂²ωε_mats1")
    # @btime fj_ε_∂ωε_∂²ωε_mats11($p_ω);
    # println("fj_ε_∂ωε_∂²ωε_mats1!")
    # @btime fj_ε_∂ωε_∂²ωε_mats11!($res96,$p_ω);
    # println("fj_ε_∂ωε_∂²ωε_mats1_T")
    # @btime fj_ε_∂ωε_∂²ωε_mats1_T($p_ω_T);
    # println("fj_ε_∂ωε_∂²ωε_mats1_T!")
    # @btime fj_ε_∂ωε_∂²ωε_mats1_T!($res96,$p_ω_T);
    # println("fj_ε_∂ωε_∂²ωε_mats2")
    # @btime fj_ε_∂ωε_∂²ωε_mats2($p_ω);
    # println("fj_ε_∂ωε_∂²ωε_mats2!")
    # @btime fj_ε_∂ωε_∂²ωε_mats2!($res912,$p_ω);
    # println("fj_ε_∂ωε_∂²ωε_mats2_T")
    # @btime fj_ε_∂ωε_∂²ωε_mats2_T($p_ω_T);
    # println("fj_ε_∂ωε_∂²ωε_mats2_T!")
    # @btime fj_ε_∂ωε_∂²ωε_mats2_T!($res912,$p_ω_T);
    # println("fj_ε_∂ωε_∂²ωε_mats3")
    # @btime fj_ε_∂ωε_∂²ωε_mats3($p_ω);
    # println("fj_ε_∂ωε_∂²ωε_mats3!")
    # @btime fj_ε_∂ωε_∂²ωε_mats3!($res920,$p_ω);
    # println("fj_ε_∂ωε_∂²ωε_mats3_T")
    # @btime fj_ε_∂ωε_∂²ωε_mats3_T($p_ω_T);
    # println("fj_ε_∂ωε_∂²ωε_mats3_T!")
    # @btime fj_ε_∂ωε_∂²ωε_mats3_T!($res920,$p_ω_T);
    return nothing
end
test_ε_fs()


# Build and compile functions for ε & ∂ωε vs. (ω, T, n̂, r):
#     building f_ε_∂ωε_mats1:
#      53.353139 seconds (87.53 M allocations: 4.772 GiB, 3.04% gc time, 85.24% compilation time)
#     f_ε_∂ωε_mats1 warmup run:
#       6.340693 seconds (43.26 M allocations: 1.734 GiB, 22.58% gc time, 100.00% compilation time)
#     f_ε_∂ωε_mats1! warmup run:
#       6.019088 seconds (43.06 M allocations: 1.723 GiB, 22.44% gc time, 99.97% compilation time)
#     building f_ε_∂ωε_mats1_T:
#      15.979448 seconds (29.02 M allocations: 1.142 GiB, 4.70% gc time, 28.28% compilation time)
#     f_ε_∂ωε_mats1_T warmup run:
#       9.274001 seconds (67.97 M allocations: 2.749 GiB, 24.59% gc time, 100.00% compilation time)
#     f_ε_∂ωε_mats1_T! warmup run:
#       8.864856 seconds (67.98 M allocations: 2.754 GiB, 24.36% gc time, 100.00% compilation time)
#     building f_ε_∂ωε_mats2:
#      22.733657 seconds (49.85 M allocations: 1.810 GiB, 6.07% gc time, 2.14% compilation time)
#     f_ε_∂ωε_mats2 warmup run:
#      19.336215 seconds (129.98 M allocations: 5.194 GiB, 27.16% gc time, 100.00% compilation time)
#     f_ε_∂ωε_mats2! warmup run:
#      20.729016 seconds (129.91 M allocations: 5.196 GiB, 28.67% gc time, 100.00% compilation time)
#     building f_ε_∂ωε_mats2_T:
#      31.287330 seconds (66.78 M allocations: 2.379 GiB, 5.83% gc time, 0.71% compilation time)
#     f_ε_∂ωε_mats2_T warmup run:
#      27.610653 seconds (182.41 M allocations: 7.339 GiB, 27.88% gc time, 100.00% compilation time)
#     f_ε_∂ωε_mats2_T! warmup run:
#      26.667212 seconds (182.44 M allocations: 7.342 GiB, 27.73% gc time, 100.00% compilation time)
#     building f_ε_∂ωε_mats3:
#       6.348701 seconds (14.19 M allocations: 563.703 MiB, 21.65% compilation time)
#     f_ε_∂ωε_mats3 warmup run:
#       4.988230 seconds (33.80 M allocations: 1.348 GiB, 23.43% gc time, 100.00% compilation time)
#     f_ε_∂ωε_mats3! warmup run:
#       5.245575 seconds (33.81 M allocations: 1.350 GiB, 28.94% gc time, 100.00% compilation time)
#     building f_ε_∂ωε_mats3_T:
#      10.083131 seconds (19.10 M allocations: 754.589 MiB, 4.54% gc time, 24.02% compilation time)
#     f_ε_∂ωε_mats3_T warmup run:
#       7.010157 seconds (49.42 M allocations: 1.975 GiB, 23.74% gc time, 100.00% compilation time)
#     f_ε_∂ωε_mats3_T! warmup run:
#       6.475240 seconds (49.44 M allocations: 1.978 GiB, 20.87% gc time, 100.00% compilation time)
# Build and compile functions for ε, ∂ωε & ∂²ωε vs. (ω, T, n̂, r):
#     building f_ε_∂ωε_∂²ωε_mats1:
#      51.302802 seconds (102.65 M allocations: 4.196 GiB, 4.76% gc time, 26.90% compilation time)
#     f_ε_∂ωε_∂²ωε_mats1 warmup run:
#      32.495999 seconds (207.32 M allocations: 8.262 GiB, 28.08% gc time, 100.00% compilation time)
#     f_ε_∂ωε_∂²ωε_mats1! warmup run:
#      34.702038 seconds (207.29 M allocations: 8.273 GiB, 29.96% gc time, 100.00% compilation time)
#     building f_ε_∂ωε_∂²ωε_mats1_T:
#      69.619757 seconds (142.79 M allocations: 5.263 GiB, 5.44% gc time, 7.49% compilation time)
#     f_ε_∂ωε_∂²ωε_mats1_T warmup run:
#      67.166632 seconds (338.22 M allocations: 13.654 GiB, 33.15% gc time, 100.00% compilation time)
#     f_ε_∂ωε_∂²ωε_mats1_T! warmup run:
#      62.374292 seconds (338.25 M allocations: 13.675 GiB, 39.82% gc time, 100.00% compilation time)
#     building f_ε_∂ωε_∂²ωε_mats2:
#      126.551227 seconds (269.87 M allocations: 9.801 GiB, 6.50% gc time, 0.50% compilation time)
#     f_ε_∂ωε_∂²ωε_mats2 warmup run:
#      150.772278 seconds (618.64 M allocations: 24.688 GiB, 43.41% gc time, 100.00% compilation time)
#     f_ε_∂ωε_∂²ωε_mats2! warmup run:
#      184.192100 seconds (618.62 M allocations: 24.721 GiB, 54.64% gc time, 100.00% compilation time)
#     building f_ε_∂ωε_∂²ωε_mats2_T:
#      171.667558 seconds (376.44 M allocations: 13.441 GiB, 6.65% gc time, 0.05% compilation time)
#     f_ε_∂ωε_∂²ωε_mats2_T warmup run: