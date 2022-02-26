using LinearAlgebra
using Symbolics
using Symbolics: Sym, Num
using SymbolicUtils: @rule, @acrule, @slots, RuleSet, numerators, denominators, flatten_pows
using SymbolicUtils.Rewriters: Chain, RestartedChain, PassThrough, Prewalk, Postwalk
using IterTools: subsets
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

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

###### Utility Functions ######
ε_views(εv,n_mats;nε=3) = ([ reshape(view(view(εv,(1+(i-1)*9*n_mats):(i*9*n_mats)), (1+9*(mat_idx-1)):(9+9*(mat_idx-1))), (3,3)) for mat_idx=1:n_mats] for i=1:nε) 


###### Kottke Smoothing Functions ######
include("f_epse.jl")

# ∂ωεₑ(r₁,n,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)   = @views reshape( fj_εₑ(vcat(r₁,n,vec(ε₁),vec(ε₂)))[:,2:23]  * vcat(zeros(4),vec(∂ω_ε₁),vec(∂ω_ε₂)), (3,3) )

# function εₑ_∂ωεₑ_∂²ωεₑ(r₁,n,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂)
#     fjh_εₑ_12 = fjh_εₑ(vcat(r₁,n,vec(ε₁),vec(ε₂)));
#     f_εₑ_12, j_εₑ_12, h_εₑ_12 = @views @inbounds fjh_εₑ_12[:,1], fjh_εₑ_12[:,2:23], reshape(fjh_εₑ_12[:,24:507],(9,22,22));
#     εₑ_12 = @views reshape(f_εₑ_12,(3,3))
#     v_∂ω, v_∂²ω = vcat(zeros(4),vec(∂ω_ε₁),vec(∂ω_ε₂)), vcat(zeros(4),vec(∂²ω_ε₁),vec(∂²ω_ε₂));
#     ∂ω_εₑ_12 = @views reshape( j_εₑ_12 * v_∂ω, (3,3) );
#     ∂ω²_εₑ_12 = @views reshape( [dot(v_∂ω,h_εₑ_12[i,:,:],v_∂ω) for i=1:9] + j_εₑ_12*v_∂²ω , (3,3) );
#     return εₑ_12, ∂ω_εₑ_12, ∂ω²_εₑ_12
# end

####

function _f_εₑ_∂ωεₑ_∂²ωεₑ_mats_sym(mats,p_syms=(:ω,))
    n_mats = length(mats)
    ε_∂ωε_∂²ωε_mats_sym, p_mats = _f_ε_mats_sym(mats,p_syms) # 27*length(mats) x 1
    ε_mats,∂ωε_mats,∂²ωε_mats = ε_views(ε_∂ωε_∂²ωε_mats_sym,n_mats)
    ε_mats_simp = simplify.(expand.(ε_mats))
    p_geom = @variables n1 n2 n3 r
    p = vcat(p_mats,p_geom)
    S = simplify.(normcart([n1, n2, n3]))
    ω = first(p_mats)
    Dom = Differential(ω)
    εₑ_∂ωεₑ_∂²ωεₑ_mats_sym = map(subsets(1:length(mats),2)) do (mat_idx1, mat_idx2)
        εₑ_vec = expand.(vec(avg_param(ε_mats_simp[mat_idx1], ε_mats_simp[mat_idx2], S, r)));
        ∂ωεₑ_vec = expand_derivatives.(Dom.(εₑ_vec));
        ∂²ωεₑ_vec = expand_derivatives.(Dom.(∂ωεₑ_vec));
        return vcat(εₑ_vec,∂ωεₑ_vec,∂²ωεₑ_vec)
    end
    return εₑ_∂ωεₑ_∂²ωεₑ_mats_sym, p
end

function _f_εₑ_∂ωεₑ_∂²ωεₑ_mats(mats,p_syms=(:ω,))
    n_mats = length(mats)
    ε_∂ωε_∂²ωε_mats_sym, p_mats = _f_ε_mats_sym(mats,p_syms) # 27*length(mats) x 1
    ε_mats,∂ωε_mats,∂²ωε_mats = ε_views(ε_∂ωε_∂²ωε_mats_sym,n_mats)
    ε_mats_simp = simplify.(expand.(ε_mats))
    p_geom = @variables n1 n2 n3 r
    p = vcat(p_mats,p_geom)
    S = simplify.(normcart([n1, n2, n3]))
    ω = first(p_mats)
    Dom = Differential(ω)
    f_εₑ_∂ωεₑ_∂²ωεₑ_mats = map(subsets(1:length(mats),2)) do (mat_idx1, mat_idx2)
        εₑ_vec = expand.(vec(avg_param(ε_mats_simp[mat_idx1], ε_mats_simp[mat_idx2], S, r)));
        ∂ωεₑ_vec = expand_derivatives.(Dom.(εₑ_vec));
        ∂²ωεₑ_vec = expand_derivatives.(Dom.(∂ωεₑ_vec));
        # return eval(fn_expr(vcat(εₑ_vec,∂ωεₑ_vec,∂²ωεₑ_vec), p))
        return eval_fn_oop(vcat(εₑ_vec,∂ωεₑ_vec,∂²ωεₑ_vec), p)
        # return eval_fn_ip(vcat(εₑ_vec,∂ωεₑ_vec,∂²ωεₑ_vec), p)
    end
    return f_εₑ_∂ωεₑ_∂²ωεₑ_mats
end

function _fj_εₑ_∂ωεₑ_∂²ωεₑ_mats(mats,p_syms=(:ω,))
    n_mats = length(mats)
    ε_∂ωε_∂²ωε_mats_sym, p_mats = _f_ε_mats_sym(mats,p_syms) # 27*length(mats) x 1
    ε_mats,∂ωε_mats,∂²ωε_mats = ε_views(ε_∂ωε_∂²ωε_mats_sym,n_mats)
    ε_mats_simp = simplify.(expand.(ε_mats))
    p_geom = @variables n1 n2 n3 r
    p = vcat(p_mats,p_geom)
    S = simplify.(normcart([n1, n2, n3]))
    ω = first(p_mats)
    Dom = Differential(ω)
    fj_εₑ_∂ωεₑ_∂²ωεₑ_mats = map(subsets(1:length(mats),2)) do (mat_idx1, mat_idx2)
        εₑ_vec = expand.(vec(avg_param(ε_mats_simp[mat_idx1], ε_mats_simp[mat_idx2], S, r)));
        ∂ωεₑ_vec = expand_derivatives.(Dom.(εₑ_vec));
        ∂²ωεₑ_vec = expand_derivatives.(Dom.(∂ωεₑ_vec));
        # return eval(fn_expr(_fj_sym(vcat(εₑ_vec,∂ωεₑ_vec,∂²ωεₑ_vec),p),p))
        return eval_fn_oop(_fj_sym(vcat(εₑ_vec,∂ωεₑ_vec,∂²ωεₑ_vec), p),p)
        # return eval_fn_ip(_fj_sym(vcat(εₑ_vec,∂ωεₑ_vec,∂²ωεₑ_vec), p),p)
    end
    return fj_εₑ_∂ωεₑ_∂²ωεₑ_mats
end

function _fjh_εₑ_∂ωεₑ_∂²ωεₑ_mats(mats,p_syms=(:ω,))
    n_mats = length(mats)
    ε_∂ωε_∂²ωε_mats_sym, p_mats = _f_ε_mats_sym(mats,p_syms) # 27*length(mats) x 1
    ε_mats,∂ωε_mats,∂²ωε_mats = ε_views(ε_∂ωε_∂²ωε_mats_sym,n_mats)
    ε_mats_simp = simplify.(expand.(ε_mats))
    p_geom = @variables n1 n2 n3 r
    p = vcat(p_mats,p_geom)
    S = simplify.(normcart([n1, n2, n3]))
    ω = first(p_mats)
    Dom = Differential(ω)
    fjh_εₑ_∂ωεₑ_∂²ωεₑ_mats = map(subsets(1:length(mats),2)) do (mat_idx1, mat_idx2)
        εₑ_vec = expand.(vec(avg_param(ε_mats_simp[mat_idx1], ε_mats_simp[mat_idx2], S, r)));
        ∂ωεₑ_vec = expand_derivatives.(Dom.(εₑ_vec));
        ∂²ωεₑ_vec = expand_derivatives.(Dom.(∂ωεₑ_vec));
        # return eval(fn_expr(_fjh_sym(vcat(εₑ_vec,∂ωεₑ_vec,∂²ωεₑ_vec),p)[2],p))
        return eval_fn_oop(_fjh_sym(vcat(εₑ_vec,∂ωεₑ_vec,∂²ωεₑ_vec), p)[2],p)
        # return eval_fn_ip(_fjh_sym(vcat(εₑ_vec,∂ωεₑ_vec,∂²ωεₑ_vec), p)[2],p)
    end
    return fjh_εₑ_∂ωεₑ_∂²ωεₑ_mats
end

####

function _f_εₑ_∂ωεₑ_mats_sym(mats,p_syms=(:ω,))
    n_mats = length(mats)
    ε_∂ωε_∂²ωε_mats_sym, p_mats = _f_ε_mats_sym(mats,p_syms) # 27*length(mats) x 1
    ε_mats,∂ωε_mats,∂²ωε_mats = ε_views(ε_∂ωε_∂²ωε_mats_sym,n_mats)
    ε_mats_simp = simplify.(expand.(ε_mats))
    p_geom = @variables n1 n2 n3 r
    p = vcat(p_mats,p_geom)
    S = simplify.(normcart([n1, n2, n3]))
    ω = first(p_mats)
    Dom = Differential(ω)
    εₑ_∂ωεₑ_mats_sym = map(subsets(1:length(mats),2)) do (mat_idx1, mat_idx2)
        εₑ_vec = expand.(vec(avg_param(ε_mats_simp[mat_idx1], ε_mats_simp[mat_idx2], S, r)));
        ∂ωεₑ_vec = expand_derivatives.(Dom.(εₑ_vec));
        return vcat(εₑ_vec,∂ωεₑ_vec)
    end
    return εₑ_∂ωεₑ_mats_sym, p
end

function _f_εₑ_∂ωεₑ_mats(mats,p_syms=(:ω,))
    n_mats = length(mats)
    ε_∂ωε_∂²ωε_mats_sym, p_mats = _f_ε_mats_sym(mats,p_syms) # 27*length(mats) x 1
    ε_mats,∂ωε_mats,∂²ωε_mats = ε_views(ε_∂ωε_∂²ωε_mats_sym,n_mats)
    ε_mats_simp = simplify.(expand.(ε_mats))
    p_geom = @variables n1 n2 n3 r
    p = vcat(p_mats,p_geom)
    S = simplify.(normcart([n1, n2, n3]))
    ω = first(p_mats)
    Dom = Differential(ω)
    f_εₑ_∂ωεₑ_mats = map(subsets(1:length(mats),2)) do (mat_idx1, mat_idx2)
        εₑ_vec = expand.(vec(avg_param(ε_mats_simp[mat_idx1], ε_mats_simp[mat_idx2], S, r)));
        ∂ωεₑ_vec = expand_derivatives.(Dom.(εₑ_vec));
        # return eval(fn_expr(vcat(εₑ_vec,∂ωεₑ_vec), p))
        return eval_fn_oop(vcat(εₑ_vec,∂ωεₑ_vec), p)
        # return eval_fn_ip(vcat(εₑ_vec,∂ωεₑ_vec), p)
    end
    return f_εₑ_∂ωεₑ_mats
end

function _fj_εₑ_∂ωεₑ_mats(mats,p_syms=(:ω,))
    n_mats = length(mats)
    ε_∂ωε_∂²ωε_mats_sym, p_mats = _f_ε_mats_sym(mats,p_syms) # 27*length(mats) x 1
    ε_mats,∂ωε_mats,∂²ωε_mats = ε_views(ε_∂ωε_∂²ωε_mats_sym,n_mats)
    ε_mats_simp = simplify.(expand.(ε_mats))
    p_geom = @variables n1 n2 n3 r
    p = vcat(p_mats,p_geom)
    S = simplify.(normcart([n1, n2, n3]))
    ω = first(p_mats)
    Dom = Differential(ω)
    fj_εₑ_∂ωεₑ_mats = map(subsets(1:length(mats),2)) do (mat_idx1, mat_idx2)
        εₑ_vec = expand.(vec(avg_param(ε_mats_simp[mat_idx1], ε_mats_simp[mat_idx2], S, r)));
        ∂ωεₑ_vec = expand_derivatives.(Dom.(εₑ_vec));
        # return eval(fn_expr(_fj_sym(vcat(εₑ_vec,∂ωεₑ_vec),p),p))
        return eval_fn_oop(_fj_sym(vcat(εₑ_vec,∂ωεₑ_vec), p),p)
        # return eval_fn_ip(_fj_sym(vcat(εₑ_vec,∂ωεₑ_vec), p),p)
    end
    return fj_εₑ_∂ωεₑ_mats
end

function _fjh_εₑ_∂ωεₑ_mats(mats,p_syms=(:ω,))
    n_mats = length(mats)
    ε_∂ωε_∂²ωε_mats_sym, p_mats = _f_ε_mats_sym(mats,p_syms) # 27*length(mats) x 1
    ε_mats,∂ωε_mats,∂²ωε_mats = ε_views(ε_∂ωε_∂²ωε_mats_sym,n_mats)
    ε_mats_simp = simplify.(expand.(ε_mats))
    p_geom = @variables n1 n2 n3 r
    p = vcat(p_mats,p_geom)
    S = simplify.(normcart([n1, n2, n3]))
    ω = first(p_mats)
    Dom = Differential(ω)
    fjh_εₑ_∂ωεₑ_mats = map(subsets(1:length(mats),2)) do (mat_idx1, mat_idx2)
        εₑ_vec = expand.(vec(avg_param(ε_mats_simp[mat_idx1], ε_mats_simp[mat_idx2], S, r)));
        ∂ωεₑ_vec = expand_derivatives.(Dom.(εₑ_vec));
        # return eval(fn_expr(_fjh_sym(vcat(εₑ_vec,∂ωεₑ_vec),p)[2],p))
        return eval_fn_oop(_fjh_sym(vcat(εₑ_vec,∂ωεₑ_vec), p)[2],p)
        # return eval_fn_ip(_fjh_sym(vcat(εₑ_vec,∂ωεₑ_vec), p)[2],p)
    end
    return fjh_εₑ_∂ωεₑ_mats
end

# _f_εₑ_∂ωεₑ_∂²ωεₑ_mats(mats,p_syms=(:ω,)) = eval(fn_expr(_f_εₑ_∂ωεₑ_∂²ωεₑ_mats_sym(mats,p_syms)...))

# _f_εₑ_∂ωεₑ_∂²ωεₑ_mats(mats,p_syms=(:ω,)) = eval(fn_expr(_f_εₑ_∂ωεₑ_∂²ωεₑ_mats_sym(mats,p_syms)...))
#     εₑ_∂ωεₑ_∂²ωεₑ_mats_sym, p = _f_εₑ_∂ωεₑ_∂²ωεₑ_mats_sym(mats,p_syms)

####### testing
mats1 = [Si₃N₄,SiO₂];
f_εₑ_sym1, p1 = _f_εₑ_∂ωεₑ_∂²ωεₑ_mats_sym(mats1);
f_εₑ1 = _f_εₑ_∂ωεₑ_∂²ωεₑ_mats(mats1);
map(ff->ff(rand(5)),f_εₑ1);
fj_εₑ1 = _fj_εₑ_∂ωεₑ_∂²ωεₑ_mats(mats1);
map(ff->ff(rand(5)),fj_εₑ1);

mats2 = [MgO_LiNbO₃,Si₃N₄,SiO₂,Vacuum];
f_εₑ_sym2, p2 = _f_εₑ_∂ωεₑ_∂²ωεₑ_mats_sym(mats2);
f_εₑ2 = _f_εₑ_∂ωεₑ_∂²ωεₑ_mats(mats2);
map(ff->ff(rand(5)),f_εₑ2);
fj_εₑ2 = _fj_εₑ_∂ωεₑ_∂²ωεₑ_mats(mats2);
map(ff->ff(rand(5)),fj_εₑ2);

# fjh_εₑ1 = _fjh_εₑ_∂ωεₑ_∂²ωεₑ_mats(mats1);
# map(ff->ff(rand(5)),fjh_εₑ1);


##### Tests
# mats = [MgO_LiNbO₃,Si₃N₄,SiO₂,Vacuum];
# # mats = [MgO_LiNbO₃,Si₃N₄,SiO₂];
# n_mats = length(mats);
# fj_ε_mats, fj_ε_mats! = _fj_ε_mats(mats,(:ω,:T))
# @show ω0,T0 = ( 0.2*rand(Float64)+0.8, 20.0*rand(Float64)+20.0 ) # frequency and temperature, in μm⁻¹ and C respectively
# p0 = [ω0,T0];
# fj = fj_ε_mats(p0);
# (ε, ∂ωε, ∂²ωε), (∂ω_ε, ∂ω_∂ωε, ∂ω_∂²ωε), (∂T_ε, ∂T_∂ωε, ∂T_∂²ωε) = ε_views.(eachcol(fj),(n_mats,));
# fjh_ε_mats, fjh_ε_mats! = _fjh_ε_mats(mats,(:ω,:T));
# fjh = fjh_ε_mats(p0);
# fjh_ε_mats_unpack(x) = collect.(ε_views.(eachcol(fjh_ε_mats(x)),(n_mats,)));
# (ε, ∂ωε, ∂²ωε), (∂ω_ε, ∂ω_∂ωε, ∂ω_∂²ωε), (∂T_ε, ∂T_∂ωε, ∂T_∂²ωε),  (∂ω∂ω_ε, ∂ω∂ω_∂ωε, ∂ω∂ω_∂²ωε), (∂ω∂T_ε, ∂ω∂T_∂ωε, ∂ω∂T_∂²ωε), (∂T∂ω_ε, ∂T_∂ωε, ∂T∂ω_∂²ωε), (∂T∂T_ε, ∂T∂T_∂ωε, ∂T∂T_∂²ωε) = ε_views.(eachcol(fjh),(n_mats,));
# @assert ∂ω∂ω_ε == ∂ω_∂ωε == ∂²ωε

# @assert jacFD(om->vcat(fjh_ε_mats_unpack(vcat(om,p0[2:end]))[1][1]...),p0[1]) ≈ vcat(fjh_ε_mats_unpack(p0)[1][2]...)
# @assert jacFD(om->vcat(fjh_ε_mats_unpack(vcat(om,p0[2:end]))[1][2]...),p0[1]) ≈ vcat(fjh_ε_mats_unpack(p0)[1][3]...)
# @assert ForwardDiff.derivative(om->vcat(fjh_ε_mats_unpack(vcat(om,p0[2:end]))[1][1]...),p0[1]) ≈ vcat(fjh_ε_mats_unpack(p0)[1][2]...)
# @assert ForwardDiff.derivative(om->vcat(fjh_ε_mats_unpack(vcat(om,p0[2:end]))[1][2]...),p0[1]) ≈ vcat(fjh_ε_mats_unpack(p0)[1][3]...)

# @assert jacFD(TT->vcat(fjh_ε_mats_unpack([p0[1],TT])[1][1]...),p0[2]) ≈ vcat(fjh_ε_mats_unpack(p0)[3][1]...)
# @assert jacFD(TT->vcat(fjh_ε_mats_unpack([p0[1],TT])[1][2]...),p0[2]) ≈ vcat(fjh_ε_mats_unpack(p0)[3][2]...)
# @assert jacFD(TT->vcat(fjh_ε_mats_unpack([p0[1],TT])[1][3]...),p0[2]) ≈ vcat(fjh_ε_mats_unpack(p0)[3][3]...)
# @assert ForwardDiff.derivative(TT->vcat(fjh_ε_mats_unpack([p0[1],TT])[1][1]...),p0[2]) ≈ vcat(fjh_ε_mats_unpack(p0)[3][1]...)
# @assert ForwardDiff.derivative(TT->vcat(fjh_ε_mats_unpack([p0[1],TT])[1][2]...),p0[2]) ≈ vcat(fjh_ε_mats_unpack(p0)[3][2]...)
# @assert ForwardDiff.derivative(TT->vcat(fjh_ε_mats_unpack([p0[1],TT])[1][3]...),p0[2]) ≈ vcat(fjh_ε_mats_unpack(p0)[3][3]...)