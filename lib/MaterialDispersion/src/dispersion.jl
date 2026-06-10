export _f_ε_mats, _fj_ε_mats, _fjh_ε_mats, ε_views, _fj_sym, _fj_fjh_sym

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
    _f_ε_mats(mats, p_syms=(:ω,))

Generate out-of-place and in-place functions `(f, f!)` mapping a parameter vector `p`
(matching `p_syms`, with frequency `ω` first) to a flat array containing the dielectric
tensors `ε` of all materials in `mats` along with their first and second frequency
derivatives `∂ωε` and `∂²ωε`.

Example:
    ```
    mats = [MgO_LiNbO₃,Si₃N₄,SiO₂,LiB₃O₅];
    n_mats = length(mats);
    f_ε_mats, f_ε_mats! = _f_ε_mats(mats,(:ω,:T))
    ω0,T0 = ( 0.2*rand(Float64)+0.8, 20.0*rand(Float64)+20.0 ) # frequency and temperature, in μm⁻¹ and C respectively
    p0 = [ω0,T0];
    f0 = f_ε_mats(p0);
    ε, ∂ωε, ∂²ωε = ε_views(f0,n_mats);
    ```
"""
function _f_ε_mats(mats,p_syms=(:ω,);expression=Val{false})
    f_ε_mats_sym, p = _f_ε_mats_sym(mats,p_syms)
    f_ε_∂ωε_∂²ωε = eval_fn_oop(f_ε_mats_sym, p)
    f_ε_∂ωε_∂²ωε! = eval_fn_ip(f_ε_mats_sym, p)
    return f_ε_∂ωε_∂²ωε, f_ε_∂ωε_∂²ωε!
end

"""
    _fj_ε_mats(mats, p_syms=(:ω,))

Like [`_f_ε_mats`](@ref) but the generated functions return the function values
horizontally concatenated with their (symbolically computed) Jacobian w.r.t. `p`.

Example:
    ```
    mats = [MgO_LiNbO₃,Si₃N₄,SiO₂,LiB₃O₅];
    n_mats = length(mats);
    fj_ε_mats, fj_ε_mats! = _fj_ε_mats(mats,(:ω,:T))
    ω0,T0 = ( 0.2*rand(Float64)+0.8, 20.0*rand(Float64)+20.0 ) # frequency and temperature, in μm⁻¹ and C respectively
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
    _fjh_ε_mats(mats, p_syms=(:ω,))

Like [`_fj_ε_mats`](@ref) but the generated functions also return the (symbolically
computed) Hessian w.r.t. `p`.

Example:
    ```
    mats = [MgO_LiNbO₃,Si₃N₄,SiO₂,LiB₃O₅];
    n_mats = length(mats);
    fjh_ε_mats, fjh_ε_mats! = _fjh_ε_mats(mats,(:ω,:T))
    ω0,T0 = ( 0.2*rand(Float64)+0.8, 20.0*rand(Float64)+20.0 ) # frequency and temperature, in μm⁻¹ and C respectively
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

"""
    ε_views(εv, n_mats; nε=3)

Split the flat output of a generated dispersion function (`_f_ε_mats` and friends, or a
column of the Jacobian/Hessian variants) into `nε` lists of `3×3` matrix views, one per
material. The data layout is material-major: each material contributes `9*nε` consecutive
entries `(ε, ∂ωε, ∂²ωε, …)`.
"""
ε_views(εv,n_mats;nε=3) = ([ reshape(view(εv, (9*nε*(mat_idx-1)+9*(i-1)+1):(9*nε*(mat_idx-1)+9*i)), (3,3)) for mat_idx=1:n_mats] for i=1:nε)

"""
    reactant_compile_dispersion(f, p0::AbstractVector{<:Real})

Compile the generated dispersion function `f` (e.g. from [`_f_ε_mats`](@ref)) with
Reactant/XLA for parameter vectors shaped like `p0`. Defined by the package extension
that loads when `Reactant` is available; see `ext/MaterialDispersionReactantExt.jl`.
"""
function reactant_compile_dispersion end
export reactant_compile_dispersion
