################################################################################
#                                                                              #
#                                 OptiMode.jl:                                 #
#                Differentiable Electromagnetic Eigenmode Solver               #
#                                                                              #
# This top-level package re-exports all four sub-packages:                     #
#   1. MaterialModels   - Dielectric material dispersion modeling              #
#   2. DielectricSmoother - Dielectric tensor smoothing on FD grids           #
#   3. EigenModeSolver  - Electromagnetic eigenmode solving                   #
#   4. ModeAnalysis     - Post-processing of mode solver results              #
#                                                                              #
################################################################################

module OptiMode

using MaterialModels
using DielectricSmoother
using EigenModeSolver
using ModeAnalysis

# Re-export all public symbols from sub-packages
using MaterialModels: ε_tensor, εᵥ, flat, Δₘ,
    AbstractMaterial, Material, RotatedMaterial, NumMat,
    get_model, generate_fn, has_model, material_name,
    n²_sym_fmt1, n²_sym_fmt1_ω, n_sym_cauchy, n_sym_cauchy_ω,
    n²_sym_NASA, n²_sym_NASA_ω, ng_model, gvd_model,
    nn̂g_model, nĝvd_model, nn̂g_fn, nĝvd_fn, ε_fn,
    nn̂g, nĝvd, χ⁽²⁾_fn, rotate, unique_axes, Δₘ_factors,
    eval_fn_oop, eval_fn_ip,
    # Materials
    vacuum, silicon, SiO₂, Si₃N₄, LiNbO₃, MgO_LiNbO₃, LiB₃O₅, αAl₂O₃, germanium

using DielectricSmoother: Grid, δx, δy, δz, δV, xvals, yvals, zvals, x⃗, xc, yc, zc,
    x⃗c, N, g⃗, _fftaxes, Nranges, corners, vxlmin, vxlmax, my_fftfreq,
    x, y, z, δ,
    Geometry, fεs, fεs!, fnn̂gs, nn̂gs, fnĝvds, nĝvds, matinds,
    εₘₐₓ, nₘₐₓ, materials,
    corner_sinds, proc_sinds, smooth_ε, smooth_ε_single, vec3D,
    normcart, τ_trans, τ⁻¹_trans, avg_param, avg_param_rot,
    _f_ε_mats, _fj_ε_mats, _fjh_ε_mats, ε_views,
    εₑ_∂ωεₑ_∂²ωεₑ, εₑ_∂ωεₑ

using EigenModeSolver: HelmholtzMap, HelmholtzPreconditioner, ModeSolver,
    solve_ω², _solve_Δω², solve_k, solve_k_single, filter_eigs,
    AbstractEigensolver, KrylovKitEigsolve, IterativeSolversLOBPCG,
    herm, herm_back, eig_adjt, my_linsolve, solve_adj!, ng_gvd_E, ng_gvd,
    _mult, _dot, _3dot, _cross, _cross_x, _cross_y, _cross_z,
    _sum_cross, _sum_cross_x, _sum_cross_y, _sum_cross_z,
    _outer, _expect, sliceinv_3x3,
    update_k!, update_ε⁻¹

using ModeAnalysis: unflat, _d2ẽ!, _H2d!, _H2e!, E⃗, E⃗x, E⃗y, E⃗z,
    H⃗, H⃗x, H⃗y, H⃗z, S⃗, S⃗x, S⃗y, S⃗z,
    normE!, Ex_norm, Ey_norm, val_magmax, ax_magmax, idx_magmax,
    group_index, canonicalize_phase, canonicalize_phase!,
    E_relpower_xyz, Eslices, count_E_nodes, mode_viable, mode_idx

end # module OptiMode
