using Revise
using OptiMode
using LinearAlgebra, Statistics, StaticArrays, HybridArrays, GeometryPrimitives, BenchmarkTools
using ChainRules, Zygote, ForwardDiff, FiniteDifferences
using UnicodePlots, OhMyREPL
using Optim, Interpolations
using Zygote: @ignore, dropgrad
using GLMakie, AbstractPlotting
using AbstractPlotting.GeometryBasics
import Colors: JULIA_LOGO_COLORS
logocolors = JULIA_LOGO_COLORS
using AbstractPlotting: lines, lines!
# p = [
#        1.7,                #   top ridge width         `w_top`         [μm]
#        0.7,                #   ridge thickness         `t_core`        [μm]
#        π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
#        # 0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
#                ];
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg(x) = ridge_wg(x[1],x[2],x[3],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy) # dispersive material model version
rwg2(x) = ridge_wg(x[1],x[2],x[3],0.5,2.2,1.4,Δx,Δy) # constant index version

rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).
p = [
       1.7,                #   top ridge width         `w_top`         [μm]
       0.7,                #   ridge thickness         `t_core`        [μm]
       0.5,                #   ridge thickness         `t_core`        [μm]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
               ];


##
# parameters used by Swedish group (Gallo) to get broadband phase-matching in
# x-cut partially etched, unclad MgO:LiNbO₃-on-SiO₂ ridge waveguides:
# Fergestad and Gallo, "Ultrabroadband second harmonic generation at telecom wavelengths in lithium niobate waveguides"
# Integrated Photonics Research, Silicon and Nanophotonics (pp. ITu4A-13). OSA 2020
p_sw = [
    0.7,        # 700 nm top width of angle sidewall ridge
    0.6,        # 600 nm MgO:LiNbO₃ ridge thickness
    5. / 6.,    # etch fraction (they say etch depth of 500 nm, full thickness 600 nm)
    0.349,      # 20° sidewall angle in radians (they call it 70° , in our params 0° is vertical)
]

λs_sw = collect(reverse(1.4:0.01:1.6))

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy)

ms = ModeSolver(kguess(1/1.55,rwg_pe(p_sw)), rwg_pe(p_sw), gr; nev=2)
ωs_sw = 1 ./ λs_sw
n_swF,ng_swF = solve_n(ms,ωs_sw,rwg_pe(p_sw))
n_swS,ng_swS = solve_n(ms,2*ωs_sw,rwg_pe(p_sw))
k1,H1 = solve_k(ms,2*ωs_sw[end],rwg_pe(p_sw))
# Ex1 = E⃗x(ms); Ey1 = E⃗y(ms)
# k2,H2 = solve_k(ms,2*ωs_sw[end],rwg_pe(p_sw))
# Ex2 = E⃗x(ms); Ey2 = E⃗y(ms)

##
fig = Figure()
ax_n = fig[1,1] = Axis(fig)
ax_ng = fig[1,2] = Axis(fig)
ax_Λ = fig[2,1] = Axis(fig)
ax_qpm = fig[2,2] = Axis(fig)

lines!(ax_n, λs_sw, n_swF; color=logocolors[:red],linewidth=2)
lines!(ax_n, λs_sw, n_swS; color=logocolors[:blue],linewidth=2)
plot!(ax_n, λs_sw, n_swF; color=logocolors[:red],markersize=2)
plot!(ax_n, λs_sw, n_swS; color=logocolors[:blue],markersize=2)

lines!(ax_ng, λs_sw, ng_swF; color=logocolors[:red],linewidth=2)
lines!(ax_ng, λs_sw, ng_swS; color=logocolors[:blue],linewidth=2)
plot!(ax_ng, λs_sw, ng_swF; color=logocolors[:red],markersize=2)
plot!(ax_ng, λs_sw, ng_swS; color=logocolors[:blue],markersize=2)

# Δk_sw = ( 4π ./ λs_sw ) .* ( n_swS .- n_swF )
# Λ_sw = 2π ./ Δk_sw
Λ_sw = ( λs_sw ./ 2 ) ./ ( n_swS .- n_swF )

lines!(ax_Λ, λs_sw, Λ_sw; color=logocolors[:green],linewidth=2)
plot!(ax_Λ, λs_sw, Λ_sw; color=logocolors[:green],markersize=2)

# Λ0_sw = 2.8548 # 128x128
Λ0_sw = 2.86275 # 256x256
L_sw = 1e3 # 1cm in μm
Δk_qpm_sw = ( 4π ./ λs_sw ) .* ( n_swS .- n_swF ) .- (2π / Λ0_sw)

Δk_qpm_sw_itp = LinearInterpolation(ωs_sw,Δk_qpm_sw)
ωs_sw_dense = collect(range(extrema(ωs_sw)...,length=3000))
λs_sw_dense = inv.(ωs_sw_dense)
Δk_qpm_sw_dense = Δk_qpm_sw_itp.(ωs_sw_dense)
sinc2Δk_sw_dense = (sinc.(Δk_qpm_sw_dense * L_sw / 2.0)).^2



lines!(ax_qpm, λs_sw_dense, sinc2Δk_sw_dense; color=logocolors[:purple],linewidth=2)
# plot!(ax_qpm, λs_sw_dense, sinc2Δk_sw_dense; color=logocolors[:purple],markersize=2)

fig

Ex_axes = fig[3, 1:2] = [Axis(fig, title = t) for t in ["|Eₓ₁|²","|Eₓ₂|²"]] #,"|Eₓ₃|²","|Eₓ₄|²"]]
Ey_axes = fig[4, 1:2] = [Axis(fig, title = t) for t in ["|Ey₁|²","|Ey₂|²"]] #,"|Ey₃|²","|Ey₄|²"]]
# Es = [Ex[1],Ey[1],Ex[2],Ey[2],Ex[3],Ey[3],Ex[4],Ey[4]]

Earr = [ε⁻¹_dot( fft( kx_tc( unflat(ms.H⃗; ms)[:,:,:,i],mn(ms),ms.M̂.mag), (2:1+ND) ), copy(flat( ms.M̂.ε⁻¹ ))) for i=1:2]
Ex = [Earr[i][1,:,:] for i=1:2]
Ey = [Earr[i][2,:,:] for i=1:2]

heatmaps_x = [heatmap!(ax, abs2.(Ex[i])) for (i, ax) in enumerate(Ex_axes)]
heatmaps_y = [heatmap!(ax, abs2.(Ey[i])) for (i, ax) in enumerate(Ey_axes)]

fig

##

# parameters used by Fejer/Loncar groups (Jankowski) to get broadband phase-matching in
# x-cut partially etched, unclad MgO:LiNbO₃-on-SiO₂ ridge waveguides:
# Jankowski et al, "Ultrabroadband second harmonic generation at telecom wavelengths in lithium niobate waveguides"
# Integrated Photonics Research, Silicon and Nanophotonics (pp. ITu4A-13). OSA 2020
p_jank = [
    1.85,        # 700 nm top width of angle sidewall ridge
    0.7,        # 600 nm MgO:LiNbO₃ ridge thickness
    3.4 / 7.0,    # etch fraction (they say etch depth of 500 nm, full thickness 600 nm)
    0.5236,      # 30° sidewall angle in radians (they call it 60° , in our params 0° is vertical)
]

# λs_jank = collect(reverse(1.9:0.02:2.4))
# ωs_jank = 1 ./ λs_jank

ωs_jank =  collect(0.416:0.003:0.527)
λs_jank = 1 ./ ωs_jank

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 512, 512, 1;

ms = ModeSolver(kguess(1/1.55,rwg_pe(p_jank)), rwg_pe(p_jank), gr; nev=1)

n_jankF,ng_jankF = solve_n(ms,ωs_jank,rwg_pe(p_jank))
n_jankS,ng_jankS = solve_n(ms,2*ωs_jank,rwg_pe(p_jank))

function ng_jank_AD(om)
    neff,neff_om_pb = Zygote.pullback(x->solve_n(ms,x,rwg_pe(p_jank))[1],om)
    ng_FD = neff + om * neff_om_pb(1)[1]
end

function ng_jank_FD(om)
    neff = solve_n(ms,om,rwg_pe(p_jank))[1]
    dneff_dom_FD = FiniteDifferences.central_fdm(7,1)(x->solve_n(ms,x,rwg_pe(p_jank))[1],om)
    ng_FD = neff + dneff_dom_FD * om
end

using Zygote: Buffer, forwarddiff, dropgrad
function slvn2(ms::ModeSolver{ND,T},ωs::Vector{T},fgeom::Function,pgeom::AbstractVector;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol,wp=nothing) where {ND,T<:Real}
	nω = length(ωs)
	ns = Buffer(ωs,nω)
	ngs = Buffer(ωs,nω)
	# geom = forwarddiff(fgeom,pgeom)
	# Srvol = forwarddiff(x->S_rvol(x;ms=dropgrad(ms)),geom)
	# Srvol = S_rvol(geom;ms=dropgrad(ms))
	Srvol = Zygote.forwarddiff(x->S_rvol(fgeom(x);ms=dropgrad(ms)),pgeom)
	ms_copies = @ignore( [ deepcopy(ms) for om in 1:length(ωs) ] )
	geom = fgeom(pgeom)
	nω = length(ωs)
	n_buff = Buffer(ωs,nω)
	ng_buff = Buffer(ωs,nω)
	for ωind=1:nω
		ωinv = inv(ωs[ωind])
		es = vcat(εs(geom,ωinv),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
		eis = inv.(es)
		ε⁻¹_ω = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)
		# ε⁻¹_ω = εₛ⁻¹(ωs[ωind],geom;ms=dropgrad(ms))
		# @ignore(update_ε⁻¹(ms,ε⁻¹_ω))
		k, H⃗ = solve_k(ms, ωs[ωind], ε⁻¹_ω; nev, eigind, maxiter, tol, log) #ω²_tol)
		(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
		nngs_ω = vcat( nn̂g.(materials(geom), ωinv) ,[εᵥ,]) # = √.(ε̂) .* nĝ (elementwise product of index and group index tensors) for each material, vacuum permittivity tensor appended
		nnginv_ω_sm = real(inv.(εₛ(nngs_ω,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)))  # new spatially smoothed ε⁻¹ tensor array
		ng_ω = ωs[ωind] / H_Mₖ_H(ms.H⃗[:,eigind],nnginv_ω_sm,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
		ns[ωind] = k/ωs[ωind]
		ngs[ωind] = ng_ω
	end
	return ( copy(ns), copy(ngs) )
end

function foo1(ms,om,fgeom,pgeom)
	eism = Zygote.forwarddiff(x->εₛ⁻¹(om,fgeom(x);ms=dropgrad(ms)), pgeom)
	sum(sum(eism))
end

# ng_jank_FD(0.44)
# 2.2626961583653573
#
# julia> ng_jank_AD(0.44)
# 2.2627035013465817
#
# ng calc'd in solve_n without material dispersion:
# julia> solve_n(ms,0.44,rwg_pe(p_jank);eigind=1)
# (1.8155414056027075, 2.1999549264309226)
#
# ng calc'd again with dispersion included
# julia> solve_n(ms,0.44,rwg_pe(p_jank);eigind=1)
# (1.8155414056027075, 2.2599594279797897)

om = 0.48
@show ng_jank_FD(om)
@show ng_jank_AD(om)
@show solve_n(ms,om,rwg_pe(p_jank);eigind=1)[2]



# re-run with material dispersion not incorporated into ng calc.
# n_jankF_old,ng_jankF_old = solve_n(ms,ωs_jank,rwg_pe(p_jank))
# n_jankS_old,ng_jankS_old = solve_n(ms,2*ωs_jank,rwg_pe(p_jank))


k1,H1 = solve_k(ms,ωs_jank[end],rwg_pe(p_jank))
Ex1 = E⃗x(ms)[1]; Ey1 = E⃗y(ms)[1]
k2,H2 = solve_k(ms,2*ωs_jank[end],rwg_pe(p_jank))
Ex2 = E⃗x(ms)[1]; Ey2 = E⃗y(ms)[1]




##
fig = Figure()
ax_n = fig[1,1] = Axis(fig)
ax_ng = fig[1,2] = Axis(fig)
ax_Λ = fig[2,1] = Axis(fig)
ax_qpm = fig[2,2] = Axis(fig)
# xs = λs_jank
xs = ωs_jank
lines!(ax_n, xs, n_jankF; color=logocolors[:red],linewidth=2)
lines!(ax_n, xs, n_jankS; color=logocolors[:blue],linewidth=2)
plot!(ax_n, xs, n_jankF; color=logocolors[:red],markersize=2)
plot!(ax_n, xs, n_jankS; color=logocolors[:blue],markersize=2)

lines!(ax_ng, xs, ng_jankF; color=logocolors[:red],linewidth=2)
lines!(ax_ng, xs, ng_jankS; color=logocolors[:blue],linewidth=2)
plot!(ax_ng, xs, ng_jankF; color=logocolors[:red],markersize=2)
plot!(ax_ng, xs, ng_jankS; color=logocolors[:blue],markersize=2)


lines!(ax_ng, xs, ng_jankF_old; color=logocolors[:red],linestyle=:dash,linewidth=2)
lines!(ax_ng, xs, ng_jankS_old; color=logocolors[:blue],linestyle=:dash,linewidth=2)
plot!(ax_ng, xs, ng_jankF_old; color=logocolors[:red],markersize=2)
plot!(ax_ng, xs, ng_jankS_old; color=logocolors[:blue],markersize=2)

# Δk_jank =  ( n_jankS .* 2π ./ ( (λs_jank/2) ) ) .-  ( n_jankF .* 4π ./ ( λs_jank) )
# Δk_jank = ( 4π ./ λs_jank ) .* ( n_jankS .- n_jankF )
# Λ_jank = 2π ./ Δk_jank
Δn_jank = ( n_jankS .- n_jankF )
Δk_jank = 4π .* ωs_jank .* Δn_jank
Λ_jank = (λs_jank ./ 2) ./ Δn_jank


# lines!(ax_Λ, xs, Δk_jank; color=logocolors[:green],linewidth=2)
# plot!(ax_Λ, xs, Δk_jank; color=logocolors[:green],markersize=2)

lines!(ax_Λ, xs, Λ_jank; color=logocolors[:green],linewidth=2)
plot!(ax_Λ, xs, Λ_jank; color=logocolors[:green],markersize=2)

Λ0_jank = 5.1201 #5.1201
L_jank = 1e4 # 1cm in μm
Δk_qpm_jank = ( 4π ./ λs_jank) .* (  n_jankS .-  n_jankF ) .- (2π / Λ0_jank)

Δk_qpm_jank_itp = LinearInterpolation(ωs_jank,Δk_qpm_jank)
ωs_jank_dense = collect(range(extrema(ωs_jank)...,length=3000))
λs_jank_dense = inv.(ωs_jank_dense)
Δk_qpm_jank_dense = Δk_qpm_jank_itp.(ωs_jank_dense)
sinc2Δk_jank_dense = (sinc.(Δk_qpm_jank_dense * L_jank / 2.0)).^2



lines!(ax_qpm, ωs_jank_dense, sinc2Δk_jank_dense; color=logocolors[:purple],linewidth=2)
# plot!(ax_qpm, λs_jank_dense, sinc2Δk_jank_dense; color=logocolors[:purple],markersize=2)
fig
##

# E_axes = fig[3:4, 1:2] = [Axis(fig, title = t) for t in ["|Eₓ₁|²","|Ey₁|²","|Eₓ₂|²","|Ey₂|²"]]
# Es = [Ex1,Ey1,Ex2,Ey2]

Ex_axes = fig[3, 1:4] = [Axis(fig, title = t) for t in ["|Eₓ₁|²","|Eₓ₂|²","|Eₓ₃|²","|Eₓ₄|²"]]
Ey_axes = fig[4, 1:4] = [Axis(fig, title = t) for t in ["|Ey₁|²","|Ey₂|²","|Ey₃|²","|Ey₄|²"]]
# Es = [Ex[1],Ey[1],Ex[2],Ey[2],Ex[3],Ey[3],Ex[4],Ey[4]]

Earr = [ε⁻¹_dot( fft( kx_tc( unflat(ms.H⃗; ms)[:,:,:,i],mn(ms),ms.M̂.mag), (2:1+ND) ), copy(flat( ms.M̂.ε⁻¹ ))) for i=1:4]
Ex = [Earr[i][1,:,:] for i=1:4]
Ey = [Earr[i][2,:,:] for i=1:4]

heatmaps_x = [heatmap!(ax, abs2.(Ex[i])) for (i, ax) in enumerate(Ex_axes)]
heatmaps_y = [heatmap!(ax, abs2.(Ey[i])) for (i, ax) in enumerate(Ey_axes)]
##
fig
##2# ωs = [0.65, 0.75]

p_lower = [0.4, 0.3, 0.]
p_upper = [2., 1.8, π/4.]

# λs = 0.7:0.05:1.1
# ωs = 1 ./ λs

λs = reverse(1.4:0.01:1.6)
ωs = 1 ./ λs

n1,ng1 = solve_n(ms,ωs,rwg(p))

n_jank,ng_sw = solve_n(ms,ωs,rwg_pe(p_sw))
# function var_ng(ωs,p)
#     ngs = solve_n(dropgrad(ms),ωs,rwg(p))[2]
#     mean( abs2.( ngs ) ) - abs2(mean(ngs))
# end

function var_ng(ωs,p)
    ngs = solve_n(ωs,rwg(p),gr)[2]
    # mean(  ngs.^2  ) - mean(ngs)^2
    var(real(ngs))
end

function sum_Δng_FHSH(ωs,p)
    ngs_FH = solve_n(ωs,rwg_pe(p),gr)[2]
    ngs_SH = solve_n(2*ωs,rwg_pe(p),gr)[2]
    Δng² = abs2.(ngs_SH .- ngs_FH)
    sum(Δng²)
end

sum_Δng_FHSH(ωs,p)

# warmup
println("warmup function runs")
p0 = copy(p)
@show var_ng(ωs,p0)
# @show vng0, vng0_pb = Zygote.pullback(x->var_ng(ωs,x),p0)
# @show grad_vng0 = vng0_pb(1)

# define function that computes value and gradient of function `f` to be optimized
# according to https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/
function fg!(F,G,x)
    value, value_pb = Zygote.pullback(x) do x
       # var_ng(ωs,x)
       sum_Δng_FHSH(ωs,x)
    end
    if G != nothing
        G .= value_pb(1)[1]
    end
    if F != nothing
        # F = value
        return value
    end
end


@show fg!(0.,[0.,0.,0.],p0)
##
opts =  Optim.Options(
                        outer_iterations = 2,
                        iterations = 4,
                        store_trace = true,
                        show_trace = true,
                        show_every = 1,
                        extended_trace = true,
                    )


# BFGS(; alphaguess = LineSearches.InitialStatic(),
#        linesearch = LineSearches.HagerZhang(),
#        initial_invH = nothing,
#        initial_stepnorm = nothing,
#        manifold = Flat())
inner_optimizer = Optim.BFGS(; alphaguess= Optim.LineSearches.InitialStatic(),
       linesearch = Optim.LineSearches.HagerZhang(),
       initial_invH = nothing,
       initial_stepnorm = 0.1,
       manifold = Flat())
     # GradientDescent() #

# results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))
# res2 = optimize( Optim.only_fg!(fg!),
#                 p_lower,
#                 p_upper,
#                 p0,
#                 Fminbox(inner_optimizer),
#                 opts,
#             )

res2 = optimize( Optim.only_fg!(fg!),
                p_lower,
                p_upper,
                p0,
                Fminbox(inner_optimizer),
                opts,
            )
# res2 = optimize( Optim.only_fg!(fg!),
#                 p_lower,
#                 p_upper,
#                 p0,
#                 inner_optimizer,
#                 opts,
#             )
#


##
# first optimization result after a few very inefficient steps:
# Δx,Δy,Δz,Nx,Ny,Nz = 6., 4., 1., 128, 128, 1
# rwg(p) = ridge_wg(p[1],p[2],p[3],0.5,2.4,1.4,Δx,Δy)
using Plots: plot, plot!, heatmap, @layout, cgrad, grid, heatmap!
using LaTeXStrings
cmap_n=cgrad(:viridis)
cmap_e=cgrad(:plasma)

p0 = [
    1.7,                #   top ridge width         `w_top`         [μm]
    0.7,                #   ridge thickness         `t_core`        [μm]
    π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
]
# p_lower = [0.4, 0.3, 0.]
# p_upper = [2., 1.8, π/4.]
p_opt0 = [1.665691811699148, 0.36154202879652847, 0.2010932097703251]
#
# optimized using ωs = collect(0.625:0.025:0.7) => λs0 = [1.6, 1.538, 1.481, 1.428]
# plot with higher density of frequency points,
ωs = collect(0.53:0.02:0.8); λs = 1. ./ ωs
# x = ms.M̂.x; y = ms.M̂.y;
mg = MaxwellGrid(Δx,Δy,Nx,Ny)

nng = solve_n(ms,ωs,rwg(p_opt0))
k,H = solve_k(ms,ωs[14],rwg(p_opt0))
mn = vcat(reshape(ms.M̂.m,(1,3,Nx,Ny,Nz)),reshape(ms.M̂.n,(1,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)))
e = ε⁻¹_dot( fft( kx_tc(reshape(H,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),mn,ms.M̂.mag), (2:4) ), ms.M̂.ε⁻¹ )
enorm = e ./ e[argmax(abs2.(e))]
x = ms.M̂.x; y = ms.M̂.y;

ε⁻¹ = ms.M̂.ε⁻¹
ε = [ inv(ε⁻¹[:,:,ix,iy,1]) for ix=1:Nx, iy=1:Ny ]
n₁ = [ √ε[ix,iy,1][1,1] for ix=1:Nx, iy=1:Ny ]


nng0 = solve_n(ms,ωs,rwg(p0))
k0,H0 = solve_k(ms,ωs[14],rwg(p0))
mn0 = vcat(reshape(ms.M̂.m,(1,3,Nx,Ny,Nz)),reshape(ms.M̂.n,(1,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)))
e0 = ε⁻¹_dot( fft( kx_tc(reshape(H0,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),mn0,ms.M̂.mag), (2:4) ), ms.M̂.ε⁻¹ )
enorm0 = e0 ./ e0[argmax(abs2.(e0))]

ε⁻¹0 = ms.M̂.ε⁻¹
ε0 = [ inv(ε⁻¹0[:,:,ix,iy,1]) for ix=1:Nx, iy=1:Ny ]
n₁0 = [ √ε0[ix,iy,1][1,1] for ix=1:Nx, iy=1:Ny ]

##
pyplot()
ylim_ng = (2.485,2.545)

hm_e = heatmap(
        x,
        y,
        real(enorm[1,:,:,1])',
        c=cmap_e, #cgrad(:cherry),
        aspect_ratio=:equal,
        legend=false,
        colorbar = true,
        clim = (0,1),
        xlabel = "x (μm)",
        ylabel = "y (μm)",
        colorbar_title = "|Eₓ|²",
    )
hm_n = heatmap(
    x,
    y,
    transpose(n₁),
    c=cmap_n, #cgrad(:cherry),
    aspect_ratio=:equal,
    legend=false,
    colorbar = true,
    clim = (1,2.5),
    xlabel = "x (μm)",
    ylabel = "y (μm)",
    colorbar_title = "nₓ",
    title = "optimized params",
)
hm_n0 = heatmap(
        x,
        y,
        transpose(n₁0),
        c=cmap_n, #cgrad(:cherry),
        aspect_ratio=:equal,
        legend=false,
        colorbar = true,
        clim = (1,2.5),
        xlabel = "x (μm)",
        ylabel = "y (μm)",
        colorbar_title = "nₓ",
        title = "intial params",
    )
hm_e0 = heatmap(
    x,
    y,
    real(enorm0[1,:,:,1])',
    c=cmap_e, #cgrad(:cherry),
    aspect_ratio=:equal,
    legend=false,
    colorbar = true,
    clim = (0,1),
    xlabel = "x (μm)",
    ylabel = "y (μm)",
    colorbar_title = "|Eₓ|²",

)

plt_n = plot(
    λs,nng0[1],
    xlabel="λ (μm)",
    ylabel="effective index n",
    label="init. params",
    legend=:bottomleft,
    m=:dot,
    msize=2,
    )
plot!(plt_n,λs,
    nng[1],
    label="opt. params",
    m=:dot,
    msize=2,
    )
plt_ng = plot(λs,
    nng0[2],
    xlabel="λ (μm)",
    ylabel="group index ng",
    legend=false,
    m=:dot,
    msize=2,
    ylim=ylim_ng,
    )
plot!(plt_ng,
    λs,
    nng[2],
    m=:dot,
    msize=2,
    fillalpha=0.2,
    )
annot_str1 = latexstring("\$ \\mathcal{L}(\\vec{p}) = \$")
annot_str2 = latexstring("\$ \\langle n_g^2 \\rangle - \\langle n_g \\rangle^2 \$")
annot_str = "minimize\n" * annot_str1 * "\n" * annot_str2 * "\n here"
plot!(plt_ng,
    [1.428,1.6],
    [2.52,2.52],
    linecolor=nothing,
    fill_between=(ylim_ng[1],ylim_ng[2]),
    fillcolor=:blue,
    fillalpha=0.2,
    annotations = (1.51,2.53, annot_str)
    )

l = @layout [   a   b
                c   d
                e   f   ]
plot(plt_n,
        plt_ng,
        hm_n0,
        hm_n,
        hm_e0,
        hm_e,
        layout=l,
        size= (900,700),
        )



## varng LN/SIO2 opt runs for
# ωs =
# [ 1.4285714285714286
#  1.3333333333333333
#  1.25
#  1.1764705882352942
#  1.1111111111111112
#  1.0526315789473684
#  1.0
#  0.9523809523809523
#  0.9090909090909091 ]
# -------
# Initial mu = 5.21952e-8
#
# Fminbox iteration 1
# -------------------
# Calling inner optimizer with mu = 5.21952e-8
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     8.498314e-05     6.594861e-04
#  * Current step size: 1.0
#  * time: 6.9141387939453125e-6
#  * g(x): [-3.4210966385493146e-5, -0.0006594861073720177, -2.3414479358497828e-5]
#  * ~inv(H): [151.5681288200004 0.0 0.0; 0.0 151.5681288200004 0.0; 0.0 0.0 151.5681288200004]
#  * x: [1.7, 0.7, 0.2243994752564138]
#      1     1.454428e-05     6.436278e-05
#  * Current step size: 4.286116026185353
#  * time: 191.1114239692688
#  * g(x): [7.2123743240460636e-6, 6.43627808472856e-5, 7.141562184584354e-6]
#  * ~inv(H): [152.66420051334478 21.936497319161738 0.7263192198957077; 21.936497319161738 590.0046758005193 14.553935286853264; 0.7263192198957077 14.553935286853264 152.04891021578507]
#  * x: [1.7222247638281714, 1.1284276222760623, 0.23961042451866174]
#      2     9.629194e-06     1.519242e-05
#  * Current step size: 2.8525073343228833
#  * time: 391.3495829105377
#  * g(x): [1.882330618620086e-6, 1.5192415134192208e-5, 1.502213107405214e-6]
#  * ~inv(H): [159.6885813777736 128.05022395038802 6.317162269724115; 128.05022395038804 2192.9678556738795 99.00681657147157; 6.317162269724115 99.00681657147157 156.49815584224172]
#  * x: [1.71504171568567, 1.0193577435257075, 0.23382600465803224]
#      3     9.413559e-06     2.616302e-06
#  * Current step size: 0.6516665843608196
#  * time: 515.9198799133301
#  * g(x): [3.9142554969633425e-7, 2.6163019873609537e-6, 5.018003868000681e-8]
#  * ~inv(H): [157.7079643965072 97.64918583786871 4.572002711613711; 97.64918583786874 1726.6588557119794 72.30389290573649; 4.572002711613711 72.30389290573649 154.9821835548046]
#  * x: [1.7135719021732212, 0.9973925130806568, 0.23268484704229064]
#      4     9.409294e-06     1.897483e-07
#  * Current step size: 0.698292756550739
#  * time: 640.4948480129242
#  * g(x): [1.5369403255934563e-7, -3.374880136407756e-8, -1.8974833175362043e-7]
#  * ~inv(H): [156.36531244037866 69.38382657038773 2.5945301959345226; 69.38382657038775 1191.6961289512915 38.27922552147466; 2.5945301959345226 38.27922552147466 153.03241267121243]
#  * x: [1.7133502360445376, 0.9942087788196526, 0.23254607153151022]
#      5     9.409267e-06     4.172450e-07
#  * Current step size: 11.728402554128527
#  * time: 886.5753040313721
#  * g(x): [3.6255166137383976e-7, 4.1724503250617465e-7, 5.239083403471206e-8]
#  * ~inv(H): [901.9707147706754 -490.7328548394383 -890.7702212105684; -490.73285483943823 888.195523376339 552.3155273802527; -890.7702212105684 552.3155273802527 1189.3707921306986]
#  * x: [1.7131016116594016, 0.9946405934445752, 0.23289711152390843]
#      6     9.409176e-06     2.768810e-07
#  * Current step size: 1.3205394106899584
#  * time: 1071.3093581199646
#  * g(x): [2.7688097875390917e-7, 1.4576299093565252e-7, -3.259230503634351e-8]
#  * ~inv(H): [1428.2474532496853 306.6957935658639 -1245.0210939215647; 306.69579356586394 1040.2219831957698 -188.56113566387057; -1245.0210939215647 -188.56113566387057 1388.386002255681]
#  * x: [1.7130017961257242, 0.9943479415361147, 0.2329369748922578]
#      7     9.408899e-06     5.473291e-07
#  * Current step size: 3.0372569228535458
#  * time: 1258.7134199142456
#  * g(x): [1.107034618256448e-7, -5.473290712635008e-7, -1.9903025112837252e-7]
#  * ~inv(H): [7933.72741698678 1927.4044116081627 -7174.76019051297; 1927.4044116081632 1018.6503381400173 -1737.5763965192405; -7174.76019051297 -1737.5763965192405 6781.237464158446]
#  * x: [1.7115416726550623, 0.9936108304399704, 0.23420490389766824]
#      8     9.384243e-06     3.170912e-06
#  * Current step size: 91.69179048820088
#  * time: 1577.5429310798645
#  * g(x): [-4.0120543779499473e-7, -3.1709120935064932e-6, -5.438774665082602e-7]
#  * ~inv(H): [821171.5279215599 -13337.743566548821 -784790.4750529989; -13337.74356654882 976.6233938899304 12811.474726614591; -784790.4750529989 12811.474726614591 750328.3636313506]
#  * x: [1.5968021469518514, 0.9934582861328269, 0.3435855180133476]
#      9     9.373474e-06     2.602562e-06
#  * Current step size: 0.673751360246807
#  * time: 1703.6139631271362
#  * g(x): [7.949202957466854e-8, -2.602562048423657e-6, 9.140412838158468e-8]
#  * ~inv(H): [663077.3454702945 -18030.98133160457 -633719.9592466753; -18030.981331604573 1473.4699379438102 17324.154779842807; -633719.9592466753 17324.154779842807 605970.7012118072]
#  * x: [1.5027032121131607, 0.9966340024568505, 0.433766156254365]
#     10     9.370656e-06     1.740643e-06
#  * Current step size: 0.4281786206235337
#  * time: 1827.946142911911
#  * g(x): [-7.198694296993818e-8, -1.7406432517445223e-6, 2.0842439616224706e-7]
#  * ~inv(H): [75319.22814722802 2223.603872094107 -71502.61733144848; 2223.6038720941106 2481.4606072395018 -1916.8824482670752; -71502.61733144848 -1916.882448267068 68194.70163517178]
#  * x: [1.484843243367148, 0.9982116766701578, 0.4509252644953674]
#
# Exiting inner optimizer with x = [1.484843243367148, 0.9982116766701578, 0.4509252644953674]
# Current distance to box: 0.334473
# Decreasing barrier term μ.
#
# Fminbox iteration 2
# -------------------
# Calling inner optimizer with mu = 5.21952e-11
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     9.211430e-06     1.730827e-06
#  * Current step size: 0.4281786206235337
#  * time: 5.9604644775390625e-6
#  * g(x): [-1.2503352768109717e-7, -1.730826785527779e-6, 1.6828298332138298e-7]
#  * ~inv(H): [57770.214325904235 0.0 0.0; 0.0 57770.214325904235 0.0; 0.0 0.0 57770.214325904235]
#  * x: [1.484843243367148, 0.9982116766701578, 0.4509252644953674]
#      1     9.210854e-06     9.089252e-07
#  * Current step size: 0.03195307995456166
#  * time: 124.53795409202576
#  * g(x): [3.46270735673455e-8, 9.089251842607265e-7, 1.0725018794008214e-7]
#  * ~inv(H): [57573.75258558891 -3394.283107841412 22.78298675315989; -3394.283107841412 1443.9232308860664 1223.439914099259; 22.78298675315989 1223.4399140992582 58064.76846026388]
#  * x: [1.4850740472917792, 1.0014066726233688, 0.45061462483159415]
#      2     9.210443e-06     7.891384e-07
#  * Current step size: 1.2845908889107083
#  * time: 504.3476300239563
#  * g(x): [4.4830862525931765e-8, -7.891384188304818e-7, 2.2321774953146828e-7]
#  * ~inv(H): [51578.94772390525 -254.69440115038742 3796.3545336404586; -254.69440115038742 2686.113682121294 24664.629878708176; 3796.3545336404623 24664.629878708172 279510.40340807126]
#  * x: [1.486473089259767, 0.9997031788826629, 0.4411853916739758]
#      3     9.210445e-06     8.765670e-07
#  * Current step size: 0.0002164291992870169
#  * time: 1708.3853678703308
#  * g(x): [-4.125993932992022e-8, -8.765669525888612e-7, 1.139295028400189e-7]
#  * ~inv(H): [47168.25695359466 -3449.1804179146807 -34390.4219347625; -3449.1804179146807 717.83458823418 2149.488338486819; -34390.42193476249 2149.4883384868117 25456.534913632902]
#  * x: [1.486472361899325, 0.9997024485532594, 0.4411760639931854]
#      4     9.209873e-06     2.727668e-07
#  * Current step size: 3.907571077377182
#  * time: 1901.2678380012512
#  * g(x): [4.095064529348949e-9, 2.727668323118462e-7, 1.230180239926332e-7]
#  * ~inv(H): [199025.49852705954 3065.3652332564884 -159462.74931989083; 3065.3652332564884 725.6190879366986 -3000.5559948438868; -159462.7493198908 -3000.555994843894 128297.47237122593]
#  * x: [1.4975730114085066, 1.0006481861213496, 0.4316610242954845]
#      5     9.209785e-06     3.367705e-07
#  * Current step size: 0.1567321746747203
#  * time: 2424.692887067795
#  * g(x): [-3.926326614099638e-8, 3.367704988844189e-7, 4.996622313671564e-8]
#  * ~inv(H): [228511.16307637878 -3449.0875124577783 -177194.95629044567; -3449.0875124577783 608.2687192005525 2239.6896687880444; -177194.95629044564 2239.689668788037 137837.65381068946]
#  * x: [1.5003888052504515, 1.0006730508628525, 0.42941796131760473]
#      6     9.209783e-06     1.117497e-06
#  * Current step size: 0.002244536104041773
#  * time: 6979.014787912369
#  * g(x): [7.665025781492648e-7, 1.1174974474350182e-6, 1.0292884497705933e-6]
#  * ~inv(H): [79233.81848158318 -2154.256576303677 -63431.01266280713; -2154.256576303677 631.2575573732286 1268.1951030574473; -63431.0126628071 1268.1951030574392 51145.23148335033]
#  * x: [1.500431423173811, 1.0006720359324792, 0.42938519391612834]
#      7     9.209783e-06     1.641101e-07
#  * Current step size: 1.594401501850564e-14
#  * time: 11317.735109090805
#  * g(x): [-6.654037667299105e-8, 1.6411014950070306e-7, -1.0963888658749017e-7]
#  * ~inv(H): [292.3343136044714 256.84440278059174 -428.8238731009624; 256.84440278059174 592.7234467979533 -684.0273305565588; -428.823873100926 -684.027330556567 886.2476494475777]
#  * x: [1.500431423173811, 1.0006720359324792, 0.4293851939161282]
#      8     9.209783e-06     4.363236e-07
#  * Current step size: 3.5772702349138456e-12
#  * time: 15351.74334692955
#  * g(x): [3.668932720505993e-8, 4.3632358698628084e-7, 1.52382136133064e-7]
#  * ~inv(H): [292.3343136044714 256.84440278059174 -428.8238731009624; 256.84440278059174 592.7234467979533 -684.0273305565588; -428.823873100926 -684.027330556567 886.2476494475777]
#  * x: [1.5004314231738107, 1.0006720359324788, 0.4293851939161289]
#      9     9.209783e-06     3.558985e-07
#  * Current step size: 2.9437838402227183e-12
#  * time: 19532.71118593216
#  * g(x): [-6.204956552365209e-8, 3.5589847414234937e-7, 4.0082704206458306e-8]
#  * ~inv(H): [292.3343136044714 256.84440278059174 -428.8238731009624; 256.84440278059174 592.7234467979533 -684.0273305565588; -428.823873100926 -684.027330556567 886.2476494475777]
#  * x: [1.5004314231738105, 1.0006720359324783, 0.4293851939161294]
#     10     9.209783e-06     3.591572e-07
#  * Current step size: 1.5308121605229496e-13
#  * time: 23449.355585098267
#  * g(x): [-3.5915720040819217e-7, 6.926503689149826e-8, -2.9118083651784196e-7]
#  * ~inv(H): [21.25224436427237 -124.78306361935245 88.91046178055115; -124.78306361935233 732.6667574479275 -522.0399134870167; 88.91046178058753 -522.0399134870249 371.9640184348657]
#  * x: [1.5004314231738105, 1.0006720359324783, 0.4293851939161294]
#
# Exiting inner optimizer with x = [1.5004314231738105, 1.0006720359324783, 0.4293851939161294]
# Current distance to box: 0.356013
# Decreasing barrier term μ.

#######################

"""
first attempt at broadband phase matching, tried minimizing Δng for FH-SH pairs
with FH wavelenghts in the 1.4-1.6μm range. ωs below = 1 ./ λs.
This was for a fully etched, unclad, MgO:LiNbO₃-on-SiO₂ waveguide with the
MgO:LiNbO₃ dielectric tensor model modified to be isotropic (matching extraordinary disp.)
as a quick hack. Need to figure out symmetry constraints soon I guess.
"""

# ωs = [   0.625,
#          0.6289308176100629,
#          0.6329113924050632,
#          0.6369426751592356,
#          0.641025641025641,
#          0.6451612903225806,
#          0.6493506493506493,
#          0.6535947712418301,
#          0.6578947368421053,
#          0.6622516556291391,
#          0.6666666666666666,
#          0.6711409395973155,
#          0.6756756756756757,
#          0.6802721088435374,
#          0.684931506849315,
#          0.6896551724137931,
#          0.6944444444444444,
#          0.6993006993006994,
#          0.7042253521126761,
#          0.7092198581560284,
#          0.7142857142857143,
#     ]

# -------
# Initial mu = 4.18104e-6
#
# Fminbox iteration 1
# -------------------
# Calling inner optimizer with mu = 4.18104e-6
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     6.131435e-03     2.147646e-02
#  * Current step size: 1.0
#  * time: 9.059906005859375e-6
#  * g(x): [-0.019815485241852157, 0.02147645734071707, -0.016183598733926382]
#  * ~inv(H): [4.6611850827316985 0.0 0.0; 0.0 4.6611850827316985 0.0; 0.0 0.0 4.6611850827316985]
#  * x: [1.7, 0.7, 0.2243994752564138]
#      1     1.834491e-03     9.487007e-03
#  * Current step size: 1.0556385968042492
#  * time: 1748.6312410831451
#  * g(x): [-0.009487006977172998, 0.005487197740983217, -0.0070639665171882806]
#  * ~inv(H): [6.380765593492846 -1.2270633729170202 1.313561109959477; -1.2270633729170202 5.301081879230213 -0.9037002520663582; 1.313561109959477 -0.9037002520663577 5.659794831354706]
#  * x: [1.7975026277763393, 0.594324514363967, 0.30403130784474214]
#      2     7.125945e-05     3.282646e-03
#  * Current step size: 1.1018768906302832
#  * time: 3510.7914052009583
#  * g(x): [-0.0006356687715907745, 0.0032826460719002286, -0.00048762113229897965]
#  * ~inv(H): [7.344123220646283 -2.6953340729335 2.0372184728519143; -2.695334072933501 6.743532590487835 -2.005504938922271; 2.0372184728519143 -2.00550493892227 6.203391939114847]
#  * x: [1.881847401756922, 0.5424118417300735, 0.36728030475652684]
#      3     6.007662e-05     2.053307e-03
#  * Current step size: 0.13182851360207842
#  * time: 4679.663056135178
#  * g(x): [-0.0003417010730138389, 0.002053307190326052, -0.0002703311990855169]
#  * ~inv(H): [5.66704805188807 -0.06121791445110247 0.7897224650368133; -0.06121791445110247 2.6393211722869943 -0.047901609092789954; 0.7897224650368133 -0.047901609092789066 5.275536116280067]
#  * x: [1.883760187134629, 0.5391388169651558, 0.36871766585300764]
#      4     5.153350e-05     6.574411e-04
#  * Current step size: 1.082717372119829
#  * time: 6433.112316131592
#  * g(x): [7.182301736507736e-5, 0.0006574411035336189, 1.9404397052517996e-5]
#  * ~inv(H): [5.3829469031192705 -0.04395107284288757 0.6092715646463928; -0.04395107284288757 4.188025475209889 -0.13868460921177173; 0.6092715646463929 -0.13868460921177084 5.167600644308727]
#  * x: [1.8862240428283517, 0.5332345375383438, 0.37066043731861265]
#
# Exiting inner optimizer with x = [1.8862240428283517, 0.5332345375383438, 0.37066043731861265]
# Current distance to box: 0.113776
# Decreasing barrier term μ.
#
# Fminbox iteration 2
# -------------------
# Calling inner optimizer with mu = 4.18104e-9
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     3.119590e-05     6.720872e-04
#  * Current step size: 1.082717372119829
#  * time: 5.0067901611328125e-6
#  * g(x): [3.790852362393901e-5, 0.0006720872473517286, 2.062356376504414e-5]
#  * ~inv(H): [148.797951312084 0.0 0.0; 0.0 148.797951312084 0.0; 0.0 0.0 148.797951312084]
#  * x: [1.8862240428283517, 0.5332345375383438, 0.37066043731861265]
#      1     3.063802e-05     5.404484e-04
#  * Current step size: 0.0091018475420365
#  * time: 2317.683361053467
#  * g(x): [8.118640004120912e-5, 0.0005404484413058811, 4.90612958082445e-5]
#  * ~inv(H): [155.05675354997553 52.17153991580341 3.725026937517393; 52.17153991580341 31.423898943826828 34.05682794508624; 3.725026937517393 34.05682794508624 150.99859813200624]
#  * x: [1.886172701939964, 0.5323243054043917, 0.3706325060782424]
#      2     2.960157e-05     2.652610e-04
#  * Current step size: 0.08678963856739261
#  * time: 4657.894564151764
#  * g(x): [7.016093581671254e-5, -0.0002652609816718826, 4.194837581484669e-5]
#  * ~inv(H): [74.96866671582583 3.8172258343801886 -48.728736690259986; 3.8172258343801886 2.3917512989112737 2.4500790468621716; -48.72873669025997 2.4500790468621716 116.66835360184209]
#  * x: [1.8826171663681932, 0.5303377349423279, 0.36836585858129717]
#      3     2.951907e-05     1.981138e-04
#  * Current step size: 0.5771086529739968
#  * time: 5819.558609008789
#  * g(x): [2.03627959740178e-5, 0.00019811382557069363, 8.954244435031253e-6]
#  * ~inv(H): [109.71393228089983 6.4107101800323285 -37.02146798977759; 6.4107101800323285 1.2723269380638902 3.5781422807809715; -37.02146798977758 3.5781422807809715 120.56385444228759]
#  * x: [1.8813456693984658, 0.5304900000534588, 0.3678895801579516]
#      4     2.870806e-05     3.363244e-03
#  * Current step size: 25.277066111779902
#  * time: 8749.481281995773
#  * g(x): [-0.0003379479478639439, 0.0033632440690423483, -0.0002342573820436077]
#  * ~inv(H): [15622.40035903521 2124.3691037457415 4960.395556114755; 2124.3691037457415 289.91030813541767 686.2361987979814; 4960.395556114755 686.2361987979813 1730.2421140645429]
#  * x: [1.8011508036004562, 0.5200089884193593, 0.3417385069329951]
#
# Exiting inner optimizer with x = [1.8011508036004562, 0.5200089884193593, 0.3417385069329951]
# Current distance to box: 0.198849
# Decreasing barrier term μ.

################################################################################

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
p_pe = [
       1.7,                #   top ridge width         `w_top`         [μm]
       0.7,                #   ridge thickness         `t_core`        [μm]
       0.5,                #   top layer etch fraction `etch_frac`     [1]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
               ];
rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).

geom_pe = rwg_pe(p_pe)
ms = ModeSolver(1.45, rwg_pe(p_pe), gr)
p_pe_lower = [0.4, 0.3, 0., 0.]
p_pe_upper = [2., 2., 1., π/4.]


# λs = reverse(1.45:0.02:1.65)
# ωs = 1 ./ λs

ωs = collect(range(0.6,0.7,length=10))
λs = inv.(ωs)

n1F,ng1F = solve_n(ms,ωs,rwg_pe(p_pe)); n1S,ng1S = solve_n(ms,2*ωs,rwg_pe(p_pe))
##

function sum_Δng_FHSH(ωs,p)
    # ms = ModeSolver(1.45, rwg_pe(p), gr)
	nω = length(ωs)
    ngs_FHSH = solve_n(vcat(ωs, 2*ωs ),rwg_pe(p),gr)[2]
    ngs_FH = ngs_FHSH[1:nω]
	ngs_SH = ngs_FHSH[nω+1:2*nω]
    Δng² = abs2.(ngs_SH .- ngs_FH)
    sum(Δng²)
end

# sum_Δng_FHSH(ωs,p_pe)

# warmup
println("warmup function runs")
p0 = copy(p_pe)
@show sum_Δng_FHSH(ωs,p0)
# @show vng0, vng0_pb = Zygote.pullback(x->var_ng(ωs,x),p0)
# @show grad_vng0 = vng0_pb(1)

# define function that computes value and gradient of function `f` to be optimized
# according to https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/
function fg!(F,G,x)
    value, value_pb = Zygote.pullback(x) do x
       # var_ng(ωs,x)
       sum_Δng_FHSH(ωs,x)
    end
    if G != nothing
        G .= value_pb(1)[1]
    end
    if F != nothing
        # F = value
        return value
    end
end

# G0 = [0.,0.,0.,0.]
# @show fg!(0.,G0,p0)
# println("G0 = $G0")
##
rand_p0() = p_pe_lower .+ [rand()*(p_pe_upper[i]-p_pe_lower[i]) for i=1:4]

opts =  Optim.Options(
                        outer_iterations = 2,
                        iterations = 3,
                        # time_limit = 3*3600,
                        store_trace = true,
                        show_trace = true,
                        show_every = 1,
                        extended_trace = true,
                        x_tol = 1e-3, # Absolute tolerance in changes of the input vector x, in infinity norm. Defaults to 0.0.
                        f_tol = 1e-4, # Relative tolerance in changes of the objective value. Defaults to 0.0.
                        g_tol = 1e-4, # Absolute tolerance in the gradient, in infinity norm. Defaults to 1e-8. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
                    )

println("########################### Opt 1 ##########################")

# res1 = optimize( Optim.only_fg!(fg!),
#                 rand_p0(),
#                 Optim.BFGS(),
#                 opts,
#             )
res4 = optimize( Optim.only_fg!(fg!),
                p_pe_lower,
                p_pe_upper,
                rand_p0(),
                Fminbox(Optim.BFGS()),
                opts,
            )


println("########################### Opt 2 ##########################")

# res2 = optimize( Optim.only_fg!(fg!),
#                 rand_p0(),
#                 Optim.BFGS(),
#                 opts,
#             )

res5 = optimize( Optim.only_fg!(fg!),
                p_pe_lower,
                p_pe_upper,
                rand_p0(),
                Fminbox(Optim.BFGS();mu0=1e-6),
                opts,
            )

println("########################### Opt 3 ##########################")

res3 = optimize( Optim.only_fg!(fg!),
                rand_p0(),
                Optim.BFGS(),
                opts,
            )

##
# ########################### Opt 1 ##########################
# Iter     Function value   Gradient norm
#      0     4.196073e-03     5.849341e-03
#  * Current step size: 1.0
#  * time: 0.018551111221313477
#  * g(x): [4.0300506162195305e-5, 0.005082681974533791, -0.005849340901323757, 2.4797956471422756e-6]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [1.3953487636667194, 1.696287815825243, 0.02471676311871973, 0.4470011635893353]
#      1     1.651162e-04     2.571512e-04
#  * Current step size: 76.46315253862139
#  * time: 1520.1866681575775
#  * g(x): [7.917883779174499e-5, 0.00025715119468382684, -8.052757895125927e-5, 7.921354740479212e-5]
#  * ~inv(H): [1.0022121101605432 0.27226156112737865 -0.3131798223789001 0.00018753393293197143; 0.27226156112737865 34.48887516966724 -38.52146259551165 0.023237656322715138; -0.3131798223789001 -38.52146259551165 45.31031857339175 -0.026733598294606576; 0.00018753393293197143 0.023237656322715138 -0.026733598294606576 1.0000147032766795]
#  * x: [1.392267259916656, 1.3076499287011645, 0.47197580870703526, 0.44681155059650324]
#      2     1.643483e-04     1.358986e-04
#  * Current step size: 0.33422291799156045
#  * time: 2117.7677550315857
#  * g(x): [4.169370053357093e-5, 0.0001358986117504111, 0.00010224434898492934, 3.9769555642456814e-5]
#  * ~inv(H): [1.000800286175414 0.06678078568463375 -0.06967301943168669 0.00022443469576099043; 0.06678078568463375 12.920162863415804 -13.354271743682553 -0.028510575236902248; -0.06967301943168658 -13.354271743682546 15.969972151100677 0.03728348103282321; 0.00022443469576099043 -0.028510575236902248 0.037283481032823154 1.000405049625367]
#  * x: [1.392208904313119, 1.3036411608732554, 0.4765150523416733, 0.4467823535695984]
#      3     1.642274e-04     1.805282e-04
#  * Current step size: 6.271698286781552
#  * time: 3310.6307611465454
#  * g(x): [2.618050635386976e-5, 8.165092961758586e-5, 0.00018052815109906642, 2.3694447739787128e-5]
#  * ~inv(H): [1.6161468968116162 3.784707672059781 -0.43953843174767726 0.5725836572231778; 3.784707672059781 30.091517953659427 -9.097556525024924 3.466997603657071; -0.43953843174767715 -9.097556525024917 8.230107302348861 -0.35250828702579684; 0.5725836572231777 3.46699760365707 -0.3525082870257969 1.5325168875186455]
#  * x: [1.3919349080100405, 1.3011821291920767, 0.47766534304145514, 0.4465331633208667]
#      4     1.274098e-04     7.147653e-04
#  * Current step size: 189.88225543968844
#  * time: 5667.0069460868835
#  * g(x): [1.2034928965232877e-5, 0.0005799880098807759, -0.000714765288893353, -9.590790443718644e-6]
#  * ~inv(H): [170.27967300064137 618.3849676402906 396.1194573547092 160.25193667077914; 618.3849676402906 2265.531548092738 1440.717401933161 585.3668430543764; 396.11945735470914 1440.717401933161 935.0864573788921 375.0397695208205; 160.25193667077914 585.3668430543764 375.0397695208205 152.7059165074372]
#  * x: [1.3377131874659247, 1.1120841800675583, 0.34036484899217845, 0.39512281240837]
#      5     1.220156e-04     1.015985e-03
#  * Current step size: 0.14336230485068682
#  * time: 6848.895653009415
#  * g(x): [9.283534528106245e-5, 0.0005378680065080088, -0.0010159848888717162, 5.72387353101034e-5]
#  * ~inv(H): [18.921755244135767 70.25204924991021 35.19190292326056 16.908031928545256; 70.2520492499101 283.48736916542157 130.03256847644116 66.22666919184144; 35.1919029232605 130.03256847644138 78.78518483834 33.258066660263296; 16.908031928545256 66.22666919184144 33.25806666026324 16.951994752456386]
#  * x: [1.3268125214499205, 1.0710776245823201, 0.3162223761376952, 0.38481443313198]
# ########################### Opt 2 ##########################
# Iter     Function value   Gradient norm
#      0     2.794650e-03     7.826270e-03
#  * Current step size: 1.0
#  * time: 2.8133392333984375e-5
#  * g(x): [6.5097528102344965e-6, 0.004930386279113436, -0.007826269921070187, 4.853541552814327e-7]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [0.4858236499543672, 1.457695458268576, 0.034171494389373835, 0.5989537685691309]
#      1     1.053719e-04     3.193261e-03
#  * Current step size: 50.184481136888806
#  * time: 1499.7813239097595
#  * g(x): [0.0007960587139937687, 8.475228445064241e-5, 0.0031932612493225316, 0.0008237424518389079]
#  * ~inv(H): [1.0001131643098313 0.05007440201671428 -0.07928910795110679 5.3621349580590746e-5; 0.05007440201671427 11.936559475419413 -17.21120971033654 0.03795514533194962; -0.07928910795110679 -17.21120971033654 28.083795757597503 -0.06023359730769429; 5.3621349580590746e-5 0.03795514533194962 -0.06023359730769429 1.0000073667326796]
#  * x: [0.4854969613872562, 1.2102665810468323, 0.4269287896155209, 0.5989294113226804]
#      2     8.657558e-05     1.252427e-03
#  * Current step size: 0.31888611860072325
#  * time: 2111.6649010181427
#  * g(x): [-0.00023697392002484637, 0.00043515813938049065, -0.001252426818261997, -0.00025997501304827445]
#  * ~inv(H): [0.9996614914752149 0.1148940563938892 -0.18385371656728328 -0.0005148479479569493; 0.11489405639388918 3.5513096271634677 -3.640002564550631 0.11752469469331014; -0.18385371656728328 -3.6400025645506275 6.12170359068271 -0.18868966033317527; -0.0005148479479569493 0.11752469469331014 -0.18868966033317525 0.9992963115854097]
#  * x: [0.4853224522172848, 1.2274472438261546, 0.3988325399680015, 0.5987270250244889]
#      3     7.779650e-05     1.303005e-04
#  * Current step size: 1.1158207304422603
#  * time: 3024.395863056183
#  * g(x): [0.00010655076932939876, 0.00013030046597319435, 0.00011394158160179808, 0.0001018826568500223]
#  * ~inv(H): [1.000289521646925 0.1600145641320593 -0.25130458642644593 -1.8388836394869746e-5; 0.16001456413205928 3.960909963084772 -4.136471155872786 0.15958688935406395; -0.25130458642644593 -4.1364711558727825 6.685961094992036 -0.25183020250345334; -1.8388836394869746e-5 0.15958688935406395 -0.25183020250345334 0.9996743260949317]
#  * x: [0.48527391339696485, 1.2207005032492637, 0.40905160488333975, 0.5986960144111407]
#      4     7.751930e-05     2.498477e-04
#  * Current step size: 14.924097812103566
#  * time: 4236.987639904022
#  * g(x): [4.990664255684426e-5, 0.0002308869264370238, -0.00024984770002419945, 5.164965236126993e-5]
#  * ~inv(H): [6.261012663840258 7.1485996335551265 4.361990519906382 5.016495797336206; 7.148599633555127 11.722670441073815 4.39101414431019 6.816259671133317; 4.361990519906382 4.391014144310193 6.95073467735101 4.1591207490542015; 5.016495797336207 6.816259671133317 4.1591207490542015 5.783279129628687]
#  * x: [0.4837994770827205, 1.2195349139084013, 0.40650867240366934, 0.5972939277851915]
#      5     5.519917e-05     5.054679e-04
#  * Current step size: 27.146088281594196
#  * time: 5736.533575057983
#  * g(x): [0.00025078519657694727, 8.968518565836075e-5, 0.0005054678726761311, 0.00024574413420107066]
#  * ~inv(H): [1057.3800390436083 2109.144303052077 -187.39204813210785 1010.9190680617553; 2109.144303052077 4212.829260841755 -375.35834545487376 2018.3826225663738; -187.39204813210787 -375.35834545487376 36.183564920334184 -179.33553651107886; 1010.9190680617553 2018.3826225663738 -179.33553651107886 968.4144739800007]
#  * x: [0.4530632837741969, 1.156600872107478, 0.4143888525596142, 0.5678757306129366]
# ########################### Opt 3 ##########################
# Iter     Function value   Gradient norm
#      0     1.162767e-03     5.674435e-03
#  * Current step size: 1.0
#  * time: 2.6941299438476562e-5
#  * g(x): [4.483601189965165e-5, 0.0034132395844401935, -0.0056744346283534326, 7.936733878288571e-6]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [0.6382558898028023, 1.1437215614013543, 0.07316373932835196, 0.24018205347692223]
#      1     5.193007e-04     2.615410e-03
#  * Current step size: 27.337992714591568
#  * time: 1529.8767039775848
#  * g(x): [-0.0003069313529873534, 0.0018381310835620095, -0.002615410273288798, -8.797435402036471e-5]
#  * ~inv(H): [1.0010755927505202 0.13155198803329124 -0.21783425332434364 0.00012409831420528695; 0.13155198803329127 14.795930915790848 -22.86934954478074 0.01823971143166509; -0.21783425332434364 -22.869349544780736 38.90993051014816 -0.030169464106457173; 0.00012409831420528695 0.01823971143166509 -0.030169464106457173 1.0000102313112789]
#  * x: [0.6370301632361384, 1.0504104425087728, 0.22829139185770422, 0.23996507910397993]
#      2     2.275973e-05     2.936016e-03
#  * Current step size: 1.3881774989823086
#  * time: 3934.581979036331
#  * g(x): [-0.00129856680501709, 9.741373169738364e-5, -0.002936016437182231, -0.0007289415496054012]
#  * ~inv(H): [0.9940014411572127 -0.23024705551897653 0.36527757335245065 -0.0031128657362764424; -0.23024705551897662 100.41483169404867 -166.93235408004202 -0.49846148837016374; 0.36527757335245065 -166.93235408004205 281.23492280543496 0.8230723889920634; -0.0031128657362764424 -0.49846148837016374 0.8230723889920634 0.999713317669812]
#  * x: [0.6363301562157482, 0.9296839628709647, 0.42781800284642346, 0.23993118097118646]
#      3     1.303306e-05     2.236529e-03
#  * Current step size: 0.004347680226574619
#  * time: 4790.511950016022
#  * g(x): [-0.00094150996058909, 0.00012063449435226322, -0.0022365288042529136, -0.0005258476184443912]
#  * ~inv(H): [0.991318699067819 0.2824924796128333 -0.49869212320932405 -0.006526982695011344; 0.2824924796128332 3.3751343170482357 -3.411250517069192 0.1507309499019201; -0.49869212320932405 -3.4112505170692202 5.6864145378535795 -0.2708510923511046; -0.006526982695011344 0.1507309499019201 -0.2708510923511046 0.9953800237861536]
#  * x: [0.6363405184696823, 0.9275076869288639, 0.43148329800378926, 0.23994504920751902]
#      4     2.101732e-06     1.264689e-04
#  * Current step size: 0.688322851644425
#  * time: 5389.005700826645
#  * g(x): [-2.3038115549164795e-5, 3.985458860338732e-5, -0.00012646894191238107, -1.316887856417867e-5]
#  * ~inv(H): [0.9914854135974992 0.27808947961381036 -0.49097573516695686 -0.006427603337579637; 0.27808947961381025 2.861922928798074 -2.5567185810014434 0.14923457227042486; -0.49097573516695686 -2.556718581001472 4.264285834117031 -0.2681481947271601; -0.006427603337579637 0.14923457227042486 -0.2681481947271601 0.9954372420642749]
#  * x: [0.6361894223771479, 0.9222135985572288, 0.4400993066019841, 0.23987162124817837]
#      5     2.061630e-06     4.083142e-05
#  * Current step size: 0.9165743556237453
#  * time: 5983.192487001419
#  * g(x): [4.0831423896805585e-5, 2.6809359514079602e-5, 3.371594756640858e-5, 2.221460688616477e-5]
#  * ~inv(H): [1.0229168678061211 0.4163982420552575 -0.6647469068400793 0.010412125138561308; 0.41639824205525744 2.939830144882044 -2.4309340744378103 0.22362470225008602; -0.6647469068400793 -2.4309340744378387 3.7309680557100355 -0.36173410406038287; 0.010412125138561308 0.223624702250086 -0.36173410406038287 1.0044591435049124]
#  * x: [0.636143209603401, 0.9218203566788058, 0.440673406907343, 0.2398469659583548]

##
p_opt1 = [1.3268125214499205, 1.0710776245823201, 0.3162223761376952, 0.38481443313198]

ωs_opt1 = collect(0.58:0.01:0.72)
λs_opt1 = inv.(ωs_opt1)

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy)

ms = ModeSolver(kguess(1/1.55,rwg_pe(p_opt1)), rwg_pe(p_opt1), gr; nev=1)

n_opt1F,ng_opt1F = solve_n(ms,ωs_opt1,rwg_pe(p_opt1))
n_opt1S,ng_opt1S = solve_n(ms,2*ωs_opt1,rwg_pe(p_opt1))
k1,H1 = solve_k(ms,2*ωs_opt1[end],rwg_pe(p_opt1))
# Ex1 = E⃗x(ms); Ey1 = E⃗y(ms)
# k2,H2 = solve_k(ms,2*ωs_opt1[end],rwg_pe(p_opt1))
# Ex2 = E⃗x(ms); Ey2 = E⃗y(ms)

##
fig = Figure()
ax_n = fig[1,1] = Axis(fig)
ax_ng = fig[1,2] = Axis(fig)
ax_Λ = fig[2,1] = Axis(fig)
ax_qpm = fig[2,2] = Axis(fig)

lines!(ax_n, λs_opt1, n_opt1F; color=logocolors[:red],linewidth=2)
lines!(ax_n, λs_opt1, n_opt1S; color=logocolors[:blue],linewidth=2)
plot!(ax_n, λs_opt1, n_opt1F; color=logocolors[:red],markersize=2)
plot!(ax_n, λs_opt1, n_opt1S; color=logocolors[:blue],markersize=2)

lines!(ax_ng, λs_opt1, ng_opt1F; color=logocolors[:red],linewidth=2)
lines!(ax_ng, λs_opt1, ng_opt1S; color=logocolors[:blue],linewidth=2)
plot!(ax_ng, λs_opt1, ng_opt1F; color=logocolors[:red],markersize=2)
plot!(ax_ng, λs_opt1, ng_opt1S; color=logocolors[:blue],markersize=2)

# Δk_opt1 = ( 4π ./ λs_opt1 ) .* ( n_opt1S .- n_opt1F )
# Δk_opt1 = ( 2 * ωs_opt1 ) .* ( n_opt1S .- n_opt1F )
Λ_opt1 = λs_opt1 ./ (2*( n_opt1S .- n_opt1F ))  #2π ./ Δk_opt1

lines!(ax_Λ, λs_opt1, Λ_opt1; color=logocolors[:green],linewidth=2)
plot!(ax_Λ, λs_opt1, Λ_opt1; color=logocolors[:green],markersize=2)

# Λ0_opt1 = 2.8548 # 128x128
Λ0_opt1 = 6.84 # 128x128
L_opt1 = 1e3 # 1cm in μm
Δk_qpm_opt1 = ( 4π ./ λs_opt1 ) .* ( n_opt1S .- n_opt1F ) .- (2π / Λ0_opt1)
Δk_qpm_opt1_itp = LinearInterpolation(ωs_opt1,Δk_qpm_opt1)
ωs_opt1_dense = collect(range(extrema(ωs_opt1)...,length=3000))
λs_opt1_dense = inv.(ωs_opt1_dense)
Δk_qpm_opt1_dense = Δk_qpm_opt1_itp.(ωs_opt1_dense)
sinc2Δk_opt1_dense = (sinc.(Δk_qpm_opt1_dense * L_opt1 / 2.0)).^2



lines!(ax_qpm, λs_opt1_dense, sinc2Δk_opt1_dense; color=logocolors[:purple],linewidth=2)
# plot!(ax_qpm, λs_sw_dense, sinc2Δk_sw_dense; color=logocolors[:purple],markersize=2)


fig

##
Ex_axes = fig[3, 1:2] = [Axis(fig, title = t) for t in ["|Eₓ₁|²","|Eₓ₂|²"]] #,"|Eₓ₃|²","|Eₓ₄|²"]]
Ey_axes = fig[4, 1:2] = [Axis(fig, title = t) for t in ["|Ey₁|²","|Ey₂|²"]] #,"|Ey₃|²","|Ey₄|²"]]
# Es = [Ex[1],Ey[1],Ex[2],Ey[2],Ex[3],Ey[3],Ex[4],Ey[4]]

Earr = ε⁻¹_dot( fft( kx_tc( unflat(ms.H⃗; ms)[:,:,:,1],mn(ms),ms.M̂.mag), (2:1+ND) ), copy(flat( ms.M̂.ε⁻¹ )))
Ex = Earr[1,:,:] # for i=1:2
Ey = Earr[2,:,:] #for i=1:2

heatmaps_x = [heatmap!(ax, abs2.(Ex)) for (i, ax) in enumerate(Ex_axes)]
heatmaps_y = [heatmap!(ax, abs2.(Ey)) for (i, ax) in enumerate(Ey_axes)]

fig
##

p_opt2 = [0.4530632837741969, 1.156600872107478, 0.4143888525596142, 0.5678757306129366]

ωs_opt2 = collect(0.58:0.01:0.72)
λs_opt2 = inv.(ωs_opt2)

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy)

ms = ModeSolver(kguess(1/1.55,rwg_pe(p_opt2)), rwg_pe(p_opt2), gr; nev=1)

n_opt2F,ng_opt2F = solve_n(ms,ωs_opt2,rwg_pe(p_opt2))
n_opt2S,ng_opt2S = solve_n(ms,2*ωs_opt2,rwg_pe(p_opt2))
k1,H1 = solve_k(ms,2*ωs_opt2[end],rwg_pe(p_opt2))
# Ex1 = E⃗x(ms); Ey1 = E⃗y(ms)
# k2,H2 = solve_k(ms,2*ωs_opt2[end],rwg_pe(p_opt2))
# Ex2 = E⃗x(ms); Ey2 = E⃗y(ms)

##
fig = Figure()
ax_n = fig[1,1] = Axis(fig)
ax_ng = fig[1,2] = Axis(fig)
ax_Λ = fig[2,1] = Axis(fig)
ax_qpm = fig[2,2] = Axis(fig)

lines!(ax_n, λs_opt2, n_opt2F; color=logocolors[:red],linewidth=2)
lines!(ax_n, λs_opt2, n_opt2S; color=logocolors[:blue],linewidth=2)
plot!(ax_n, λs_opt2, n_opt2F; color=logocolors[:red],markersize=2)
plot!(ax_n, λs_opt2, n_opt2S; color=logocolors[:blue],markersize=2)

lines!(ax_ng, λs_opt2, ng_opt2F; color=logocolors[:red],linewidth=2)
lines!(ax_ng, λs_opt2, ng_opt2S; color=logocolors[:blue],linewidth=2)
plot!(ax_ng, λs_opt2, ng_opt2F; color=logocolors[:red],markersize=2)
plot!(ax_ng, λs_opt2, ng_opt2S; color=logocolors[:blue],markersize=2)

# Δk_opt2 = ( 4π ./ λs_opt2 ) .* ( n_opt2S .- n_opt2F )
# Δk_opt2 = ( 2 * ωs_opt2 ) .* ( n_opt2S .- n_opt2F )
Λ_opt2 = λs_opt2 ./ (2*( n_opt2S .- n_opt2F ))  #2π ./ Δk_opt2

lines!(ax_Λ, λs_opt2, Λ_opt2; color=logocolors[:green],linewidth=2)
plot!(ax_Λ, λs_opt2, Λ_opt2; color=logocolors[:green],markersize=2)

# Λ0_opt2 = 2.8548 # 128x128
Λ0_opt2 = 5.86 # 128x128
L_opt2 = 1e3 # 1cm in μm
Δk_qpm_opt2 = ( 4π ./ λs_opt2 ) .* ( n_opt2S .- n_opt2F ) .- (2π / Λ0_opt2)
Δk_qpm_opt2_itp = LinearInterpolation(ωs_opt2,Δk_qpm_opt2)
ωs_opt2_dense = collect(range(extrema(ωs_opt2)...,length=3000))
λs_opt2_dense = inv.(ωs_opt2_dense)
Δk_qpm_opt2_dense = Δk_qpm_opt2_itp.(ωs_opt2_dense)
sinc2Δk_opt2_dense = (sinc.(Δk_qpm_opt2_dense * L_opt2 / 2.0)).^2



lines!(ax_qpm, λs_opt2_dense, sinc2Δk_opt2_dense; color=logocolors[:purple],linewidth=2)
# plot!(ax_qpm, λs_sw_dense, sinc2Δk_sw_dense; color=logocolors[:purple],markersize=2)


fig

##
Ex_axes = fig[3, 1:2] = [Axis(fig, title = t) for t in ["|Eₓ₁|²","|Eₓ₂|²"]] #,"|Eₓ₃|²","|Eₓ₄|²"]]
Ey_axes = fig[4, 1:2] = [Axis(fig, title = t) for t in ["|Ey₁|²","|Ey₂|²"]] #,"|Ey₃|²","|Ey₄|²"]]
# Es = [Ex[1],Ey[1],Ex[2],Ey[2],Ex[3],Ey[3],Ex[4],Ey[4]]

Earr = ε⁻¹_dot( fft( kx_tc( unflat(ms.H⃗; ms)[:,:,:,1],mn(ms),ms.M̂.mag), (2:1+ND) ), copy(flat( ms.M̂.ε⁻¹ )))
Ex = Earr[1,:,:] # for i=1:2
Ey = Earr[2,:,:] #for i=1:2

heatmaps_x = [heatmap!(ax, abs2.(Ex)) for (i, ax) in enumerate(Ex_axes)]
heatmaps_y = [heatmap!(ax, abs2.(Ey)) for (i, ax) in enumerate(Ey_axes)]

fig

##
p_opt3 = [ 0.636143209603401, 0.9218203566788058, 0.440673406907343, 0.2398469659583548 ]
#     0.7,        # 700 nm top width of angle sidewall ridge
#     0.6,        # 600 nm MgO:LiNbO₃ ridge thickness
#     5. / 6.,    # etch fraction (they say etch depth of 500 nm, full thickness 600 nm)
#     0.349,      # 20° sidewall angle in radians (they call it 70° , in our params 0° is vertical)
# ]

# λs_opt3 = collect(reverse(1.4:0.02:1.7))
# ωs_opt3 = 1 ./ λs_opt3

ωs_opt3 = collect(0.58:0.01:0.72)
λs_opt3 = inv.(ωs_opt3)

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy)

ms = ModeSolver(kguess(1/1.55,rwg_pe(p_opt3)), rwg_pe(p_opt3), gr; nev=1)

n_opt3F,ng_opt3F = solve_n(ms,ωs_opt3,rwg_pe(p_opt3))
n_opt3S,ng_opt3S = solve_n(ms,2*ωs_opt3,rwg_pe(p_opt3))
k1,H1 = solve_k(ms,2*ωs_opt3[end],rwg_pe(p_opt3))
# Ex1 = E⃗x(ms); Ey1 = E⃗y(ms)
# k2,H2 = solve_k(ms,2*ωs_opt3[end],rwg_pe(p_opt3))
# Ex2 = E⃗x(ms); Ey2 = E⃗y(ms)

##
fig = Figure()
ax_n = fig[1,1] = Axis(fig)
ax_ng = fig[1,2] = Axis(fig)
ax_Λ = fig[2,1] = Axis(fig)
ax_qpm = fig[2,2] = Axis(fig)

lines!(ax_n, λs_opt3, n_opt3F; color=logocolors[:red],linewidth=2)
lines!(ax_n, λs_opt3, n_opt3S; color=logocolors[:blue],linewidth=2)
plot!(ax_n, λs_opt3, n_opt3F; color=logocolors[:red],markersize=2)
plot!(ax_n, λs_opt3, n_opt3S; color=logocolors[:blue],markersize=2)

lines!(ax_ng, λs_opt3, ng_opt3F; color=logocolors[:red],linewidth=2)
lines!(ax_ng, λs_opt3, ng_opt3S; color=logocolors[:blue],linewidth=2)
plot!(ax_ng, λs_opt3, ng_opt3F; color=logocolors[:red],markersize=2)
plot!(ax_ng, λs_opt3, ng_opt3S; color=logocolors[:blue],markersize=2)

# Δk_opt3 = ( 4π ./ λs_opt3 ) .* ( n_opt3S .- n_opt3F )
# Δk_opt3 = ( 2 * ωs_opt3 ) .* ( n_opt3S .- n_opt3F )
Λ_opt3 = λs_opt3 ./ (2*( n_opt3S .- n_opt3F ))  #2π ./ Δk_opt3

lines!(ax_Λ, λs_opt3, Λ_opt3; color=logocolors[:green],linewidth=2)
plot!(ax_Λ, λs_opt3, Λ_opt3; color=logocolors[:green],markersize=2)

# Λ0_opt3 = 2.8548 # 128x128
Λ0_opt3 = 4.73 # 128x128
L_opt3 = 1e3 # 1cm in μm
Δk_qpm_opt3 = ( 4π ./ λs_opt3 ) .* ( n_opt3S .- n_opt3F ) .- (2π / Λ0_opt3)
Δk_qpm_opt3_itp = LinearInterpolation(ωs_opt3,Δk_qpm_opt3)
ωs_opt3_dense = collect(range(extrema(ωs_opt3)...,length=3000))
λs_opt3_dense = inv.(ωs_opt3_dense)
Δk_qpm_opt3_dense = Δk_qpm_opt3_itp.(ωs_opt3_dense)
sinc2Δk_opt3_dense = (sinc.(Δk_qpm_opt3_dense * L_opt3 / 2.0)).^2



lines!(ax_qpm, λs_opt3_dense, sinc2Δk_opt3_dense; color=logocolors[:purple],linewidth=2)
# plot!(ax_qpm, λs_sw_dense, sinc2Δk_sw_dense; color=logocolors[:purple],markersize=2)


fig

##
Ex_axes = fig[3, 1:2] = [Axis(fig, title = t) for t in ["|Eₓ₁|²","|Eₓ₂|²"]] #,"|Eₓ₃|²","|Eₓ₄|²"]]
Ey_axes = fig[4, 1:2] = [Axis(fig, title = t) for t in ["|Ey₁|²","|Ey₂|²"]] #,"|Ey₃|²","|Ey₄|²"]]
# Es = [Ex[1],Ey[1],Ex[2],Ey[2],Ex[3],Ey[3],Ex[4],Ey[4]]

Earr = ε⁻¹_dot( fft( kx_tc( unflat(ms.H⃗; ms)[:,:,:,1],mn(ms),ms.M̂.mag), (2:1+ND) ), copy(flat( ms.M̂.ε⁻¹ )))
Ex = Earr[1,:,:] # for i=1:2
Ey = Earr[2,:,:] #for i=1:2

heatmaps_x = [heatmap!(ax, abs2.(Ex)) for (i, ax) in enumerate(Ex_axes)]
heatmaps_y = [heatmap!(ax, abs2.(Ey)) for (i, ax) in enumerate(Ey_axes)]

fig
##
p_opt

## SHG QPM bandwidth runs from 03/09/21-03/10/21 that were not let to run long enough and which
# appear to have needed box constraints....

# ωs = [
#  0.6,
#  0.6111111111111112,
#  0.6222222222222222,
#  0.6333333333333333,
#  0.6444444444444445,
#  0.6555555555555556,
#  0.6666666666666666,
#  0.6777777777777778,
#  0.6888888888888889,
#  0.7,
#  ]
#
# ########################### Opt 1 ##########################
# Iter     Function value   Gradient norm
#      0     3.332078e-02     3.932040e-02
#  * Current step size: 1.0
#  * time: 0.01732802391052246
#  * g(x): [-0.0005741405388116682, -0.0048572994613995234, -0.039320398848935706, 0.00010192824627240238]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [1.0427730870238208, 0.8907727905412024, 0.18189664400222494, 0.22441652304885823]
#      1     3.583271e-03     1.153236e-03
#  * Current step size: 24.92322727777843
#  * time: 1031.8714709281921
#  * g(x): [0.0011532359427948683, -0.0002627029861411958, 0.0, 0.0009097606751414886]
#  * ~inv(H): [1.0041814719519704 0.03904132762820769 0.31526601901375756 -0.0011500865022875; 0.039041327628207695 1.3613055258186304 2.918227661902613 -0.01038062703191859; 0.31526601901375756 2.918227661902613 24.570105500052854 -0.083894247415978; -0.0011500865022875 -0.01038062703191859 -0.083894247415978 1.000276564205286]
#  * x: [1.0570825221620102, 1.0118323689738935, 1.161887881167147, 0.22187614220098578]
#
 # * Status: failure (line search failed)
 #
 # * Candidate solution
 #    Final objective value:     3.578920e-03
 #
 # * Found with
 #    Algorithm:     BFGS
 #
 # * Convergence measures
 #    |x - x'|               = 1.15e-03 ≰ 1.0e-03
 #    |x - x'|/|x'|          = 9.87e-04 ≰ 0.0e+00
 #    |f(x) - f(x')|         = 4.35e-06 ≰ 0.0e+00
 #    |f(x) - f(x')|/|f(x')| = 1.22e-03 ≰ 1.0e-04
 #    |g(x)|                 = 2.22e-03 ≰ 1.0e-04
 #
 # * Work counters
 #    Seconds run:   1032  (vs limit 10800)
 #    Iterations:    2
 #    f(x) calls:    55  <-------------- 00
 #    ∇f(x) calls:   55					 o

# ########################### Opt 2 ##########################
# Iter     Function value   Gradient norm
#      0     4.153378e-02     1.714674e-02
#  * Current step size: 1.0
#  * time: 2.9087066650390625e-5
#  * g(x): [0.0034337841588441536, 0.017146742132521126, -3.335747663876001e-5, 0.008494598604769968]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [1.9393158506489665, 1.6370881387874332, 0.957304784740935, 0.6900344143756805]
#      1     5.226171e-05     0.000000e+00
#  * Current step size: 625.0
#  * time: 1356.3452801704407
#  * g(x): [0.0, -0.0, 0.0, 0.0]
#  * ~inv(H): [20.466309970997067 97.20581781027452 -0.18910535725663447 48.156343517908454; 97.20581781027451 486.40124092559614 -0.9443053630514929 240.4706778729589; -0.18910535725663444 -0.9443053630514929 1.0018370629151823 -0.467814524616904; 48.156343517908454 240.4706778729589 -0.467814524616904 120.13061204049156]
#  * x: [-0.20679924862862942, -9.07962569403827, 0.97815320764016, -4.6190897136055495]
#
# * Status: "success" (yeah... right)
#
# * Candidate solution
#    Final objective value:     5.226171e-05
#
# * Found with
#    Algorithm:     BFGS
#
# * Convergence measures
#    |x - x'|               = 1.07e+01 ≰ 1.0e-03
#    |x - x'|/|x'|          = 1.18e+00 ≰ 0.0e+00
#    |f(x) - f(x')|         = 4.15e-02 ≰ 0.0e+00
#    |f(x) - f(x')|/|f(x')| = 7.94e+02 ≰ 1.0e-04
#    |g(x)|                 = 0.00e+00 ≤ 1.0e-04
#
# * Work counters
#    Seconds run:   1356  (vs limit 10800)
#    Iterations:    1
#    f(x) calls:    6
#    ∇f(x) calls:   6
# ########################### Opt 3 ##########################
# Iter     Function value   Gradient norm
#      0     6.310561e-03     2.132570e-02
#  * Current step size: 1.0
#  * time: 2.9087066650390625e-5
#  * g(x): [0.015698184070288924, -0.021325702461053093, -0.018443114057898866, 0.009375727570433659]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [1.2454420572592633, 0.6608024510679751, 0.7212028977120437, 0.16904614839522777]
#      1     1.684884e-05     3.332685e-04
#  * Current step size: 54.47389180603591
#  * time: 1538.3423640727997
#  * g(x): [0.0, -0.0003332684851285998, 0.0, 0.0]
#  * ~inv(H): [12.741771282183354 -15.955648568880392 -13.794896659985673 7.012763275240821; -15.955648568880392 22.681796843050776 18.745597905210175 -9.529498031210746; -13.794896659985673 18.745597905210175 17.20702441619168 -8.23899072448132; 7.01276327524082 -9.529498031210746 -8.238990724481319 5.188367118782923]
#  * x: [0.390300876663108, 1.822496459619095, 1.7258710974684064, -0.3416862208794432]

# ########################### Opt 4 ########################## (added Box constraints)
# Fminbox
# -------
# Initial mu = 2.84548e-5
#
# Fminbox iteration 1
# -------------------
# Calling inner optimizer with mu = 2.84548e-5
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     5.744935e-02     5.697679e-01
#  * Current step size: 1.0
#  * time: 1.9073486328125e-5
#  * g(x): [0.0329055546222017, -0.5697678659463177, -0.047814353138851205, 0.01277416905287158]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [1.676716762547048, 0.43474974558777724, 0.7058578266295201, 0.20717224874415013]
#      1     1.521949e-02     3.004023e-02
#  * Current step size: 0.9790012384387733
#  * time: 539.232057094574
#  * g(x): [0.01635291099808244, 0.030040227236247823, -0.010509332918751972, 0.014213542378415168]
#  * ~inv(H): [1.00322079032787 -0.025815253960548998 -0.00341258832054201 0.002002552989180584; -0.02581525396054901 0.9283446865042082 0.015564983331412291 -0.023046530308788695; -0.00341258832054201 0.015564983331412263 1.0031170212214415 -0.0024178247465377143; 0.002002552989180584 -0.023046530308788695 -0.0024178247465377143 1.0010694221275922]
#  * x: [1.644502183820398, 0.9925531919718392, 0.7526681375676043, 0.19466632142136261]
#      2     5.953687e-03     1.956094e-02
#  * Current step size: 12.097080427568066
#  * time: 1862.8806681632996
#  * g(x): [0.019560943932913362, -0.0011989780786397818, -0.010437880826112996, 0.01167669608836972]
#  * ~inv(H): [5.088820037124012 6.314730855802797 -2.6080845475547347 3.4405454042424655; 6.314730855802797 10.654730654731026 -4.0197835144771545 5.296565603957517; -2.6080845475547347 -4.0197835144771545 2.663240810818114 -2.1935570418710983; 3.4405454042424655 5.296565603957519 -2.1935570418710983 3.8925649807194502]
#  * x: [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425]
#
# Exiting inner optimizer with x = [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425]
# Current distance to box: 0.0302116
# Decreasing barrier term μ.
#
# Fminbox iteration 2
# -------------------
# Calling inner optimizer with mu = 2.84548e-8
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     5.747107e-03     1.953595e-02
#  * Current step size: 12.097080427568066
#  * time: 0.00020384788513183594
#  * g(x): [0.019535954448720257, -0.0011424116158813686, -0.01063397131537939, 0.012581072645450187]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425]
#      1     4.163141e-03     1.651796e-02
#  * Current step size: 2.3984801515923606
#  * time: 2149.4501798152924
#  * g(x): [0.016517961282915727, 0.0007971545772888044, -0.007372684517568671, 0.010511361366639925]
#  * ~inv(H): [8.256094375729432 -0.14175009082682638 -3.6903002835198455 4.652678567016319; -0.14175009082682632 0.9917653733861643 0.06198973552038578 -0.0901042992579506; -3.6903002835198446 0.06198973552038578 2.867537666528222 -2.3655338876602063; 4.652678567016318 -0.0901042992579506 -2.3655338876602063 3.9832866071795143]
#  * x: [1.4077890254553842, 0.6689819948730713, 0.9011365392777256, 3.6177729235559175e-5]
#      2     4.161747e-03     1.644100e-02
#  * Current step size: 0.0002575511156360406
#  * time: 3467.3261349201202
#  * g(x): [0.016440997699906108, 0.00024324353309534223, -0.0073153721995303745, -0.014006999658463852]
#  * ~inv(H): [3.400045856735412 -0.06935209758117244 -1.2078382588814502 -0.009698511346722505; -0.06935209758117239 0.9906908097982596 0.025016773309692167 -0.02213623839086537; -1.2078382588814494 0.025016773309692167 1.5987722847592214 0.005840366359258997; -0.009698511346723393 -0.02213623839086537 0.005840366359259441 0.001973747857787256]
#  * x: [1.4077343282288142, 0.6689827559319353, 0.9011640749055386, 1.1273307247269671e-6]
#
# Exiting inner optimizer with x = [1.4077343282288142, 0.6689827559319353, 0.9011640749055386, 1.1273307247269671e-6]
# Current distance to box: 1.12733e-6
# Decreasing barrier term μ.
#
# * Status: failure (reached maximum number of iterations)
#
#  * Candidate solution
#     Final objective value:     4.161247e-03
#
#  * Found with
#     Algorithm:     Fminbox with BFGS
#
#  * Convergence measures
#     |x - x'|               = 6.14e-02 ≰ 1.0e-03
#     |x - x'|/|x'|          = 3.41e-02 ≰ 0.0e+00
#     |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
#     |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 1.0e-04
#     |g(x)|                 = 1.64e-02 ≰ 1.0e-04
#
#  * Work counters
#     Seconds run:   5588  (vs limit Inf)
#     Iterations:    2
#     f(x) calls:    21
#     ∇f(x) calls:   21

##
#
# # BFGS(; alphaguess = LineSearches.InitialStatic(),
# #        linesearch = LineSearches.HagerZhang(),
# #        initial_invH = nothing,
# #        initial_stepnorm = nothing,
# #        manifold = Flat())
# inner_optimizer = Optim.BFGS(; alphaguess= Optim.LineSearches.InitialStatic(),
#        linesearch = Optim.LineSearches.HagerZhang(),
#        initial_invH = nothing,
#        initial_stepnorm = nothing,
#        manifold = Flat())
#      # GradientDescent() #
#
# # results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))
# # res2 = optimize( Optim.only_fg!(fg!),
# #                 p_lower,
# #                 p_upper,
# #                 p0,
# #                 Fminbox(inner_optimizer),
# #                 opts,
# #             )
#
# res2 = optimize( Optim.only_fg!(fg!),
#                 p_lower,
#                 p_upper,
#                 p0,
#                 Fminbox(inner_optimizer),
#                 opts,
#             )
