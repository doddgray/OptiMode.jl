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

##
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

using Rotations: RotY, MRP
LNx = rotate(MgO_LiNbO₃,Matrix(MRP(RotY(π/2))))

rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).
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
_,ng_swF_old = solve_n(ms,ωs_sw,rwg_pe(p_sw);ng_nodisp=true)
_,ng_swS_old = solve_n(ms,2*ωs_sw,rwg_pe(p_sw);ng_nodisp=true)
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

lines!(ax_ng, λs_sw, ng_swF_old; color=logocolors[:red],linewidth=2,linestyle=:dash)
lines!(ax_ng, λs_sw, ng_swS_old; color=logocolors[:blue],linewidth=2,linestyle=:dash)
plot!(ax_ng, λs_sw, ng_swF_old; color=logocolors[:red],markersize=2)
plot!(ax_ng, λs_sw, ng_swS_old; color=logocolors[:blue],markersize=2)

# Δk_sw = ( 4π ./ λs_sw ) .* ( n_swS .- n_swF )
# Λ_sw = 2π ./ Δk_sw
Λ_sw = ( λs_sw ./ 2 ) ./ ( n_swS .- n_swF )

lines!(ax_Λ, λs_sw, Λ_sw; color=logocolors[:green],linewidth=2)
plot!(ax_Λ, λs_sw, Λ_sw; color=logocolors[:green],markersize=2)

Λ0_sw = 2.8548 # 128x128
# Λ0_sw = 2.86275 # 256x256
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
##
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

# ωs_jank =  collect(0.416:0.003:0.527)
# λs_jank = 1 ./ ωs_jank

nω_jank = 20
ωs_jank = collect(range(1/2.25,1/1.95,length=nω_jank)) # collect(0.416:0.01:0.527)
λs_jank = 1 ./ ωs_jank

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 512, 512, 1;

ms = ModeSolver(kguess(1/1.55,rwg_pe(p_jank)), rwg_pe(p_jank), gr; nev=1)

n_jankF,ng_jankF = solve_n(ms,ωs_jank,rwg_pe(p_jank))
n_jankS,ng_jankS = solve_n(ms,2*ωs_jank,rwg_pe(p_jank))

_,ng_jankF_old = solve_n(ms,ωs_jank,rwg_pe(p_jank);ng_nodisp=true)
_,ng_jankS_old = solve_n(ms,2*ωs_jank,rwg_pe(p_jank);ng_nodisp=true)

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
xs = λs_jank
# xs = ωs_jank
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
L_jank = 3e3 # 1cm in μm
Δk_qpm_jank = ( 4π ./ λs_jank) .* (  n_jankS .-  n_jankF ) .- (2π / Λ0_jank)

Δk_qpm_jank_itp = LinearInterpolation(ωs_jank,Δk_qpm_jank)
ωs_jank_dense = collect(range(extrema(ωs_jank)...,length=3000))
λs_jank_dense = inv.(ωs_jank_dense)
Δk_qpm_jank_dense = Δk_qpm_jank_itp.(ωs_jank_dense)
sinc2Δk_jank_dense = (sinc.(Δk_qpm_jank_dense * L_jank / 2.0)).^2



# lines!(ax_qpm, ωs_jank_dense, sinc2Δk_jank_dense; color=logocolors[:purple],linewidth=2)
lines!(ax_qpm, λs_jank_dense, sinc2Δk_jank_dense; color=logocolors[:purple],markersize=2)
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
	println("")
	println("p: $p")
	println("\tngs_FH: $ngs_FH")
	println("\tngs_SH: $ngs_SH")
    Δng² = abs2.(ngs_SH .- ngs_FH)
	println("\tsum(Δng²): $(sum(Δng²))")
	# println("")
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
		println("\tgrad_p sum(Δng²): $G")
		println("")
    end
    if F != nothing
        # F = value
        return value
    end
end


fg!(0.,[0.,0.,0.,0.],p0)
## parameter sweep for comparison with optimization trajectories

p_pe_lower = [0.4, 0.3, 0., 0.]
p_pe_upper = [2., 2., 1., π/4.]

np1 = 20
np2 = 20
np3 = 5
np4 = 3

range1 = range(p_pe_lower[1],p_pe_upper[1],length=np1)
range2 = range(p_pe_lower[2],p_pe_upper[2],length=np2)
range3 = range(p_pe_lower[3],p_pe_upper[3],length=np3)
range4 = range(p_pe_lower[4],p_pe_upper[4],length=np4)

for p4 in range4, p3 in range3, p2 in range2, p1 in range1
	try
		sum_Δng_FHSH(ωs,[p1,p2,p3,p4])
	catch
		println("")
		println("p: $([p1,p2,p3,p4])")
		println("\tngs_FH: Error")
		println("\tngs_SH: Error")
		println("\tsum(Δng²): Error")
	end
end

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
##
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
p_pe = [
       1.7,                #   top ridge width         `w_top`         [μm]
       0.7,                #   ridge thickness         `t_core`        [μm]
       0.5,                #   top layer etch fraction `etch_frac`     [1]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
               ];
rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).

geom_pe = rwg_pe(p_pe)
ms = ModeSolver(1.45, rwg_pe(p_pe), gr)
p_pe_lower = [0.4, 0.3, 0., 0.]
p_pe_upper = [2., 2., 1., π/4.]


# λs = reverse(1.45:0.02:1.65)
# ωs = 1 ./ λs

nω_jank = 20
ωs_jank = collect(range(1/2.25,1/1.95,length=nω_jank)) # collect(0.416:0.01:0.527)
λs_jank = 1 ./ ωs_jank

ωs = ωs_jank #collect(range(0.6,0.7,length=10))
λs = λs_jank #inv.(ωs)

# n1F,ng1F = solve_n(ms,ωs,rwg_pe(p_pe)); n1S,ng1S = solve_n(ms,2*ωs,rwg_pe(p_pe))
##

function sum_Δng²_FHSH(ωs,p)
    # ms = ModeSolver(1.45, rwg_pe(p), gr)
	nω = length(ωs)
    # ngs_FHSH = solve_n(vcat(ωs, 2*ωs ),rwg_pe(p),gr)[2]
	ms = Zygote.@ignore ModeSolver(kguess(ωs[1],rwg_pe(p)), rwg_pe(p), grid; nev=4)

	nFS,ngFS,gvdFS,EFS = solve_n(
	    Zygote.dropgrad(ms),
	    vcat(ωs,2*ωs),
	    rwg_pe(p);
	    f_filter=TE_filter,
	)

    ngs_FH = ngFS[1:nω]
	ngs_SH = ngFS[nω+1:2*nω]
    Δng² = abs2.(ngs_SH .- ngs_FH)
    sum(Δng²)
end

# sum_Δng_FHSH(ωs,p_pe)

# warmup
println("warmup function runs")
p0 = copy(p_pe)
# @show sum_Δng_FHSH(ωs,p0)
# @show vng0, vng0_pb = Zygote.pullback(x->var_ng(ωs,x),p0)
# @show grad_vng0 = vng0_pb(1)

# define function that computes value and gradient of function `f` to be optimized
# according to https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/
function fg!(F,G,x)
    value, value_pb = Zygote.pullback(x) do x
       # var_ng(ωs,x)
       sum_Δng²_FHSH(ωs,x)
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
                        outer_iterations = 4,
                        iterations = 6,
                        # time_limit = 3*3600,
                        store_trace = true,
                        show_trace = true,
                        show_every = 1,
                        extended_trace = true,
                        x_tol = 1e-4, # Absolute tolerance in changes of the input vector x, in infinity norm. Defaults to 0.0.
                        f_tol = 1e-5, # Relative tolerance in changes of the objective value. Defaults to 0.0.
                        g_tol = 1e-5, # Absolute tolerance in the gradient, in infinity norm. Defaults to 1e-8. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
                    )

println("########################### Opt 1 ##########################")

# p_opt1 = 0.62929, 0.71422,0.7658459,0.125366
res1 = optimize( Optim.only_fg!(fg!),
                p_pe_lower,
                p_pe_upper,
                rand_p0(),
				# p_opt1,
                Fminbox(Optim.BFGS()),
                opts,
            )


println("########################### Opt 2 ##########################")

# res2 = optimize( Optim.only_fg!(fg!),
#                 rand_p0(),
#                 Optim.BFGS(),
#                 opts,
#             )

res2 = optimize( Optim.only_fg!(fg!),
                p_pe_lower,
                p_pe_upper,
                rand_p0(),
                Fminbox(Optim.BFGS();mu0=1e-6),
                opts,
            )

println("########################### Opt 3 ##########################")

# res3 = optimize( Optim.only_fg!(fg!),
#                 rand_p0(),
#                 Optim.BFGS(),
#                 opts,
#             )

res3 = optimize( Optim.only_fg!(fg!),
                p_pe_lower,
                p_pe_upper,
                rand_p0(),
                Fminbox(Optim.BFGS();mu0=1e-6),
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
_, ng_opt1F_nodisp = solve_n(ms,ωs_opt1,rwg_pe(p_opt1);ng_nodisp=true)
_, ng_opt1S_nodisp = solve_n(ms,2*ωs_opt1,rwg_pe(p_opt1);ng_nodisp=true)
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

lines!(ax_ng, λs_opt1, ng_opt1F_nodisp; color=logocolors[:red],linewidth=2, alpha=0.3)
lines!(ax_ng, λs_opt1, ng_opt1S_nodisp; color=logocolors[:blue],linewidth=2, alpha=0.3)
plot!(ax_ng, λs_opt1, ng_opt1F_nodisp; color=logocolors[:red],markersize=2)
plot!(ax_ng, λs_opt1, ng_opt1S_nodisp; color=logocolors[:blue],markersize=2)

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
#
##### overnight runs between March 10 and 11



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
#
#starting from
# p_opt1 = [ 0.629290654535625,
#  0.7142246705802344,
#  0.7658459012111655,
#  0.12536671438304348, ]
# Fminbox
# -------
# Initial mu = 1.06695e-6
#
# Fminbox iteration 1
# -------------------
# Calling inner optimizer with mu = 1.06695e-6
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     1.149235e-04     1.052225e-02
#  * Current step size: 1.0
#  * time: 0.03420400619506836
#  * g(x): [0.005672843354504562, 0.005694646039948283, 0.01052224560191127, 0.003050123474544317]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [0.629290654535625, 0.7142246705802344, 0.7658459012111655, 0.12536671438304348]
#      1     9.851611e-05     1.203712e-03
#  * Current step size: 0.17107087871970592
#  * time: 598.3605408668518
#  * g(x): [0.0012037118318105552, 0.0006191757774280639, 0.0006031229476605116, 0.0007586303493860311]
#  * ~inv(H): [0.9023336563641802 -0.11822222355879555 -0.23697363126203408 -0.04869558392843025; -0.11822222355879555 0.8610653325624109 -0.27531613063052024 -0.05973321566873577; -0.23697363126203408 -0.27531613063052024 0.45691745001893136 -0.12033432089609272; -0.04869558392843025 -0.05973321566873577 -0.12033432089609272 0.9758699670048809]
#  * x: [0.6283201962381307, 0.7132504824781828, 0.764045851409942, 0.12484492708004957]
#      2     9.848512e-05     5.307978e-03
#  * Current step size: 2.5372610522583807
#  * time: 1494.1609208583832
#  * g(x): [-0.001336278109880404, -0.0025610404916637263, -0.005307978329116837, -0.0005613318158930096]
#  * ~inv(H): [3.530312241878714 0.1602134185271394 -1.6424376496859217 1.777243277147924; 0.16021341852713952 0.8590609857746432 -0.4843619263253224 0.1360348364742461; -1.6424376496859217 -0.4843619263253223 1.093783064960201 -1.0924538046131953; 1.777243277147924 0.13603483647424605 -1.0924538046131953 2.2443765091482515]
#  * x: [0.6262064467363574, 0.7127950938473154, 0.7647345374158281, 0.12339323983458993]
#      3     9.431438e-05     2.383508e-03
#  * Current step size: 1.8631869064578785
#  * time: 3111.2775659561157
#  * g(x): [-0.0013904516026559015, -0.002383507882856247, -0.0020537486642922247, -0.0006358494114057974]
#  * ~inv(H): [3.025270439094615 -0.4049722406252899 -1.3777857225110965 1.490886798617244; -0.4049722406252898 0.8244745449792459 -0.10367743283791275 -0.25767484484783876; -1.3777857225110965 -0.1036774328379127 0.9670468299249135 -0.9527517358535873; 1.490886798617244 -0.25767484484783876 -0.9527517358535873 2.09098804429504]
#  * x: [0.6213759330223262, 0.7126452148572809, 0.7680087600052793, 0.12001043923861131]
#      4     9.306942e-05     3.617122e-04
#  * Current step size: 1.4008554539467293
#  * time: 4018.2419760227203
#  * g(x): [-1.3064609872065097e-5, -0.00021300967294684853, -0.0003617121562952686, 1.8378031972508157e-5]
#  * ~inv(H): [2.8146836543530607 -0.33050119309322357 -1.2575202203083498 1.334100097679968; -0.33050119309322346 1.0565487648574405 -0.14726061863757267 -0.2332113324639964; -1.2575202203083498 -0.14726061863757262 0.8983678410903528 -0.8630850583480367; 1.334100097679968 -0.2332113324639964 -0.8630850583480367 1.9779712976964725]
#  * x: [0.6232805242024614, 0.7140814804188901, 0.7669124563551152, 0.12117549825751972]
#      5     9.301278e-05     5.402580e-04
#  * Current step size: 0.4082637770143645
#  * time: 4631.104068994522
#  * g(x): [0.0004961588141846286, -4.406861653291081e-5, 0.0005402580407514264, 0.000249936484500065]
#  * ~inv(H): [2.8146836543530607 -0.33050119309322357 -1.2575202203083498 1.334100097679968; -0.33050119309322346 1.0565487648574405 -0.14726061863757267 -0.2332113324639964; -1.2575202203083498 -0.14726061863757262 0.8983678410903528 -0.8630850583480367; 1.334100097679968 -0.2332113324639964 -0.8630850583480367 1.9779712976964725]
#  * x: [0.6230710825610506, 0.7141516026813306, 0.7670320839398395, 0.12102003698418824]
#      6     9.291662e-05     4.193888e-04
#  * Current step size: 1.5760934094864436
#  * time: 6165.188092947006
#  * g(x): [-0.00035686639316645735, -0.0004193888068910373, -0.00028019648216327407, -0.00019691885548799048]
#  * ~inv(H): [2.6831943955109243 -0.9379747908982712 -0.9827441300022488 1.226975893097002; -0.9379747908982711 1.46799974411201 -0.02473090862926053 -0.625831166126642; -0.9827441300022486 -0.024730908629260473 0.732940962627447 -0.6756822195851524; 1.226975893097002 -0.625831166126642 -0.6756822195851524 1.8939487150825673]
#  * x: [0.6213923071189889, 0.7147006954627761, 0.7675802589445816, 0.11991632812708061]
#
# Exiting inner optimizer with x = [0.6213923071189889, 0.7147006954627761, 0.7675802589445816, 0.11991632812708061]
# Current distance to box: 0.119916
# Decreasing barrier term μ.
#
# Fminbox iteration 2
# -------------------
# Calling inner optimizer with mu = 1.06695e-9
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     8.644895e-05     4.176396e-04
#  * Current step size: 1.5760934094864436
#  * time: 1.1205673217773438e-5
#  * g(x): [-0.0003528145641642496, -0.00041763962492077954, -0.000283390619090616, -0.0001896181753252915]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [0.6213923071189889, 0.7147006954627761, 0.7675802589445816, 0.11991632812708061]
#      1     8.643662e-05     4.763774e-04
#  * Current step size: 0.22845458489883178
#  * time: 602.0583760738373
#  * g(x): [-0.0004763774493603344, -0.00037759803439752816, 6.0427221373495045e-5, -0.00022487429228148232]
#  * ~inv(H): [6.968943438779687 6.036629906555775 2.3472403467086336 3.035913092215052; 6.036629906555775 6.927682295794504 1.9519700651893803 3.0406767454569756; 2.3472403467086336 1.9519700651893803 0.9197227998919209 1.1233062694075935; 3.035913092215052 3.0406767454569756 1.1233062694075933 2.539160696058574]
#  * x: [0.6214729092237913, 0.7147961071499247, 0.7676450008308302, 0.11995964726861383]
#
# Exiting inner optimizer with x = [0.6214729092237913, 0.7147961071499247, 0.7676450008308302, 0.11995964726861383]
# Current distance to box: 0.11996
# Decreasing barrier term μ.
#
# Fminbox iteration 3
# -------------------
# Calling inner optimizer with mu = 1.06695e-12
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     8.643015e-05     4.763734e-04
#  * Current step size: 0.22845458489883178
#  * time: 7.867813110351562e-6
#  * g(x): [-0.0004763733993311013, -0.0003775962858695896, 6.0424025839630345e-5, -0.0002248669949193012]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [0.6214729092237913, 0.7147961071499247, 0.7676450008308302, 0.11995964726861383]
#      1     8.643015e-05     2.389342e-04
#  * Current step size: 8.403908368305993e-8
#  * time: 4321.518954992294
#  * g(x): [-0.00016033946012437, -0.00021337290095060582, 0.00023893418477299261, -7.74940881866708e-5]
#  * ~inv(H): [0.460228819164178 -0.2528834623063271 -0.3748194512117264 -0.25112776470494264; -0.2528834623063271 0.9382380536116095 -0.3192926354031945 -0.11646551477205463; -0.3748194512117264 -0.3192926354031945 1.1037697785615541 -0.17739446979160167; -0.25112776470494264 -0.11646551477205463 -0.17739446979160167 0.8831880605858922]
#  * x: [0.6214729092638253, 0.7147961071816575, 0.7676450008257523, 0.11995964728751145]
#
# Exiting inner optimizer with x = [0.6214729092638253, 0.7147961071816575, 0.7676450008257523, 0.11995964728751145]
# Current distance to box: 0.11996
# Decreasing barrier term μ.
#############
#
#
# Fminbox
# -------
# Initial mu = 8.56589e-7
#
# Fminbox iteration 1
# -------------------
# Calling inner optimizer with mu = 8.56589e-7
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     4.188360e-02     2.489634e-02
#  * Current step size: 1.0
#  * time: 2.09808349609375e-5
#  * g(x): [0.0005561019573532833, 0.024896338264477885, -0.022551759643157088, 0.0001550797912775847]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [1.3145139631841207, 1.2917050766494207, 0.08635033020434402, 0.026851696251419598]
#      1     8.306225e-03     3.731688e-02
#  * Current step size: 25.262718314364825
#  * time: 1081.4288759231567
#  * g(x): [0.01647430839777716, -0.014353921573339196, -0.037316882138055625, 0.007260694420400828]
#  * ~inv(H): [1.0418166590243743 1.212855562647482 -1.1427553249530977 0.01399933378206751; 1.2128555626474817 25.784566535409922 -24.425684025299507 0.4428967747425916; -1.1427553249530975 -24.425684025299507 24.914589835185485 -0.4134909510682464; 0.01399933378206751 0.44289677474259165 -0.4134909510682464 1.0045559675020534]
#  * x: [1.300465316081438, 0.6627558960147735, 0.6560690815626822, 0.022933959168123484]
#      2     7.586104e-03     4.334062e-02
#  * Current step size: 0.10466256308535091
#  * time: 1688.3990149497986
#  * g(x): [0.01672435078025475, -0.043340624044110376, -0.020793630099045494, 0.007246424798281104]
#  * ~inv(H): [1.010624457402458 0.09304015484903982 -0.1212740267311807 0.003953494144916921; 0.0930401548490396 1.5372470648917442 -0.8807694848610765 0.03394768446146168; -0.12127402673118093 -0.8807694848610801 2.263293534848465 -0.04477215196800366; 0.0039534941449169275 0.033947684461461736 -0.04477215196800366 1.0014667902645873]
#  * x: [1.2960171900177808, 0.6036657760954169, 0.7189669693153617, 0.02119684417620386]
#      3     3.640689e-03     1.660801e-02
#  * Current step size: 3.002624294028121
#  * time: 2510.6881470680237
#  * g(x): [0.009862122974027934, 0.004029205277150647, -0.016608005364687893, 0.004919123481357019]
#  * ~inv(H): [1.4044780869900624 -0.7097808897026127 -0.6264119424162391 0.17981325801288678; -0.7097808897026129 2.7971647703535267 0.3620490592237282 -0.3270764738020113; -0.6264119424162392 0.36204905922372466 2.790488225449513 -0.268874441327485; 0.17981325801288678 -0.32707647380201116 -0.268874441327485 1.0799726605895044]
#  * x: [1.2497167368830475, 0.7433142790137055, 0.7527214412848778, 0.0008305385908495486]
#      4     3.556630e-03     8.257868e-02
#  * Current step size: 0.08021281605870471
#  * time: 3970.0733559131622
#  * g(x): [0.009744800167240757, -0.003307466916315965, -0.014629015725614158, -0.0825786767626484]
#  * ~inv(H): [5.109809609921585 -1.1052403790602983 -9.375071885890998 -0.10579464660249088; -1.1052403790602985 2.8322945354453712 1.2973495822342072 -0.20973479878551216; -9.375071885890994 1.2973495822342036 23.44660000129857 0.3860694417363382; -0.10579464660249088 -0.209734798785512 0.3860694417363382 0.03584147708699681]
#  * x: [1.2479296547611718, 0.7435831072650915, 0.7569234740078943, 9.683204535018285e-6]
#      5     3.103137e-03     9.623687e-03
#  * Current step size: 0.05000000220434187
#  * time: 4849.537552118301
#  * g(x): [0.009456297485526469, 0.005672272551545391, -0.009623687242075156, 0.004598365440658601]
#  * ~inv(H): [2.0814262846623155 -0.3872401772246964 -1.878071839009559 0.04027981604507022; -0.3872401772246966 2.711478730341565 -0.4356043848934037 -0.24306720166668905; -1.8780718390095554 -0.435604384893407 4.927284355127753 0.025622064141330225; 0.04027981604507022 -0.24306720166668888 0.025622064141330225 0.028829779161991434]
#  * x: [1.2379629505466654, 0.7446729769326049, 0.7804500213787015, 0.000456923947608688]
#      6     2.628155e-03     4.423001e-03
#  * Current step size: 0.755481127681599
#  * time: 5268.751177072525
#  * g(x): [0.0024627263700052727, 0.0044230006336515365, 0.0010046715403585079, 0.0009144239721225867]
#  * ~inv(H): [1.8277307356220707 -0.13800877760953423 -1.3487305204496458 0.016318594015878568; -0.13800877760953434 2.961211011146246 -0.8857492201597521 -0.26400131765463963; -1.3487305204496423 -0.8857492201597554 3.8326761452128117 0.06933371907762716; 0.016318594015878582 -0.26400131765463947 0.06933371907762716 0.030565859555263957]
#  * x: [1.2109580875399983, 0.7334972982729376, 0.8314686416215006, 0.001296908936826641]
#
# Exiting inner optimizer with x = [1.2109580875399983, 0.7334972982729376, 0.8314686416215006, 0.001296908936826641]
# Current distance to box: 0.00129691
# Decreasing barrier term μ.
#
# Fminbox iteration 2
# -------------------
# Calling inner optimizer with mu = 8.56589e-10
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     2.619681e-03     4.424309e-03
#  * Current step size: 0.755481127681599
#  * time: 1.0013580322265625e-5
#  * g(x): [0.0024627055136223885, 0.004424308769909549, 0.0010006275664114087, 0.0015738250545551714]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [1.2109580875399983, 0.7334972982729376, 0.8314686416215006, 0.001296908936826641]
#      1     2.598207e-03     3.998734e-02
#  * Current step size: 0.7764087499407638
#  * time: 1045.4263379573822
#  * g(x): [0.006952456128019986, -0.039987341441955725, 0.003465251200690705, 0.003980512941488638]
#  * ~inv(H): [1.528343773692885 0.22791394961333933 0.2234741954608865 0.33128779303892814; 0.22791394961333944 0.11367749584644304 0.10841655534752975 0.13423026013740563; 0.2234741954608865 0.10841655534752975 1.094376521635021 0.14023109964272654; 0.33128779303892814 0.13423026013740558 0.1402310996427265 1.2076511369994969]
#  * x: [1.2090460214306946, 0.7300622262315402, 0.8306917456235068, 7.497739359400607e-5]
#      2     2.597981e-03     3.602340e-03
#  * Current step size: 0.0330601257089042
#  * time: 2073.4983570575714
#  * g(x): [0.0025617802413303156, 0.0036023403014904557, 0.0011889129392874639, 0.0010445861370424901]
#  * ~inv(H): [1.7712296742269116 0.22428113405119032 0.3317759680897685 0.4643850568255825; 0.22428113405119043 0.039192024453169105 0.10535458963805408 0.14168939477082104; 0.3317759680897685 0.10535458963805408 1.1426399134361716 0.19976141868247543; 0.4643850568255825 0.141689394770821 0.19976141868247538 1.27938811644588]
#  * x: [1.2089268343983326, 0.7301300359617963, 0.8306398780556008, 1.2943220700641046e-6]
#      3     2.597973e-03     1.100453e-02
#  * Current step size: 0.00037499999999999556
#  * time: 3416.016249895096
#  * g(x): [0.0028530179227386652, 0.001118123001399854, 0.001325557689962269, -0.011004526996868977]
#  * ~inv(H): [4.696175366440394 0.5578539023012339 1.660894754439307 0.01752492696903385; 0.5578539023012339 0.07135492096042988 0.2576079671051076 0.0017244922099273752; 1.660894754439307 0.2576079671051076 1.746524598944994 0.00692691002330359; 0.01752492696903385 0.0017244922099273752 0.006926910023303534 0.00024849796329573515]
#  * x: [1.208924500030426, 0.7301296650844715, 0.8306388293202543, 6.657360538918008e-8]
#
# Exiting inner optimizer with x = [1.208924500030426, 0.7301296650844715, 0.8306388293202543, 6.657360538918008e-8]
# Current distance to box: 6.65736e-8
# Decreasing barrier term μ.
#
# Fminbox iteration 3
# -------------------
# Calling inner optimizer with mu = 8.56589e-13
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     2.597956e-03     2.853018e-03
#  * Current step size: 0.00037499999999999556
#  * time: 1.811981201171875e-5
#  * g(x): [0.0028530179157886756, 0.0011181243352607634, 0.0013255536803811316, 0.0018622673395933909]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [1.208924500030426, 0.7301296650844715, 0.8306388293202543, 6.657360538918008e-8]
#      1     2.597955e-03     8.594327e-03
#  * Current step size: 3.5703319815295144e-5
#  * time: 1472.8807380199432
#  * g(x): [0.002246273578144664, 0.007166784938561111, 0.001005653784052369, -0.008594326726379381]
#  * ~inv(H): [6.159944021436862 3.228786138920341 2.3900922783633467 1.4371631771389084; 3.228786138920341 2.738251936484808 1.497282925242097 1.3508000867154344; 2.390092278363347 1.497282925242097 2.1070833112455523 0.6629666252384148; 1.4371631771389088 1.3508000867154344 0.6629666252384148 0.6777086436734296]
#  * x: [1.208924398168215, 0.7301296251637208, 0.8306387819935873, 8.447898209840182e-11]
#
# Exiting inner optimizer with x = [1.208924398168215, 0.7301296251637208, 0.8306387819935873, 8.447898209840182e-11]
# Current distance to box: 8.4479e-11
# Decreasing barrier term μ.
#
# Fminbox iteration 4
# -------------------
# Calling inner optimizer with mu = 8.56589e-16
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     2.597955e-03     7.166785e-03
#  * Current step size: 3.5703319815295144e-5
#  * time: 1.5020370483398438e-5
#  * g(x): [0.0022462735781434272, 0.007166784939900685, 0.0010056537800485024, 0.0015453427972367696]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [1.208924398168215, 0.7301296251637208, 0.8306387819935873, 8.447898209840182e-11]
#      1     2.597955e-03     1.662631e-02
#  * Current step size: 5.351363917142236e-8
#  * time: 1062.3177318572998
#  * g(x): [0.0013646727580223976, 0.01662631171513314, 0.00048018626584739175, 0.0006160280451670596]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [1.2089243980480087, 0.7301296247802, 0.8306387819397711, 1.7820652509168142e-12]
#
# Exiting inner optimizer with x = [1.2089243980480087, 0.7301296247802, 0.8306387819397711, 1.7820652509168142e-12]
# Current distance to box: 1.78207e-12
# Decreasing barrier term μ.
#
#  * Status: failure
#
#  * Candidate solution
#     Final objective value:     2.597955e-03
#
#  * Found with
#     Algorithm:     Fminbox with BFGS
#
#  * Convergence measures
#     |x - x'|               = 4.14e-10 ≤ 1.0e-04
#     |x - x'|/|x'|          = 2.53e-10 ≰ 0.0e+00
#     |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
#     |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 1.0e-05
#     |g(x)|                 = 1.66e-02 ≰ 1.0e-05
#
#  * Work counters
#     Seconds run:   11444  (vs limit Inf)
#     Iterations:    4
#     f(x) calls:    54
#     ∇f(x) calls:   54
#
#####################
# SiN-LiNbO3 loaded slab SHG BW opt 1 (went poorly)
######################

########################### Opt 1 ##########################
# Fminbox
# -------
# Initial mu = 1.78554e-6
#
# Fminbox iteration 1
# -------------------
# Calling inner optimizer with mu = 1.78554e-6
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     3.713344e-02     1.207498e-02
#  * Current step size: 1.0
#  * time: 0.01929306983947754
#  * g(x): [9.160848269122319e-8, 0.008668237285021039, 0.012074978925583157]
#  * ~inv(H): [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
#  * x: [1.8669221448035564, 1.966838838658595, 1.458853077190889]
#      1     2.246619e-02     1.391815e-02
#  * Current step size: 57.08445815228756
#  * time: 1554.0992879867554
#  * g(x): [0.0024840314890478337, 0.013918146226521013, -0.00394634307064296]
#  * ~inv(H): [1.0000030793996582 0.14585218839213837 0.20315974293190475; 0.14585218839213837 31.603274907567375 41.26365054593753; 0.20315974293190475 41.26365054593753 56.57639699947109]
#  * x: [1.86691691538296, 1.472017210107713, 0.769559448023683]
#      2     2.244687e-02     1.043937e-02
#  * Current step size: 0.017687572784124164
#  * time: 2931.7107009887695
#  * g(x): [0.0029536245076588934, 0.008625000363829958, -0.010439367288927326]
#  * ~inv(H): [1.0028428584975773 0.04714321903409985 0.04420952648554477; 0.04714321903409985 0.9977928127268214 -0.05438636907575045; 0.04420952648554477 -0.05438636907575045 1.0051820974652088]
#  * x: [1.8668512539271405, 1.4671110173472917, 0.7633414117905094]
#      3     2.075105e-02     1.365278e-02
#  * Current step size: 22.081667094422848
#  * time: 3779.0340700149536
#  * g(x): [7.774967195283574e-6, 0.011106526868829982, 0.013652775597205357]
#  * ~inv(H): [1.7997478182206232 2.679126545547567 -2.7204063105533756; 2.679126545547567 9.68274453881034 -9.205553027109707; -2.720406310553376 -9.205553027109708 10.543597372831721]
#  * x: [1.8026573736347213, 1.2614652019834502, 1.0025293500804864]
#      4     2.033448e-02     1.337140e-02
#  * Current step size: 2.62293061190059
#  * time: 4418.596955060959
#  * g(x): [0.00030594778383603703, 0.013371401016355665, 0.0042225167844978555]
#  * ~inv(H): [1.435887467943016 1.2737778662699155 -1.6989303824302047; 1.273777866269915 4.972385228713014 -3.805078236202668; -1.698930382430205 -3.80507823620267 10.627087736533051]
#  * x: [1.821991816015511, 1.308989707243653, 0.893188564115736]
#      5     1.731740e-02     6.332947e-03
#  * Current step size: 5.493820846981495
#  * time: 5260.995810985565
#  * g(x): [0.0002532078322778134, -0.00038475093940457485, 0.0063329474082572034]
#  * ~inv(H): [1.8935037768427947 3.9363083321770542 -1.1019624840656528; 3.9363083321770533 20.10597269161535 -1.1159554023861995; -1.1019624840656532 -1.1159554023862017 9.686675265118371]
#  * x: [1.7654178817832862, 1.0298461445127236, 0.9290409897767588]
#      6     1.719069e-02     1.462414e-02
#  * Current step size: 0.7461479312473024
#  * time: 5676.696197986603
#  * g(x): [0.000690471941154833, 0.014624140235002199, 0.006278047711317521]
#  * ~inv(H): [4.049140084477565 0.15531512330315955 -34.203618493737444; 0.1553151233031569 0.6795322594182664 -0.6313045058557316; -34.20361849373744 -0.6313045058557196 390.77770958633516]
#  * x: [1.7713973014869868, 1.0401477423988654, 0.8831562165623505]
#
# Exiting inner optimizer with x = [1.7713973014869868, 1.0401477423988654, 0.8831562165623505]
# Current distance to box: 0.228603
# Decreasing barrier term μ.
#
# Fminbox iteration 2
# -------------------
# Calling inner optimizer with mu = 1.78554e-9
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     1.719133e-02     1.462476e-02
#  * Current step size: 0.7461479312473024
#  * time: 4.792213439941406e-5
#  * g(x): [0.0006839626023709488, 0.014624756982062921, 0.006279893287580216]
#  * ~inv(H): [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
#  * x: [1.7713973014869868, 1.0401477423988654, 0.8831562165623505]
#      1     1.706742e-02     1.479769e-03
#  * Current step size: 0.9999999999999982
#  * time: 714.5850930213928
#  * g(x): [0.0006880785829956403, -0.0010186112241614735, 0.0014797686823208298]
#  * ~inv(H): [1.0036968723844235 0.03749230548436748 0.02116358803420036; 0.03749230548436748 0.9131155068409662 0.07097924311508286; 0.02116358803420036 0.07097924311508286 1.0769774509157326]
#  * x: [1.770713338884616, 1.0255229854168024, 0.8768763232747704]
#      2     1.705343e-02     1.399794e-02
#  * Current step size: 8.866116283121784
#  * time: 2277.958687067032
#  * g(x): [0.000811655851328241, 0.013997944513934515, 0.0020799946846218028]
#  * ~inv(H): [2.270873973110514 -0.5373687886220692 2.876579481453531; -0.5373687886220694 0.5255293608662097 -1.230756618150179; 2.876579481453531 -1.2307566181501786 7.51121683555217]
#  * x: [1.764651136723138, 1.032609487664553, 0.8632585054063666]
#      3     1.672646e-02     6.941481e-03
#  * Current step size: 26.141197524598926
#  * time: 3346.68789601326
#  * g(x): [0.0008539218244902244, 0.006941480750891473, -0.0009588695113974447]
#  * ~inv(H): [2.258323121979547 -0.07525619793436134 2.8245232331428203; -0.07525619793436178 16.340416898184543 -0.4369257530189581; 2.8245232331428203 -0.4369257530189572 7.333710464691206]
#  * x: [1.7566942870572486, 0.9186285052002235, 0.8441748872177308]
#      4     1.666411e-02     1.357449e-02
#  * Current step size: 0.13716341545868177
#  * time: 4616.230960845947
#  * g(x): [0.0007604327782132275, -0.013574491090918725, 0.00019125367673724755]
#  * ~inv(H): [2.2554634452800983 0.13872065230877156 2.8131586845325405; 0.1387206523087711 0.7794261627019559 0.3451901559012347; 2.8131586845325405 0.3451901559012356 7.298897299668143]
#  * x: [1.7568729159691758, 0.9030218623345947, 0.8452246072089102]
#      5     1.665031e-02     8.896612e-03
#  * Current step size: 0.627566261065167
#  * time: 5030.288840055466
#  * g(x): [0.0008130701484829345, 0.008896611752375586, -0.0003938856392609978]
#  * ~inv(H): [2.261853204809662 0.05745664192328623 2.8069042023133557; 0.057456641923285756 0.2961586187566878 0.21502422533727683; 2.8069042023133557 0.21502422533727772 7.27602483182757]
#  * x: [1.7566406580784875, 0.9095540783530027, 0.8459466985527908]
#      6     1.664347e-02     1.435200e-02
#  * Current step size: 2.001903482716772
#  * time: 6515.201367855072
#  * g(x): [0.0007484327044794361, -0.01435199814067794, -0.00020551902627280008]
#  * ~inv(H): [2.3417188111536804 0.12362717084237698 2.8344857481305974; 0.12362717084237651 0.22460453079033948 0.16527287895751663; 2.8344857481305974 0.16527287895751752 7.243839973131015]
#  * x: [1.7541490579728019, 0.9043554761845412, 0.8432856175509493]
#
# Exiting inner optimizer with x = [1.7541490579728019, 0.9043554761845412, 0.8432856175509493]
# Current distance to box: 0.245851
# Decreasing barrier term μ.
#
# Fminbox iteration 3
# -------------------
# Calling inner optimizer with mu = 1.78554e-12
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     1.664347e-02     1.435200e-02
#  * Current step size: 2.001903482716772
#  * time: 2.288818359375e-5
#  * g(x): [0.0007484267597806431, -0.014351997120222412, -0.00020551705415918758]
#  * ~inv(H): [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
#  * x: [1.7541490579728019, 0.9043554761845412, 0.8432856175509493]
#      1     1.662662e-02     9.208984e-03
#  * Current step size: 0.19527558766164116
#  * time: 831.7164759635925
#  * g(x): [0.0007973971105547839, 0.009208983856440778, -3.0422069470690452e-5]
#  * ~inv(H): [1.003259827905723 -0.008284705708524152 -0.00047784025549823205; -0.008284705708524152 0.11901046098650059 -0.00572735201081585; -0.00047784025549823194 -0.00572735201081585 1.0000166230651577]
#  * x: [1.754002908497464, 0.9071580708563107, 0.8433257500144747]
#      2     1.662213e-02     9.339377e-03
#  * Current step size: 0.41738883022520806
#  * time: 1252.748780965805
#  * g(x): [0.0007954225227335379, 0.00933937694237356, -3.8540972086779134e-5]
#  * ~inv(H): [1.003259827905723 -0.008284705708524152 -0.00047784025549823205; -0.008284705708524152 0.11901046098650059 -0.00572735201081585; -0.00047784025549823194 -0.00572735201081585 1.0000166230651577]
#  * x: [1.7537008369786706, 0.9067033117635066, 0.8433606214721145]
#      3     1.661689e-02     1.398623e-02
#  * Current step size: 0.4337962676578905
#  * time: 1672.6721909046173
#  * g(x): [0.0007720447558694296, -0.013986231102608137, -3.621529715230504e-5]
#  * ~inv(H): [1.0720176806081028 0.01232751664665583 -0.0048887971177560665; 0.01232751664665583 0.020539768136724312 -0.0016140115853294258; -0.0048887971177560665 -0.0016140115853294258 1.000017259439037]
#  * x: [1.7533882174080384, 0.9062239172391381, 0.8434007092779676]
#      4     1.661671e-02     6.851416e-03
#  * Current step size: 0.09392488137278064
#  * time: 12688.334417819977
#  * g(x): [0.0007710173951143572, -0.006851415958467631, -0.00010842568387537807]
#  * ~inv(H): [1.2653112110809572 -0.008819864439402442 -0.03696132712279363; -0.008819864439402442 0.003759292962241343 0.010360845758426657; -0.03696132712279363 0.010360845758426657 1.0015853148001193]
#  * x: [1.753326658366311, 0.9062499999999999, 0.84340234510689]
#
# Exiting inner optimizer with x = [1.753326658366311, 0.9062499999999999, 0.84340234510689]
# Current distance to box: 0.246673
# Decreasing barrier term μ.
#
# Fminbox iteration 4
# -------------------
# Calling inner optimizer with mu = 1.78554e-15
#
# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     1.661671e-02     6.851416e-03
#  * Current step size: 0.09392488137278064
#  * time: 7.867813110351562e-6
#  * g(x): [0.0007710173891946662, -0.006851415957453361, -0.00010842568190366075]
#  * ~inv(H): [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
#  * x: [1.753326658366311, 0.9062499999999999, 0.84340234510689]

###########################################################################################
###### old "opt1" SHG QPM output copied from supercloud ##########
#######################################################################################
#
#
#  * x: [0.6349909188228182, 0.5057835978979331, 0.30985461586616436, 0.24741736682864388]
#      1     2.954875e-02     1.081495e-01
#  * Current step size: 0.5
#  * time: 727.6242680549622
#  * g(x): [-0.04484091724653931, -0.021542481578597505, -0.10814952758994184, -0.020618055580778887]
#  * ~inv(H): [0.9903891051276432 -0.0884753335048219 0.031872640178480426 0.004003652318971165; -0.0884753335048219 0.5381669106929468 0.06189875628141872 0.0014363276600221908; 0.031872640178480426 0.06189875628141872 1.0462880487467827 0.009975990725897138; 0.004003652318971165 0.0014363276600221908 0.009975990725897138 1.0018898229254998]
#  * x: [0.7334970625238433, 0.8159109163565503, 0.37490689234678776, 0.2663147502490992]
#      2     3.607659e-03     5.146272e-03
#  * Current step size: 4.853538816480108
#  * time: 1378.214926958084
#  * g(x): [0.004708744960648093, 0.00407197958549979, 0.00028945605677111255, 0.005146272381787064]
#  * ~inv(H): [1.4858189830801543 0.06465672370080261 1.3112716363553227 0.2311185944186338; 0.06465672370080261 0.585494928847007 0.4574255235987997 0.07161113273600246; 1.3112716363553227 0.4574255235987997 4.3484381145555435 0.5970210907173434; 0.2311185944186338 0.07161113273600248 0.5970210907173434 1.1058374240799353]
#  * x: [0.9569226171707059, 0.8855595438747766, 0.9385187240745907, 0.3728323974710411]
#      3     3.265571e-03     3.794239e-03
#  * Current step size: 5.327718493239273
#  * time: 2229.869994163513
#  * g(x): [-0.0007756439373469636, 0.002039038070293253, 0.003794239069306582, -0.0013188777503567112]
#  * ~inv(H): [8.91771213079452 2.971294939477272 13.312626030332783 5.992958279171792; 2.971294939477272 1.7156767422291523 5.10376740423309 2.335104845359737; 13.31262603033278 5.103767404233089 23.388670992695907 9.973341642541195; 5.992958279171791 2.3351048453597367 9.973341642541195 5.557732043665583]
#  * x: [0.9098864374815023, 0.8685667303188279, 0.8726245107734417, 0.3342403783465787]
#      4     2.394051e-03     4.920033e-03
#  * Current step size: 2.156167350043872
#  * time: 3697.8732891082764
#  * g(x): [0.004920032541861801, 0.0025029149779460204, -0.0035038243573527294, 0.004089938444368879]
#  * ~inv(H): [42.77984700135157 16.66307244294332 70.00775568825783 31.340240242722047; 16.66307244294332 7.2448437569029105 27.960765656940204 12.591278360325004; 70.00775568825783 27.960765656940204 117.66620377243066 52.48235508956417; 31.34024024272205 12.591278360325004 52.48235508956417 24.52364696463175]
#  * x: [0.8198687779485274, 0.8308793849071482, 0.7094683855581461, 0.26820939345092315]

# Exiting inner optimizer with x = [0.8198687779485274, 0.8308793849071482, 0.7094683855581461, 0.26820939345092315]
# Current distance to box: 0.268209
# Decreasing barrier term μ.

# Fminbox iteration 2
# -------------------
# Calling inner optimizer with mu = 4.67565e-8

# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     2.172938e-03     4.991994e-03
#  * Current step size: 2.156167350043872
#  * time: 1.9073486328125e-5
#  * g(x): [0.004991993925428592, 0.002551217113011675, -0.003598633748967474, 0.004174083033529]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [0.8198687779485274, 0.8308793849071482, 0.7094683855581461, 0.26820939345092315]
#      1     2.021996e-03     1.297026e-02
#  * Current step size: 7.2376603398020904
#  * time: 847.0312190055847
#  * g(x): [0.0032762160924712457, 0.003687013589298837, 0.01297025732524611, 0.0018633143117199997]
#  * ~inv(H): [4.4389351396619645 1.8915814717904647 -1.457723575316473 2.8171224186736206; 1.8915814717904647 2.035234650658385 -0.8416376406146253 1.5518296045933546; -1.457723575316473 -0.8416376406146253 1.3145809691298014 -1.1768114741980462; 2.8171224186736206 1.5518296045933544 -1.1768114741980458 3.3067526065384127]
#  * x: [0.78373842149792, 0.8124145419900792, 0.7355140743205213, 0.23799879822410952]
#      2     1.383538e-03     6.810664e-03
#  * Current step size: 5.0582115717180605
#  * time: 1725.374402999878
#  * g(x): [0.003928193347619528, 0.0030802095967869, -0.006810664311312841, 0.0027158652285788607]
#  * ~inv(H): [12.826482940041675 7.599432772877158 2.596120084835526 9.205637921882815; 7.599432772877158 5.904898103632529 1.7753940658169638 5.905417116138272; 2.596120084835526 1.7753940658169642 1.9005589966354612 1.9700707820131529; 9.205637921882815 5.905417116138272 1.9700707820131529 8.170119495114061]
#  * x: [0.7439842575843102, 0.7837018214522238, 0.7002141464334269, 0.20841286119371594]
#      3     1.163760e-03     1.073360e-02
#  * Current step size: 0.26878110336516636
#  * time: 2570.7323439121246
#  * g(x): [0.003681418008572685, 0.0013107525274692303, -0.010733602209913986, 0.0032074974821944408]
#  * ~inv(H): [31.035682026326572 17.749721648512036 -1.3902939851965623 24.02325703416952; 17.749721648512036 11.435986473150544 -0.9309184107635513 14.21917243083576; -1.3902939851965623 -0.9309184107635509 0.9266568149752612 -1.0678099898818514; 24.023257034169514 14.21917243083576 -1.0678099898818518 20.204873257782477]
#  * x: [0.7221826911399528, 0.7697286796656979, 0.6980442698956598, 0.19144661826806325]
#      4     1.135939e-03     1.253340e-02
#  * Current step size: 0.017144183745201817
#  * time: 3229.3979160785675
#  * g(x): [0.003197782986824333, 0.0002819330966671423, -0.012533398020053522, 0.002780356321465857]
#  * ~inv(H): [5.786442526556105 2.4462565521600226 -1.648455961389378 3.7134353551908426; 2.446256552160026 2.1654154066849607 -1.0539344367430452 1.9113746750042662; -1.648455961389378 -1.0539344367430448 1.1572321836948807 -1.262184979415326; 3.7134353551908355 1.9113746750042662 -1.2621849794153264 3.868945499759164]
#  * x: [0.7182481316537885, 0.7673982028648796, 0.6983821786792992, 0.18830330084011704]

# Exiting inner optimizer with x = [0.7182481316537885, 0.7673982028648796, 0.6983821786792992, 0.18830330084011704]
# Current distance to box: 0.188303
# Decreasing barrier term μ.

# Fminbox iteration 3
# -------------------
# Calling inner optimizer with mu = 4.67565e-11

# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     1.135697e-03     1.253349e-02
#  * Current step size: 0.017144183745201817
#  * time: 9.775161743164062e-6
#  * g(x): [0.003197893669337823, 0.00028199544189876747, -0.01253348584662986, 0.002780526561735807]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [0.7182481316537885, 0.7673982028648796, 0.6983821786792992, 0.18830330084011704]
#      1     1.026789e-03     5.756140e-03
#  * Current step size: 1.1909636361666165
#  * time: 632.4637649059296
#  * g(x): [0.005756140158227877, 0.003919115672652746, 0.001785785338688995, 0.0037801841510116422]
#  * ~inv(H): [1.2527376318075742 0.08742557137985534 -0.5257016651060856 0.19636790877371513; 0.08742557137985534 1.0134533695813386 -0.3016551741957248 0.0739533348601604; -0.5257016651060855 -0.3016551741957248 1.2384898959595527 -0.3654413580769169; 0.19636790877371513 0.0739533348601604 -0.3654413580769168 1.1504071124416515]
#  * x: [0.7144395565812798, 0.7670623565480135, 0.7133091045570443, 0.1849917948156943]
#      2     2.448063e-04     2.230679e-02
#  * Current step size: 10.19100898469945
#  * time: 2318.543219804764
#  * g(x): [-0.011295990981815272, -0.0007646127447830264, -0.022306793526050292, -0.007043092850548948]
#  * ~inv(H): [6.945812230905965 3.7612548169070004 -4.4150783036179355 4.184416801837248; 3.7612548169070004 3.354497489953761 -2.7193560580168166 2.6454117229083525; -4.4150783036179355 -2.7193560580168166 3.6098938821546214 -3.0835010205929825; 4.184416801837248 2.645411722908352 -3.083501020592982 3.9439233909829303]
#  * x: [0.6394634795622367, 0.7240976673679469, 0.7477342566612786, 0.13285149688836012]
#      3     2.117484e-04     9.503506e-03
#  * Current step size: 0.42562886951431206
#  * time: 2736.890150785446
#  * g(x): [0.009503506102719224, 0.0034206079319491806, 0.008341861907001164, 0.006491664224558904]
#  * ~inv(H): [4.352868089521303 2.609636622868252 -4.197365974783797 2.3959285041791185; 2.609636622868252 2.94541240909181 -2.9526382066820456 1.8600303988052613; -4.197365974783797 -2.9526382066820456 4.655046473473991 -2.962175845284554; 2.3959285041791185 1.8600303988052609 -2.9621758452845537 2.711092345751353]
#  * x: [0.6447075386382123, 0.7253846661981084, 0.7506522977431904, 0.13637751200391168]
#      4     1.085242e-04     1.022087e-02
#  * Current step size: 0.5000000000000001
#  * time: 3434.6742498874664
#  * g(x): [0.00485871031060522, 0.005304453669038327, 0.010220869627939314, 0.002742914519724142]
#  * ~inv(H): [2.424575769764026 1.7508161702971377 -2.030660835663216 0.9704208725530687; 1.7508161702971377 2.7124775349790777 -1.9134573517064486 1.2116840276459673; -2.030660835663216 -1.9134573517064495 2.2572387816009414 -1.367092589070745; 0.9704208725530687 1.2116840276459668 -1.3670925890707446 1.658483383810435]
#  * x: [0.629290654535625, 0.7142246705802344, 0.7658459012111655, 0.12536671438304348]

# Exiting inner optimizer with x = [0.629290654535625, 0.7142246705802344, 0.7658459012111655, 0.12536671438304348]
# Current distance to box: 0.125367
# Decreasing barrier term μ.

# Fminbox iteration 4
# -------------------
# Calling inner optimizer with mu = 4.67565e-14

# (numbers below include barrier contribution)
# Iter     Function value   Gradient norm
#      0     1.085239e-04     1.022087e-02
#  * Current step size: 0.5000000000000001
#  * time: 1.3113021850585938e-5
#  * g(x): [0.004858710480692536, 0.005304453745831483, 0.010220869489589248, 0.0027429148221226394]
#  * ~inv(H): [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
#  * x: [0.629290654535625, 0.7142246705802344, 0.7658459012111655, 0.12536671438304348]
