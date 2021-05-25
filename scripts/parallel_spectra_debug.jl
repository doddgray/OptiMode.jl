using Revise
using OptiMode
using GeometryPrimitives
using LinearAlgebra
using StaticArrays
using LoopVectorization
using ChainRules
using Zygote
using ForwardDiff
using UnicodePlots
using FFTW
using OhMyREPL
using Crayons.Box       # for color printing
using Rotations: RotY, MRP
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)
LNx = rotate(MgO_LiNbO₃,Matrix(MRP(RotY(π/2))),name=:LiNbO₃)
AD_style = BOLD*BLUE_FG #NEGATIVE*BOLD*BLUE_FG      # defined in Crayons.Box
FD_style = BOLD*RED_FG
MAN_style = BOLD*GREEN_FG
AD_style_N = NEGATIVE*BOLD*BLUE_FG #NEGATIVE*BOLD*BLUE_FG      # defined in Crayons.Box
FD_style_N = NEGATIVE*BOLD*RED_FG
MAN_style_N = NEGATIVE*BOLD*GREEN_FG

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
grid = Grid(Δx,Δy,Nx,Ny)
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).
p = [
       1.7,                #   top ridge width         `w_top`         [μm]
       0.7,                #   ridge thickness         `t_core`        [μm]
       0.5,                #   ridge thickness         `t_core`        [μm]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
               ];
geom = rwg(p)
##
ms = ModeSolver(1.45, geom, grid, nev=4)
ωs = range(0.6,0.8,length=16) |> collect
using OptiMode: _solve_n_serial
##

function E_relpower_xyz(ms::ModeSolver{ND,T},ω²H) where {ND,T<:Real}
    E = 1im * ε⁻¹_dot( fft( kx_tc( reshape(ω²H[2],(2,size(ms.grid)...)),mn(ms),ms.M̂.mag), (2:1+ND) ), flat( ms.M̂.ε⁻¹ ))
    Es = reinterpret(reshape, SVector{3,Complex{T}},  E)
    Pₑ_xyz_rel = normalize([mapreduce((ee,epss)->(abs2(ee[a])*inv(epss)[a,a]),+,Es,ms.M̂.ε⁻¹) for a=1:3],1)
    return Pₑ_xyz_rel
end
# E_relpower_xyz(ms,(real(ms.ω²[eigind]),ms.H⃗[:,eigind]))
TE_filter = (ms,ω²H)->E_relpower_xyz(ms,ω²H)[1]>0.7
TM_filter = (ms,ω²H)->E_relpower_xyz(ms,ω²H)[2]>0.7
oddX_filter = αX->sum(abs2,𝓟x̄(ms.grid)*αX[2])>0.9
evenX_filter = αX->sum(abs2,𝓟x(ms.grid)*αX[2])>0.9
ns_TE,ngs_TE,gvds_TE,Evs_TE = _solve_n_serial(ms,ωs,geom;f_filter=TE_filter)
ns_TM,ngs_TM,gvds_TM,Evs_TM = _solve_n_serial(ms,ωs,geom;f_filter=TM_filter)
##
using GLMakie, AbstractPlotting
using AbstractPlotting: lines, lines!, scatterlines, scatterlines!, GeometryBasics, Point, PointBased
using Colors
import Colors: JULIA_LOGO_COLORS
logocolors = JULIA_LOGO_COLORS


##
# interactions

function indicator(ax::Axis,ob)
    register_interaction!(ax, :indicator) do event::MouseEvent, axis
    if event.type === MouseEventTypes.over
        ob[] = event.data
    end
    end
end
function indicator(grid::GridLayout,ob)
    foreach(Axis,grid;recursive=true) do ax
    indicator(ax,ob)
    end
end
function indicator(grid::GridLayout)
    ob = Observable(Point2f0(0.,0.))
    indicator(grid,ob)
    ob
end
function indicator(fig,args...; tellwidth=false, kwargs...)
    Label(
        fig,
        lift(ind->"x: $(ind[1])  y: $(ind[2])",indicator(fig.layout)),
        args...; tellwidth=tellwidth, kwargs...
    )
end



##

ωind = 5
Es = [ view(real(Evs_TE),i,:,:,ωind) for i=1:3 ]
Emagmax = [sqrt(maximum(abs2.(Evs_TE[:,:,:,ωind]))) for i=1:3]

fig = Figure()

# dispersion plots
disp = ( (ns_TE,ns_TM), (ngs_TE,ngs_TM), (gvds_TE,gvds_TM) )
# labels_disp = [la*lb for la in ["neff", "ng", "gvd"], lb in [" TE₀₀", " TM₀₀"]]
labels_disp = [" TE₀₀", " TM₀₀"]
# labels_disp = ["n", "ng" ] .* [",TE",",TM"]

n_disp = length(disp)
n_modes = 2
λs = inv.(ωs)
ax_disp = fig[1:n_disp,1] = [Axis(fig) for i=1:n_disp]
colors_disp = [logocolors[:red],logocolors[:blue]]
sls_disp 	= 	[scatterlines!(
					ax_disp[i],
					λs,
					disp[i][j],
					color=colors_disp[j],
					markercolor=colors_disp[j],
					strokecolor=colors_disp[j],
					markersize=2,
					linewidth=2,
					label=labels_disp[j],
				) for i=1:n_disp, j=1:n_modes ]
disp_models = [:n,:ng,:gvd]
lns_bulk = [plot_model!(ax_disp[i],[LNx,];model=disp_models[i],linewidth=3) for i=1:3]

fig[4,1] = Legend(fig,ax_disp[1],orientation=:horizontal)

hidexdecorations!.(ax_disp[1:end-1])
[axx.xlabel= "λ [μm]" for axx in ax_disp[end,:]]
[axx.ylabel= lbl for (axx,lbl) in zip(ax_disp,["neff","ng","gvd"])]
linkxaxes!(ax_disp...)


# spatial plots
xs = x(ms.grid)
ys = y(ms.grid)
ax_geom = fig[1,2] = Axis(fig)
plot_shapes(geom,ax_geom)

ax_n	= fig[1,3] = Axis(fig)
ax_E 	= fig[2,2:3] = [Axis(fig) for i=1:2]

# poly!(ax_geom,geom.shapes[1],color=:red,strokecolor=:black,strokewidth=1)
# poly!(ax_geom,geom.shapes[2],color=:blue,strokecolor=:black,strokewidth=1)


label_n = "nₓ"
cmap_n 	  = :viridis
n_spatial = sqrt.(getindex.(inv.(ms.M̂.ε⁻¹),1,1))
n_spatial_max = maximum(n_spatial)
heatmap_n = heatmap!(ax_n, xs, ys, n_spatial, colormap=cmap_n,label=label_n,colorrange=(1,n_spatial_max);overdraw=false)
# wf_n = wireframe!(ax_n, xs, ys, n_spatial, colormap=cmap_n,linewidth=0.1,color=:white)
cbar_n = Colorbar(fig[1,4], heatmap_n,  width=20 )
text!(ax_n,label_n,position=(-1.4,1.1),textsize=0.7,color=:white)

cmap_E = :diverging_bkr_55_10_c35_n256
label_base = ["x","y"]
labels_E = "E".*label_base
heatmaps_E = [heatmap!(ax_E[j], xs, ys, Es[j],colormap=cmap_E,label=labels_E[j],colorrange=(-Emagmax[j],Emagmax[j])) for j=1:2]
cbars_E = Colorbar(fig[2,4], heatmaps_E[2],  width=20 )
# wfs_E = [wireframe!(ax_E[j], xs, ys, Es[j], colormap=cmap_E,linewidth=0.02,color=:white) for j=1:2]
map( (axx,ll)->text!(axx,ll,position=(-1.4,1.1),textsize=0.7,color=:white), ax_E, labels_E )

ax_spatial = vcat(ax_geom, ax_n, ax_E)
hidexdecorations!.(ax_spatial[1:end-1,:])
hideydecorations!.(ax_spatial[:,2:end])
[axx.xlabel= "x [μm]" for axx in ax_spatial[end,:]]
[axx.ylabel= "y [μm]" for axx in ax_spatial[:,1]]
[ axx.aspect=DataAspect() for axx in ax_spatial ]
linkaxes!(ax_spatial...)
txt= fig[4,2] = indicator(fig)
# for axx in ax_all
# 	on(mouseposition) do mpos
#
# end

fig

##

using Distributed
pids = addprocs(8)
@show wp = CachingPool(workers()) #default_worker_pool()
@everywhere begin
	using LinearAlgebra, Statistics, FFTW, StaticArrays, HybridArrays, ChainRules, Zygote, ForwardDiff, GeometryPrimitives, OptiMode
	using Zygote: dropgrad, @ignore
	using Rotations: RotY, MRP
	LNx = rotate(MgO_LiNbO₃,Matrix(MRP(RotY(π/2))))
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


	λs = collect(reverse(1.45:0.02:1.65))
	ωs = 1 ./ λs

	geom1 = rwg_pe(p_pe)
	# numngs1(oms) = sum(solve_n(oms,rwg_pe(p_pe),gr)[2])
	# numngs2(oms,x) = sum(solve_n(oms,rwg_pe(x),gr)[2])
	# numngs3(oms) = sum(pmap(x->solve_n(x,rwg_pe(p_pe),gr)[2]),oms))

	# numngs1([0.6,0.7])
	# n1F,ng1F = solve_n(ms,ωs,rwg_pe(p_pe)); n1S,ng1S = solve_n(ms,2*ωs,rwg_pe(p_pe))
end

##
@everywhere function foo1(x)
	Zygote.@ignore begin
	Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
	p_pe = [
	      1.7,                #   top ridge width         `w_top`         [μm]
	      0.7,                #   ridge thickness         `t_core`        [μm]
	      0.5,                #   top layer etch fraction `etch_frac`     [1]
	      π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
	               ];
	ms = ModeSolver(1.45, rwg_pe(p_pe), gr)
	end
	return x^2
end

@everywhere function foo2(x)
	Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
	p_pe = [
	      1.7,                #   top ridge width         `w_top`         [μm]
	      0.7,                #   ridge thickness         `t_core`        [μm]
	      0.5,                #   top layer etch fraction `etch_frac`     [1]
	      π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
	               ];
	ms = ModeSolver(1.45, rwg_pe(p_pe), gr)
	return x^2
end

@everywhere function foo3(x)
	sumeps = Zygote.@ignore begin
	Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
	p_pe = [
	      1.7,                #   top ridge width         `w_top`         [μm]
	      0.7,                #   ridge thickness         `t_core`        [μm]
	      0.5,                #   top layer etch fraction `etch_frac`     [1]
	      π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
	               ];
	ms = ModeSolver(1.45, rwg_pe(p_pe), gr)
	sum(sum(ms.M̂.ε⁻¹))
	end
	return sumeps*x^2
end

function _solve_n_parallel1(ωs::Vector{T},geom::Vector{<:Shape},gr::Grid;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol,wp=nothing) where {ND,T<:Real}
	wp = CachingPool(workers())
	nω = length(ωs)
	ind0 = Int(ceil(nω/2))
	ω0 = ωs[ind0]
	ms = @ignore(ModeSolver(kguess(ω0,geom), geom, gr))
	nng0 = solve_n(ms,ω0,geom)
	ms_copies = [ deepcopy(ms) for om in 1:nω ]
	geoms = [deepcopy(geom) for om in 1:nω ]
	nng = pmap(wp,ms_copies,ωs,geoms) do m,om,s
		@ignore( replan_ffts!(m) );
		solve_n(m,om,s)
	end
	n = [res[1] for res in nng]; ng = [res[2] for res in nng]
	return n, ng
end

function _solve_n_parallel2(ωs::Vector{T},geom::Vector{<:Shape},gr::Grid;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol,wp=nothing) where {ND,T<:Real}
	wp = CachingPool(workers())
	nω = length(ωs)
	# ind0 = Int(ceil(nω/2))
	# ω0 = ωs[ind0]
	# ms = @ignore(ModeSolver(kguess(ω0,geom), geom, gr))
	# nng0 = solve_n(ms,ω0,geom)
	# ms_copies = [ deepcopy(ms) for om in 1:nω ]
	geoms = [deepcopy(geom) for om in 1:nω ]
	nng = pmap(wp,ωs,geoms) do om,s
		@ignore( replan_ffts!(m) );
		solve_n(om,s,gr)
	end
	n = [res[1] for res in nng]; ng = [res[2] for res in nng]
	return n, ng
end

n1,ng1 = _solve_n_parallel1(ωs,rwg_pe(p_pe),gr)
gr2 = Grid(6.0, 4.0, 256, 256)
n2,ng2 = _solve_n_parallel1(ωs,rwg_pe(p_pe),gr2)

n1,ng1 = _solve_n_parallel2(ωs,rwg_pe(p_pe),gr)

##
gradient(var_ng2,ωs,p)

##

solve_n(ms,ωs,shapes;n_procs=10)
var_ng2(ωs,p)
var_ng1(ωs,p)

gradient(var_ng2,ωs,p)

@everywhere begin
	function var_ng1(ms,ωs,p,procs)
		ng = solve_n(ms,ωs,rwg(p);n_procs=procs)[2]
		mean( abs2.( ng ) ) - abs2(mean(ng))
	end

	function var_ng_pb(ms,ωs,p)
		var_ng,var_ng_pb = Zygote.pullback(ωs,p) do ωs,p
			ng = solve_n(ms,ωs,rwg(p))[2]
			mean( abs2.( ng ) ) - abs2(mean(ng))
		end
		return (var_ng, var_ng_pb(1))
	end
end

var_ng1(ms,ωs,p,10)
gradient(p) do p
	var_ng1(ms,ωs,p,10)
end

gradient((om,x)->var_ng1(ms,om,x,10), ωs, p)

gradient((om,x)->var_ng1(ms,om,x,1), ωs, p)


ωs2 = collect(0.55:0.05:1.0)
(n2,ng2), nng_pb = pullback(ωs2,p) do om, p
	s = rwg(p)
	solve_n(ms,om,s;n_procs=10)
end


plot_nng(ωs2,solve_n(ms,ωs2,shapes;n_procs=10)...;c_ng=:green,m=".")


# n,ng = solve_n(ms,ωs,shapes;n_procs=10)
using Plots
function plot_nng(ωs,n,ng;c_n=:blue,c_ng=:red,ls_n=:solid,ls_ng=:dash,
		legend=:bottomleft,m=nothing,xlabel="λ (μm)",ylabel="n, ng")
	p_nng = plot(
		(1 ./ ωs),
		n;
		ls=ls_n,
		label="n",
		color=c_n,
		legend,
		xlabel,
		ylabel,
		m
		)
	plot!(p_nng,
		(1 ./ ωs),
		ng;
		ls=ls_ng,
		label="ng",
		color=c_ng,
		m
		)
	return p_nng
end

plot_nng(ωs,n,ng;c_ng=:green)



vng,∂vng = var_ng(ms,ωs,p)
ε⁻¹ = make_εₛ⁻¹(rwg(p),dropgrad(ms))
nω = length(ωs)
ms_copies = [ deepcopy(ms) for om in 1:nω ]
vmap(ms_copies,ωs) do m,om
	solve_n(m,om,ε⁻¹)
end


solve_ω²(ms,ks)
solve_k(ms,ωs)
solve_k(ms,ωs,rwg(p))
solve_n(ms,ωs,rwg(p))

@btime solve_ω²($ms,$ks) # 1.982 s (26246 allocations: 19.68 MiB)
@btime solve_k($ms,$ωs) # 4.873 s (57515 allocations: 279.31 MiB)
@btime solve_k($ms,$ωs,rwg($p)) # 5.074 s (3335372 allocations: 450.57 MiB)
@btime solve_n($ms,$ωs,rwg($p)) # 5.147 s (3337492 allocations: 514.33 MiB)

@everywhere using OptiMode, ArrayInterface, StaticArrays, HybridArrays, ChainRules, Zygote

@everywhere p = [
    # 1.45,               #   propagation constant    `kz`            [μm⁻¹]
    1.7,                #   top ridge width         `w_top`         [μm]
    0.7,                #   ridge thickness         `t_core`        [μm]
    π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
    2.4,                #   core index              `n_core`        [1]
    1.4,                #   substrate index         `n_subs`        [1]
    0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
]

m.M̂.𝓕! = plan_fft!(randn(ComplexF64, (3,m.M̂.Nx,m.M̂.Ny,m.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
m.M̂.𝓕⁻¹! = plan_bfft!(randn(ComplexF64, (3,m.M̂.Nx,m.M̂.Ny,m.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
m.M̂.𝓕 = plan_fft(randn(ComplexF64, (3,m.M̂.Nx,m.M̂.Ny,m.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
m.M̂.𝓕⁻¹ = plan_bfft(randn(ComplexF64, (3,m.M̂.Nx,m.M̂.Ny,m.M̂.Nz)),(2:4),flags=FFTW.PATIENT);

@everywhere rwg2(p) = ridge_wg(p[1],p[2],p[3],p[6],p[4],p[5],6.0,4.0)

pmap(ms_copies,ωs) do m,om
	solve_n(m,om,rwg2(p))
end

n,ng = solve_n(ms,ωs,rwg(p))
var(ng)
mean( abs2.( ng .- mean(ng) ) )
mean( abs2.( ng ) ) - abs2(mean(ng))
var_ng
# solve_ω²(ms,1.5,ε⁻¹)
# (ω²,H⃗), ω²H⃗_pb = Zygote.pullback(1.5,ε⁻¹) do x,y
# 	solve_ω²(ms,x,y)
# end

function calc_ng(p)
	# ε⁻¹ = make_εₛ⁻¹(ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),g)
	ε⁻¹ = HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},Float64,5,5,Array{Float64,5}}(make_εₛ⁻¹(ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),ms))
	solve_nω(ms,p[1],ε⁻¹;eigind=1)[2]
end

function calc_ng(ms,p)
	ng,ng_pb = Zygote.pullback(p) do p
		solve_nω(ms,p[1],ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],ms.M̂.Δx,ms.M̂.Δy);eigind=1)[2]
	end
	return (ng, real(ng_pb(1)[1]))
end

function calc_ng(p)
	solve_n(ms,p[1],ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],ms.M̂.Δx,ms.M̂.Δy);eigind=1)[2]
end

function calc_ng_pb(ms,p)
	ng,ng_pb = Zygote.pullback(p) do p
		solve_n(ms,p[1],ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],ms.M̂.Δx,ms.M̂.Δy);eigind=1)[2]
	end
	return (ng, real(ng_pb(1)[1]))
end


p̄ω_FD3 = FiniteDifferences.jacobian(central_fdm(3,1),x->calc_ng2(x),pω)[1][1,:]
p̄_FD3 = FiniteDifferences.jacobian(central_fdm(3,1),x->calc_ng(x),p)[1][1,:]


## minimal working pmap + Zygote example from:
# https://discourse.julialang.org/t/passing-constructed-closures-to-child-processes/34723
# where issues with closures are discussed
using Distributed
addprocs(8)
@everywhere begin
  using OptiMode, ChainRules, Zygote
  function f_pmap_zygote_solve(ωs, p)
	  Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
  	# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
  	gr = Grid(Δx,Δy,Nx,Ny)
  	rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy)
	geom = rwg_pe(p)
	nng = pmap(ωs) do om
  	  # @ignore( replan_ffts!(m) );
  	  solve_n(om,geom,gr)
    end
    # return sum(sum(xs))
	# n = [res[1] for res in nng]
	ng = [res[2] for res in nng]
	# return n, ng
	return sum(ng)
  end

end
##
# wp = default_worker_pool()
# A = randn(8,8) #sprand(200, 200, 0.01) + 200*I
# b0s = [randn(8) for i=1:8]
# Zygote.gradient(f_pmap_zygote_solve, A, b0s)
ωs = collect(range(0.6,0.7,length=10))
Zygote.gradient(f_pmap_zygote_solve, ωs, p_pe)
