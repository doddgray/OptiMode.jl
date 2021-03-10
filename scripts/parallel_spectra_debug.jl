# using Revise
using Distributed
pids = addprocs(8)
@show wp = CachingPool(workers()) #default_worker_pool()
@everywhere begin
	using Revise, Distributed, LinearAlgebra, Statistics, FFTW, StaticArrays, HybridArrays, ChainRules, Zygote, ForwardDiff, GeometryPrimitives, OptiMode
	using Zygote: dropgrad, @ignore
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


	λs = collect(reverse(1.45:0.02:1.65))
	ωs = 1 ./ λs

	n1F,ng1F = solve_n(ms,ωs,rwg_pe(p_pe)); n1S,ng1S = solve_n(ms,2*ωs,rwg_pe(p_pe))
end

##

function foo2()

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
  using Zygote
  function f_pmap_zygote_solve(A, bs)
    xs = pmap((b) -> A \ b, wp, bs)
    return sum(sum(xs))
  end
end
wp = default_worker_pool()
A = randn(8,8) #sprand(200, 200, 0.01) + 200*I
b0s = [randn(8) for i=1:8]
Zygote.gradient(f_pmap_zygote_solve, A, b0s)
