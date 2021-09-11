using Revise
using OptiMode
using LinearAlgebra
using Statistics
using ArrayInterface
using RecursiveArrayTools
using StaticArrays
using HybridArrays
using SparseArrays
using FFTW
using LinearMaps
using GeometryPrimitives
using BenchmarkTools
using ChainRules
using Zygote
using ForwardDiff
using FiniteDifferences
using FiniteDiff
using UnicodePlots
using OhMyREPL
using Crayons.Box       # for color printing
using Zygote: @ignore, dropgrad
using Setfield: @set
using StaticArrays: Dynamic
using IterativeSolvers: bicgstabl
using Rotations: RotY, MRP
using RuntimeGeneratedFunctions
using Tullio
RuntimeGeneratedFunctions.init(@__MODULE__)
LNx = rotate(MgO_LiNbO₃,Matrix(MRP(RotY(π/2))),name=:LiNbO₃_X);

gradRM(fn,in) 			= 	Zygote.gradient(fn,in)[1]
gradFM(fn,in) 			= 	ForwardDiff.gradient(fn,in)
gradFD(fn,in;n=3)		=	FiniteDifferences.grad(central_fdm(n,1),fn,in)[1]
gradFD2(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_gradient(fn,in;relstep=rs)

derivRM(fn,in) 			= 	Zygote.gradient(fn,in)[1]
derivFM(fn,in) 			= 	ForwardDiff.gradient(fn,in)
derivFD(fn,in;n=3)		=	FiniteDifferences.grad(central_fdm(n,1),fn,in)[1]
derivFD2(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_derivative(fn,in;relstep=rs)

AD_style = BOLD*BLUE_FG #NEGATIVE*BOLD*BLUE_FG      # defined in Crayons.Box
FD_style = BOLD*RED_FG
MAN_style = BOLD*GREEN_FG
AD_style_N = NEGATIVE*BOLD*BLUE_FG #NEGATIVE*BOLD*BLUE_FG      # defined in Crayons.Box
FD_style_N = NEGATIVE*BOLD*RED_FG
MAN_style_N = NEGATIVE*BOLD*GREEN_FG


Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
grid = Grid(Δx,Δy,Nx,Ny)
# rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).
LNxN = NumMat(LNx;expr_module=@__MODULE__())
SiO₂N = NumMat(SiO₂;expr_module=@__MODULE__())
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNxN,SiO₂N,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).

##
p = [
       1.7,                #   top ridge width         `w_top`         [μm]
       0.7,                #   ridge thickness         `t_core`        [μm]
       0.9, #0.5,                #   ridge thickness         `t_core`        [μm]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
               ];
om = 0.65
k1,Hv1 = solve_k(om,p,rwg,grid;nev=1,eigind=1);
k2,Hv2 = solve_k(1.1*om,p,rwg,grid;nev=1,eigind=1,kguess=k1,Hguess=Hv1);
k3,Hv3 = solve_k(1.1*om,p,rwg,grid;nev=1,eigind=1);
ε⁻¹,nng,nng⁻¹ = copy(smooth(om,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
ngvd = deepcopy(first(smooth(om,p,(:fnĝvds,),[false,],rwg,grid,volfrac_smoothing)));
# k2,Hv2 = solve_k(0.75,[1.7,0.7,0.9,0.22],rwg,Grid(6.0,4.0,128,128);nev=1,eigind=1);
E1 = E⃗(k1,Hv1,om,ε⁻¹,nng,grid; normalized=true, nnginv=false)
neff,ng,gvd,E = solve(om,p,rwg,grid;nev=1);
# neff, ng, gvd = neff_ng_gvd(om,ε⁻¹,nng,nng⁻¹,ngvd,k1,Hv1,grid)
##
fig =Figure();
#ax = fig[1,1]= Axis(fig,yscale=log10);
# l1s = [ lines!(ax,res1.residual_history[i,:]) for i=1:length(res1.λ)];
# l2s = [ lines!(ax,res2.residual_history[i,:],color=:red) for i=1:size(res.λ,1)];
# l3s = [ lines!(ax,res3.residual_history[i,:],color=:green) for i=1:size(res.λ,1)];
ax2 = fig[2,1][1,1] = Axis(fig,aspect=DataAspect()); hm2 = heatmap!(ax2,x(grid),y(grid),real(E1[1,:,:]),colormap=:bwr); cb2 = Colorbar(fig[2,1][1,2],hm2,width=30);
ax3 = fig[2,2][1,1] = Axis(fig,aspect=DataAspect()); hm3 = heatmap!(ax3,x(grid),y(grid),real(E1[2,:,:]),colormap=:bwr); cb3 = Colorbar(fig[2,2][1,2],hm3,width=30);
fig
##
using DFTK: LOBPCG
nev=1
ñₘₐₓ(ε⁻¹)
k_g = ñₘₐₓ(ε⁻¹)*om
M̂ = HelmholtzMap(SVector(0.,0.,k_g), ε⁻¹, grid);
##
X1 = randn(ComplexF64,size(M̂,1),2) #/N(grid)
res1 = LOBPCG(M̂,X1,I,HelmholtzPreconditioner(M̂),1e-10,200;display_progress=true)
EE1 = E⃗(k_g,copy(X1[:,1]),om,ε⁻¹,nng,grid; normalized=true, nnginv=false)
EE2 = E⃗(k_g,copy(X1[:,2]),om,ε⁻¹,nng,grid; normalized=true, nnginv=false)

##
eigind=1
p = [
       1.7,                #   top ridge width         `w_top`         [μm]
       0.7,                #   ridge thickness         `t_core`        [μm]
       0.9, #0.5,                #   ridge thickness         `t_core`        [μm]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
               ];
om = 0.75
# om = 0.65
grid = Grid(6.0,4.0,128,128)
# grid = Grid(6.0,4.0,256,256)
k,Hv = solve_k(om,p,rwg,grid;nev=1,eigind=1);
# (mag,m⃗,n⃗) = mag_m_n(k,g⃗(grid));
neff,ng,gvd,E = solve(om,p,rwg,grid;nev=1,eigind=1);
@show neff
@show ng
@show gvd
##
fneff(oo) = solve(oo,p,rwg,grid;nev=1,eigind=1)[1]
fng(oo) = solve(oo,p,rwg,grid;nev=1,eigind=1)[2]
fgvd(oo) = solve(oo,p,rwg,grid;nev=1,eigind=1)[3]
fk(oo) = solve_k(oo,p,rwg,grid;nev=1,eigind=1)[1]


fneff(oo_pp) = solve(oo_pp[1],oo_pp[2:5],rwg,grid;nev=1,eigind=1)[1]
fng(oo_pp) = solve(oo_pp[1],oo_pp[2:5],rwg,grid;nev=1,eigind=1)[2]
fgvd(oo_pp) = solve(oo_pp[1],oo_pp[2:5],rwg,grid;nev=1,eigind=1)[3]
fk(oo_pp) = solve_k(oo_pp[1],oo_pp[2:5],rwg,grid;nev=1,eigind=1)[1]

fneff_ng_gvd(oo_pp) = [ solve(oo_pp[1],oo_pp[2:5],rwg,grid;nev=1,eigind=1)[1:3]... ]
rrss= 1e-4

fneff_ng_gvd(vcat(0.9,p))




jac_fneff_ng_gvd_FD = FiniteDiff.finite_difference_jacobian(fneff_ng_gvd,vcat(0.83,p))





jac_fneff_ng_gvd_FD2 = FiniteDiff.finite_difference_jacobian(fneff_ng_gvd,vcat(0.83,p);relstep=1e-3)

(neff,ng,gvd),fneff_ng_gvd_pb = Zygote.pullback(fneff_ng_gvd,vcat(0.83,p))
jac_fneff_ng_gvd_AD = mapreduce(Δ->fneff_ng_gvd_pb(Δ)[1],hcat,([1.,0.,0.],[0.,1.,0.],[0.,0.,1.])) |> transpose
jac_fneff_ng_gvd_AD_tuples = jac_fneff_ng_gvd_AD
mapreduce(tt->getindex(tt,1),hcat,jac_fneff_ng_gvd_AD_tuples) |> transpose |> copy


##
@show dk_dom_FD 	= derivFD2(fk,om;rs=rrss)
@show dneff_dom_FD 	= derivFD2(fneff,om;rs=rrss)
@show dng_dom_FD 	= derivFD2(fng,om;rs=rrss)
@show dgvd_dom_FD 	= derivFD2(fgvd,om;rs=rrss)
@show dneff_dom_AD	= Zygote.gradient(fneff,om)[1]
@show dng_dom_AD	= Zygote.gradient(fng,om)[1]
@show dgvd_dom_AD	= Zygote.gradient(fgvd,om)[1]
##
@show dgvd_dom_FD_1em2 	= derivFD2(fgvd,om;rs=1e-2)
@show dgvd_dom_FD_1em3 	= derivFD2(fgvd,om;rs=1e-3)
@show dgvd_dom_FD_1em4 	= derivFD2(fgvd,om;rs=1e-4)
@show dgvd_dom_FD_1em5 	= derivFD2(fgvd,om;rs=1e-5)
@show dgvd_dom_FD_1em6 	= derivFD2(fgvd,om;rs=1e-6)
##
@show dgvd_dom_AD1	= Zygote.gradient(fgvd,om)[1]
@show dgvd_dom_AD2	= Zygote.gradient(fgvd,om)[1]
@show dgvd_dom_AD3	= Zygote.gradient(fgvd,om)[1]

@show dgvd_dom_FD 	= derivFD2(fgvd,1.5;rs=1e-4)
@show dgvd_dom_AD	= Zygote.gradient(fgvd,1.5)[1]
##
oms = 0.7:0.05:1.6
rrss = 1e-3
nom = length(oms)
dgvd_dom_FDs = zeros(Float64,nom)
dgvd_dom_ADs = zeros(ComplexF64,nom)

# iom = 2
# oo = oms[iom]
# println("")
# println("oo: $oo")
# @show dgvd_dom_FDs[iom] 	= derivFD2(fgvd,oo;rs=rrss)
# @show dgvd_dom_ADs[iom]		= Zygote.gradient(fgvd,oo)[1]
# println("")

for iom in 1:nom
	oo = oms[iom]
	println("")
	println("oo: $oo")
	@show dgvd_dom_FDs[iom] 	= derivFD2(fgvd,oo;rs=rrss)
	@show dgvd_dom_ADs[iom]		= Zygote.gradient(fgvd,oo)[1]
	println("")
end

##
function E_relpower_xyz(ms::ModeSolver{ND,T},ω²H) where {ND,T<:Real}
	mns = copy(vcat(reshape(flat(ms.M̂.m⃗),1,3,Ns...),reshape(flat(ms.M̂.n⃗),1,3,Ns...)))
    D = 1im * fft( kx_tc( reshape(ω²H[2],(2,size(ms.grid)...)),mns,ms.M̂.mag), (2:1+ND) )
    E = ε⁻¹_dot( D, flat( ms.M̂.ε⁻¹ ))
    Pe = real(_dot(E,D))
    Pe_tot = sum(Pe)
    Pₑ_xyz_rel = [sum(Pe[1,:,:]),sum(Pe[2,:,:]),sum(Pe[3,:,:])] ./ Pe_tot
    # Pₑ_xyz_rel = normalize([mapreduce((ee,epss)->(abs2(ee[a])*inv(epss)[a,a]),+,Es,ms.M̂.ε⁻¹) for a=1:3],1)
    return Pₑ_xyz_rel
end
TE_filter = (ms,ω²H)->E_relpower_xyz(ms,ω²H)[1]>0.7
TM_filter = (ms,ω²H)->E_relpower_xyz(ms,ω²H)[2]>0.7
function fneff_ng_gvd2(oo_pp;neff_guess=nothing,Hv=Hv)
	if !isnothing(neff_guess)
		kguess=neff_guess*oo_pp[1]
	else
		kguess=nothing
	end
	neff,ng,gvd,E = solve(oo_pp[1],oo_pp[2:5],rwg,Grid(6.0,4.0,128,128);nev=3,eigind=1,f_filter=(ms,ω²H)->E_relpower_xyz(ms,ω²H)[1]>0.51,Hguess=Hv,kguess)
	return [ neff,ng,gvd ]
end
oms = 0.7:0.05:1.6
ws = 0.7:0.05:2.0
ts = 0.4:0.05:1.0
rrss = 1e-3
nom = length(oms)
nw = length(ws)
nt = length(ts)
neff_ng_gvds = zeros(Float64,(nom,nw,nt,3))
jac_fneff_ng_gvd_FDs = zeros(Float64,(nom,nw,nt,3,5))
jac_fneff_ng_gvd_ADs = zeros(Float64,(nom,nw,nt,3,5))
for wind in 1:nw
	ww = ws[wind]
	for tind in 1:nt
		tt = ts[tind]
		for iom in 1:nom
			oo = oms[iom]
			oo_pp = [
				oo,
			       ww,                #   top ridge width         `w_top`         [μm]
			       tt,                #   ridge thickness         `t_core`        [μm]
			       0.9, #0.5,              partial etch fraction
			       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
			               ];
			println("")
			println("")
			println("om: $oo")
			println("w: $ww")
			println("t: $tt")
			println("")
			try
				neff_ng_gvd,fneff_ng_gvd_pb = Zygote.pullback(x->fneff_ng_gvd2(x;neff_guess=neff_last),oo_pp)
				@show neff_ng_gvds[iom,wind,tind,:]	.=	neff_ng_gvd
				neff_last = neff_ng_gvd[1]
			catch
				println("primal calc. exception, oo_pp: $oo_pp")
			end
			println("")
			try
				@show jac_fneff_ng_gvd_ADs[iom,wind,tind,:,:] .= real(mapreduce(Δ->fneff_ng_gvd_pb(Δ)[1],hcat,([1.,0.,0.],[0.,1.,0.],[0.,0.,1.]))) |> transpose
			catch
				println("AD jacobian calc. exception, oo_pp: $oo_pp")
			end
			println("")
			try
				@show jac_fneff_ng_gvd_FDs[iom,wind,tind,:,:] .= FiniteDiff.finite_difference_jacobian(x->fneff_ng_gvd2(x;neff_guess=neff_last),oo_pp;relstep=rrss)
			catch
				println("FD jacobian calc. exception, oo_pp: $oo_pp")
			end
			println("")
			println("")
		end
	end
end
##

neffs 		= view(neff_ng_gvds,:,1)
ngs 		= view(neff_ng_gvds,:,2)
gvds 		= view(neff_ng_gvds,:,3)

∂neff∂ω_FD 	= view(jac_fneff_ng_gvd_FDs,:,1,1)
∂neff∂w_FD 	= view(jac_fneff_ng_gvd_FDs,:,1,2)
∂neff∂t_FD 	= view(jac_fneff_ng_gvd_FDs,:,1,3)
∂neff∂ef_FD 	= view(jac_fneff_ng_gvd_FDs,:,1,4)
∂neff∂sw_FD 	= view(jac_fneff_ng_gvd_FDs,:,1,5)
∂ng∂ω_FD 	= view(jac_fneff_ng_gvd_FDs,:,2,1)
∂ng∂w_FD 	= view(jac_fneff_ng_gvd_FDs,:,2,2)
∂ng∂t_FD 	= view(jac_fneff_ng_gvd_FDs,:,2,3)
∂ng∂ef_FD 	= view(jac_fneff_ng_gvd_FDs,:,2,4)
∂ng∂sw_FD 	= view(jac_fneff_ng_gvd_FDs,:,2,5)
∂gvd∂ω_FD 	= view(jac_fneff_ng_gvd_FDs,:,3,1)
∂gvd∂w_FD 	= view(jac_fneff_ng_gvd_FDs,:,3,2)
∂gvd∂t_FD 	= view(jac_fneff_ng_gvd_FDs,:,3,3)
∂gvd∂ef_FD 	= view(jac_fneff_ng_gvd_FDs,:,3,4)
∂gvd∂sw_FD 	= view(jac_fneff_ng_gvd_FDs,:,3,5)

∂neff∂ω_AD 	= view(jac_fneff_ng_gvd_ADs,:,1,1)
∂neff∂w_AD 	= view(jac_fneff_ng_gvd_ADs,:,1,2)
∂neff∂t_AD 	= view(jac_fneff_ng_gvd_ADs,:,1,3)
∂neff∂ef_AD 	= view(jac_fneff_ng_gvd_ADs,:,1,4)
∂neff∂sw_AD 	= view(jac_fneff_ng_gvd_ADs,:,1,5)
∂ng∂ω_AD 	= view(jac_fneff_ng_gvd_ADs,:,2,1)
∂ng∂w_AD 	= view(jac_fneff_ng_gvd_ADs,:,2,2)
∂ng∂t_AD 	= view(jac_fneff_ng_gvd_ADs,:,2,3)
∂ng∂ef_AD 	= view(jac_fneff_ng_gvd_ADs,:,2,4)
∂ng∂sw_AD 	= view(jac_fneff_ng_gvd_ADs,:,2,5)
∂gvd∂ω_AD 	= view(jac_fneff_ng_gvd_ADs,:,3,1)
∂gvd∂w_AD 	= view(jac_fneff_ng_gvd_ADs,:,3,2)
∂gvd∂t_AD 	= view(jac_fneff_ng_gvd_ADs,:,3,3)
∂gvd∂ef_AD 	= view(jac_fneff_ng_gvd_ADs,:,3,4)
∂gvd∂sw_AD 	= view(jac_fneff_ng_gvd_ADs,:,3,5)


##
using GLMakie

fig = Figure()
ax = fig[1,1] = Axis(fig)

x1,x2 = oms,oms
# x1,x2 = inv.(oms),inv.(oms)

# y1,y2 = ∂gvd∂ω_FD, real(∂gvd∂ω_AD)
y1,y2 = ∂gvd∂w_FD, real(∂gvd∂w_AD)
# v1,v2 = ∂ng∂ω_FD, real(∂ng∂ω_AD)
# v1,v2 = ∂ng∂w_FD, real(∂ng∂w_AD)
# v1,v2 = ∂ng∂t_FD, real(∂ng∂t_AD)
# v1,v2 = neff_ng_gvds[:,1], neff_ng_gvds[:,2]

sc1 = scatterlines!(ax,x1,y1,color=:red)
sc2 = scatterlines!(ax,x2,y2,color=:blue)

fig
##
##
# neff = 2.02087795094446
# ng = 2.4305416802127993
# gvd = -0.22318870671473665
# dk_dom_FD = derivFD2(fk, om) = 2.430519837299261
# dneff_dom_FD = derivFD2(fneff, om) = 0.546306176369793
# dng_dom_FD = derivFD2(fng, om) = -0.2236921865319763
# dgvd_dom_FD = derivFD2(fgvd, om) = 1.0843918092994203
# dneff_dom_AD = (Zygote.gradient(fneff, om))[1] = 0.5462439827404987 + 0.0im
# dng_dom_AD = (Zygote.gradient(fng, om))[1] = -0.22386684737786888 + 5.29268205376241e-7im
# dgvd_dom_AD = (Zygote.gradient(fgvd, om))[1] = 11.440791322143077 + 289.7195137517479im
# dgvd_dom_AD = (Zygote.gradient(fgvd, om))[1] = 201090.8618860823 + 2.3986973829772325im
##
# ∂²ω²∂k²(om,p,rwg,k,Hv,grid)
##
(k,Hv), k_Hv_pb = pullback((oo,pp)->solve_k(oo,pp,rwg,grid;nev=1,eigind=1),om,p)
mag,m⃗,n⃗ = mag_m_n(k,grid)
m=flat(m⃗);
n=flat(n⃗);
mns = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3])));
ei,ei_pb = Zygote.pullback(om) do ω
        ε⁻¹,nng = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs),[true,false],rwg,grid));
        return ε⁻¹
end
eps,eps_pb = Zygote.pullback(om) do ω
        ε,nng = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs),[false,false],rwg,grid));
        return ε
end
nng,nng_pb = Zygote.pullback(om) do ω
        ε⁻¹,nng = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs),[true,false],rwg,grid));
        return nng
end
nngi,nngi_pb = Zygote.pullback(om) do ω
        ε⁻¹,nngi = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs),[true,true],rwg,grid));
        return nngi
end
ngvd,ngvd_pb = Zygote.pullback(om) do ω
        # ngvd,nng2,nngi2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs,:fnn̂gs),[false,false,true],geom_fn,grid,volfrac_smoothing));
        ngvd,nng2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,false],rwg,grid,volfrac_smoothing));
        return ngvd
end

function calc_ng(ω,p,grid::Grid{ND,T}) where {ND,T<:Real}
    ε⁻¹,nng = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs),[true,false],rwg,grid));
    k,Hv = solve_k(ω,p,rwg,grid;nev=1,eigind=1);
    Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
    Hₜ = reshape(Hv,(2,Ns...))
	E = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
	H = inv(ω) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
	P = 2*real(_sum_cross_z(conj(E),H))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
	# W = dot(E,_dot((ε+nng),E))             # energy density per unit length
    W = real(dot(E,_dot(nng,E))) + prod(Ns)*(ω^2)     # energy density per unit length
	ng = W / P #real( W / P )
end

function calc_ng(ω,nng,E,H,grid::Grid{ND,T}) where {ND,T<:Real}
	P = 2*real(_sum_cross_z(conj(E),H))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
        W = real(dot(E,_dot(nng,E))) + N(grid)*ω^2     # energy density per unit length
	ng = real( W / P )
end

function calc_ng(ω,ε⁻¹,nng,mag,mns,Hv,grid::Grid{ND,T}) where {ND,T<:Real}
    Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
    Hₜ = reshape(Hv,(2,Ns...))
	E = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
	# H = inv(ω) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
    H = fft( tc(Hₜ,mns), (2:3) ) * (-1im * ω)
	P = 2*real(_sum_cross_z(conj(E),H))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
	# W = dot(E,_dot((ε+nng),E))             # energy density per unit length
    W = real(dot(E,_dot(nng,E))) + (N(grid)* (ω^2))     # energy density per unit length
	ng = real( W / P )
    return ng
end

function calc_ng(ω,ε⁻¹,nng,k,Hv,grid::Grid{ND,T}) where {ND,T<:Real}
    Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
    Hₜ = reshape(Hv,(2,Ns...))
	E = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
	# H = inv(ω) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
    H = fft( tc(Hₜ,mns), (2:3) ) * (-1im * ω)
	P = 2*real(_sum_cross_z(conj(E),H))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
	# W = dot(E,_dot((ε+nng),E))             # energy density per unit length
    W = real(dot(E,_dot(nng,E))) + (N(grid)* (ω^2))     # energy density per unit length
	ng = real( W / P )
    return ng
end

function ∇ng(ω,ε⁻¹,nng,k,Hv,grid::Grid{ND,T};dk̂=SVector(0.,0.,1.)) where {ND,T<:Real}
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
	Hₜ = reshape(Hv,(2,Ns...))
	D = 1im * fft( kx_tc( Hₜ,mns,mag), (2:1+ND) )
	E = ε⁻¹_dot( D, ε⁻¹)
	H = fft( tc(Hₜ,mns), (2:3) ) * (-1im * ω)
	P = 2*real(_sum_cross_z(conj(E),H))
	nḡ = 1.0
	W̄ = nḡ / P #PPz1
	om̄₁₁ = 2*ω * N(grid) * W̄
	nnḡ = _outer(E,E) * W̄
	H̄ = (-2*ng*W̄) * _cross(repeat([0.,0.,1.],outer=(1,Ns...)), E)
	Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,repeat([0.,0.,1.],outer=(1,Ns...))) )
	om̄₁₂ = dot(H,H̄) / ω
	om̄₁ = om̄₁₁ + om̄₁₂
	eī₁ = _outer(Ē,D)
	𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),(2:3))
	𝓕⁻¹_H̄ = bfft( H̄ ,(2:3))
	H̄ₜ = 1im*( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mns,mag) + ω*ct(𝓕⁻¹_H̄,mns) )
	𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ = 1im *_outer(_dot(repeat([0. 1. ;-1. 0. ],outer=(1,1,Ns...)), Hₜ), 𝓕⁻¹_ε⁻¹_Ē )
	@tullio māg2[ix,iy] := mns[a,b,ix,iy] * -conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ[a,b,ix,iy])
	mn̄s2 = -conj( 1im*ω*_outer(Hₜ,𝓕⁻¹_H̄) + _mult(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ,mag))
	m̄ = reinterpret(reshape,SVector{3,T},real(view(mn̄s,1,:,:,:)))
	n̄ = reinterpret(reshape,SVector{3,T},real(view(mn̄s,2,:,:,:)))
	k̄ = ∇ₖmag_m_n(māg,m̄,n̄,mag,m⃗,n⃗;dk̂)
	return NoTangent(),om̄₁,eī₁,nnḡ,k̄,vec(H̄ₜ),NoTangent()
end

function ng_gvd(ω,ε,ε⁻¹,nng,ngvd,k,Hv,grid::Grid{ND,T};eigind=1,dk̂=SVector(0.,0.,1.)) where {ND,T<:Real}
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
    Hₜ = reshape(Hv,(2,Ns...))
	# E = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
	# H = inv(ω) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
    D = 1im * fft( kx_tc( Hₜ,mns,mag), (2:1+ND) )
	E = ε⁻¹_dot( D, ε⁻¹)
	H = fft( tc(Hₜ,mns), (2:3) ) * (-1im * ω)
	P = 2*real(_sum_cross_z(conj(E),H))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
	# W = dot(E,_dot((ε+nng),E))             # energy density per unit length
    W = real(dot(E,_dot(nng,E))) + (N(grid)* (ω^2))     # energy density per unit length
	ng = real( W / P )
	# calculate GVD = ∂(ng) / ∂ω = (∂²k)/(∂ω²)
	W̄ = inv(P)
	om̄₁₁ = 2*ω * N(grid) * W̄
	nnḡ = _outer(E,E) * W̄
	# H̄ = (-2*ng*W̄) * _cross(repeat([0.,0.,1.],outer=(1,Ns...)), E)
	# Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,repeat([0.,0.,1.],outer=(1,Ns...))) )
	H̄ = (-2*ng*W̄) * _cross(dk̂, E)
	Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,dk̂) )
	om̄₁₂ = dot(H,H̄) / ω
	om̄₁ = om̄₁₁ + om̄₁₂
	eī₁ = _outer(Ē,D)
	𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),(2:3))
	𝓕⁻¹_H̄ = bfft( H̄ ,(2:3))
	H̄ₜ = 1im*( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mns,mag) + ω*ct(𝓕⁻¹_H̄,mns) )
	# 𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ = 1im *_outer(_dot(repeat([0.0+0.0im 1.0+0.0im ;-1.0+0.0im 0.0+0.0im ],outer=(1,1,Ns...)), Hₜ), 𝓕⁻¹_ε⁻¹_Ē )
	local one_mone = [1.0im, -1.0im]
	@tullio 𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ[i,j,ix,iy] := one_mone[i] * reverse(Hₜ;dims=1)[i,ix,iy] * conj(𝓕⁻¹_ε⁻¹_Ē)[j,ix,iy] nograd=one_mone
	@tullio māg[ix,iy] := mns[a,b,ix,iy] * -conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ)[a,b,ix,iy]
	mn̄s = -conj( 1im*ω*_outer(Hₜ,𝓕⁻¹_H̄) + _mult(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ,mag))
	# m̄ = reinterpret(reshape,SVector{3,T},view(mn̄s,1,:,:,:))
	# n̄ = reinterpret(reshape,SVector{3,T},view(mn̄s,2,:,:,:))
	m̄ = reinterpret(reshape,SVector{3,eltype(mn̄s)},view(mn̄s,1,:,:,:))
	n̄ = reinterpret(reshape,SVector{3,eltype(mn̄s)},view(mn̄s,2,:,:,:))
	k̄ = ∇ₖmag_m_n(māg,m̄,n̄,mag,m⃗,n⃗;dk̂)
	∂ω²∂k_nd = 2 * HMₖH(Hv,ε⁻¹,mag,flat(m⃗),flat(n⃗))
	( _, _, om̄₂, eī₂ ) = ∇solve_k(
		(k̄,vec(H̄ₜ)),
		(k,Hv),
		∂ω²∂k_nd,
		ω,
		ε⁻¹,
		grid;
		eigind,
	)
	om̄₃ = dot(herm(nnḡ), ngvd)
	om̄₄ = dot(herm(eī₁+eī₂), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
	gvd = real( om̄₁ + om̄₂ + om̄₃ + om̄₄ )
	return ng, gvd
end

##
ng,gvd = ng_gvd(om,eps,ei,nng,ngvd,k,Hv,grid)
om̄_gvd,eps̄_gvd,eī_gvd,nnḡ_gvd,ngvd̄_gvd,k̄_gvd,Hv̄_gvd,grid̄_gvd = Zygote.gradient((aa,bb,cc,dd,ee,ff,gg,hh)->ng_gvd(aa,bb,cc,dd,ee,ff,gg,hh)[2],om,eps,ei,nng,ngvd,k,Hv,grid)
om̄_gvd
##
ω = 0.75
geom_fn = rwg
# kguess=nothing
# Hguess=nothing
dk̂=SVector(0.0,0.0,1.0)
nev=1
eigind=1
maxiter=500
tol=1e-8
log=false
f_filter=nothing
using Zygote: ignore
##
ε,ε⁻¹,nng,nng⁻¹ = smooth(ω,p,(:fεs,:fεs,:fnn̂gs,:fnn̂gs),[false,true,false,true],geom_fn,grid);
ngvd,nng2 = smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,false],geom_fn,grid,volfrac_smoothing);
# ε,ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fεs,:fnn̂gs,:fnn̂gs),[false,true,false,true],geom_fn,grid));
# ngvd,nng2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,false],geom_fn,grid,volfrac_smoothing));
ms = ignore() do
	kguess = k_guess(ω,ε⁻¹)
	ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, tol)
	return ms
end
# update_ε⁻¹(ms,ε⁻¹)
k, Hv = solve_k(ms,ω,ε⁻¹;nev,eigind,maxiter,tol,log, f_filter); #ω²_tol)
neff = k/ω
# calculate effective group index `ng`
Ninv 		= 		1. / N(grid)
# M̂ = HelmholtzMap(k,ε⁻¹,grid)
# Nranges = eachindex(grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
mag,m⃗,n⃗ = mag_m_n(k,grid)
m = flat(m⃗)
n = flat(n⃗)
mns = vcat(reshape(m,1,3,Ns...),reshape(n,1,3,Ns...))
Hₜ = reshape(Hv,(2,Ns...))
D = 1im * fft( kx_tc( Hₜ,mns,mag), _fftaxes(grid) )
E = ε⁻¹_dot( D, ε⁻¹)
# E = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
# H = inv(ω) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
H = fft( tc(Hₜ,mns), (2:3) ) * (-1im * ω)
@show P = 2*real(_sum_cross_z(conj(E),H))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
# W = dot(E,_dot((ε+nng),E))             # energy density per unit length
@show W = real(dot(E,_dot(nng,E))) + (N(grid)* (ω^2))     # energy density per unit length
@show ng = real( W / P )

# calculate GVD = ∂(ng) / ∂ω = (∂²k)/(∂ω²)
W̄ = inv(P)
om̄₁₁ = 2*ω * N(grid) * W̄
nnḡ = _outer(E,E) * W̄
# H̄ = (-2*ng*W̄) * _cross(repeat([0.,0.,1.],outer=(1,Ns...)), E)
# Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,repeat([0.,0.,1.],outer=(1,Ns...))) )
H̄ = (-2*ng*W̄) * _cross(dk̂, E)
Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,dk̂) )
om̄₁₂ = dot(H,H̄) / ω
om̄₁ = om̄₁₁ + om̄₁₂
# eī₁ = _outer(Ē,D)
𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),(2:3))
𝓕⁻¹_H̄ = bfft( H̄ ,(2:3))
H̄ₜ = 1im*( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mns,mag) + ω*ct(𝓕⁻¹_H̄,mns) )
one_mone = [1.0im, -1.0im]
@tullio 𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ[i,j,ix,iy] := one_mone[i] * reverse(Hₜ;dims=1)[i,ix,iy] * conj(𝓕⁻¹_ε⁻¹_Ē)[j,ix,iy] nograd=one_mone
∂ω²∂k_nd = 2 * HMₖH(Hv,ε⁻¹,mag,m,n)

##### grad solve k
# solve_adj!(λ⃗,M̂,H̄,ω^2,H⃗,eigind)
# λ⃗	= eig_adjt(
# 	ms.M̂,								 # Â
# 	ω^2, 							# α
# 	Hv, 					 		 # x⃗
# 	0.0, 							# ᾱ
# 	vec(H̄ₜ);								 # x̄
# 	# λ⃗₀,
# 	P̂	= HelmholtzPreconditioner(M̂),
# )
M̂2 = HelmholtzMap(k,ε⁻¹,dropgrad(grid))
λ⃗	= eig_adjt(
	M̂2,								 # Â
	ω^2, 							# α
	Hv, 					 		 # x⃗
	0.0, 							# ᾱ
	vec(H̄ₜ);								 # x̄
	# λ⃗₀,
	P̂	= HelmholtzPreconditioner(M̂2),
)


### k̄ₕ, eīₕ = ∇M̂(k,ε⁻¹,λ⃗,H⃗,grid)
λ = reshape(λ⃗,(2,Ns...))
λd 	= 	fft(kx_tc( λ , mns, mag ),_fftaxes(grid))
# eīₕ	 = 	 ε⁻¹_bar(vec(D * (Ninv * -1.0im)), vec(λd), Ns...)
λẽ  =   bfft(ε⁻¹_dot(λd , ε⁻¹),_fftaxes(grid))
ẽ 	 =   bfft(E * -1.0im,_fftaxes(grid))
@tullio mn̄s_kx0[i,j,ix,iy] := -1.0im * one_mone[i] * reverse(conj(Hₜ);dims=1)[i,ix,iy] * (Ninv*λẽ)[j,ix,iy] + -1.0im * one_mone[i] * reverse(conj(λ);dims=1)[i,ix,iy] * (Ninv*ẽ)[j,ix,iy]  nograd=one_mone
@tullio mn̄s[i,j,ix,iy] := mag[ix,iy] * (mn̄s_kx0-conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ))[i,j,ix,iy]  + 1im*ω*conj(Hₜ)[i,ix,iy]*𝓕⁻¹_H̄[j,ix,iy]
@tullio māg[ix,iy] := mns[a,b,ix,iy] * (mn̄s_kx0-conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ))[a,b,ix,iy]
# k̄ = ∇ₖmag_mn(māg,mn̄s,mag,mns)
@tullio kp̂g_over_mag[i,ix,iy] := m[mod(i-2),ix,iy] * n[mod(i-1),ix,iy] / mag[ix,iy] - m[mod(i-1),ix,iy] * n[mod(i-2),ix,iy] / mag[ix,iy] (i in 1:3)
kp̂g_over_mag_x_dk̂ = _cross(kp̂g_over_mag,dk̂)
@tullio k̄_mag := māg[ix,iy] * mag[ix,iy] * kp̂g_over_mag[j,ix,iy] * dk̂[j]
@tullio k̄_mn := -conj(mn̄s)[imn,i,ix,iy] * mns[imn,mod(i-2),ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-1),ix,iy] + conj(mn̄s)[imn,i,ix,iy] * mns[imn,mod(i-1),ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-2),ix,iy] (i in 1:3)
k̄ = k̄_mag + k̄_mn
### \ k̄ₕ, eīₕ = ∇M̂(k,ε⁻¹,λ⃗,H⃗,grid)

# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
λₖ  = ( k̄ / ∂ω²∂k_nd ) * Hₜ #reshape(λ⃗ₖ, (2,Ns...))
λdₖ	=	fft(kx_tc( λₖ , mns, mag ),_fftaxes(grid))
# eīₖ = ε⁻¹_bar(vec(D* (Ninv * -1.0im)), vec(λdₖ), Ns...)
# eī₂ = eīₕ + eīₖ
@show om̄₂  =  2ω * k̄ / ∂ω²∂k_nd
##### \grad solve k

om̄₃ = dot(herm(nnḡ), ngvd)
# @show om̄₄ = dot(herm(eī₁+eī₂), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
eī₁ = _outer(Ē,D) ####################################
eīₖ = ε⁻¹_bar(vec(D* (Ninv * -1.0im)), vec(λdₖ), Ns...) ####################################
eīₕ	 = 	 ε⁻¹_bar(vec(D * (Ninv * -1.0im)), vec(λd), Ns...) ##########################
@show om̄₄_new = dot( herm(_outer(Ē+(λd+λdₖ)*(Ninv * -1.0im),D) ), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
@show om̄₄ = dot( ( eīₖ+ eīₕ+ eī₁ ), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
# @show om̄₄ = dot( herm(_outer(Ē+(λd+λdₖ)*(Ninv * -1.0im),D) ), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
@show gvd = real( om̄₁ + om̄₂ + om̄₃ + om̄₄ )

#
# P = 2 * real(_sum_cross_z(conj(E), H)) = 7758.43246359937
# W = real(dot(E, _dot(nng, E))) + N(grid) * ω ^ 2 = 18857.193475894397
# ng = real(W / P) = 2.430541680212807
# om̄₂ = ((2ω) * (k̄ₖ + k̄ₕ)) / ∂ω²∂k_nd = -0.2526356610951862 + 4.111339425628389e-7im
# gvd = real(om̄₁ + om̄₂ + om̄₃ + om̄₄) = -0.22318870673735686
##
using Zygote: @showgrad
function ff1(oo)
	M̂2 = HelmholtzMap(k,ε⁻¹,dropgrad(grid))
	λ⃗	= eig_adjt(
		M̂2,								 # Â
		(oo^2), 							# α
		Hv, 					 		 # x⃗
		0.0, 							# ᾱ
		vec(H̄ₜ);								 # x̄
		# λ⃗₀,
		P̂	= HelmholtzPreconditioner(M̂2),
	)
	# @showgrad
	lmmax = maximum(abs2.(λ⃗))
	return lmmax
	# lmsum = sum(abs2.(λ⃗).^2)
	# return lmsum
end

function ff2(kk)
	M̂2 = HelmholtzMap(kk,ε⁻¹,dropgrad(grid))
	λ⃗	= eig_adjt(
		M̂2,								 # Â
		(om^2), 							# α
		Hv, 					 		 # x⃗
		0.0, 							# ᾱ
		vec(H̄ₜ);								 # x̄
		# λ⃗₀,
		P̂	= HelmholtzPreconditioner(M̂2),
	)
	lmmax = maximum(abs2.(λ⃗))
	return lmmax
	# lmsum = sum(abs2.(λ⃗).^2)
	# return lmsum
end

function ff3(oo)
	M̂2 = HelmholtzMap(k,ε⁻¹,dropgrad(grid))
	λ⃗	= eig_adjt(
		M̂2,								 # Â
		(oo^2), 							# α
		Hv, 					 		 # x⃗
		0.0, 							# ᾱ
		vec(H̄ₜ);								 # x̄
		# λ⃗₀,
		P̂	= HelmholtzPreconditioner(M̂2),
	)
	# @showgrad

	lmsum = sum(abs2.(λ⃗).^2)
	return lmsum
end

function ff4(kk)
	M̂2 = HelmholtzMap(kk,ε⁻¹,dropgrad(grid))
	λ⃗	= eig_adjt(
		M̂2,								 # Â
		(om^2), 							# α
		Hv, 					 		 # x⃗
		0.0, 							# ᾱ
		vec(H̄ₜ);								 # x̄
		# λ⃗₀,
		P̂	= HelmholtzPreconditioner(M̂2),
	)

	lmsum = sum(abs2.(λ⃗).^2)
	return lmsum
end
##
rrss = 1e-5
println("")
@show ff1(0.75)
@show derivFD2(ff1,0.75;rs=rrss)
@show derivRM(ff1,0.75)
println("")
@show ff2(k)
@show derivFD2(ff2,k;rs=rrss)
@show derivRM(ff2,k)
println("")
@show ff3(0.75)
@show derivFD2(ff3,0.75;rs=rrss)
@show derivRM(ff3,0.75)
println("")
@show ff4(k)
@show derivFD2(ff4,k;rs=rrss)
@show derivRM(ff4,k)
println("")
##
eib11 = _outer(Ē,D);
eib12 = ε⁻¹_bar(vec(D* (Ninv * -1.0im)), vec(Ē), Ns...) ;
eibk1 = _outer(λdₖ,D);
eibk2 = ε⁻¹_bar(vec(D* (Ninv * -1.0im)), vec(λdₖ), Ns...) ;
eibh1 = _outer(λd,D);
eibh2 = ε⁻¹_bar(vec(D* (Ninv * -1.0im)), vec(λd), Ns...) ;

##
ng,ng_pb = pullback(calc_ng,om,ei,nng,k,Hv,grid)
om̄₁,eī₁,nnḡ,k̄,Hv̄,grid̄ = ng_pb(1.0)
_,om̄₁2,eī₁2,nnḡ2,k̄2,Hv̄2,grid̄2 = ∇ng(om,ei,nng,k,Hv,grid;dk̂=SVector(0.,0.,1.))
∂ω²∂k_nd = 2 * HMₖH(Hv,ei,real(mag),real(flat(m⃗)),real(flat(n⃗)))
# k̄, H̄, nngī  = ∇HMₖH(k,Hv,nng⁻¹,grid; eigind)
( _, _, om̄₂, eī₂ ) = ∇solve_k(	  (k̄,Hv̄),
									(k,Hv),
									∂ω²∂k_nd,
									om,
									ei,
									grid; eigind=1)
om̄₃ = dot(herm(nnḡ), ngvd) #dot(herm(nngī), ∂nng⁻¹_∂ω(ei,nng,nngi,ngvd,om))
# om̄₄ = dot(herm(eī₂), ∂ε⁻¹_∂ω(eps,ei,nng,om))
om̄₄ = dot(herm(eī₁+eī₂), ∂ε⁻¹_∂ω(eps,ei,nng,om))
println("om̄₁: $(om̄₁)")
println("om̄₂: $(om̄₂)")
println("om̄₃: $(om̄₃)")
om̄ = om̄₁ + om̄₂ + om̄₃ + om̄₄
println("om̄: $(om̄)")
println("dng_dom_FD = derivFD2(fng, om) = -0.22376369476611035")
##
ng2,ng2_pb = pullback(calc_ng,om,ei,nng,k,Hv,grid)
Ns = size(grid)
ND = 2
Hₜ = reshape(Hv,(2,Ns...))
mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
DD = 1im * fft( kx_tc( Hₜ,mns,mag), (2:1+ND) )
EE = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ei)
HH = fft(tc(kx_ct( ifft( EE, (2:1+ND) ), mns,mag), mns),(2:1+ND) ) / om
EEs = copy(reinterpret(reshape,SVector{3,ComplexF64},EE))
HHs = copy(reinterpret(reshape,SVector{3,ComplexF64},HH))
Sz = dot.(cross.(conj.(EEs),HHs),(SVector(0.,0.,1.),))
PP = 2*real(sum(Sz))
WW = dot(EE,_dot((eps+nng),EE))
WW2 = dot(EE,_dot(nng,EE)) + om^2 * N(grid)
dot(EE,_dot(nng,EE))
ng = WW / PP

SSs = cross.(conj.(EEs),HHs)
SSsr = copy(flat(SSs))
SSz = dot.((SVector(0.,0.,1.),),cross.(conj.(EEs),HHs)) #dot.(cross.(conj.(EEs),HHs),(SVector(0.,0.,1.),))
PPz = 2*real(sum(SSz))
SS1 = _cross(conj(EE),HH)
SSz1 = _cross_z(conj(EE),HH)
PPz1 = 2*real(_sum_cross_z(conj(EE),HH))
@assert SS1 ≈ SSsr
@assert SSz1 ≈ SSz
@assert PPz1 ≈ PPz
##
ng,ng_pb = pullback(calc_ng,om,nng,EE,HH,grid)
om̄₁,nnḡ,EĒ,HH̄,grid̄ = ng_pb(1.0)
nḡ = 1.0
W̄ = nḡ / PPz1
om̄₁₁ = 2*om * N(grid) * W̄
nnḡ2 = _outer(EE,EE) * W̄
H̄ = (-2*ng*W̄) * _cross(repeat([0.,0.,1.],outer=(1,Ns...)), EE)
Ē = 2W̄*( _dot(nng,EE) - ng * _cross(HH,repeat([0.,0.,1.],outer=(1,Ns...))) )
@assert nnḡ2 ≈ nnḡ
@assert H̄ ≈ HH̄
@assert Ē ≈ EĒ
@assert om̄₁₁ ≈ om̄₁
##
mag,m⃗,n⃗ = mag_m_n(k,grid)
mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
ng,ng_pb = pullback(calc_ng,om,ei,nng,mag,mns,Hv,grid)
om̄₁,eī₁,nnḡ,māg,mn̄s,Hv̄,grid̄ = ng_pb(1.0)
Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D

Hₜ = reshape(Hv,(2,Ns...))
D = 1im * fft( kx_tc( Hₜ,mns,mag), (2:1+ND) )
E = ε⁻¹_dot( D, ei)
# E = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ei)
# H = inv(om) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
H = fft( tc(Hₜ,mns), (2:3) ) * (-1im * om)
P = 2*real(_sum_cross_z(conj(E),H))

nḡ = 1.0
W̄ = nḡ / PPz1
om̄₁₁ = 2*om * N(grid) * W̄
nnḡ2 = _outer(E,E) * W̄
H̄ = (-2*ng*W̄) * _cross(repeat([0.,0.,1.],outer=(1,Ns...)), E)
Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,repeat([0.,0.,1.],outer=(1,Ns...))) )
om̄₁₂ = dot(H,H̄) / om
om̄₁2 = om̄₁₁ + om̄₁₂
eī₁2 = _outer(Ē,D)
𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ei),(2:3))
𝓕⁻¹_H̄ = bfft( H̄ ,(2:3))
H̄ₜ = 1im*( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mns,mag) + om*ct(𝓕⁻¹_H̄,mns) )
# 𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ = 1im *_outer(𝓕⁻¹_ε⁻¹_Ē,_dot(repeat([0. 1. ;-1. 0. ],outer=(1,1,Ns...)), Hₜ) )
# @tullio māg2[ix,iy] := mns[a,b,ix,iy] * 𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ[b,a,ix,iy]
# mn̄s2 = permutedims((1im*om*_outer(𝓕⁻¹_H̄,Hₜ))+_mult(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ,mag),(2,1,3,4))
𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ = 1im *_outer(_dot(repeat([0. 1. ;-1. 0. ],outer=(1,1,Ns...)), Hₜ), 𝓕⁻¹_ε⁻¹_Ē )
using OffsetArrays
one_mone = [1.0im, -1.0im]
@tullio 𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ2[i,j,ix,iy] := one_mone[i] * reverse(Hₜ;dims=1)[i,ix,iy] * conj(𝓕⁻¹_ε⁻¹_Ē)[j,ix,iy]  nograd=one_mone verbose=true
𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ2 ≈ 𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ
@tullio māg2[ix,iy] := mns[a,b,ix,iy] * -conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ[a,b,ix,iy])
mn̄s2 = -conj( 1im*om*_outer(Hₜ,𝓕⁻¹_H̄) + _mult(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ,mag))

m̄ = reinterpret(reshape,SVector{3,Float64},real(view(mn̄s,1,:,:,:)))
n̄ = reinterpret(reshape,SVector{3,Float64},real(view(mn̄s,2,:,:,:)))
m̄2 = reinterpret(reshape,SVector{3,Float64},real(view(mn̄s2,1,:,:,:)))
n̄2 = reinterpret(reshape,SVector{3,Float64},real(view(mn̄s2,2,:,:,:)))
∇ₖmag_m_n(māg,m̄,n̄,mag,m⃗,n⃗;dk̂=SVector(0.,0.,1.))
∇ₖmag_m_n(māg2,m̄2,n̄2,mag,m⃗,n⃗;dk̂=SVector(0.,0.,1.))
##
using LoopVectorization
using Zygote: Buffer
function ε⁻¹_bar2(d⃗::AbstractVector{Complex{T}}, λ⃗d, Nx, Ny) where T<:Real
	# # capture 3x3 block diagonal elements of outer product -| λ⃗d X d⃗ |
	# # into (3,3,Nx,Ny,Nz) array. This is the gradient of ε⁻¹ tensor field

	# eīf = flat(eī)
	eīf = Buffer(Array{ComplexF64,1}([2., 2.]),3,3,Nx,Ny) # bufferfrom(zero(T),3,3,Nx,Ny)
	# eīf = bufferfrom(zero(eltype(real(d⃗)),3,3,Nx,Ny))
	@avx for iy=1:Ny,ix=1:Nx
		q = (Ny * (iy-1) + ix) # (Ny * (iy-1) + i)
		for a=1:3 # loop over diagonal elements: {11, 22, 33}
			eīf[a,a,ix,iy] =  -λ⃗d[3*q-2+a-1] * conj(d⃗[3*q-2+a-1])
		end
		for a2=1:2 # loop over first off diagonal
			eīf[a2,a2+1,ix,iy] =  -conj(λ⃗d[3*q-2+a2]) * d⃗[3*q-2+a2-1] - λ⃗d[3*q-2+a2-1] * conj(d⃗[3*q-2+a2])
			eīf[a2+1,a2,ix,iy] = eīf[a2,a2+1,ix,iy]
		end
		# a = 1, set 1,3 and 3,1, second off-diagonal
		eīf[1,3,ix,iy] =  -conj(λ⃗d[3*q]) * d⃗[3*q-2] - λ⃗d[3*q-2] * conj(d⃗[3*q])
		eīf[3,1,ix,iy] = eīf[1,3,ix,iy]
	end
	# eī = reinterpret(reshape,SMatrix{3,3,T,9},reshape(copy(eīf),9,Nx,Ny))
	eī = copy(eīf)
	return eī # inv( (eps' + eps) / 2)
end
##
# A1, A2 = fftshift(māg,(1:2)), fftshift(māg2,(1:2))
idx1,idx2 = 1,2

eib11 = herm(_outer(Ē,D)* (Ninv * -1.0im));
eib12 = herm(ε⁻¹_bar2(vec(D* (Ninv * -1.0im)), vec(Ē), Ns...)) ;
eibk1 = herm(_outer(λdₖ,D)* (Ninv * -1.0im));
eibk2 = herm(ε⁻¹_bar2(vec(D* (Ninv * -1.0im)), vec(λdₖ), Ns...)) ;
eibh1 = herm(_outer(λd,D)* (Ninv * -1.0im));
eibh2 = herm(ε⁻¹_bar2(vec(D* (Ninv * -1.0im)), vec(λd), Ns...)) ;
eib1 = herm(_outer(Ē+λd+λdₖ,D)* (Ninv * -1.0im));
eib2 = herm(ε⁻¹_bar2(vec(D* (Ninv * -1.0im)), vec(Ē+λd+λdₖ), Ns...)) ;

# A1, A2 = eib11[idx1,idx2,:,:], eib12[idx1,idx2,:,:]
# A1, A2 = eibk1[idx1,idx2,:,:], eibk2[idx1,idx2,:,:]
# A1, A2 = eibh1[idx1,idx2,:,:], eibh2[idx1,idx2,:,:]
A1, A2 = eib1[idx1,idx2,:,:], eib2[idx1,idx2,:,:]

# A1, A2 = fftshift(māg_kx,(1:2))[:,:], fftshift(māg_kx2,(1:2))[:,:]
# A1, A2 = fftshift(māg_kx,(1:2))[:,:], fftshift(māg_kx3,(1:2))[:,:]

# A1, A2 = fftshift(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ,(3:4))[idx1,idx2,:,:], fftshift(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ2,(3:4))[idx1,idx2,:,:]
# A1, A2 = fftshift(māg,(1:2))[:,:], fftshift(māg2,(1:2))[:,:]
# A1, A2 = fftshift(mn̄s,(3:4))[idx1,idx2,:,:], fftshift(mn̄s2,(3:4))[idx1,idx2,:,:]
# A1, A2 = fftshift(mn̄s-mn̄s₁,(3:4))[idx1,idx2,:,:], fftshift(mn̄s₂,(3:4))[idx1,idx2,:,:]
# A1, A2 = fftshift(mn̄s-mn̄s₂,(3:4))[idx1,idx2,:,:], fftshift(mn̄s₁,(3:4))[idx1,idx2,:,:]

# Dx,Dy = 0.8,0.8 #6.0,4.0
Dx,Dy = 6.0,4.0
A12_diff = A1 .- A2
println("")
@show indmax = argmax(abs2.(A1))
idx = indmax
@show A1[idx]
@show A2[idx]
@show r12 = A1[idx] / A2[idx]
@show rsumabs12 = sum(abs,A1) / sum(abs,A2)


xlimits = -0.5*Dx,0.5*Dx
ylimits = -0.5*Dy,0.5*Dy

Z1r = real(A1[:,:])
Z2r = real(A2[:,:])
Z3r = real(A12_diff[:,:])
Z1i = imag(A1[:,:])
Z2i = imag(A2[:,:])
Z3i = imag(A12_diff[:,:])
fig = Figure()

ax11 = fig[1,1][1,1] = Axis(fig,aspect=DataAspect())
hm11 = heatmap!(ax11,x(grid),y(grid),Z1r,colormap=:cividis)
cb11 = Colorbar(fig[1,1][1,2],hm11,width=30)

ax21 = fig[2,1][1,1] = Axis(fig,aspect=DataAspect())
hm21 = heatmap!(ax21,x(grid),y(grid),Z2r,colormap=:cividis)
cb21 = Colorbar(fig[2,1][1,2],hm21,width=30)

ax31 = fig[3,1][1,1] = Axis(fig,aspect=DataAspect())
hm31 = heatmap!(ax31,x(grid),y(grid),Z3r,colormap=:cividis)
cb31 = Colorbar(fig[3,1][1,2],hm31,width=30)

ax12 = fig[1,2][1,1] = Axis(fig,aspect=DataAspect())
hm12 = heatmap!(ax12,x(grid),y(grid),Z1i,colormap=:cividis)
cb12 = Colorbar(fig[1,2][1,2],hm12,width=30)

ax22 = fig[2,2][1,1] = Axis(fig,aspect=DataAspect())
hm22 = heatmap!(ax22,x(grid),y(grid),Z2i,colormap=:cividis)
cb22 = Colorbar(fig[2,2][1,2],hm22,width=30)

ax32 = fig[3,2][1,1] = Axis(fig,aspect=DataAspect())
hm32 = heatmap!(ax32,x(grid),y(grid),Z3i,colormap=:cividis)
cb32 = Colorbar(fig[3,2][1,2],hm32,width=30)

axs = [ax11,ax21,ax31,ax12,ax22,ax32]
xlims!.(axs,(xlimits,))
ylims!.(axs,(ylimits,))

fig

##
ng,ng_pb = pullback(calc_ng,om,ei,nng,k,Hv,grid)
om̄₁,eī₁,nnḡ,k̄,Hv̄,grid̄ = ng_pb(1.0)
Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
mag,m⃗,n⃗ = mag_m_n(k,grid)
mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
Hₜ = reshape(Hv,(2,Ns...))
D = 1im * fft( kx_tc( Hₜ,mns,mag), _fftaxes(grid) )
E = ε⁻¹_dot( D, ei)
# E = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ei)
# H = inv(om) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
H = fft( tc(Hₜ,mns), (2:3) ) * (-1im * om)
P = 2*real(_sum_cross_z(conj(E),H))

nḡ = 1.0
W̄ = nḡ / P
om̄₁₁ = 2*om * N(grid) * W̄
nnḡ2 = _outer(E,E) * W̄
H̄ = (-2*ng*W̄) * _cross(repeat([0.,0.,1.],outer=(1,Ns...)), E)
Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,repeat([0.,0.,1.],outer=(1,Ns...))) )
om̄₁₂ = dot(H,H̄) / om
om̄₁2 = om̄₁₁ + om̄₁₂
eī₁2 = _outer(Ē,D)
𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ei),(2:3))
𝓕⁻¹_H̄ = bfft( H̄ ,(2:3))
H̄ₜ = 1im*( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mns,mag) + om*ct(𝓕⁻¹_H̄,mns) )
𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ = 1im *_outer(_dot(repeat([0. 1. ;-1. 0. ],outer=(1,1,Ns...)), Hₜ), 𝓕⁻¹_ε⁻¹_Ē )
@tullio māg2[ix,iy] := mns[a,b,ix,iy] * -conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ[a,b,ix,iy])
mn̄s2 = -conj( 1im*om*_outer(Hₜ,𝓕⁻¹_H̄) + _mult(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ,mag))
m̄2 = reinterpret(reshape,SVector{3,Float64},real(view(mn̄s2,1,:,:,:)))
n̄2 = reinterpret(reshape,SVector{3,Float64},real(view(mn̄s2,2,:,:,:)))
@show k̄2 = ∇ₖmag_m_n(māg2,m̄2,n̄2,mag,m⃗,n⃗;dk̂=SVector(0.,0.,1.))
@show k̄

##
m̄ = real(mn̄s[:,1,:,:])
n̄ = real(mn̄s[:,2,:,:])
m̄s = reinterpret(reshape,SVector{3,Float64},m̄)
n̄s = reinterpret(reshape,SVector{3,Float64},m̄)
k̄ = ∇ₖmag_m_n(māg,m̄s,n̄s,mag,m⃗,n⃗)

##
k̄ = ∇ₖmag_m_n(māg, view(mn̄s,:,1,:,:), view(mn̄s,:,2,:,:),mag,m⃗,n⃗)
# mns_T = permutedims(mns,(2,1,3,4))


# kx̄_m⃗ =
# kx̄_n⃗ =  -real.( λẽ_sv .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
# māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
# k̄ₕ = -mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1]


# kx̄_m⃗ = real.( λẽ_sv .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
# kx̄_n⃗ =  -real.( λẽ_sv .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
# māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
# k̄ₕ = -mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1]


# H̄ₜ = 1im*( kx_ct(bfft(ε⁻¹_dot( Ē, ei),(2:3)),mns,mag) + om*ct(bfft( H̄ ,(2:3)),mns) )
Hv̄2 = vec(H̄ₜ)

using Tullio
function _outer2(v1::TA1,v2::TA2) where {TA1<:AbstractArray{<:Number,3},TA2<:AbstractArray{<:Number,3}}
        @tullio A[i,j,ix,iy] := v1[i,ix,iy] * conj(v2[j,ix,iy])
end



##
eī₁2 = _outer(Ē,D)

A1, A2 = eī₁,eī₁2
A12_diff = A1 .- A2
println("")
@show indmax = argmax(abs2.(A1))
idx = indmax
@show A1[idx]
@show A2[idx]
@show r12 = A1[idx] / A2[idx]
@show rsumabs12 = sum(abs,A1) / sum(abs,A2)

Dx,Dy = 6.0,4.0
idx1,idx2 = 1,3
xlimits = -0.5*Dx,0.5*Dx
ylimits = -0.5*Dy,0.5*Dy

Z1r = real(A1[idx1,idx2,:,:])
Z2r = real(A2[idx1,idx2,:,:])
Z3r = real(A12_diff[idx1,idx2,:,:])
Z1i = imag(A1[idx1,idx2,:,:])
Z2i = imag(A2[idx1,idx2,:,:])
Z3i = imag(A12_diff[idx1,idx2,:,:])
fig = Figure()

ax11 = fig[1,1][1,1] = Axis(fig,aspect=DataAspect())
hm11 = heatmap!(ax11,x(grid),y(grid),Z1r,colormap=:cividis)
cb11 = Colorbar(fig[1,1][1,2],hm11,width=30)

ax21 = fig[2,1][1,1] = Axis(fig,aspect=DataAspect())
hm21 = heatmap!(ax21,x(grid),y(grid),Z2r,colormap=:cividis)
cb21 = Colorbar(fig[2,1][1,2],hm21,width=30)

ax31 = fig[3,1][1,1] = Axis(fig,aspect=DataAspect())
hm31 = heatmap!(ax31,x(grid),y(grid),Z3r,colormap=:cividis)
cb31 = Colorbar(fig[3,1][1,2],hm31,width=30)

ax12 = fig[1,2][1,1] = Axis(fig,aspect=DataAspect())
hm12 = heatmap!(ax12,x(grid),y(grid),Z1i,colormap=:cividis)
cb12 = Colorbar(fig[1,2][1,2],hm12,width=30)

ax22 = fig[2,2][1,1] = Axis(fig,aspect=DataAspect())
hm22 = heatmap!(ax22,x(grid),y(grid),Z2i,colormap=:cividis)
cb22 = Colorbar(fig[2,2][1,2],hm22,width=30)

ax32 = fig[3,2][1,1] = Axis(fig,aspect=DataAspect())
hm32 = heatmap!(ax32,x(grid),y(grid),Z3i,colormap=:cividis)
cb32 = Colorbar(fig[3,2][1,2],hm32,width=30)

axs = [ax11,ax21,ax31,ax12,ax22,ax32]
xlims!.(axs,(xlimits,))
ylims!.(axs,(ylimits,))

fig

##
# H̄ₜ₁ = 1im*N(grid)* kx_ct(ifft(ε⁻¹_dot( Ē, ei),(2:3)),mns,mag)
# H̄ₜ₂ = 1im*N(grid)*om*ct(ifft( H̄ ,(2:3)),mns)
# H̄ₜ = H̄ₜ₁ + H̄ₜ₂

# H̄ₜ = 1im*( kx_ct(bfft(ε⁻¹_dot( Ē, ei),(2:3)),mns,mag) + om*ct(bfft( H̄ ,(2:3)),mns) )

𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ei),(2:3))
𝓕⁻¹_H̄ = bfft( H̄ ,(2:3))
H̄ₜ = 1im*( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mns,mag) + om*ct(𝓕⁻¹_H̄,mns) )


A1, A2 = fftshift(H̄ₜ_pb,(2:3)), fftshift(H̄ₜ,(2:3))
A12_diff = A1 .- fftshift(H̄ₜ,(2:3))
println("")
@show indmax = argmax(abs2.(A1))
idx = indmax
@show A1[idx]
@show A2[idx]
@show r12 = A1[idx] / A2[idx]
@show rsumabs12 = sum(abs,A1) / sum(abs,A2)

Dx,Dy = 0.8,0.8
idx = 2
xlimits = -0.5*Dx,0.5*Dx
ylimits = -0.5*Dy,0.5*Dy

Z1r = real(A1[idx,:,:])
Z2r = real(A2[idx,:,:])
Z3r = real(A12_diff[idx,:,:])
Z1i = imag(A1[idx,:,:])
Z2i = imag(A2[idx,:,:])
Z3i = imag(A12_diff[idx,:,:])
fig = Figure()

ax11 = fig[1,1][1,1] = Axis(fig,aspect=DataAspect())
hm11 = heatmap!(ax11,x(grid),y(grid),Z1r,colormap=:cividis)
cb11 = Colorbar(fig[1,1][1,2],hm11,width=30)

ax21 = fig[2,1][1,1] = Axis(fig,aspect=DataAspect())
hm21 = heatmap!(ax21,x(grid),y(grid),Z2r,colormap=:cividis)
cb21 = Colorbar(fig[2,1][1,2],hm21,width=30)

ax31 = fig[3,1][1,1] = Axis(fig,aspect=DataAspect())
hm31 = heatmap!(ax31,x(grid),y(grid),Z3r,colormap=:cividis)
cb31 = Colorbar(fig[3,1][1,2],hm31,width=30)

ax12 = fig[1,2][1,1] = Axis(fig,aspect=DataAspect())
hm12 = heatmap!(ax12,x(grid),y(grid),Z1i,colormap=:cividis)
cb12 = Colorbar(fig[1,2][1,2],hm12,width=30)

ax22 = fig[2,2][1,1] = Axis(fig,aspect=DataAspect())
hm22 = heatmap!(ax22,x(grid),y(grid),Z2i,colormap=:cividis)
cb22 = Colorbar(fig[2,2][1,2],hm22,width=30)

ax32 = fig[3,2][1,1] = Axis(fig,aspect=DataAspect())
hm32 = heatmap!(ax32,x(grid),y(grid),Z3i,colormap=:cividis)
cb32 = Colorbar(fig[3,2][1,2],hm32,width=30)

axs = [ax11,ax21,ax31,ax12,ax22,ax32]
xlims!.(axs,(xlimits,))
ylims!.(axs,(ylimits,))

fig

##
eī₁2 ≈ eī₁
(2*eī₁2) ≈ eī₁
eī₁[:,:,64,64]
eī₁2[:,:,64,64]

eī₁2 = _outer(Ē,-DD)
eī₁2 ≈ eī₁
(2*eī₁2) ≈ eī₁
eī₁[:,:,64,64]
eī₁2[:,:,64,64]

eī₁2 = _outer(Ē,conj(DD))
eī₁2 ≈ eī₁
(2*eī₁2) ≈ eī₁
eī₁[:,:,64,64]
eī₁2[:,:,64,64]

eī₁2 = _outer(Ē,-conj(DD))
eī₁2 ≈ eī₁
(2*eī₁2) ≈ eī₁
eī₁[:,:,64,64]
eī₁2[:,:,64,64]

eī₁2 = _outer(conj(Ē),DD)
eī₁2 ≈ eī₁
(2*eī₁2) ≈ eī₁
eī₁[:,:,64,64]
eī₁2[:,:,64,64]

eī₁2 = _outer(-conj(Ē),DD)
eī₁2 ≈ eī₁
(2*eī₁2) ≈ eī₁
eī₁[:,:,64,64]
eī₁2[:,:,64,64]

eī₁2 = _outer(conj(Ē),conj(DD))
eī₁2 ≈ eī₁
(2*eī₁2) ≈ eī₁
eī₁[:,:,64,64]
eī₁2[:,:,64,64]

eī₁2 = _outer(conj(Ē),-conj(DD))
eī₁2 ≈ eī₁
(2*eī₁2) ≈ eī₁
eī₁[:,:,64,64]
eī₁2[:,:,64,64]


@assert nnḡ2 ≈ nnḡ
@assert eī₁2 ≈ eī₁
##


@show om̄₁
@show om̄₂ = ei_pb( eī₁ )[1]
@show om̄₂2 = dot(herm(eī₁), ∂ε⁻¹_∂ω(eps,ei,nng,om))
@show om̄₃ = nng_pb( nnḡ )[1]
@show om̄₃2 = dot(herm(nnḡ), ngvd )
@show om̄₃3 = dot(nnḡ, ngvd )
∂ω²∂k_nd = 2 * HMₖH(Hv,ei,mag,m,n)
( _, _, om̄₄, eī₂ ) = ∇solve_k(
        (k̄,Hv̄),
        (k,Hv),
        ∂ω²∂k_nd,
        om,
        ei,
        grid; eigind=1)
@show om̄₅ = ei_pb( eī₂ )[1]
@show om̄₅2 = dot(herm(eī₂), ∂ε⁻¹_∂ω(eps,ei,nng,om)) # ei_pb( eī₂ )[1]
@show om̄1 = om̄₁ + om̄₂ + om̄₃ + om̄₄ + om̄₅
@show om̄2 = om̄₁ + om̄₂2 + om̄₃2 + om̄₄ + om̄₅2

(om̄₂ + om̄₃ + om̄₅) - (om̄₂2 + om̄₃2 + om̄₅2)
om̄₂ - om̄₂2
om̄₃ - om̄₃2
om̄₅ - om̄₅2

om̄₄2,p̄2 = k_Hv_pb( ( k̄, Hv̄ ) )
# om̄₅2 = ei_pb( eī₂2 )[1]
@show om̄3 = om̄₁ + om̄₂ + om̄₃ + om̄₄2  #+ om̄₅
om̄₁
om̄₂
om̄₃
om̄₄2

##
@btime _sum_cross($EE,$HH)[3] # 189.506 μs (1 allocation: 128 bytes)
@btime _sum_cross_z($EE,$HH) # 42.425 μs (1 allocation: 32 bytes)
@btime gradient((a,b)->abs2(_sum_cross(a,b)[3]),$EE,$HH) # 2.281 ms (29 allocations: 4.50 MiB)
@btime gradient((a,b)->abs2(_sum_cross_z(a,b)),$EE,$HH) # 2.121 ms (41 allocations: 4.50 MiB)

ff1(v1,v2) = abs2(sum(sum(gradient((a,b)->abs2(_sum_cross_z(a,b)),v1,v2))))
ff1(EE,HH)
EEbar,HHbar = gradient(ff1,EE,HH)

PP = 2*real(_sum_cross_z(conj(EE),HH))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
# W = dot(E,_dot((ε+nng),E))             # energy density per unit length
WW = dot(EE,_dot(nng,EE)) + N(grid)*om^2     # energy density per unit length
nngg = real( WW / PP )



ng1,ng1_pb = pullback(calc_ng,om,p,grid)
dng_dom_RM,dng_dp_RM,_ = ng1_pb(1.0)
dng_dom_FD = derivFD(oo->calc_ng(oo,p,grid),om)
dng_dom_FD2 = derivFD2(oo->calc_ng(oo,p,grid),om)
dng_dp_FD = gradFD2(pp->calc_ng(om,pp,grid),p)

function ∇calc_ng(Δng,ng,ω,ε⁻¹,nng,k,Hv,grid::Grid{ND,T}) where {ND,T<:Real}
        Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))

        return NoTangent(),ω̄,ε̄⁻¹,nnḡ,k̄,Hv̄,NoTangent()
end




##

function ng_gvd(ω,ε,ε⁻¹,nng,nng⁻¹,ngvd,k,Hv,grid::Grid{ND,T};eigind=1,log=true) where {ND,T<:Real}
	# calculate om̄ = ∂²ω²/∂k²
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	m = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,m⃗)))
	n = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,n⃗)))
	∂ω²∂k_nd = 2 * HMₖH(Hv,ε⁻¹,mag,m,n)
	k̄, H̄, nngī  = ∇HMₖH(k,Hv,nng⁻¹,grid; eigind)
	( _, _, om̄₁, eī₁ ) = ∇solve_k(	  (k̄,H̄),
									 	(k,Hv),
									  	∂ω²∂k_nd,
									   	ω,
									    ε⁻¹,
										grid; eigind)
	# nngī2 = copy(reinterpret(SMatrix{3,3,T,9},copy(reshape( nngī , 9*Ns[1], Ns[2:end]...))))
	# nngī_herm = (real.(nngī2) .+ transpose.(real.(nngī)) ) ./ 2
	# eī_herm = (real.(eī₁) .+ transpose.(real.(eī₁)) ) ./ 2
	om̄₂ = dot(herm(nngī), ∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,ω))
	om̄₃ = dot(herm(eī₁), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
	om̄ = om̄₁ + om̄₂ + om̄₃
	# calculate and return neff = k/ω, ng = ∂k/∂ω, gvd = ∂²k/∂ω²
	∂ω²∂k_disp = 2 * HMₖH(Hv,nng⁻¹,mag,m,n)
	neff = k / ω
	# ng = 2 * ω / ∂ω²∂k_disp # HMₖH(H⃗,nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗))) # ng = ∂k/∂ω
	gvd = 2 / ∂ω²∂k_disp - ω * 4 / ∂ω²∂k_disp^2 * om̄ #( ng / ω ) * ( 1. - ( ng * om̄ ) )

	Hₜ = reshape(Hv,(2,Ns...))
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
	EE = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
	HH = inv(ω) * fft(tc(kx_ct( ifft( EE, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
	EEs = copy(reinterpret(reshape,SVector{3,Complex{T}},EE))
	HHs = copy(reinterpret(reshape,SVector{3,Complex{T}},HH))
	# Sz = dot.(cross.(conj.(EEs),HHs),(SVector(0.,0.,1.),))
	Sz = getindex.(cross.(conj.(EEs),HHs),(3,))
	PP = 2*sum(Sz)
	# PP = 2*real( mapreduce((a,b)->dot(cross(conj(a),b),SVector(0.,0.,1.)),+,zip(EEs,HHs)))
	WW = dot(EE,_dot((ε+nng),EE))
	ng = real( WW / PP )

	return neff, ng, gvd
end

##
# A1, A2 = real(HH), real(HH2)
A1, A2 = real(H̄), imag(H̄)
# A1, A2 = real(H1), real(H12)
# A1, A2 = imag(H1), imag(H12)
A12_diff = A1 .- A2
idx = 2
Z1 = A1[idx,:,:]
Z2 = A2[idx,:,:]
Z3 = A12_diff[idx,:,:]
fig = Figure()

ax1 = fig[1,1][1,1] = Axis(fig,aspect=DataAspect())
hm1 = heatmap!(ax1,x(grid),y(grid),Z1,colormap=:cividis)
cb1 = Colorbar(fig[1,1][1,2],hm1,width=30)

ax2 = fig[2,1][1,1] = Axis(fig,aspect=DataAspect())
hm2 = heatmap!(ax2,x(grid),y(grid),Z2,colormap=:cividis)
cb2 = Colorbar(fig[2,1][1,2],hm2,width=30)

ax3 = fig[3,1][1,1] = Axis(fig,aspect=DataAspect())
hm3 = heatmap!(ax3,x(grid),y(grid),Z3,colormap=:cividis)
cb3 = Colorbar(fig[3,1][1,2],hm3,width=30)

fig


##
op1 = ei + _dot(ei,nng,ei)
op2 =  _dot(ei,(eps+nng),ei)

ng1 = om / HMₖH(Hv,op1,real(mag),real(flat(m⃗)),real(flat(n⃗)))
HMₖH(Hv,op1,real(mag),real(flat(m⃗)),real(flat(n⃗)))
HMH(Hv,op1,real(mag),real(flat(m⃗)),real(flat(n⃗)))*N(grid)
(HMH(Hv,_dot(ei,nng,ei),real(mag),real(flat(m⃗)),real(flat(n⃗)))+om^2)*N(grid)
om / HMₖH(Hv,nngi,real(mag),real(flat(m⃗)),real(flat(n⃗)))

m=flat(m⃗);
n=flat(n⃗);
mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3])));
H = reshape(Hv,(2,grid.Nx,grid.Ny))
om / real( dot(H, -kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mn), (2:3) ), real(op1)), (2:3)),mn,mag) ) )
real( dot(H, -kx_ct( ifft( ε⁻¹_dot( fft( kx_tc(H,mn,mag), (2:3) ), real(ei)), (2:3)),mn,mag) ) )
M̂1 = HelmholtzMap(k,ei,grid)
M̂_w = HelmholtzMap(k,op2,grid)
dot(Hv,M̂1,Hv)
dot(Hv,M̂1,Hv) / (om^2)
E1 = 1im * ε⁻¹_dot( fft( kx_tc(H,mn,mag), (2:3) ), real(ei))
E1s = copy(reinterpret(reshape,SVector{3,ComplexF64},E1))
H1 = fft( tc(H,mn), (2:3) ) * (-1im * om)
H12 =   inv(om) * fft( tc( kx_ct(ifft(E1, (2:3)),mn,mag),mn), (2:3) )
H1s = copy(reinterpret(reshape,SVector{3,ComplexF64},H1))
P1s = cross.(conj.(E1s),H1s)
P1 = copy(reinterpret(reshape,ComplexF64,P1s))
P1z = getindex.(cross.(conj.(E1s),H1s),(3,))
@tullio P12[i,ix,iy] := conj(E1)[mod(i-2),ix,iy] * H1[mod(i-1),ix,iy] - conj(E1)[mod(i-1),ix,iy] * H1[mod(i-2),ix,iy] (i in 1:3) verbose=true
@tullio Pz := conj(E1)[1,ix,iy] * H1[2,ix,iy] - conj(E1)[2,ix,iy] * H1[1,ix,iy]
(2*real(sum(P1z)) )
P1 = (2*real(Pz) )
# W1 = dot(Hv,M̂_w,Hv) * N(grid)
W1 = (HMH(Hv,_dot(ei,nng,ei),real(mag),real(flat(m⃗)),real(flat(n⃗)))+om^2)*N(grid)
W1/(2*Pz)
ng1 =  W1 / (2*real(Pz))
inv(ng1)
W1
sum(P1z)
W1 / (sum(P1z)*δ(grid))

ng_z(Hₜ,om,ei,nng,mag,m,n)
H̄ₜ,om̄,eī,nn̄g,māg,m̄,n̄ = Zygote.gradient(ng_z,Hv,om,ei,nng,mag,m,n)

##
E1 = E⃗(k,Hv,om,ei,nng,grid; normalized=true, nnginv=false)
H1 = H⃗(k,Hv,om,ei,nng,grid; normalized=true, nnginv=false)
E1s = copy(reinterpret(reshape,SVector{3,ComplexF64},E1))
H1s = copy(reinterpret(reshape,SVector{3,ComplexF64},H1))

Sz1 = dot.(cross.(conj.(E1s),H1s),(SVector(0.,0.,1.),))
P1 = 2*real(sum(Sz1)) * δ(grid)
dot(E1,_dot(nng,E1)) * δ(grid)
dot(E1,_dot(eps,E1)) * δ(grid)
dot(H1,H1) * δ(grid)
W12 = dot(E1,_dot((2*eps+Deps_FD*om),E1)) * δ(grid)
W13 = dot(E1,_dot((eps+nng),E1)) * δ(grid)
W14 = (dot(E1,_dot(nng,E1)) + dot(E1,_dot(eps,E1))) * δ(grid)
W1 = (dot(E1,_dot(nng,E1)) + dot(H1,H1)) * δ(grid)

W1/P1
W12/P1
W13/P1
W14/P1
Hnorm2 = sqrt( dot(H1,H1)  / dot(E1,_dot(eps,E1)) )
dot(H1./Hnorm2,H1./Hnorm2) * δ(grid)
Z0 = 376.730313668

ng1 = om / HMₖH(Hv,real(nngi),real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗))) #  material disp. included
2*om / HMₖH(Hv,real(ei+nngi),real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗)))
##
nngi,nngi_pb = Zygote.pullback(om) do ω
	ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
	return nng⁻¹
end

Domeps_FD = FiniteDifferences.central_fdm(5,1)(om) do ω
	ε,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[false,false,true],rwg,grid));
	return ε * ω
end

Domeps_FD2 = FiniteDiff.finite_difference_derivative(om) do ω
	ε,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[false,false,true],rwg,grid));
	return ε * ω
end

Domeps_FM = ForwardDiff.derivative(om) do ω
	ε,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[false,false,true],rwg,grid));
	return ε * ω
end

Deps_FD = FiniteDifferences.central_fdm(5,1)(om) do ω
	ε,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[false,false,true],rwg,grid));
	return ε
end

Deps_FD2 = FiniteDiff.finite_difference_derivative(om) do ω
	ε,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[false,false,true],rwg,grid));
	return ε
end

Deps_FM = ForwardDiff.derivative(om) do ω
	ε,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[false,false,true],rwg,grid));
	return ε
end

Dei_FD = FiniteDifferences.central_fdm(5,1)(om) do ω
	ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
	return ε⁻¹
end

Dei_FD2 = FiniteDiff.finite_difference_derivative(om) do ω
	ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
	return ε⁻¹
end

Dei_FM = ForwardDiff.derivative(om) do ω
	ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
	return ε⁻¹
end

Dnng_FD = FiniteDifferences.central_fdm(5,1)(om) do ω
	ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
	return nng
end

Dnng_FD2 = FiniteDiff.finite_difference_derivative(om) do ω
	ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
	return nng
end

Dnng_FM = ForwardDiff.derivative(om) do ω
	ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
	return nng
end


Dnngi_FD = FiniteDifferences.central_fdm(5,1)(om) do ω
	ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
	return nng⁻¹
end

Dnngi_FD2 = FiniteDiff.finite_difference_derivative(om) do ω
	ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
	return nng⁻¹
end

Dnngi_FM = ForwardDiff.derivative(om) do ω
	ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
	return nng⁻¹
end



ε⁻¹2,nng2,nng⁻¹2 = deepcopy(smooth(om,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));

nng,nng_pb = Zygote.pullback(om) do ω
	ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
	return nng
end

ei,ei_pb = Zygote.pullback(om) do ω
	ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid));
	return ε⁻¹
end

eps,eps_pb = Zygote.pullback(om) do ω
	ε,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[false,false,true],rwg,grid));
	return ε
end

ngvd,ngvd_pb = Zygote.pullback(om) do ω
	ngvd,nng2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,false],rwg,grid,volfrac_smoothing));
	return ngvd
end

nngi2,nngi2_pb = Zygote.pullback(om) do ω
	ngvd,nngi2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,true],rwg,grid,volfrac_smoothing));
	return nngi2
end

nng2,nng2_pb = Zygote.pullback(om) do ω
	ngvd,nng2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,false],rwg,grid,volfrac_smoothing));
	return nng2
end

# k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nngi,grid; eigind=1)
# k̄_nd, H̄_nd, nngī_nd  = ∇HMₖH(k,H⃗,ei,grid; eigind=1)
#
#
# omb2 = dot(herm(nngī), ∂nng⁻¹_∂ω(ei,nng,nngi,ngvd,om))
# omb2_pb = nngi_pb(herm(nngī))[1]
#
# sum(herm(nngī) .* herm(∂nng⁻¹_∂ω(ei,nng,nngi,ngvd,om)))
#
# using Symbolics: Sym, Num, Differential, expand_derivatives, simplify, jacobian, sparsejacobian, hessian, sparsehessian
# function nn̂g_model2(mat::AbstractMaterial; symbol=:λ)
# 	λ = Num(Sym{Real}(symbol))
# 	Dλ = Differential(λ)
# 	# n_model = sqrt.(get_model(mat,:ε,symbol))
# 	# ng_model = n_model - ( λ * expand_derivatives(Dλ(n_model)) )
# 	ε_model = get_model(mat,:ε,symbol)
# 	ω∂ε∂ω_model =   -1 * λ .* expand_derivatives.(Dλ.(ε_model),(true,))
# 	return ω∂ε∂ω_model ./ 2.0
# end
#
# λ = Num(Sym{Real}(:λ))
# eps_ln = get_model(LNx,:ε,:λ)
# Dλ = Differential(λ)
#
# ω∂ε∂ω_model =   -1 * λ * expand_derivatives(Dλ.(eps_ln))
# ω∂ε∂ω_model |> simplify
# nng_ln1 = nn̂g_model(LNx)
# nng_ln2 = nn̂g_model2(LNx)
#
# nng_sin1 = nn̂g_model(Si₃N₄)
# nng_sin2 = nn̂g_model2(Si₃N₄)
#
# nng_sin1 - nng_sin2

##
eps_mod = get_model(LNx,:ε,:λ)
nng_mod = nn̂g_model(LNx)

vac = Material(1.0)
eps_vac_mod = get_model(vac,:ε,:λ)
nng_vac_mod = nn̂g_model(vac)

λ = Num(Sym{Real}(:λ))
Dλ = Differential(λ)

deps_dom_mod = expand_derivatives.(Dλ.(eps_mod),(true,))
deps_dom = substitute(deps_dom_mod, [λ=>inv(om),])

nng_mod .== ( -1 * λ .* expand_derivatives.(Dλ.(eps_mod),(true,)) )
##
nngs = copy(reinterpret(reshape,SMatrix{3,3,Float64,9},reshape(nng,(9,128,128))))
nngis = copy(reinterpret(reshape,SMatrix{3,3,Float64,9},reshape(nngi,(9,128,128))))
epss = copy(reinterpret(reshape,SMatrix{3,3,Float64,9},reshape(eps,(9,128,128))))
eis = copy(reinterpret(reshape,SMatrix{3,3,Float64,9},reshape(ei,(9,128,128))))
ngvds = copy(reinterpret(reshape,SMatrix{3,3,Float64,9},reshape(ngvd,(9,128,128))))
nng2s = copy(reinterpret(reshape,SMatrix{3,3,Float64,9},reshape(nng2,(9,128,128))))

deps_dom = inv(om) * (nng - eps)
deps_doms = inv(om) .* (nngs .- epss)

dei_dom_s = -1 .* (eis .* deps_doms .* eis) # -(2.0/om) * (  eis.^2 .* inv.(nngis) .- eis )
dei_dom_sr = copy(flat(dei_dom_s))
dei_dom = -1.0 * _dot(ei,deps_dom,ei)  #-(2.0/om) * ( _dot(ei,ei,nng) - ei )

dnng_dom = ngvd
dnngi_dom = -1*_dot(nngi, ngvd, nngi)
dnngi_dom_s = -(nngis.^2 ) .* ( om*(eis.*inv.(nngis).^2 .- inv.(nngis)) .+ ngvds)
dnngi_dom_sr = copy(flat(dnngi_dom_s)) #copy(reinterpret(reshape,SMatrix{3,3,Float64,9},reshape(dnngi_dom,(9,128,128))))
dnngi_dom = _dot(nngi) #_dot( -nngi, nngi, ( om*( _dot(ei,nng,nng) - nng ) + ngvd ) )
dnngi_dom2 = _dot( _dot(-nngi, nngi), ( om*( _dot(_dot(ei,nng), nng) - nng ) + ngvd ) ) # ∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,om)

dnngi_dom ≈ dnngi_dom_sr
dnngi_dom_diff = dnngi_dom .- dnngi_dom_sr
maximum(abs.(dnngi_dom_diff))
argmax(abs.(dnngi_dom_diff))
##
(mag,m⃗,n⃗) = mag_m_n(k,g⃗(grid))
ng2 = om / HMₖH(H⃗,real(nngi),real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗))) #  material disp. included
ng_nd2 = om / HMₖH(H⃗,real(ei),real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗))) #  material disp. included
nngi2 = copy(flat(inv.(nngs)))
nngi3 = copy(flat(inv.(nng2s)))
ng3= om / HMₖH(H⃗,real(nngi2),real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗)))
ng4= om / HMₖH(H⃗,real(nngi3),real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗)))
@show ng2

##
using GLMakie
# A1,A2 = op1,op2
A1, A2 = abs2.(HH1), abs2.(HH2)
# A1, A2 = imag(H1), imag(H12)
# A1,A2 = eps, nng
# A1,A2 = eps, ngvd
# A1,A2 = om*Deps_FM + eps, nng
# A1,A2 = Dei_FD,Dei_FM
# A1,A2 = Dei_FD,dei_dom
# A1,A2 = Dei_FM,dei_dom
# A1,A2 = dei_dom,dei_dom_sr
# A1,A2 = Dnng_FD2,Dnng_FM
# A1,A2 = Dnng_FD2,dnng_dom
# A1,A2 = Dnngi_FD2,dnngi_dom
# A1,A2 = Dnngi_FD2,Dnngi_FM
# A1,A2 = dei_dom , dei_dom_sr
# A1,A2 = dnngi_dom , dnngi_dom_sr
# A1,A2 = dnngi_dom2 , dnngi_dom_sr
# A1,A2 = nng⁻¹2, nngi
# A1,A2 = flat(inv.(nngis)), nng
# A1,A2 = flat( ( om*(eis.*inv.(nngis).^2 .- inv.(nngis)) .+ ngvds) ) , ( om*( _dot(_dot(ε⁻¹,nng), nng) - nng ) + ngvd )
# A1,A2 = flat( ( om*(eis.*inv.(nngis).^2 .- inv.(nngis)) ) ) , ( om*( _dot(_dot(ε⁻¹,nng), nng) - nng ) )
# A1,A2 = flat(  om*(eis.*inv.(nngis).^2 )  ) , ( om*( _dot(_dot(ε⁻¹,nng), nng) ) )
# A1,A2 = flat(  om*(eis.*(nngs.^2) )  ) , ( om*( _dot(_dot(ε⁻¹,nng), nng) ) )
# A1,A2 = flat(  (nngs.^2)  ) , ( _dot(nng, nng) )
# A1,A2 = flat(  om*(eis.*(nngs.^2) )  ) , ( om*_dot(ε⁻¹,_dot(nng, nng) ) )
A12_diff = A1 .- A2
i1, i2 = 1,1
# Z1 = A1[i1,i2,:,:]
# Z2 = A2[i1,i2,:,:]
# Z3 = A12_diff[i1,i2,:,:]
Z1 = A1[i1,:,:]
Z2 = A2[i2,:,:]
Z3 = A12_diff[i1,:,:]
fig = Figure()

ax1 = fig[1,1][1,1] = Axis(fig,aspect=DataAspect())
hm1 = heatmap!(ax1,x(grid),y(grid),Z1,colormap=:cividis)
cb1 = Colorbar(fig[1,1][1,2],hm1,width=30)

ax2 = fig[2,1][1,1] = Axis(fig,aspect=DataAspect())
hm2 = heatmap!(ax2,x(grid),y(grid),Z2,colormap=:cividis)
cb2 = Colorbar(fig[2,1][1,2],hm2,width=30)

ax3 = fig[3,1][1,1] = Axis(fig,aspect=DataAspect())
hm3 = heatmap!(ax3,x(grid),y(grid),Z3,colormap=:cividis)
cb3 = Colorbar(fig[3,1][1,2],hm3,width=30)

fig


##
# om = 0.75

# neff = 2.0208779422976213
# ng = 2.4309145662550935
# gvd = 2314.1908086749054

# dneff_dom_AD = Zygote.gradient(fneff, om) = (0.5462650756617584 + 0.0im,)
# dneff_dom_FD = derivFD2(fneff, om) = 0.5463055868334399

# dng_dom_AD = (Zygote.gradient(fng, om))[1] = -0.22448595697309354 - 1.8225891892495094e-7im
# dng_dom_FD = derivFD2(fng, om) = -0.22391245888055966

# dgvd_dom_FD = derivFD2(fgvd, om) = 8555.769479283072


ngvd1 = smooth((om,),p,:fnĝvds,false,SMatrix{3,3,Float64,9}(0.,0.,0.,0.,0.,0.,0.,0.,0.),rwg,grid,volfrac_smoothing)
ngvd2 = first(smooth(om,p,(:fnĝvds,),[false,],rwg,grid,volfrac_smoothing))
##
om = 0.72
geom_fn = rwg
ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],geom_fn,grid));
ngvd,nng2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,false],geom_fn,grid,volfrac_smoothing));
@show dng_dom_FD = derivFD2(fng,om)
k1,Hv1 = solve_k(om,p,rwg,grid;nev=1,eigind=1)
k = k1
eigind = 1
ω = om
Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
mag,m⃗,n⃗ = mag_m_n(k,grid)
m = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,m⃗)))
n = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,n⃗)))
∂ω²∂k_nd = 2 * HMₖH(Hv1,ε⁻¹,mag,m,n)
k̄, H̄, nngī  = ∇HMₖH(k,Hv1,nng⁻¹,grid; eigind)
( _, _, om̄₁, eī₁ ) = ∇solve_k(	  (k̄,H̄),
									(k,Hv1),
									∂ω²∂k_nd,
									ω,
									ε⁻¹,
									grid; eigind)
# nngī2 = copy(reinterpret(SMatrix{3,3,T,9},copy(reshape( nngī , 9*Ns[1], Ns[2:end]...))))
# nngī_herm = (real.(nngī2) .+ transpose.(real.(nngī)) ) ./ 2
# eī_herm = (real.(eī₁) .+ transpose.(real.(eī₁)) ) ./ 2
@show om̄₁
# @show om̄₂ = dot(herm(nng), ∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,ω))
@show om̄₂ = dot(herm(nngī), ∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,ω))
@show om̄₃ = dot(herm(eī₁), ∂ε⁻¹_∂ω(ε⁻¹,nng,ω))
@show om̄ = om̄₁ + om̄₂ + om̄₃
# calculate and return neff = k/ω, ng = ∂k/∂ω, gvd = ∂²k/∂ω²
@show ∂ω²∂k_disp = 2 * HMₖH(Hv1,nng⁻¹,mag,m,n)
@show neff = k / ω
@show ng = 2 * ω / ∂ω²∂k_disp # HMₖH(H⃗,nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗))) # ng = ∂k/∂ω
@show gvd = 2 / ∂ω²∂k_disp - ω * 4 / ∂ω²∂k_disp^2 * om̄ #( ng / ω ) * ( 1. - ( ng * om̄ ) )
@show gvd2 = ( ng / ω ) * ( 1. - ( ng * om̄ ) )

# dng_dom_FD = derivFD2(fng, om) = -0.25771376890388886
# om̄₁ = 0.4522141123589637
# om̄₂ = dot(herm(nngī), ∂nng⁻¹_∂ω(ε⁻¹, nng, nng⁻¹, ngvd, ω)) = -0.01724246403745816
# om̄₃ = dot(herm(eī₁), ∂ε⁻¹_∂ω(ε⁻¹, nng, ω)) = 0.009524537893156076
# om̄ = om̄₁ + om̄₂ + om̄₃ = 0.4444961862146616
# ∂ω²∂k_disp = 2 * HMₖH(Hv1, nng⁻¹, mag, m, n) = 0.5893404966803133
# neff = k / ω = 2.0036637648712854
# ng = (2ω) / ∂ω²∂k_disp = 2.4434092143868495
# gvd = 2 / ∂ω²∂k_disp - ((ω * 4) / ∂ω²∂k_disp ^ 2) * om̄ = -0.2921437696599165
# gvd2 = (ng / ω) * (1.0 - ng * om̄) = -0.2921437696599165
# ng_nd = (2ω) / ∂ω²∂k_nd = 2.388157523115816

@show ng_nd = 2 * ω / ∂ω²∂k_nd
( ng / ω ) * ( 1. - ( ng_nd * om̄ ) )
( ng_nd / ω ) * ( 1. - ( ng * om̄ ) )
( ng_nd / ω ) * ( 1. - ( ng_nd * om̄ ) )
( ng / ω ) * ( 1. - ( ng * om̄₁ ) )
( ng / ω ) * ( 1. - ( ng * (om̄₂ + om̄₃) ) )

2 / ∂ω²∂k_nd - ω * 4 / ∂ω²∂k_disp^2 * om̄
2 / ∂ω²∂k_disp - ω * 4 / ∂ω²∂k_nd^2 * om̄
2 / ∂ω²∂k_nd - ω * 4 / ∂ω²∂k_nd^2 * om̄

##
ε⁻¹,nng,nng⁻¹ = smooth(om,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],rwg,grid);
ngvd = smooth((om,),p,:fnĝvds,false,SMatrix{3,3,Float64,9}(0.,0.,0.,0.,0.,0.,0.,0.,0.,),rwg,grid,volfrac_smoothing);
Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
mag,m⃗,n⃗ = mag_m_n(k,grid)
∂ω²∂k_nd = 2 * HMₖH(H⃗,ε⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗)))
k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind)
@btime ∇HMₖH($k,$H⃗,$nng⁻¹,$grid; eigind=1)
H= reshape(H⃗,(2,Ns...))
m = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,m⃗)))
n = real(HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,n⃗)))
mns = vcat(reshape(m,(1,3,Ns...)),reshape(n,(1,3,Ns...)))
d0 = randn(Complex{Float64}, (3,Ns...))
𝓕	 =	plan_fft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place FFT operator 𝓕
𝓕⁻¹ =	plan_bfft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place iFFT operator 𝓕⁻¹
Y = zx_ct( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * kx_tc(H,real(mns),real(mag))	, real(nng⁻¹)), real(mns) )
real(sum(zx_ct( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * kx_tc(H,mns,mag)	, nng⁻¹), mns )))
zx_ct(d0,mns)
real(sum(zx_ct(d0,mns)))
Zygote.gradient((a,b)->sum(real.(zx_ct(a,b))),d0,mns)
Zygote.gradient((a,b)->sum(real.(zx_ct(a,real.(b)))),d0,mns)

neff_ng_gvd(om,ε⁻¹,nng,nng⁻¹,ngvd,k,H⃗,grid)

Zygote.gradient((om,ε⁻¹,nng,nng⁻¹,ngvd,k,H⃗,grid)->neff_ng_gvd(om,ε⁻¹,nng,nng⁻¹,ngvd,k,H⃗,grid)[3],om,ε⁻¹,nng,nng⁻¹,ngvd,k,H⃗,grid)
dneff_dom = -2.6945

using Tullio
function zx_ct3(e⃗,mn)
	zxinds = [2; 1; 3]
	zxscales = [-1.; 1.; 0.]
	@tullio zxe⃗[b,i,j] := zxscales[a] * e⃗[a,i,j] * mn[b,zxinds[a],i,j] nograd=(zxscales,zxinds) threads=false # fastmath=false
	# @tullio zxe⃗[b,i,j] := zxscales[a] * e⃗[a,i,j] * mn[b,a,i,j] nograd=zxscales  # fastmath=false
end
zx_ct3(d0,mns)
Zygote.gradient((a,b)->abs(sum(zx_ct3(a,b))),d0,mns)
Zygote.gradient((a,b)->sum(abs2,zx_ct2(a,b)),d0,mns)

Mₖᵀ_plus_Mₖ(H⃗,k,nng⁻¹,grid)

sum(abs2,Mₖᵀ_plus_Mₖ(H⃗,k,nng⁻¹,grid))

Zygote.gradient((H,mns,mag,nng⁻¹)->sum(real.(zx_ct( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * kx_tc(H,real.(mns),real.(mag))	, real.(nng⁻¹)), real.(mns) ))),H,mns,mag,nng⁻¹)
Zygote.gradient((H,mns,mag,nng⁻¹)->real(sum(zx_ct( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * kx_tc(H,mns,mag)	, nng⁻¹), mns ))),H,mns,mag,nng⁻¹)
Zygote.gradient((k,H⃗,nng⁻¹)->(out=∇HMₖH(k,H⃗,nng⁻¹,grid; eigind); abs2(out[1]+sum(out[2])+sum(out[3]))),k,H⃗,nng⁻¹)
(out=∇HMₖH(k,H⃗,nng⁻¹,grid; eigind); real(out[1]+sum(out[2])+sum(out[3])))
(out=∇HMₖH(k,H⃗,nng⁻¹,grid; eigind); real(out[1]+sum(out[2])+sum(out[3])))
zx_ct(e⃗::AbstractArray{T,3},mn)
( _, _, om̄₁, eī₁ ) = ∇solve_k(	  (k̄,H̄),
									(k,H⃗),
									∂ω²∂k_nd,
									ω,
									ε⁻¹,
									grid; eigind)
# nngī2 = copy(reinterpret(SMatrix{3,3,T,9},copy(reshape( nngī , 9*Ns[1], Ns[2:end]...))))
# nngī_herm = (real.(nngī2) .+ transpose.(real.(nngī)) ) ./ 2
# eī_herm = (real.(eī₁) .+ transpose.(real.(eī₁)) ) ./ 2
om̄₂ = dot(herm(nng), ∂nng⁻¹_∂ω(ε⁻¹,nng,nng⁻¹,ngvd,ω))
om̄₃ = dot(herm(eī₁), ∂ε⁻¹_∂ω(ε⁻¹,nng,ω))
om̄ = om̄₁ + om̄₂ + om̄₃
# calculate and return neff = k/ω, ng = ∂k/∂ω, gvd = ∂²k/∂ω²
∂ω²∂k_disp = 2 * HMₖH(H⃗,nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗)))
neff = k / ω
ng = 2 * ω / ∂ω²∂k_disp # HMₖH(H⃗,nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗))) # ng = ∂k/∂ω
gvd = 2 / ∂ω²∂k_disp - ω * 4 / ∂ω²∂k_disp^2 * om̄ #( ng / ω ) * ( 1. - ( ng * om̄ ) )


##
ε⁻¹ |>size
eis = reinterpret(reshape,SMatrix{3,3,Float64,9},copy(reshape(ε⁻¹.data,(9,128,128))))
nngs = reinterpret(reshape,SMatrix{3,3,Float64,9},copy(reshape(nng.data,(9,128,128))))
nngis = reinterpret(reshape,SMatrix{3,3,Float64,9},copy(reshape(nng⁻¹.data,(9,128,128))))
ngvds = reinterpret(reshape,SMatrix{3,3,Float64,9},copy(reshape(ngvd.data,(9,128,128))))

dei_dom1 = ∂ε⁻¹_∂ω(eis,nngis,om)
dnngi_dom1 = ∂nng⁻¹_∂ω(eis,nngis,ngvds,om)

dei_dom1r = copy(reshape(reinterpret(Float64,dei_dom1),(3,3,128,128)))
dnngi_dom1r = copy(reshape(reinterpret(Float64,dnngi_dom1),(3,3,128,128)))

# ∂ε⁻¹_∂ω(ε⁻¹,nng⁻¹,ω) = -(2.0/ω) * (  ε⁻¹.^2 .* inv.(nng⁻¹) .- ε⁻¹ )
function ∂ε⁻¹_∂ω2(ε⁻¹,nng,ω)
	-(2.0/ω) * ( _dot(ε⁻¹,ε⁻¹,nng) - ε⁻¹ )
end
using Tullio


dei_dom2 = ∂ε⁻¹_∂ω2(ε⁻¹,nng,om)

@btime ∂ε⁻¹_∂ω2($ε⁻¹,$nng,$om)

dei_dom2 ≈ dei_dom1r
dei_dom3 ≈ dei_dom1r
dei_dom2.data ≈ dei_dom3
# ∂nng⁻¹_∂ω(ε⁻¹,nng⁻¹,ngvd,ω) = -(nng⁻¹.^2 ) .* ( ω*(ε⁻¹.*inv.(nng⁻¹).^2 .- inv.(nng⁻¹)) .+ ngvd) # (1.0/ω) * (nng⁻¹ .- ε⁻¹ ) .- (  ngvd .* (nng⁻¹).^2  )
function ∂nng⁻¹_∂ω2(ε⁻¹,nng,nng⁻¹,ngvd,ω)
	# -(nng⁻¹.^2 ) .* ( ω*(ε⁻¹.*inv.(nng⁻¹).^2 .- inv.(nng⁻¹)) .+ ngvd)
	_dot( -_dot(nng⁻¹,nng⁻¹), ( ω*( _dot(ε⁻¹,nng,nng) - nng ) + ngvd ) )
end

function ∂nng⁻¹_∂ω3(ε⁻¹,nng,nng⁻¹,ngvd,ω)
	# -(nng⁻¹.^2 ) .* ( ω*(ε⁻¹.*inv.(nng⁻¹).^2 .- inv.(nng⁻¹)) .+ ngvd)
	_dot( -nng⁻¹, nng⁻¹, ( ω*( _dot(ε⁻¹,nng,nng) - nng ) + ngvd ) )
end


dnngi_dom2 = ∂nng⁻¹_∂ω2(ε⁻¹,nng,nng⁻¹,ngvd,om)
dnngi_dom3 = ∂nng⁻¹_∂ω3(ε⁻¹,nng,nng⁻¹,ngvd,om)
@btime ∂nng⁻¹_∂ω2($ε⁻¹,$nng,$nng⁻¹,$ngvd,$om)
@btime ∂nng⁻¹_∂ω3($ε⁻¹,$nng,$nng⁻¹,$ngvd,$om)
dnngi_dom2 ≈ dnngi_dom1r
dnngi_dom3 ≈ dnngi_dom1r
eieinng1 = _dot(_dot(ε⁻¹,ε⁻¹),nng)
eieinng2 = _dot(ε⁻¹,ε⁻¹,nng)

@btime _dot(_dot($ε⁻¹,$ε⁻¹),$nng)
@btime _dot($ε⁻¹,$ε⁻¹,$nng)
eieinng1 ≈ eieinng2
##
EE1 = E⃗(k1,Hv1,om,ε⁻¹,nng,grid; normalized=true, nnginv=false)
EE2 = E⃗(k1,Hv1,om,ε⁻¹,nng⁻¹,grid; normalized=true, nnginv=true)
EE1 ≈ EE2
Zygote.gradient(oo->abs2(sum(E⃗(k1,Hv1,oo,ε⁻¹,nng,grid; normalized=true, nnginv=false))),om)
Zygote.gradient((ddx,ddy)->δ(Grid(ddx,ddy,128,128)),6.0,4.0)

##
Zygote.gradient((oo,pp)->sum(smooth(oo,pp,:fεs,true,rwg,grid)),om,p)
Zygote.gradient(oo->sum(smooth(oo,p,:fεs,true,rwg,grid)),om)
Zygote.gradient((oo,pp)->sum(sum(smooth(oo,pp,(:fεs,:fnn̂gs),rwg,grid))),om,p)
Zygote.gradient((oo,pp)->sum(sum(smooth(oo,pp,(:fεs,:fnn̂gs),[true,true,],rwg,grid))),om,p)
Zygote.gradient((oo,pp)->sum(sum(smooth(oo,pp,(:fεs,),[true,],rwg,grid))),om,p)
Zygote.gradient((oo,pp)->sum(sum(smooth(oo,pp,(:fεs,:fnn̂gs),[true,false],rwg,grid))),om,p)
Zygote.gradient(oo->sum(sum(smooth(oo,p,(:fεs,:fnn̂gs,:fnĝvds),[true,false,false],rwg,grid))),om)

ei1, ei1_pb = Zygote.pullback(om,p) do om,p
	# smooth(ω,p,:fεs,true,geom_fn,grid)
	smooth(om,p,(:fεs,:fnn̂gs),[true,false,],rwg,grid)[1]
end


gradRM(x->(solve_k(x,p,rwg,grid;nev=1)[1]/x),om)
gradFD(x->(solve_k(x,p,rwg,grid;nev=1)[1]/x),om)


FiniteDiff.finite_difference_derivative(x->solve(x,p,rwg,grid;nev=1)[1],om; absstep=0.06)
FiniteDiff.finite_difference_derivative(x->solve(x,p,rwg,grid;nev=1)[1],om; absstep=0.03)
FiniteDiff.finite_difference_derivative(x->solve(x,p,rwg,grid;nev=1)[1],om; absstep=0.01)
FiniteDiff.finite_difference_derivative(x->solve(x,p,rwg,grid;nev=1)[1],om; absstep=0.003)

solve(om,p,rwg,grid;nev=1)[2]
gradRM(x->solve(x,p,rwg,grid;nev=1)[2],om)
gradRM(x->solve(x,p,rwg,grid;nev=1)[2],om)
gradRM(x->solve(x,p,rwg,grid;nev=1)[2],om)
gradRM(x->solve(x,p,rwg,grid;nev=1)[2],om)
gradRM(x->solve(x,p,rwg,grid;nev=1)[2],om)
gradRM(x->solve(x,p,rwg,grid;nev=1)[2],om)
gradRM(x->solve(x,p,rwg,grid;nev=1)[2],om)
gradRM(x->solve(x,p,rwg,grid;nev=1)[2],om)

gradRM(x->solve(x[1],x[2:5],rwg,grid;nev=1)[2],vcat(om,p))





FiniteDiff.finite_difference_derivative(x->solve(x,p,rwg,grid;nev=1)[2],om; inplace=Val{false})
FiniteDiff.finite_difference_derivative(x->solve(x,p,rwg,grid;nev=1)[2],om; inplace=Val{false})
FiniteDiff.finite_difference_derivative(x->solve(x,p,rwg,grid;nev=1)[2],om; absstep=0.06)
FiniteDiff.finite_difference_derivative(x->solve(x,p,rwg,grid;nev=1)[2],om; absstep=0.03)
FiniteDiff.finite_difference_derivative(x->solve(x,p,rwg,grid;nev=1)[2],om; absstep=0.01)
FiniteDiff.finite_difference_derivative(x->solve(x,p,rwg,grid;nev=1)[2],om; absstep=0.003)

FiniteDiff.finite_difference_gradient(x->solve(x[1],x[2:5],rwg,grid;nev=1)[2],vcat(om,p); relstep=0.02)

gradFD(x->solve(x,p,rwg,grid;nev=1)[2],om)
gradFD(x->solve(x,p,rwg,grid;nev=1)[2],om)




neff,ng,gvd,E = solve(om,p,rwg,grid;nev=1);
dneff_dom = 0.54626
@show ng2 = neff + dneff_dom * om


Zygote.gradient(0.6) do om
	pp = [1.7,0.7, 0.5, π / 14.0]
	gr = Grid(6.0,4.0,128,128)
	gfn = xx->ridge_wg_partial_etch(xx[1],xx[2],xx[3],xx[4],0.5,LNxN,SiO₂N,6.0,4.0)
	ε⁻¹,nng⁻¹ = copy(smooth(om,pp,(:fεs,:fnn̂gs),[true,false],gfn,gr));
	kk,HH = solve_k(om,pp,gfn,gr;nev=1)
	nefff = kk / om
	(mag,m⃗,n⃗) = mag_m_n(kk,g⃗(gr))
	nngg = om / HMₖH(HH,real(nng⁻¹),real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗)))
	return nngg
end


Zygote.gradient(0.6) do om
	pp = [1.7,0.7, 0.5, π / 14.0]
	gr = Grid(6.0,4.0,128,128)
	gfn = xx->ridge_wg_partial_etch(xx[1],xx[2],xx[3],xx[4],0.5,LNxN,SiO₂N,6.0,4.0)
	ε⁻¹,nng⁻¹ = copy(smooth(om,pp,(:fεs,:fnn̂gs),[true,false],gfn,gr));
	kk,HH = solve_k(om,pp,gfn,gr;nev=1)
	nefff = kk / om
	(mag,m⃗,n⃗) = mag_m_n(kk,g⃗(gr))
	nngg = om  / HMₖH(HH,real(nng⁻¹),real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗)))
	return nngg
end

Zygote.gradient((oo,pp)->solve_k(oo,pp,rwg,grid;nev=1)[1],om,p)




FiniteDifferences.grad(central_fdm(3,1),x->solve_k(x,p,rwg,grid;nev=1)[1],om)
FiniteDifferences.grad(central_fdm(3,1),om_p->solve_k(om_p[1],om_p[2:5],rwg,grid;nev=1)[1],vcat(om,p))


Zygote.gradient((oo,pp)->solve_k(oo,pp,rwg,grid;nev=1)[1],1.1*om,p)
FiniteDifferences.grad(central_fdm(5,1),om_p->solve_k(om_p[1],om_p[2:5],rwg,grid;nev=1,Hguess=Hv1,kguess=k1)[1],vcat(1.1*om,p))


FiniteDifferences.estimate_step(central_fdm(3,1),oo->solve_k(oo,p,rwg,grid;nev=1,Hguess=Hv1,kguess=k1)[1],1.1*om)

FiniteDifferences.estimate_step(central_fdm(3,1),oo->first(solve_k(oo,p,rwg,grid;nev=1,Hguess=Hv1,kguess=k1)),1.1*om)
FiniteDifferences.estimate_step(central_fdm(5,1),oo->first(solve_k(oo,p,rwg,grid;nev=1,Hguess=Hv1,kguess=k1)),1.1*om)

solve(om,p,rwg,grid;nev=1)[1:3]
Zygote.gradient(oo->solve(oo,p,rwg,grid;nev=1)[1],om)
Zygote.gradient(oo->solve(oo,p,rwg,grid;nev=1)[2],om)
Zygote.gradient((oo,pp)->solve(oo,pp,rwg,grid;nev=1)[1],om,p)
Zygote.gradient((oo,pp)->solve(oo,pp,rwg,grid;nev=1)[2],om,p)


solve(om,p,rwg,grid;nev=1)[1:3]
solve(om,p,rwg,grid;nev=1)[1:3]
solve(om,p,rwg,grid;nev=1)[1:3]
Zygote.gradient(oo->solve(oo,p,rwg,grid;nev=1)[1],om)
Zygote.gradient(oo->solve(oo,p,rwg,grid;nev=1)[2],om)
Zygote.gradient(oo->solve(oo,p,rwg,grid;nev=1)[2],om)
Zygote.gradient(oo->solve(oo,p,rwg,grid;nev=1)[2],om)

solve(om,p,rwg,grid;nev=1)[1:3]
solve(om,p,rwg,grid;nev=1)[1:3]
Zygote.gradient(oo->solve(oo,p,rwg,grid;nev=1)[1],om)
Zygote.gradient(oo->solve(oo,p,rwg,grid;nev=1)[2],om)
Zygote.gradient(oo->solve(oo,p,rwg,grid;nev=1)[2],om)
Zygote.gradient(oo->solve(oo,p,rwg,grid;nev=1)[2],om)

Zygote.gradient(oo->solve_k(oo,p,rwg,Grid(6.,4.,128,128))[1],0.6)
Zygote.gradient(oo->solve_k(oo,p,rwg,Grid(6.,4.,128,128))[1],0.6)
Zygote.gradient(oo->solve_k(oo,p,rwg,Grid(6.,4.,128,128))[1],0.6)

Zygote.gradient(oo->solve_k(oo,[1.7,0.7, 0.5, π / 14.0],rwg,Grid(6.,4.,128,128);nev=2,eigind=1)[1]*inv(oo),0.6)
Zygote.gradient(oo->solve_k(oo,[1.7,0.7, 0.5, π / 14.0],rwg,Grid(6.,4.,128,128);nev=2,eigind=1)[1]*inv(oo),0.6)
Zygote.gradient(oo->solve_k(oo,[1.7,0.7, 0.5, π / 14.0],rwg,Grid(6.,4.,128,128);nev=2,eigind=1)[1]*inv(oo),0.6)


Zygote.gradient(oo->solve(oo,[1.7,0.7, 0.5, π / 14.0],rwg,Grid(6.,4.,128,128);nev=2,eigind=1)[2],0.6)
Zygote.gradient(oo->solve(oo,[1.7,0.7, 0.5, π / 14.0],rwg,Grid(6.,4.,128,128);nev=2,eigind=1)[2],0.6)
Zygote.gradient(oo->solve(oo,[1.7,0.7, 0.5, π / 14.0],rwg,Grid(6.,4.,128,128);nev=2,eigind=1)[2],0.6)

rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNxN,SiO₂N,Δx,Δy)
p = [1.9,0.5, 1.0,π / 14.0]

central_fdm(5,1)(x->solve(x,p,rwg,grid;nev=1)[1],om)
central_fdm(5,1)(x->solve(x,p,rwg,grid;nev=1)[2],om)
central_fdm(5,1)(x->solve(x,p,rwg,grid;nev=1)[2],om)
central_fdm(5,1)(x->solve(x,p,rwg,grid;nev=1)[2],om)

##
ω = 0.55
ε⁻¹,nng,nng⁻¹ = smooth(ω ,p,(:fεs,:fnn̂gs,:fnĝvds),[true,false,true],rwg,grid);
ms = ModeSolver(ω,p,rwg,grid; nev=2);
k2,Hv2 = solve_k(ms,ω,ε⁻¹)
Zygote.gradient(a->solve_k(ms,a,ε⁻¹)[1],ω)
ms.M̂.k⃗ = SVector(0., 0., ω*ñₘₐₓ(ms.M̂.ε⁻¹))
kz = Roots.find_zero(x -> _solve_Δω²(ms,x,ω;nev=1,eigind=2,maxiter=300,tol=1e-8,f_filter=nothing), ms.M̂.k⃗[3], Roots.Newton(); verbose=true, atol=tol, maxevals=60)



##
ω = 0.6 #1.0/1.9
ε⁻¹,nng,ngvd = smooth(ω,p,(:fεs,:fnn̂gs,:fnĝvds),[true,false,false],rwg,grid);
ε,nng⁻¹ = smooth(ω,p,(:fεs,:fnn̂gs),[false,true],rwg,grid);
ñₘₐₓ(ε⁻¹)
k_g = ñₘₐₓ(ε⁻¹)*ω
M̂ = HelmholtzMap(SVector(0.,0.,k_g), ε⁻¹, grid);
# ms = ModeSolver(k_g,ε⁻¹,grid; nev=3);

ms = ModeSolver(ω,p,rwg,grid; nev=2);
k1,Hv1 = solve_k(ω,p,rwg,grid;nev=1)
(mag,m⃗,n⃗) = mag_m_n(k1,dropgrad(ms.M̂.g⃗))
ng = ω / HMₖH(Hv1,real(nng⁻¹),real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗)))
neff = k1/ω
dneff_dom = central_fdm(5,1)(oo->(solve_k(oo,p,rwg,grid;nev=1)[1]/oo),ω)
neff + om * dneff_dom

##
ωs = [0.65, 0.75]

omsq1,Hv1 = solve_ω²(ms,1.4);

ω = 0.65
eigind = 1
# k,Hv = solve_k(ms,ω,rwg(p))
k,Hv = solve_k(ms,0.71,p,rwg)
# k,Hv = solve_k(ω,rwg(p),grid)
# ε⁻¹ = ms.M̂.ε⁻¹
# Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# Nranges = eachindex(grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
# H = reshape(H⃗[:,eigind],(2,Ns...))
g⃗s = g⃗(dropgrad(grid))
(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(x->mag_m_n(x,g⃗s),k)
# m = ms.M̂.m
# n = ms.M̂.n
# mns = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3])))
#


##
function solve_omsq!(k⃗,ε⁻¹,grid; nev=1,eigind=1,maxiter=3000,tol=1.6e-8,log=false,f_filter=nothing)::Tuple{Vector{T},Matrix{Complex{T}}} where {ND,T<:Real}
	M̂ = HelmholtzMap(k⃗, ε⁻¹, grid)
		# res = lobpcg!(ms.eigs_itr; log,not_zeros=false,maxiter,tol)
		res = LOBPCG(ms.M̂,ms.H⃗,I,ms.P̂)
		copyto!(ms.H⃗,res.X)
		copyto!(ms.ω²,res.λ)

	if isnothing(f_filter)
		return (copy(real(ms.ω²)), copy(ms.H⃗))
	else
		return filter_eigs(ms, f_filter)
	end
end

##
using Base.Iterators: product
using Symbolics: Num, Differential, Sym, build_function

mats = [LNx, SiO₂, Si₃N₄];
models = (:ε, (nn̂g_model,:ε), (nĝvd_model,:ε))
args = (:λ,)

mods1 = map( ij->get_model(mats[ij[1]],models[ij[2]],args...), Iterators.product(eachindex(mats),eachindex(models)) )
mods2 = mapreduce( ij->get_model(mats[ij[1]],models[ij[2]],args...), hcat, Iterators.product(eachindex(mats),eachindex(models)) )
mods3 = mapreduce( ij->get_model(mats[ij[1]],models[ij[2]],args...), vcat, Iterators.product(eachindex(mats),eachindex(models)) )
function generate_geometry_materials_fn(mats,models,args...)
	models_vcat = mapreduce(
		ij->get_model(mats[ij[1]],models[ij[2]],args...),
		vcat,
		Iterators.product(eachindex(mats),eachindex(models)),
	)
	geom_mats_fn = build_function(
		models_vcat,
		[Num(Sym{Real}(arg)) for arg in args]...;
		# force_SA=true,
		# convert_oop=false,
		expression=Val{false},
	)[1]
	return geom_mats_fn
end


geom_fn1 = generate_geometry_materials_fn(mats,models,args...)
geom_fn2 = generate_geometry_materials_fn(mats,models,args...)
geom_fn3 = eval(generate_geometry_materials_fn(mats,models,args...))
##


struct Geometry2{TF1,TF2,TP,TM1,TM2}
	shapes_fn::TF1
	materials_fn::TF2
	params::TP
	materials::TM1
	models::TM2
end

function Geometry2(shapes_fn,param_defaults,material_models)  #where S<:Shape{N} where N
	shapes = shapes_fn(param_defaults)
	mats =  materials(shapes)
	material_inds = matinds(shapes)
	shapes_fn_matinds = p -> [ (@set shp.data = matidx) for (shp,matidx) in zip(shapes_fn(p),material_inds) ]
	mats_fn = generate_geometry_materials_fn(mats,material_models,:λ)
	mat_names = getfield.(mats,(:name,))
	return Geometry2(
		shapes_fn_matinds,
		mats_fn,
		param_defaults,
		material_models,
		material_models,
	)
end


gg21 = Geometry2(x->rwg(x).shapes,p,(:ε,))
gg22 = Geometry2(x->rwg(x).shapes,p,(:ε, (nn̂g_model,:ε), (nĝvd_model,:ε)))

## check gradients



##

ff1 = x->sum(sum(foo10(x[1],x[2:5],:fεs,rwg,Grid(6.,4.,128,128))))
ff2 = x->sum(sum(foo11(x[1],x[2:5],(:fεs,:fnn̂gs),rwg,Grid(6.,4.,128,128))))
in1 = [0.6,p...]
##
println("ff1 grads: ")
@show dff1RM = gradRM(ff1,in1)
@show dff1FM = gradFM(ff1,in1)
@show dff1FD = gradFD(ff1,in1)
println("\nff2 grads: ")
@show dff2RM = gradRM(ff2,in1)
@show dff2FM = gradFM(ff2,in1)
@show dff2FD = gradFD(ff2,in1)

##
Zygote.gradient(x->sum(sum(foo10(x[1],x[2:5],:fεs,rwg,Grid(6.,4.,128,128)))),[0.6,p...])[1]
# ([3463.713922016181, 2490.3847632820093, 23034.200708066717, -16297.824941034361, 909.190141685666],)
Zygote.gradient(x->sum(sum(foo11(x[1],x[2:5],(:fεs,:fnn̂gs),rwg,Grid(6.,4.,128,128)))),[0.6,p...])
# ([5138.615248004345, 5055.878458463492, 46735.540925488545, -33011.18691659402, 1845.5821824636203],)

ForwardDiff.gradient(x->sum(sum(foo10(x[1],x[2:5],:fεs,rwg,Grid(6.,4.,128,128)))),[0.6,p...])
ForwardDiff.gradient(x->sum(sum(foo11(x[1],x[2:5],(:fεs,:fnn̂gs),rwg,Grid(6.,4.,128,128)))),[0.6,p...])

FiniteDifferences.grad()

##
geom = rwg(p)
om=0.6  # frequency ω = 1/λ
ix,iy = 84,71	 # smoothed pixel inds on rwg sidewall boundry using default params `p`

sinds = proc_sinds(corner_sinds(geom.shapes,xyzc))
mat_vals1 = map(f->SMatrix{3,3}(f(inv(om))),geom.fεs)
mat_vals2 = mapreduce(ss->[ map(f->SMatrix{3,3}(f(inv(om))),getfield(geom,ss))... ], hcat, [:fεs,:fnn̂gs,:fnĝvds]);

sv1 = smooth_val(sinds[ix,iy],geom.shapes,geom.material_inds,mat_vals1,xyz[ix,iy],vxlmin[ix,iy],vxlmax[ix,iy])
sv2 = smooth_val(sinds[ix,iy],geom.shapes,geom.material_inds,mat_vals2,xyz[ix,iy],vxlmin[ix,iy],vxlmax[ix,iy])


function fsv1(om_p; ix,iy,grid=Grid(6.,4.,128,128))
	geom = rwg(om_p[2:5])
	xyz,xyzc,vxlmin,vxlmax,sinds,fs = Zygote.@ignore begin
		xyz = x⃗(grid)
		xyzc = x⃗c(grid)
		vxlmin = @view xyzc[1:max((end-1),1),1:max((end-1),1)]
		vxlmax = @view xyzc[min(2,end):end,min(2,end):end]
		sinds = proc_sinds(corner_sinds(geom.shapes,xyzc))
		fs = geom.fεs
		return xyz,xyzc,vxlmin,vxlmax,sinds,fs
	end
	mat_vals = map(f->SMatrix{3,3}(f(inv(first(om_p)))),fs)
	sv1 = smooth_val(sinds[ix,iy],geom.shapes,geom.material_inds,mat_vals,xyz[ix,iy],vxlmin[ix,iy],vxlmax[ix,iy])
end

fsv1([0.6,p...];ix=84,iy=71)[1,2]
Zygote.gradient(a->fsv1(a;ix=84,iy=71)[1,2],[0.6,p...])
ForwardDiff.gradient(a->fsv1(a;ix=84,iy=71)[1,2],[0.6,p...])
FiniteDifferences.grad(central_fdm(3,1),a->fsv1(a;ix=84,iy=71)[1,2],[0.6,p...])

##


##

Srvol = S_rvol(geom,grid)
ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
minds = geom.material_inds
es = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fεs),[εᵥ,])
eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
ei_new = εₛ⁻¹(es,eis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
##
LNxN = NumMat(LNx;expr_module=@__MODULE__())
SiO2N = NumMat(SiO₂;expr_module=@__MODULE__())
rwg1(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNxN,SiO2N,Δx,Δy)
rwg2(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,NumMat(LNx),NumMat(SiO₂),Δx,Δy)

geom1 = rwg1(p)
geom2 = rwg2(p)

## single ω solve_n gradient checks, ms created within solve_n
function gradtest_solve_n(ω0)
        err_style = NEGATIVE*BOLD*BLUE_FG      # defined in Crayons.Box
        println("...............................................................")
        println("solve_n (single ω) gradient checks, ms created within solve_n: ")
        @show ω0
        neff1,ng1,gvd1,E1 = solve_n(ω0+rand()*0.1,rwg(p),grid)

        println("∂n_om, dispersive materials:")
        om = ω0 #+rand()*0.1
        println("\t∂n_om (Zygote):")
        ∂n_om_RAD = Zygote.gradient(x->solve_n(x,rwg(p),grid)[1],om)[1]
        println("\t$∂n_om_RAD")
        solve_n(om+rand()*0.2,rwg(p),grid)
        println("\t∂n_om (FD):")
        ∂n_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(x,rwg(p),grid)[1],om)[1]
        println("\t$∂n_om_FD")
        println(err_style("∂n_om_err:"))
        ∂n_om_err = abs(∂n_om_RAD - ∂n_om_FD) / abs(∂n_om_FD)
        println("$∂n_om_err")
        n_disp = solve_n(om,rwg(p),grid)[1]
        ng_manual_disp = n_disp + om * ∂n_om_FD
        println("ng_manual: $ng_manual_disp")

        println("∂ng_om, dispersive materials:")
        # om = ω0+rand()*0.1
        println("\t∂ng_om (Zygote):")
        ∂ng_om_RAD = Zygote.gradient(x->solve_n(x,rwg(p),grid)[2],om)[1]
        println("\t$∂ng_om_RAD")
        solve_n(om+rand()*0.2,rwg(p),grid)
        println("\t∂ng_om (FD):")
        ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(x,rwg(p),grid)[2],om)[1]
        println("\t$∂ng_om_FD")
        println(err_style("∂ng_om_err:"))
        ∂ng_om_err = abs( ∂ng_om_RAD -  ∂ng_om_FD) /  abs.(∂ng_om_FD)
        println("$∂ng_om_err")

        println("∂n_p, dispersive materials:")
        # om = ω0+rand()*0.1
        println("\t∂n_p (Zygote):")
        ∂n_p_RAD =  Zygote.gradient(x->solve_n(om,rwg(x),grid)[1],p)[1]
        println("\t$∂n_p_RAD")
        solve_n(om+rand()*0.2,rwg(p),grid)
        println("\t∂n_p (FD):")
        ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(om,rwg(x),grid)[1],p)[1]
        println("\t$∂n_p_FD")
        println(err_style("∂n_p_err:"))
        ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)
        println("$∂n_p_err")

        println("∂ng_p, dispersive materials:")
        # om = ω0+rand()*0.1
        println("\t∂ng_p (Zygote):")
        ∂ng_p_RAD = Zygote.gradient(x->solve_n(om,rwg(x),grid)[2],p)[1]
        println("\t$∂ng_p_RAD")
        solve_n(om+rand()*0.2,rwg(p),grid)
        println("\t∂ng_p (FD):")
        ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(om,rwg(x),grid)[2],p)[1]
        println("\t$∂ng_p_FD")
        println(err_style("∂ng_p_err:"))
        ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD
                println("$∂ng_p_err")
                println("...............................................................")
end

gradtest_solve_n(0.5)
gradtest_solve_n(0.7)
gradtest_solve_n(0.8)
gradtest_solve_n(0.9)

##
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 32, 32, 1;
grid = Grid(Δx,Δy,Nx,Ny)
kxtcsp = kx_tc_sp(k,grid)
# vec(kx_tc(H,mns,mag)) ≈ kxtcsp * H⃗
# vec(kx_ct(tc(H,mns),mns,mag)) ≈ -kxtcsp' * vec(tc(H,mns))
# @btime $kxtcsp * $H⃗ # 163.864 μs (2 allocations: 768.08 KiB)
# @btime vec(kx_tc($H,$mns,$mag)) # 378.265 μs (6 allocations: 768.34 KiB)
zxtcsp = zx_tc_sp(k,grid)
# vec(zx_tc(H,mns)) ≈ zxtcsp * H⃗
# vec(zx_ct(tc(H,mns),mns)) ≈ zxtcsp' * vec(tc(H,mns))
# @btime $zxtcsp * $H⃗ # 151.754 μs (2 allocations: 768.08 KiB)
# @btime vec(zx_tc($H,$mns)) # 296.939 μs (6 allocations: 768.38 KiB)
# zx_tc_sp(k,grid) == zx_ct_sp(k,grid)'
# vec(zx_tc(H,mns)) ≈ zx_tc_sp_coo(mag,mns) * H⃗
eisp = ε⁻¹_sp(0.75,rwg(p),grid)
# vec(ε⁻¹_dot(tc(H,mns),flat(εₛ⁻¹(0.75,rwg(p);ms)))) ≈ eisp * vec(tc(H,mns))
Mop = M̂_sp(ω,k,rwg(p),grid)
# ms.M̂ * H⃗[:,eigind] ≈ Mop * H⃗[:,eigind]
# ms.M̂ * ms.H⃗[:,eigind] ≈ Mop * ms.H⃗[:,eigind]
# @btime $Mop * $H⃗[:,eigind] # 1.225 ms (122 allocations: 4.01 MiB)
# @btime $ms.M̂ * $H⃗[:,eigind] # 4.734 ms (1535 allocations: 1.22 MiB)
Fdense = 𝓕_dense(grid)
# image(complex_to_rgb(Fdense))

Mkop = M̂ₖ_sp(ω,k,rwg(p),grid)
# Mkop * H⃗[:,eigind] ≈ vec(-kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mns), (2:3) ), real(flat(ε⁻¹))), (2:3)),mns,mag))
# @btime $Mkop * $H⃗[:,eigind] # 1.261 ms (122 allocations: 4.01 MiB)
# @btime vec(-kx_ct( ifft( ε⁻¹_dot( fft( zx_tc($H,$mns), (2:3) ), real(flat(ε⁻¹))), (2:3)),$mns,$mag)) # 2.095 ms (94 allocations: 4.01 MiB)

# nnginv = nngₛ⁻¹(ω,rwg(p),grid)
# real(dot(H⃗[:,eigind],Mkop,H⃗[:,eigind])) ≈ HMₖH(H,ε⁻¹,mag,m,n)
# real(dot(H⃗[:,eigind],Mkop,H⃗[:,eigind])) ≈ HMₖH(H,nnginv,mag,m,n)
# @btime real(dot($H⃗[:,eigind],$Mkop,$H⃗[:,eigind])) # 1.465 ms (134 allocations: 4.51 MiB)
# @btime HMₖH($H,$ε⁻¹,$mag,$m,$n) # 3.697 ms (122 allocations: 4.76 MiB)
#
# Zygote.gradient((om,kk,pp,HH)->real(dot(HH,M̂ₖ_sp(om,kk,rwg(pp),grid),HH)),ω,k,p,H⃗[:,eigind])
# Zygote.gradient((om,kk,pp,HH)->real(dot(HH,M̂ₖ_sp(om,kk,rwg(pp),grid)*HH)),ω,k,p,H⃗[:,eigind])

# ⟨H|Mₖ|H⟩

# real(dot(H⃗[:,eigind],M̂ₖ_sp(ω,k,rwg(p),grid)*H⃗[:,eigind]))

# Zygote.gradient((a,b)->sum(foo2(a,b)),mag,mns)
# Zygote.gradient((a,b)->sum(abs2.(foo2(a,b))),mag,mns)


##
fig = GLMakie.Figure()
@show H̄2_magmax = sqrt(maximum(abs2.(H̄2)))
@show H̄1_magmax = sqrt(maximum(abs2.(H̄1)))
H̄2_rel = H̄2 / H̄2_magmax
H̄1_rel = H̄1 / H̄1_magmax

axes_pb = fig[1,1:2] = [Axis(fig,title=t) for t in "|H̄_pb".*["1","2"].*"|²" ]
hms_pb = [GLMakie.heatmap!(axes_pb[axind],abs2.(fftshift(H̄2_rel[axind,:,:]))';colorrange=(0,1)) for axind=1:2]
cbar_pb = fig[1,3] = Colorbar(fig,hms_pb[1],label="relative mag. [1]")
cbar_pb.width = 30
axes_foo = fig[2,1:2] = [Axis(fig,title=t) for t in "|H̄_foo".*["1","2"].*"|²" ]
hms_foo = [GLMakie.heatmap!(axes_foo[axind],abs2.(fftshift(H̄1_rel[axind,:,:]))';colorrange=(0,1)) for axind=1:2]
cbar_foo = fig[2,3] = Colorbar(fig,hms_foo[1],label="relative mag. [1]")
cbar_foo.width = 30

axes = vcat(axes_pb,axes_foo) #,axes_Hi)
linkaxes!(axes...)
fig
##
# ω = 0.75
ω = 0.85
println("")
println(AD_style_N("∂²ω²∂k²_AD:"))
println("")
∂²ω²∂k²_AD = Zygote.gradient(om->(om / solve_n(om,rwg(p),grid)[2]),ω)[1]
println(AD_style("∂²ω²∂k²_AD= $∂²ω²∂k²_AD"))
println("")

println("")
println(FD_style_N("∂²ω²∂k²_FD:"))
println("")
∂²ω²∂k²_FD = FiniteDifferences.central_fdm(5,1)(om->(om / solve_n(om,rwg(p),grid)[2]),ω)
println(FD_style("∂²ω²∂k²_FD: $∂²ω²∂k²_FD"))
println("")

println("")
println(MAN_style_N("∂²ω²∂k²_MAN:"))
println("")
∂²ω²∂k²_MAN = ∂²ω²∂k²(ω,rwg(p),k,H⃗,grid) #om0^2,H⃗,k,rwg(p),gr)
println(MAN_style("∂²ω²∂k²_MAN: $∂²ω²∂k²_MAN"))
println("")

##

# ∂ei∂ω_RAD = Zygote.gradient(x->εₛ⁻¹(x,geom,grid),ω)
∂ei∂ω_FAD = copy(flat(ForwardDiff.derivative(x->εₛ⁻¹(x,geom,grid),ω)))
∂ei∂ω_FD = FiniteDifferences.central_fdm(5,1)(x->flat(εₛ⁻¹(x,geom,grid)),ω)
#∂ei∂ω_FD = copy(reinterpret(reshape,SMatrix{3,3,Float64,9},reshape(∂ei∂ω_FD_flat,9,128,128)))

nng = inv.(nnginv)
ε = inv.(ε⁻¹)
∂ε∂ω_man = (2/ω) * (nng .- ε)
∂ei∂ω_man = copy(flat(-(ε⁻¹.^2) .* ∂ε∂ω_man ))
# view(∂ei∂ω_FAD,1,1,:,:) - view(∂ei∂ω_FAD,1,1,:,:)
∂ei∂ω_FAD_man_err = abs.(∂ei∂ω_FAD .- ∂ei∂ω_man) ./ abs.(∂ei∂ω_FAD.+1e-10)
∂ei∂ω_FD_man_err = abs.(∂ei∂ω_FD .- ∂ei∂ω_man) ./ abs.(∂ei∂ω_FD.+1e-10)
maximum(∂ei∂ω_FD_man_err)


# nngₛ⁻¹(ω,geom,grid)
# ∂ei∂ω_RAD = Zygote.gradient(x->εₛ⁻¹(x,geom,grid),ω)
∂nngi∂ω_FAD = copy(flat(ForwardDiff.derivative(x->nngₛ⁻¹(x,geom,grid),ω)))
∂nngi∂ω_FD = FiniteDifferences.central_fdm(5,1)(x->flat(nngₛ⁻¹(x,geom,grid)),ω)



# check GVD formula works: GVD = ∂²k/∂ω² = -(λ² / 2π) * (∂ng/∂λ) = -(λ² / 2π) * D_λ
lm = 0.8
gvd_man(x) = - x^2  * Zygote.gradient(a->ng_MgO_LiNbO₃(a)[1,1],x)[1]
gvd_man(lm)
gvd_MgO_LiNbO₃(lm)[1,1] #/ (2π)

nng_LN(x) =  ng_MgO_LiNbO₃(x)[1,1] * sqrt(ε_MgO_LiNbO₃(x)[1,1])
function ∂nng∂ω_man_LN(om)
	 ng = ng_MgO_LiNbO₃(inv(om))[1,1]
	 n = sqrt(ε_MgO_LiNbO₃(inv(om))[1,1])
	 gvd = gvd_MgO_LiNbO₃(inv(om))[1,1]  #/ (2π)
	 # om = 1/om
	 om*(ng^2 - n*ng) + n * gvd
end
∂nng∂ω_FD_LN(x) = central_fdm(5,1)(a->nng_LN(inv(a)),x)
∂nng∂ω_RAD_LN(x) = Zygote.gradient(a->nng_LN(inv(a)),x)[1]

n1 = sqrt(ε_MgO_LiNbO₃(1.0)[1,1])
ng1 = ng_MgO_LiNbO₃(1.0)[1,1]
gvd1 = gvd_MgO_LiNbO₃(1.0)[1,1]

-(ng1^2 - n1*ng1) - n1 * gvd1

∂nng∂ω_man_LN(1.0)
∂nng∂ω_FD_LN(1.0)
∂nng∂ω_RAD_LN(1.0)

∂nng∂ω_man_LN(1/0.7)
∂nng∂ω_FD_LN(0.7)
∂nng∂ω_RAD_LN(0.7)


( ∂nng∂ω_RAD_LN(0.8) / ∂nng∂ω_man_LN(0.8) ) / (2π)

nng_LN(0.8)



ng² =  nng.^2 .* ε⁻¹
# = ( ng² .- nng ) / ω +
∂nngi∂ω_man = copy(flat(-(ε⁻¹.^2) .* ∂ε∂ω_man ))
# view(∂nngi∂ω_FAD,1,1,:,:) - view(∂nngi∂ω_FAD,1,1,:,:)
∂nngi∂ω_FAD_man_err = abs.(∂nngi∂ω_FAD .- ∂nngi∂ω_man) ./ abs.(∂nngi∂ω_FAD.+1e-10)
∂nngi∂ω_FD_man_err = abs.(∂nngi∂ω_FD .- ∂nngi∂ω_man) ./ abs.(∂nngi∂ω_FD.+1e-10)
maximum(∂nngi∂ω_FD_man_err)

##
fig = Figure()
diagind = 2
ax_FAD = fig[1, 1] = Axis(fig, title = "∂ε∂ω, Fwd AD")
ax_FD = fig[1, 2] = Axis(fig, title = "∂ε∂ω, FD")
ax_man = fig[1, 3] = Axis(fig, title = "∂ε∂ω, manual")
axes_∂ei∂ω = [ax_FAD,ax_FD,ax_man]
Zs_∂ei∂ω = [∂ei∂ω_FAD,∂ei∂ω_FD,∂ei∂ω_man]
hms_∂ei∂ω = [GLMakie.heatmap!(axes_∂ei∂ω[i], Zs_∂ei∂ω[i][diagind,diagind,:,:])  for i=1:3]
for hm in hms_∂ei∂ω
    hm.colorrange = extrema(Zs_∂ei∂ω[1][diagind,diagind,:,:])
end

cbar = fig[1, 4] = Colorbar(fig,hms_∂ei∂ω[1], label = "∂ε⁻¹/∂ω")
cbar.width = 30
fig



##

∂ω_²ω²∂k²_RAD1, ∂p_²ω²∂k²_RAD1 = Zygote.gradient((om,x)->∂²ω²∂k²(om,εₛ⁻¹(om,rwg(x),grid),nngₛ⁻¹(om,rwg(x),grid),k,H⃗,grid)[1],
	ω,p)







∂ω_²ω²∂k²_RAD2, ∂p_²ω²∂k²_RAD2 = Zygote.gradient((om,x)->∂²ω²∂k²(real(om),rwg(real(x)),k,H⃗,grid),ω,p)





∂ω_²ω²∂k²_FD, ∂p_²ω²∂k²_FD = FiniteDifferences.grad(central_fdm(9,1),
		(om,x)->∂²ω²∂k²(om,εₛ⁻¹(om,rwg(x),grid),nngₛ⁻¹(om,rwg(x),grid),k,H⃗,grid)[1],
		ω,
		p,
		)






eig_err = ms.M̂ * H⃗[:,1] - ( ω^2 * H⃗[:,1] )
sum(abs2,eig_err)
sum(abs2,H⃗[:,1])
ε⁻¹ = εₛ⁻¹(ω,rwg(p),grid)
nng⁻¹ = nngₛ⁻¹(ω,rwg(p),grid)
k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind=1)
lm = eig_adjt(ms.M̂,ω^2,H⃗[:,1],0.0,H̄)
adj_err = ( (ms.M̂ - (ω^2)*I) * lm ) - ( H̄ - H⃗[:,1] * dot(H⃗[:,1],H̄) )
sum(abs2,adj_err)

lm̄0 = randn(ComplexF64,size(H⃗,1))
lm̄ = lm̄0 ./ dot(lm̄0,lm̄0)
ξ⃗ = linsolve( (ms.M̂ - (ω^2)*I), lm̄ - H⃗[:,1] * dot(H⃗[:,1],lm̄) ; P̂=HelmholtzPreconditioner(ms.M̂) )
adj2_err = ( (ms.M̂ - (ω^2)*I) * ξ⃗ ) - (lm̄ - H⃗[:,1] * dot(H⃗[:,1],lm̄)) #( lm̄ - H⃗[:,1] * dot(H⃗[:,1],lm̄) )
sum(abs2,adj2_err)


sum(eig_adjt(ms.M̂,ω^2,H⃗[:,1],0.0,H̄))
Zygote.gradient(x->abs2(sum(eig_adjt(ms.M̂,x^2,H⃗[:,1],0.0,H̄))),ω)
FiniteDifferences.central_fdm(5,1)(x->abs2(sum(eig_adjt(ms.M̂,x^2,H⃗[:,1],0.0,H̄))),ω)

function foo1(k,p,ω,H⃗,grid)
	ε⁻¹ = εₛ⁻¹(ω,rwg(p),grid)
	nng⁻¹ = nngₛ⁻¹(ω,rwg(p),grid)
	M̂ = HelmholtzMap(real(k),real.(ε⁻¹),grid)
	k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind)
	lm = eig_adjt(
			M̂,								 # Â
			real(ω^2), 							# α
			H⃗[:,1], 					 # x⃗
			0.0, 							# ᾱ
			H̄ ;								 # x̄
			# λ⃗₀=nothing,
			P̂	= HelmholtzPreconditioner(M̂),
		)
	# lm2 = similar(H⃗)
	# solve_adj!(lm2,M̂,H̄,ω^2,H⃗,eigind)
	# println("")
	# println("magmax lm: $(maximum(abs2.(lm)))")
	# println("magmax lm2: $(maximum(abs2.(lm2)))")
	# println("out2: $(sum(abs2.(lm2).^2))")
	return abs2(sum(lm))
end

foo1(k,p,ω,H⃗,grid)
Zygote.gradient(foo1,k,p,ω,H⃗,grid)
Zygote.gradient((a,b,c)->foo1(a,b,c,H⃗,grid),k,p,ω)
Zygote.gradient((a,b,c)->foo1(a,b,c,H⃗,grid),k,p,ω)






FiniteDifferences.grad(central_fdm(9,1),(a,b,c)->foo1(a,b,c,H⃗,grid),k,p,ω)


ε⁻¹ = εₛ⁻¹(ω,rwg(p),grid)
nng⁻¹ = nngₛ⁻¹(ω,rwg(p),grid)
k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind)
function foo2(k,ω)
	M̂ = HelmholtzMap(k,ε⁻¹,grid)
	lm = eig_adjt(
			M̂,								 # Â
			real(ω^2), 							# α
			H⃗[:,1], 					 # x⃗
			0.0, 							# ᾱ
			H̄ ;								 # x̄
			# λ⃗₀=nothing,
			P̂	= HelmholtzPreconditioner(M̂),
		)
	return abs2(sum(lm))
end
foo2(k,ω)
Zygote.gradient(foo2,k,ω)
FiniteDifferences.grad(central_fdm(9,1),foo2,k,ω)


M̂ = HelmholtzMap(k,ε⁻¹,grid)
function foo3(k,ω)
	lm = eig_adjt(
			M̂,								 # Â
			real(ω^2), 							# α
			H⃗[:,1], 					 # x⃗
			0.0, 							# ᾱ
			H̄ ;								 # x̄
			# λ⃗₀=nothing,
			P̂	= HelmholtzPreconditioner(M̂),
		)
	return abs2(sum(lm))
end
foo3(k,ω)
Zygote.gradient(foo3,k,ω)
FiniteDifferences.grad(central_fdm(9,1),foo3,k,ω)




FiniteDifferences.grad(central_fdm(9,1),(a,b,c)->foo1(a,b,c,H⃗,grid),k,p,ω)


ε⁻¹ = εₛ⁻¹(ω,rwg(p),grid)
nng⁻¹ = nngₛ⁻¹(ω,rwg(p),grid)
M̂ = HelmholtzMap(k,ε⁻¹,grid)
k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind)
lm = eig_adjt(
		M̂,								 # Â
		ω^2, 							# α
		H⃗[:,1], 					 # x⃗
		0.1, 							# ᾱ
		H̄ )								 # x̄
		# λ⃗₀=nothing,
		# P̂	= HelmholtzPreconditioner(M̂),
	# )

lm2 = eig_adjt(
		M̂,								 # Â
		ω^2, 							# α
		H⃗[:,1], 					 # x⃗
		0.1, 							# ᾱ
		H̄ )

lm3 = eig_adjt(
		M̂,								 # Â
		ω^2, 							# α
		H⃗[:,1], 					 # x⃗
		0.1, 							# ᾱ
		H̄ )

lm4 = linsolve(
	M̂ + (-ω^2*I),
	H̄ - H⃗[:,1] * dot(H⃗[:,1],H̄))

lm5 = linsolve(
	M̂ + (-ω^2*I),
	H̄ - H⃗[:,1] * dot(H⃗[:,1],H̄))


lm2 ≈ lm3
lm4 ≈ lm5

using Zygote: @showgrad
btest = randn(ComplexF64,length(H⃗[:,1]))
function foo2(kk,pp,om)
	ε⁻¹ = εₛ⁻¹(om,rwg(pp),grid)
	M̂ = HelmholtzMap(kk,ε⁻¹,grid)
	# M̂ = HelmholtzMap(@showgrad(kk),@showgrad(ε⁻¹),grid)
	# Â = M̂  - om^2*I
	USM = UniformScalingMap(-(om^2),size(M̂,1))
	Â = M̂ + USM #- ω^2*I
	bt2 = btest - H⃗[:,1] * dot(H⃗[:,1],btest)
	lm = linsolve(
		Â,
		bt2,
	)
	# sum(sin.(abs2.(lm)))
	abs2(sum(lm))
end

foo2(k,p,ω)
k̄_foo2_RAD,p̄_foo2_RAD,om̄_foo2_RAD = Zygote.gradient(foo2,k,p,ω)
k̄_foo2_FD,p̄_foo2_FD,om̄_foo2_FD = FiniteDifferences.grad(central_fdm(9,1),foo2,k,p,ω)


foo2(k,p,ω)
k̄_foo2_RAD,p̄_foo2_RAD,om̄_foo2_RAD = Zygote.gradient(foo2,k,p,ω)
k̄_foo2_FD,p̄_foo2_FD,om̄_foo2_FD = FiniteDifferences.grad(central_fdm(9,1),foo2,k,p,ω)

abs(k̄_foo2_FD - k̄_foo2_RAD) / abs(k̄_foo2_FD)
abs.(p̄_foo2_FD .- p̄_foo2_RAD) ./ abs.(p̄_foo2_FD)
abs(om̄_foo2_FD - om̄_foo2_RAD) / abs(om̄_foo2_FD)

function foo3(kk,pp,om)
	ε⁻¹ = εₛ⁻¹(om,rwg(pp),grid)
	# M̂ = HelmholtzMap(@showgrad(kk),@showgrad(ε⁻¹),grid)
	M̂ = HelmholtzMap(kk,ε⁻¹,grid)
	k̄, H̄, nngī  = ∇HMₖH(k,H⃗,ε⁻¹,grid; eigind=1)
	USM = UniformScalingMap(-om^2,size(M̂,1))
	Â = Zygote.@showgrad(M̂) + USM #- ω^2*I
	lm = linsolve(
		Â,
		H̄ - H⃗[:,1] * dot(H⃗[:,1],H̄),
	)
	sum(sin.(abs2.(lm)))
end

foo3(k,p,ω)
k̄_foo3_RAD,p̄_foo3_RAD,om̄_foo3_RAD = Zygote.gradient(foo3,k,p,ω)


Zygote.gradient(x->sum(reshape(reinterpret(reshape,Float64,reshape(x,)),(3,3,size(x)...))),eic)
##
ω = 0.75
geom = rwg(p)
nng⁻¹, nnginv_pb = Zygote.pullback(nngₛ⁻¹,ω,geom,grid)
ε⁻¹, epsi_pb = Zygote.pullback(εₛ⁻¹,ω,geom,grid)
om̄₁, eī_herm, nngī_herm = ∂²ω²∂k²(ω,ε⁻¹,nng⁻¹,k,H⃗,grid)
om̄₂,geombar_Mₖ,grīd_Mₖ = nnginv_pb(nngī_herm) #nngī2)
om̄₃,geombar_H,grīd_H = epsi_pb(eī_herm) #eī₁)

∂²ω²∂k²_AD
om̄₁
om̄₂
om̄₃

om̄₁ + om̄₂ + om̄₃
om̄₂ + om̄₃
om̄₂ + 0.000663
∂²ω²∂k²_AD - om̄₁
∂²ω²∂k²_AD - ( om̄₁ + om̄₂ + om̄₃ )

##
Â = ms.M̂
α = real(ms.ω²[eigind])
X⃗ = H⃗[:,eigind]
ᾱ = 0
X̄ = Mₖᵀ_plus_Mₖ(H⃗[:,eigind],k,ε⁻¹,grid)
P̂ = HelmholtzPreconditioner(ms.M̂)
λ⃗ = eig_adjt(Â, α, X⃗, ᾱ, X̄)


λ⃗ = eig_adjt(Â, α, X⃗, ᾱ, X̄; P̂)

A = randn(10,10)
A = A + A'
b = randn(10)
x1 = linsolve(A,b)
@assert A * x1 ≈ b

Av = copy(vec(A))
sum(sin.(linsolve(A,b)))
∂A_RAD,∂b_RAD = Zygote.gradient((aa,bb)->sum(sin.(linsolve(aa,bb))),A,b)
∂A_FD,∂b_FD = FiniteDifferences.grad(central_fdm(7,1),(aa,bb)->sum(sin.(linsolve(aa,bb))),A,b)
∂A_err = abs.(∂A_FD .- Matrix(∂A_RAD)) ./ abs.(∂A_FD)
∂b_err = abs.(∂b_FD .- ∂b_RAD) ./ abs.(∂b_FD)

v1 = randn(10)
v2 = randn(10)
v1' * v2
v1 * v2'

outer(v1,v2) = v1 * v2'
delayed_outer = () -> outer(v1,v2)
delayed_outer

using IterativeSolvers
A * x1 - b
x2 = similar(b)

x3,ch = gmres(A,b;verbose=true,log=true,maxiter=1000)
x3,ch = bicgstabl(A,b;verbose=true,log=true,max_mv_products=1000)
A * x3 - b


##

om0 = 0.75
ω = om0
geom = rwg(p)
grid = gr
nnginv,nnginv_pb = Zygote.pullback(nngₛ⁻¹,ω,geom,grid)
epsi,epsi_pb = Zygote.pullback(εₛ⁻¹,ω,geom,grid)

∇HMₖH(k,H⃗,nnginv,grid)
sum(sum.(∇HMₖH(k,H⃗,nnginv,grid)[2]))
Zygote.gradient((a,b,c)->∇HMₖH(a,b,c,grid)[1],k,H⃗,nnginv)
# Zygote.gradient((a,b,c)->sum(∇HMₖH(a,b,c,grid)[2]),k,H⃗,nnginv)
Zygote.gradient((a,b,c)->sum(sum(∇HMₖH(a,b,c,grid)[2])),k,H⃗,nnginv)


d0 = randn(Complex{Float64}, (3,Ns...))
𝓕	 =	plan_fft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place FFT operator 𝓕
𝓕⁻¹ =	plan_bfft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place iFFT operator 𝓕⁻¹

using StaticArrays: Dynamic
m2 = HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,m⃗))
n2 = HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,n⃗))
mns1 = mapreduce(x->reshape(flat(x),(1,3,size(x)...)),vcat,(m⃗,n⃗))
mns2 = vcat(reshape(m2,(1,3,Ns...)),reshape(n2,(1,3,Ns...)))
mns1 ≈ mns2

Ninv = 1. / N(grid)
𝓕 * zx_tc( H * Ninv ,mns)
using Tullio
B̄₁ = 𝓕 * kx_tc( conj.(H) ,mns,mag)
B̄₂ = 𝓕 * zx_tc( H * Ninv ,mns)
@tullio B̄[a,b,i,j] := real(B̄₁[a,i,j] * B̄₂[b,i,j])/2 + real(B̄₁[b,i,j] * B̄₂[a,i,j])/2
Bv = reshape(B̄,3,3,128*128)

B̄₁ = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  kxtcsp	 *	vec(H),
	𝓕 * kx_tc( conj.(H) ,mns,mag),
	)
B̄₂ = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  zxtcsp	 *	vec(H),
	𝓕 * zx_tc( H * Ninv ,mns),
	)
B̄ 	= 	real.( B̄₁  .*  transpose.( B̄₂ ) )

B̄₂ = transpose.(reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  zxtcsp	 *	vec(H),
	𝓕 * zx_tc( H * Ninv ,mns),
	))
B̄ 	= 	real.( B̄₁  .*   B̄₂  )


B̄₁1 = reshape( 𝓕 * kx_tc( conj.(H) ,mns,mag), (3*128,128))
B̄₁2 = reinterpret(
	SVector{3,Complex{Float64}},
	B̄₁1
	)

B̄₂1 = reshape(𝓕 * zx_tc( H * Ninv ,mns), (3*128,128) )
B̄₂2 = reinterpret(
	SVector{3,Complex{Float64}},
	B̄₂1
	)
B̄2 	= 	Hermitian.( real.( B̄₁2  .*  transpose.( B̄₂2 ) ) )

Hsv
H
function foo3(x)
	# B̄₁ = reinterpret(
	# 	reshape,
	# 	SVector{3,Complex{Float64}},
	# 	# 𝓕  *  kxtcsp	 *	vec(H),
	# 	𝓕 * kx_tc( conj.(x) ,mns,mag),
	# 	)
	# B̄₂ = reinterpret(
	# 	reshape,
	# 	SVector{3,Complex{Float64}},
	# 	# 𝓕  *  zxtcsp	 *	vec(H),
	# 	𝓕 * zx_tc( x * Ninv ,mns),
	# 	)
	# B̄ 	= 	real.( B̄₁  .*  transpose.( B̄₂ ) )
	# B̄₁1 = reshape( 𝓕 * kx_tc( conj.(H) ,mns,mag), (3*128,128))
	# B̄₁1 = 𝓕 * kx_tc( conj.(x) ,mns,mag)
	# # B̄₁2 = reinterpret(
	# # 	SVector{3,Complex{Float64}},
	# # 	B̄₁1
	# # 	)
	# B̄₁2 = [ SVector{3,Complex{Float64}}(B̄₁1[1,i,j],B̄₁1[2,i,j],B̄₁1[3,i,j]) for i=1:128,j=1:128]
	#
	#
	# # B̄₂1 = reshape(𝓕 * zx_tc( H * Ninv ,mns), (3*128,128) )
	# B̄₂1 = 𝓕 * zx_tc( x * Ninv ,mns)
	# # B̄₂2 = reinterpret(
	# # 	SVector{3,Complex{Float64}},
	# # 	B̄₂1
	# # 	)
	# B̄₂2 = [ SVector{3,Complex{Float64}}(B̄₂1[1,i,j],B̄₂1[2,i,j],B̄₂1[3,i,j]) for i=1:128,j=1:128]
	# B̄ 	= 	Hermitian.( real.( B̄₁2  .*  transpose.( B̄₂2 ) ) )
	B̄₁4 = 𝓕 * kx_tc( conj.(x) ,mns,mag)
	B̄₂4 = 𝓕 * zx_tc( x * Ninv ,mns)
	@tullio B̄[a,b,i,j] := real(B̄₁4[a,i,j] * B̄₂4[b,i,j])
	# return B̄
	return reinterpret(SMatrix{3,3,Float64,9},reshape(B̄,9*128,128))
end

function foo4(A::AbstractArray{SMatrix{3,3,Float64,9}})
	sum(sum.(A))
end

Hc = copy(H)
foo3(Hc)
reinterpret(reshape,SMatrix{3,3,Float64,9},foo3(Hc))
foo4(copy(foo3(Hc)))
Zygote.gradient(x->foo4(foo3(x)),Hc)

B̄₁2 = vec( 𝓕 * kx_tc( conj.(H) ,mns,mag) )
B̄₂2 = vec( 𝓕 * zx_tc( H * Ninv ,mns) )

B̄₁3 = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  kxtcsp	 *	vec(H),
	𝓕 * kx_tc( conj.(H) ,mns,mag),
	) |> copy
B̄₂3 = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  zxtcsp	 *	vec(H),
	𝓕 * zx_tc( H * Ninv ,mns),
	) |> copy
B̄3 	= 	real.( B̄₁3  .*  transpose.( B̄₂3 ) )
using Tullio
B̄₁4 = 𝓕 * kx_tc( conj.(H) ,mns,mag)
B̄₂4 = 𝓕 * zx_tc( H * Ninv ,mns)
@tullio B̄4[a,b,i,j] := real(B̄₁4[a,i,j] * B̄₂4[b,i,j])
reinterpret(reshape,SMatrix{3,3,Float64,9},B̄4)
reinterpret(SMatrix{3,3,Float64,9},copy(reshape(B̄4,(9,128,128))))
Bb41 = reinterpret(reshape,SMatrix{3,3,Float64,9},copy(reshape(B̄4,(9,128,128))))
Bb42 = reinterpret(SMatrix{3,3,Float64,9},reshape(B̄4,(9*128,128)))
Bb41 ≈ Bb42
Bb43 = [SMatrix{3,3,Float64,9}(B̄4[:,:,i,j]) for i=1:128,j=1:128]
Bb41 ≈ Bb43
unflat(B̄4)
##
om0 = 0.75
M̂ = ms.M̂
kxtcsp 	= kx_tc_sp(k,gr)
zxtcsp 	= zx_tc_sp(k,gr)
eisp 	= ε⁻¹_sp(om0,rwg(p),gr)
nngsp 	= nng⁻¹_sp(om0,rwg(p),gr)
𝓕 = LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(fft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=true,ismutating=false)
𝓕⁻¹ = LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(ifft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=true,ismutating=false)
𝓕⁻¹b = LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(bfft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=true,ismutating=false)
Hsv = reinterpret(reshape, SVector{2,Complex{Float64}}, H )
A_sp 	=	-transpose(kxtcsp)
B_sp 	=	𝓕⁻¹b	*	nngsp	*	𝓕
C_sp	=	zxtcsp

zxtc_to_mn = SMatrix{3,3}(	[	0 	-1	  0
								1 	 0	  0
								0 	 0	  0	  ]	)

kxtc_to_mn = SMatrix{2,2}(	[	0 	-1
								1 	 0	  ]	)



Ā₁		=	conj.(Hsv)
Ā₂ = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# reshape(
	# 	𝓕⁻¹ * nngsp * 𝓕 * zxtcsp * vec(H),
	# 	(3,size(gr)...),
	# 	),
	M̂.𝓕⁻¹ * ε⁻¹_dot(  M̂.𝓕 * zx_tc(H * M̂.Ninv,mns) , real(flat(nnginv))),
	)
Ā 	= 	Ā₁  .*  transpose.( Ā₂ )
m̄n̄_Ā = transpose.( (kxtc_to_mn,) .* real.(Ā) )
m̄_Ā = 		view.( m̄n̄_Ā, (1:3,), (1,) )
n̄_Ā = 		view.( m̄n̄_Ā, (1:3,), (2,) )
māg_Ā = dot.(n⃗, n̄_Ā) + dot.(m⃗, m̄_Ā)
k̄_Mₖ_Ā = mag_m_n_pb( ( māg_Ā, m̄_Ā.*mag, n̄_Ā.*mag ) )[1]

B̄₁ = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  kxtcsp	 *	vec(H),
	M̂.𝓕 * kx_tc( conj.(H) ,mns,mag),
	)
B̄₂ = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  zxtcsp	 *	vec(H),
	M̂.𝓕 * zx_tc( H * M̂.Ninv ,mns),
	)
B̄ 	= 	real.( B̄₁  .*  transpose.( B̄₂ ) )


C̄₁ = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# reshape(
	# 	𝓕⁻¹ * nngsp * 𝓕 * kxtcsp * -vec(H),
	# 	(3,size(gr)...),
	# 	),
	M̂.𝓕⁻¹ * ε⁻¹_dot(  M̂.𝓕 * -kx_tc(H* M̂.Ninv,mns,mag) , real(flat(nnginv))),
	)
C̄₂ =   conj.(Hsv)
C̄ 	= 	C̄₁  .*  transpose.( C̄₂ )
m̄n̄_C̄ = 			 (zxtc_to_mn,) .* real.(C̄)
m̄_C̄ = 		view.( m̄n̄_C̄, (1:3,), (1,) )
n̄_C̄ = 		view.( m̄n̄_C̄, (1:3,), (2,) )
k̄_Mₖ_C̄ = mag_m_n_pb( ( nothing, m̄_C̄, n̄_C̄ ) )[1]

nngī_Mₖ = ( B̄ .+ transpose.(B̄) ) ./ 2
nngī_Mₖ_magmax = maximum(abs.(flat(nngī_Mₖ)))
k̄_Mₖ = k̄_Mₖ_Ā + k̄_Mₖ_C̄

println("")
println("magmax(nngī_Mₖ) = $(nngī_Mₖ_magmax)")
println("k̄_Mₖ = $k̄_Mₖ")

# @btime begin
# 	C̄ = 	reinterpret(reshape, SVector{3,Complex{Float64}}, reshape( 𝓕⁻¹ * nngsp * 𝓕 * kxtcsp * -vec(H), (3,size(gr)...)) )  .*  transpose.( conj.(Hsv) )
# 	m̄n̄_C̄ = 			 (zxtc_to_mn,) .* real.(C̄)
# 	m̄_C̄ = 		view.( m̄n̄_C̄, (1:3,), (1,) )
# 	n̄_C̄ = 		view.( m̄n̄_C̄, (1:3,), (2,) )
# 	k̄_Mₖ_C̄ = mag_m_n_pb( ( nothing, m̄_C̄, n̄_C̄ ) )[1]
#
# 	Ā = 	conj.(Hsv)   .*  transpose.( reinterpret(reshape, SVector{3,Complex{Float64}}, reshape( 𝓕⁻¹ * nngsp * 𝓕 * zxtcsp * vec(H), (3,size(gr)...)) ) )
# 	m̄n̄_Ā = transpose.( (kxtc_to_mn,) .* real.(Ā) )
# 	m̄_Ā = 		view.( m̄n̄_Ā, (1:3,), (1,) )
# 	n̄_Ā = 		view.( m̄n̄_Ā, (1:3,), (2,) )
# 	māg_Ā = dot.(n⃗, n̄_Ā) + dot.(m⃗, m̄_Ā)
# end
# 2.022 s (1683353 allocations: 4.09 GiB)
# @btime mag_m_n_pb( ( māg_Ā, m̄_Ā.*mag, n̄_Ā.*mag ) )
# 1.932 s (1650232 allocations: 4.06 GiB)
##
g⃗s = collect(g⃗(gr))
(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(x->mag_m_n(x,g⃗s),k)
m = M̂.m
n = M̂.n
# HMₖH, HMₖH_pb = Zygote.pullback(HMₖH,H,ε⁻¹,mag,m,n)
HMₖH, HMₖH_pb = Zygote.pullback(HMₖH,H,nnginv,mag,m,n)
# @btime HMₖH_pb(1) # 4.553 ms (237 allocations: 15.89 MiB)
H̄2, eī2, māg2,m̄2,n̄2 = HMₖH_pb(1)
m̄v2 = copy(reinterpret(reshape,SVector{3,Float64},real(m̄2)))
n̄v2 = copy(reinterpret(reshape,SVector{3,Float64},real(n̄2)))
k̄_Mₖ_AD = mag_m_n_pb( (real(māg2), m̄v2, n̄v2) )[1]

nngī_Mₖ_AD_magmax = maximum(abs.(flat(eī2)))
println("magmax(nngī_Mₖ_AD) = $(nngī_Mₖ_AD_magmax)")
println("magmax(nngī_Mₖ)_err = $( abs( nngī_Mₖ_magmax - nngī_Mₖ_AD_magmax ) / abs(nngī_Mₖ_AD_magmax) )")

println("k̄_Mₖ_AD = $k̄_Mₖ_AD")
println("k̄_Mₖ_err = $( abs( k̄_Mₖ - k̄_Mₖ_AD ) / abs(k̄_Mₖ_AD) )")
##

māg_Ā
k̄_Mₖ_Ā
mag
māg_A_man = (k̄_Mₖ_Ā / k) .* mag
k
ẑ = SVector(0,0,1)
k⃗ = SVector(0,0,k)
kp⃗g = (k⃗,) .- g⃗s
kp̂g = kp⃗g ./ mag
kp⃗gxz = cross.(kp⃗g,(ẑ,))
kp̂gxz = cross.(kp̂g,(ẑ,))
mxkp⃗gxz = cross.(m⃗,kp⃗gxz)
nxkp⃗gxz = cross.(n⃗,kp⃗gxz)
mxkp̂gxz = cross.(m⃗,kp̂gxz)
nxkp̂gxz = cross.(n⃗,kp̂gxz)


using Zygote: Buffer, dropgrad
function mag_m_n3(k⃗::SVector{3,T},grid::Grid) where T <: Real
	local ẑ = SVector(0.,0.,1.)
	local ŷ = SVector(0.,1.,0.)
	g⃗s = g⃗(dropgrad(grid))
	n = Buffer(g⃗s,size(g⃗s))
	m = Buffer(g⃗s,size(g⃗s))
	mag = Buffer(zeros(T,size(g⃗s)),size(g⃗s))
	@fastmath @inbounds for i ∈ eachindex(g⃗s)
		@inbounds kpg::SVector{3,T} = k⃗ - g⃗s[i]
		@inbounds mag[i] = norm(kpg)
		@inbounds n[i] =   ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : ŷ
		@inbounds m[i] =  normalize( cross( n[i], kpg )  )
	end
	return copy(mag), copy(m), copy(n)
end

mag3,m3,n3 = mag_m_n3(k⃗,gr)

(mag, m⃗, n⃗), mag_m_n_pb5 = Zygote.pullback(x->mag_m_n(x,g⃗s),k)
mag_m_n4(k⃗,g⃗s) .≈ mag_m_n(k⃗,g⃗s)
(mag4,m4,n4), mag_m_n4_pb = pullback(x->mag_m_n4(SVector(0.,0.,x),g⃗s),k)
(mag4,m4,n4), mag_m_n4_pb = Zygote.pullback(mag_m_n4,k⃗,g⃗s)
mag_m_n_pb((māg_Ā,mag.*m̄_Ā,n̄_Ā))[1]
mag_m_n4_pb((māg_Ā,mag.*m̄_Ā,n̄_Ā))[1]

ΔmagmnA = (māg_Ā,mag.*m̄_Ā,n̄_Ā)
@btime mag_m_n_pb5($ΔmagmnA)[1]
@btime mag_m_n4_pb($ΔmagmnA)[1]
@btime mag_m_n($k⃗,$g⃗s)
∇ₖmag_m_n((māg_Ā,mag.*m̄_Ā,n̄_Ā),(mag,m⃗,n⃗);dk̂=SVector(0.,0.,1.))
∇ₖmag_m_n(māg_Ā,mag.*m̄_Ā,n̄_Ā,mag,m⃗,n⃗;dk̂=SVector(0.,0.,1.))

Zygote.gradient(∇ₖmag_m_n,māg_Ā,mag.*m̄_Ā,n̄_Ā,mag,m⃗,n⃗)


mag3 ≈ mag
m3 ≈ m⃗
n3 ≈ n⃗

k̄_Ā_mag_man = dot(vec(māg_Ā),inv.(vec(mag))) * k
k̄_Ā_m_man = sum( dot.( m̄_Ā .* mag , cross.(m⃗, cross.(kp⃗g, (ẑ,) ) ) ./ mag.^2 ) )
k̄_Ā_n_man = sum( dot.( n̄_Ā , cross.(n⃗, cross.(kp⃗g, (ẑ,) ) ) ./ mag.^2 ) )
k̄_Ā_man = k̄_Ā_mag_man + k̄_Ā_m_man + k̄_Ā_n_man
k̄_Ā_man / k̄_Mₖ_Ā

function ∇ₖmag_m_n(māg,m̄,n̄,mag,m⃗,n⃗;dk̂=ẑ)
	kp̂g_over_mag = cross.(m⃗,n⃗)./mag
	k̄_mag = sum( māg .* dot.( kp̂g_over_mag, (dk̂,) ) .* mag )
	k̄_m = -sum( dot.( m̄ , cross.(m⃗, cross.( kp̂g_over_mag, (dk̂,) ) ) ) )
	k̄_n = -sum( dot.( n̄ , cross.(n⃗, cross.( kp̂g_over_mag, (dk̂,) ) ) ) )
	return +( k̄_mag, k̄_m, k̄_n )
end

kp⃗g1 = fill(k⃗,size(gr)...) - g⃗(dropgrad(gr))
kp⃗g2 = cross.(m⃗,n⃗).*mag

kp⃗g1 ≈ kp⃗g2
kp⃗g ≈ kp⃗g2

∇ₖmag_m_n(māg_Ā,mag.*m̄_Ā,n̄_Ā,mag,m⃗,n⃗;dk̂=ẑ)

∇ₖmag_m_n(māg_Ā,mag.*m̄_Ā,n̄_Ā,mag,m⃗,n⃗;dk̂=ẑ)

g⃗s = g⃗(dropgrad(grid))


foo1(x) = sum(sin.(x))
foo1_mag, foo1_mag_pb = Zygote.pullback(foo1,mag)
māg_foo1 = foo1_mag_pb(1)[1]
m̄v_foo1 = nothing
n̄v_foo1 = nothing
k̄_foo1 = mag_m_n_pb((māg_foo1,m̄v_foo1,n̄v_foo1))[1]
k̄_foo1_man = dot(vec(māg_foo1),inv.(vec(mag))) * k
k̄_foo1 / k̄_foo1_man

foo2(x) = sum(sin.(vec(flat(x))))
foo2_m, foo2_m_pb = Zygote.pullback(foo2,m⃗)
m̄_foo2 = foo2_m_pb(1)[1]
māg_foo2 = nothing
n̄_foo2 = nothing
k̄_foo2_m = mag_m_n_pb((māg_foo2,m̄_foo2,n̄_foo2))[1]
k̄_foo2_m_man = sum( dot.( m̄_foo2 , cross.(m⃗, cross.(kp⃗g, (ẑ,) ) ) ./ mag.^2 ) )

foo2_n, foo2_n_pb = Zygote.pullback(foo2,n⃗)
n̄_foo2_n = foo2_n_pb(1)[1]
k̄_foo2_n = mag_m_n_pb((nothing,nothing,n̄_foo2_n))[1]
k̄_foo2_n_man = sum( dot.( n̄_foo2_n , cross.(n⃗, cross.(kp⃗g, (ẑ,) ) ) ./ mag.^2 ) )



abs.(flat(kp⃗gxz)) |> maximum
abs.(flat(mxkp⃗gxz)) |> maximum
abs.(flat(nxkp⃗gxz)) |> maximum

abs.(flat(kp̂gxz)) |> maximum
abs.(flat(mxkp̂gxz)) |> maximum
abs.(flat(nxkp̂gxz)) |> maximum
abs.(flat(n⃗)) |> maximum

flat(dm3) ./ flat(mxkp̂gxz)
( flat(mxkp̂gxz ./ mag )  ) ./ flat(dm3)
( flat(mxkp⃗gxz ./ mag.^2 )  ) ./ flat(dm3)

# ( flat(mxkp⃗gxz )  ) ./ flat(dm3)

kp̂gxz ≈ n⃗
kp⃗gxz ≈ -n⃗
# k̄_foo1_man = dot(vec(māg_foo1),inv.(vec(mag))) * k
# k̄_foo1 / k̄_foo1_man

function dmagmn_dk_FD(k0,dk)
	mag0,m0,n0 = mag_m_n(k0-dk/2,g⃗s)
	mag1,m1,n1 = mag_m_n(k0+dk/2,g⃗s)
	dmag = ( mag1 .- mag0 ) ./ dk
	dm = ( m1 .- m0 ) ./ dk
	dn = ( n1 .- n0 ) ./ dk
	return dmag, dm, dn
end

dmag1,dm1,dn1 = dmagmn_dk_FD(k,1e-3)
dmag2,dm2,dn2 = dmagmn_dk_FD(k,1e-5)
dmag3,dm3,dn3 = dmagmn_dk_FD(k,1e-7)

dmag3 ≈ dmag2
dmag3

##
m̄2r = real(m̄2)
n̄2r = real(n̄2)
m̄f = copy(flat(SVector{3}.(m̄)))
m̄mf = copy(flat(SVector{3}.(m̄).*mag))
n̄f = copy(flat(SVector{3}.(n̄)))
n̄mf = copy(flat(SVector{3}.(n̄).*mag))

m̄2r ./ m̄f
m̄2r ./ m̄mf
n̄2r ./ n̄f
n̄2r ./ n̄mf

@show maximum(abs.(m̄2r))
@show maximum(abs.(m̄f))
@show maximum(abs.(m̄mf))
@show maximum(abs.(n̄2r))
@show maximum(abs.(n̄f))
@show maximum(abs.(n̄mf))



##

## single ω solve_n gradient checks, ms created within solve_n
function gradtest_solve_n_sweep(ω0;om_grads=false)
        println("...............................................................")
        println("ω sweep solve_n gradient checks, ms created within solve_n: ")
        @show ω0
        neff1,ng1 = solve_n(ω0.+rand()*0.1,rwg(p),gr)
        neff2,ng2 = solve_n(ω0.+rand()*0.1,rwg2(p),gr)

        if om_grads
                println("∂n_om, non-dispersive materials:")
                om = ω0.+rand()*0.1
                println("\t∂n_om (Zygote):")
                ∂n_om_RAD = Zygote.gradient(x->sum(solve_n(x,rwg2(p),gr)[1]),om)[1]
                println("\t$∂n_om_RAD")
                # solve_n(om+rand()*0.2,rwg(p),gr)
                println("\t∂n_om (FD):")
                ∂n_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(x,rwg2(p),gr)[1]),om)[1]
                println("\t$∂n_om_FD")
                @show ∂n_om_err = abs.(∂n_om_RAD .- ∂n_om_FD) ./ abs.(∂n_om_FD)

                println("∂ng_om, non-dispersive materials:")
                om = ω0.+rand()*0.1
                println("\t∂ng_om (Zygote):")
                ∂ng_om_RAD = Zygote.gradient(x->sum(solve_n(x,rwg2(p),gr)[2]),om)[1]
                println("\t$∂ng_om_RAD")
                # solve_n(om+rand()*0.2,rwg(p),gr)
                println("\t∂ng_om (FD):")
                ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(x,rwg2(p),gr)[2]),om)[1]
                println("\t$∂ng_om_FD")
                @show ∂ng_om_err = abs.( ∂ng_om_RAD .-  ∂ng_om_FD) ./  abs.(∂ng_om_FD)

                println("∂ng_om, dispersive materials:")
                om = ω0.+rand()*0.1
                println("\t∂ng_om (Zygote):")
                ∂ng_om_RAD = Zygote.gradient(x->sum(solve_n(x,rwg(p),gr)[2]),om)[1]
                println("\t$∂ng_om_RAD")
                # solve_n(om+rand()*0.2,rwg(p),gr)
                println("\t∂ng_om (FD):")
                ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(x,rwg(p),gr)[2]),om)[1]
                println("\t$∂ng_om_FD")
                @show ∂ng_om_err = abs.( ∂ng_om_RAD .-  ∂ng_om_FD) ./  abs.(∂ng_om_FD)
        end

        println("∂n_p, non-dispersive materials:")
        om = ω0.+rand()*0.1
        println("\t∂n_p (Zygote):")
        ∂n_p_RAD =  Zygote.gradient(x->sum(solve_n(om,rwg2(x),gr)[1]),p)[1]
        println("\t$∂n_p_RAD")
        # solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂n_p (FD):")
        ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(om,rwg2(x),gr)[1]),p)[1]
        println("\t$∂n_p_FD")
        @show ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)

        println("∂n_p, dispersive materials:")
        om = ω0.+rand()*0.1
        println("\t∂n_p (Zygote):")
        ∂n_p_RAD =  Zygote.gradient(x->sum(solve_n(om,rwg(x),gr)[1]),p)[1]
        println("\t$∂n_p_RAD")
        # solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂n_p (FD):")
        ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(om,rwg(x),gr)[1]),p)[1]
        println("\t$∂n_p_FD")
        @show ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)

        println("∂ng_p, non-dispersive materials:")
        om = ω0.+rand()*0.1
        println("\t∂ng_p (Zygote):")
        ∂ng_p_RAD = Zygote.gradient(x->sum(solve_n(om,rwg2(x),gr)[2]),p)[1]
        println("\t$∂ng_p_RAD")
        # solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂ng_p (FD):")
        ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(om,rwg2(x),gr)[2]),p)[1]
        println("\t$∂ng_p_FD")
        @show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD

        println("∂ng_p, dispersive materials:")
        om = ω0.+rand()*0.1
        println("\t∂ng_p (Zygote):")
        ∂ng_p_RAD = Zygote.gradient(x->sum(solve_n(om,rwg(x),gr)[2]),p)[1]
        println("\t$∂ng_p_RAD")
        # solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂ng_p (FD):")
        ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(om,rwg(x),gr)[2]),p)[1]
        println("\t$∂ng_p_FD")
        @show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD
        println("...............................................................")
end

gradtest_solve_n_sweep([0.65, 0.75, 0.85])
gradtest_solve_n_sweep(collect(0.55:0.03:0.85))

##
ns,ngs = solve_n(ms,ωs,rwg(p))

##

@show ∂sumng_RAD = Zygote.gradient(x->sum(solve_n(ms,[0.6,0.7],rwg(x))[2]),p)[1]




@show ∂sumng_FD = FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(ms,[0.6,0.7],rwg(x))[2]),p)[1]
@show ∂sumng_err = abs.(∂sumng_RAD[2] .- ∂sumng_FD) ./ abs.(∂sumng_FD)


##
λs = 1 ./ ωs
fig,ax,sc1 = scatter(λs,ng1,color=logocolors[:red])
lines!(ax,λs,ng1,color=logocolors[:red],lw=2)
lines!(ax,λs,n1,color=logocolors[:blue],lw=2)
scatter!(ax,λs,n1,color=logocolors[:blue])
fig
##

solve_n(ms,ωs,rwg(p))
solve_n(ωs,rwg(p),gr)

function var_ng(ωs,p)
    ngs = solve_n(ωs,rwg(p),gr)[2]
    # mean(  ngs.^2  ) - mean(ngs)^2
    var(real(ngs))
end

var_ng(ωs,p)
@show ∂vng_RAD = Zygote.gradient(var_ng,ωs,p)
@show ∂vng_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng(ωs,x),p)[1]
@show ∂vng_err = abs.(∂vng_RAD[2] .- ∂vng_FD) ./ abs.(∂vng_FD)

ωs = collect(0.55:0.03:0.85)
@show ∂vng_RAD = Zygote.gradient(var_ng,ωs,p)
@show ∂vng_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng(ωs,x),p)[1]
@show ∂vng_err = abs.(∂vng_RAD[2] .- ∂vng_FD) ./ abs.(∂vng_FD)


@show ∂sumng_RAD = Zygote.gradient(x->sum([solve_n(ms,om,rwg(x))[2] for om in [0.6,0.7] ]),p)[1]
@show ∂sumng_FD = FiniteDifferences.grad(central_fdm(3,1),x->sum([solve_n(ms,om,rwg(x))[2] for om in [0.6,0.7] ]),p)[1]
@show ∂sumng_err = abs.(∂sumng_RAD[2] .- ∂sumng_FD) ./ abs.(∂sumng_FD)

# @time ∂vng_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng(ωs,x),p)
#
# Zygote.gradient((x,y)->solve_n(ms,x,rwg(y))[2],1/0.85,p)
# Zygote.gradient(ωs,p) do oms,x
# 	ngs = solve_n(Zygote.dropgrad(ms),oms,rwg(x))[2]
#     mean( abs2.( ngs ) ) - abs2(mean(ngs))
# end

## Define with constant indices


##
fig,ax,sc1 = scatter(λs,ng2,color=logocolors[:red])
lines!(ax,λs,ng2,color=logocolors[:red],lw=2)
lines!(ax,λs,n2,color=logocolors[:blue],lw=2)
scatter!(ax,λs,n2,color=logocolors[:blue])
fig
##
function var_ng2(ωs,p)
    ngs = solve_n(Zygote.dropgrad(ms),ωs,rwg2(p))[2]
    mean( abs2.( ngs ) ) - abs2(mean(ngs))
end
var_ng2(ωs,p)

@show ∂vng2_RAD = Zygote.gradient(var_ng2,ωs,p)
@show ∂vng2_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng2(ωs,x),p)[1]
@show ∂vng2_err = abs.(∂vng2_RAD[2] .- ∂vng2_FD) ./ abs.(∂vng2_FD)


var_ng(ωs,p)
@show ∂vng_RAD = Zygote.gradient(var_ng,ωs,p)
@show ∂vng_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng(ωs,x),p)[1]
@show ∂vng_err = abs.(∂vng_RAD[2] .- ∂vng_FD) ./ abs.(∂vng_FD)

##

∂n_RAD = zeros(length(ωs),3)
∂n_FD = zeros(length(ωs),3)
∂n_err = zeros(length(ωs),3)

∂ng_RAD = zeros(length(ωs),3)
∂ng_FD = zeros(length(ωs),3)
∂ng_err = zeros(length(ωs),3)

for omind in 1:length(ωs)
    ∂n_RAD[omind,:] = Zygote.gradient(x->solve_n(ms,ωs[omind],rwg2(x))[1],p)[1]
    ∂ng_RAD[omind,:] = Zygote.gradient(x->solve_n(ms,ωs[omind],rwg2(x))[2],p)[1]
    ∂n_FD[omind,:] = FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,ωs[omind],rwg2(x))[1],p)[1]
    ∂ng_FD[omind,:] = FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,ωs[omind],rwg2(x))[2],p)[1]
end
∂n_err = abs.(∂n_RAD .- ∂n_FD) ./ abs.(∂n_FD)
∂ng_err = abs.(∂ng_RAD .- ∂ng_FD) ./ abs.(∂ng_FD)

##
ln = lines(collect(λs),∂n_err[:,1],color=logocolors[:green])
lines!(collect(λs),∂n_err[:,2],color=logocolors[:blue])
lines!(collect(λs),∂n_err[:,3],color=logocolors[:red])

lng = lines(collect(λs),∂ng_err[:,1],color=logocolors[:green])
lines!(collect(λs),∂ng_err[:,2],color=logocolors[:blue])
lines!(collect(λs),∂ng_err[:,3],color=logocolors[:red])

##
ei2 = εₛ⁻¹(1/1.55,rwg2(p);ms)
Zygote.gradient((x,y)->sum(sum(εₛ⁻¹(x,rwg2(y);ms))),1/1.55,p)



@time ∂sumei_FD = FiniteDifferences.grad(central_fdm(3,1),x->sum(sum(εₛ⁻¹(1/1.55,rwg2(x);ms))),p)




Zygote.gradient((x,y)->solve_n(ms,x,rwg2(y))[1],1/1.55,p)




@time ∂n2_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,1/1.55,rwg2(x))[1],p)





Zygote.gradient((x,y)->solve_n(ms,x,rwg2(y))[2],1/1.55,p)




@time ∂ng2_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,1/1.55,rwg2(x))[2],p)





omsq2,H2 = solve_ω²(ms,1.45,rwg2(p))
summag4(HH) = sum(abs2.(HH).^2)

@show ∂omsq_k_RAD = Zygote.gradient(x->solve_ω²(ms,x,rwg2(p))[1],1.45)[1]
@show ∂omsq_k_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_ω²(ms,x,rwg2(p))[1],1.45)[1]
@show ∂omsq_k_err = abs(∂omsq_k_RAD - ∂omsq_k_FD) / ∂omsq_k_FD

@show ∂sm4_k_RAD = Zygote.gradient(x->summag4(solve_ω²(ms,x,rwg2(p))[2]),1.45)[1]
@show ∂sm4_k_FD =  FiniteDifferences.grad(central_fdm(3,1),x->summag4(solve_ω²(ms,x,rwg2(p))[2]),1.45)[1]
@show ∂omsq_k_err = abs( ∂sm4_k_RAD -  ∂sm4_k_FD) /  ∂sm4_k_FD

@show ∂omsq_p_RAD =  Zygote.gradient(x->solve_ω²(ms,1.45,rwg2(x))[1],p)[1]
@show ∂omsq_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_ω²(ms,1.45,rwg2(x))[1],p)[1]
@show ∂omsq_p_err = abs.(∂omsq_p_RAD .- ∂omsq_p_FD) ./ ∂omsq_p_FD

# @show ∂omsq_p_RAD =  Zygote.gradient(x->solve_ω²(ms,1.45,rwg(x))[1],p)[1]
# @show ∂omsq_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_ω²(ms,1.45,rwg(x))[1],p)[1]
# @show ∂omsq_p_err = abs.(∂omsq_p_RAD .- ∂omsq_p_FD) ./ ∂omsq_p_FD

@show ∂sm4_p_RAD = Zygote.gradient(x->summag4(solve_ω²(ms,1.45,rwg2(x))[2]),p)[1]
@show ∂sm4_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->summag4(solve_ω²(ms,1.45,rwg2(x))[2]),p)[1]
@show ∂sm4_p_err = abs.(∂sm4_p_RAD .- ∂sm4_p_FD) ./ ∂sm4_p_FD

k2,H22 = solve_k(ms,0.7,rwg2(p))

@show ∂k_om_RAD = Zygote.gradient(x->solve_k(ms,x,rwg2(p))[1],0.7)[1]
@show ∂k_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,x,rwg2(p))[1],0.7)[1]
@show ∂k_om_err = abs(∂k_om_RAD - ∂k_om_FD) / abs(∂k_om_FD)

@show ∂sm4_om_RAD = Zygote.gradient(x->summag4(solve_k(ms,x,rwg2(p))[2]),0.7)[1]
@show ∂sm4_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->summag4(solve_k(ms,x,rwg2(p))[2]),0.7)[1]
@show ∂sm4_om_err = abs( ∂sm4_om_RAD -  ∂sm4_om_FD) /  abs(∂sm4_om_FD)

@show ∂k_p_RAD =  Zygote.gradient(x->solve_k(ms,0.7,rwg2(x))[1],p)[1]
@show ∂k_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,0.7,rwg2(x))[1],p)[1]
@show ∂k_p_err = abs.(∂k_p_RAD .- ∂k_p_FD) ./ abs.(∂k_p_FD)

@show ∂k_p_RAD =  Zygote.gradient(x->solve_k(ms,0.7,rwg(x))[1],p)[1]
@show ∂k_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,0.7,rwg(x))[1],p)[1]
@show ∂k_p_err = abs.(∂k_p_RAD .- ∂k_p_FD) ./ abs.(∂k_p_FD)

@show ∂sm4k_p_RAD = Zygote.gradient(x->summag4(solve_k(ms,0.7,rwg2(x))[2]),p)[1]
@show ∂sm4k_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->summag4(solve_k(ms,0.7,rwg2(x))[2]),p)[1]
@show ∂sm4k_p_err = abs.(∂sm4k_p_RAD .- ∂sm4k_p_FD) ./ ∂sm4k_p_FD

@show ∂sm4k_p_RAD = Zygote.gradient(x->summag4(solve_k(ms,0.7,rwg(x))[2]),p)[1]
@show ∂sm4k_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->summag4(solve_k(ms,0.7,rwg(x))[2]),p)[1]
@show ∂sm4k_p_err = abs.(∂sm4k_p_RAD .- ∂sm4k_p_FD) ./ ∂sm4k_p_FD

@show ∂sm4k_p_RAD = Zygote.gradient(x->summag4(solve_k(ms,0.7,rwg(x))[2]),p)[1]
@show ∂sm4k_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->summag4(solve_k(ms,0.7,rwg(x))[2]),p)[1]
@show ∂sm4k_p_err = abs.(∂sm4k_p_RAD .- ∂sm4k_p_FD) ./ ∂sm4k_p_FD




##
neff1,ng1 = solve_n(ms,om,rwg(p))
neff2,ng2 = solve_n(ms,0.7,rwg2(p))

@show ∂n_om_RAD = Zygote.gradient(x->solve_n(ms,x,rwg2(p))[1],0.7)[1]
@show ∂n_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,x,rwg2(p))[1],0.7)[1]
@show ∂n_om_err = abs(∂n_om_RAD - ∂n_om_FD) / abs(∂n_om_FD)

@show ∂ng_om_RAD = Zygote.gradient(x->solve_n(ms,x,rwg2(p))[2],0.7)[1]
@show ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,x,rwg2(p))[2],0.7)[1]
@show ∂ng_om_err = abs( ∂ng_om_RAD -  ∂ng_om_FD) /  abs(∂ng_om_FD)

@show ∂ng_om_RAD = Zygote.gradient(x->solve_n(ms,x,rwg(p))[2],0.7)[1]
@show ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(5,1),x->solve_n(ms,x,rwg(p))[2],0.7)[1]
@show ∂ng_om_err = abs( ∂ng_om_RAD -  ∂ng_om_FD) /  abs(∂ng_om_FD)

@show ∂n_p_RAD =  Zygote.gradient(x->solve_n(ms,0.7,rwg2(x))[1],p)[1]
@show ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,0.7,rwg2(x))[1],p)[1]
@show ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)

@show ∂n_p_RAD =  Zygote.gradient(x->solve_n(ms,0.7,rwg(x))[1],p)[1]
@show ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,0.7,rwg(x))[1],p)[1]
@show ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)

@show ∂ng_p_RAD = Zygote.gradient(x->solve_n(ms,0.7,rwg2(x))[2],p)[1]
@show ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->solve_n(ms,0.7,rwg2(x))[2],p)[1]
@show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD

@show ∂ng_p_RAD = Zygote.gradient(x->solve_n(ms,0.7,rwg(x))[2],p)[1]
@show ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->solve_n(ms,0.7,rwg(x))[2],p)[1]
@show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD


f1(om,pp) = sum(sum(εₛ⁻¹(om,rwg(pp);ms)))
f1(0.7,p)
@show ∂f1_om_RAD = Zygote.gradient(x->f1(x,p),0.7)[1]
@show ∂f1_om_FD =  FiniteDifferences.grad(central_fdm(5,1),x->f1(x,p),0.7)[1]
@show ∂f1_om_err = abs( ∂f1_om_RAD -  ∂f1_om_FD) /  abs(∂f1_om_FD)

@show ∂f1_p_RAD =  Zygote.gradient(x->f1(0.7,x),p)[1]
@show ∂f1_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->f1(0.7,x),p)[1]
@show ∂f1_p_err = abs.(∂f1_p_RAD .- ∂f1_p_FD) ./ abs.(∂f1_p_FD)

using Zygote: dropgrad
function f2(om,pp)
    ε⁻¹ = εₛ⁻¹(om,rwg(pp);ms)
    k, H⃗ = solve_k(ms,om,ε⁻¹)
    (mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
    om / HMₖH(H⃗[:,1],ε⁻¹,real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗)))
end
f2(0.7,p)
@show ∂f2_om_RAD = Zygote.gradient(x->f2(x,p),0.7)[1]
@show ∂f2_om_FD =  FiniteDifferences.grad(central_fdm(5,1),x->f2(x,p),0.7)[1]
@show ∂f2_om_err = abs( ∂f2_om_RAD -  ∂f2_om_FD) /  abs(∂f2_om_FD)

@show ∂f2_p_RAD =  Zygote.gradient(x->f2(0.7,x),p)[1]
@show ∂f2_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->f2(0.7,x),p)[1]
@show ∂f2_p_err = abs.(∂f2_p_RAD .- ∂f2_p_FD) ./ abs.(∂f2_p_FD)



∂omsq_p_RAD = Zygote.gradient(x->solve_k(ms,0.7,rwg2(x))[1],p)




∂omsq_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_ω²(ms,1.45,rwg2(x))[1],p)










∂omsq_p_RAD = Zygote.gradient(x->solve_k(ms,0.65,rwg2(x))[1],p)




∂omsq_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,0.65,rwg2(x))[1],p)





∂smm_p_RAD = Zygote.gradient(x->summag4(solve_k(ms,0.65,rwg2(x))[2]),p)[1]




∂smm_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->summag4(solve_k(ms,0.65,rwg2(x))[2]),p)



##
println("### εₛ: ")
println("## primal: ")
esm = εₛ(0.8,geom,gr)
@btime εₛ(0.8,$geom,$gr) # 2.352 ms (66436 allocations: 7.75 MiB)
println("## gradients: ")
println("# Zygote: ")
@show Zygote.gradient(x->sum(sum(εₛ(x,rwg(p),gr))),0.9)[1]
@show Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)[1]
@btime Zygote.gradient(x->sum(sum(εₛ(x,rwg($p),$gr))),0.9)[1]
@btime Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),$gr))),$p)[1]
println("# ForwardDiff: ")
@show ForwardDiff.derivative(x->sum(sum(εₛ(x,rwg(p),gr))),0.9)
@show ForwardDiff.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)
@btime ForwardDiff.derivative(x->sum(sum(εₛ(x,rwg($p),$gr))),0.9)
@btime ForwardDiff.gradient(x->sum(sum(εₛ(0.8,rwg(x),$gr))),$p)
println("# ForwardDiff over Zygote (2nd order): ")
@show ForwardDiff.derivative(y->Zygote.gradient(x->sum(sum(εₛ(x,rwg(p),gr))),y)[1],0.8)
@show ForwardDiff.gradient(y->Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),y)[1],p)
@btime ForwardDiff.derivative(y->Zygote.gradient(x->sum(sum(εₛ(x,rwg($p),$gr))),y)[1],0.8)
@btime ForwardDiff.gradient(y->Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),$gr))),y)[1],$p)


println("### εₛ⁻¹: ")
println("## primal: ")
eism = εₛ⁻¹(0.8,geom,gr)
@btime εₛ⁻¹(0.8,$geom,$gr) # 2.439 ms (66439 allocations: 7.75 MiB)
println("## gradients: ")
println("# Zygote: ")
@show Zygote.gradient(x->sum(sum(εₛ⁻¹(x,rwg(p),gr))),0.9)[1]
@show Zygote.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),gr))),p)[1]
@btime Zygote.gradient(x->sum(sum(εₛ⁻¹(x,rwg($p),$gr))),0.9)[1]
@btime Zygote.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),$gr))),$p)[1]
println("# ForwardDiff: ")
@show ForwardDiff.derivative(x->sum(sum(εₛ⁻¹(x,rwg(p),gr))),0.9)
@show ForwardDiff.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),gr))),p)
@btime ForwardDiff.derivative(x->sum(sum(εₛ⁻¹(x,rwg($p),$gr))),0.9)
@btime ForwardDiff.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),$gr))),$p)
println("# ForwardDiff over Zygote (2nd order): ")
@show ForwardDiff.derivative(y->Zygote.gradient(x->sum(sum(εₛ⁻¹(x,rwg(p),gr))),y)[1],0.8)
@show ForwardDiff.gradient(y->Zygote.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),gr))),y)[1],p)
@btime ForwardDiff.derivative(y->Zygote.gradient(x->sum(sum(εₛ⁻¹(x,rwg($p),$gr))),y)[1],0.8)
@btime ForwardDiff.gradient(y->Zygote.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),$gr))),y)[1],$p)

SMatrix
using ChainRulesCore: NO_FIELDS
ChainRulesCore.rrule(T::Type{<:SMatrix}, x::AbstractMatrix) = ( T(x), dv -> (NO_FIELDS, dv) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, xs::Number...) = ( T(xs...), dv -> (NO_FIELDS, dv...) )

@Zygote.adjoint (T::Type{<:SMatrix})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
@Zygote.adjoint (T::Type{<:SMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
@Zygote.adjoint (T::Type{SMatrix{2,2,Float64,4}})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)

ChainRules.refresh_rules()
Zygote.refresh()
Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)[1]
Zygote.hessian(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)[1]

##



Zygote.gradient(x->Zygote.forwarddiff(y->sum(sum(εₛ(y...))),[0.8,rwg(x),gr]),p)
Zygote.forwarddiff(y->sum(sum(εₛ(y...))),[0.8,rwg(p),gr])
Zygote.gradient(x->sum(sum(εₛ(0.8,Zygote.forwarddiff(rwg,x),gr))),p)
ForwardDiff.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)

f1(lm,p) = εₛ(lm,rwg(p),gr)
f1(0.8,p)
Zygote.gradient(x->sum(sum(f1(0.8,x))),p)

shapes1 = rwg3(p)
geom2 = rwg2(p)
convert.(Material,getfield.(shapes1,:data))
mats0 = getfield.(shapes1,:data)
similar(mats0,Material)
Material(3.5)
Material.(getfield.(shapes1,:data))

import Base.convert
convert(::Type{Material}, x) = Material(x)
materials(rwg(p))
materials2(shapes::Vector{S}) where S<:Shape{N,N²,D,T} where {N,N²,D<:Material,T} = unique!(getfield.(shapes,:data))
materials2(shapes1)
materials(shapes1)
rwg(p)
eltype(shapes1)<:Shape{N,N²,D,T} where {N,N²,D<:Material,T}
e1 = ε_tensor(3.5)
Material(e1)
##
εs_sym = getfield.(materials(shapes2),:ε)
ε_exprs = build_function.(getfield.(materials(shapes2),:ε),λ)
εs = [ eval(εe[1]) for εe in ε_exprs ]
εs! = [ eval(εe[2]) for εe in ε_exprs ]

εs[1](0.8)

struct Geometry3{N}
	shapes::Vector{Shape{N}}
	# materials::Vector{Material}
end
Geometry3(s::Vector{S}) where S<:Shape{N} where N = Geometry3{N}(s)

shapes1 =
Geometry3(shapes1)


mats = materials(shapes2)
sinds2minds = map(s->findfirst(m->isequal(s.data,m), mats),shapes2)

csinds = corner_sinds(shapes2,ms.M̂.xyz,ms.M̂.xyzc)
sinds_pr = proc_sinds(csinds)
vxl_min = @view ms.M̂.xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
vxl_max = @view ms.M̂.xyzc[min(2,end):end,min(2,end):end,min(2,end):end]



sr1 = S_rvol(ms.M̂.corner_sinds_proc,ms.M̂.xyz,vxl_min,vxl_max,shapes)
@btime S_rvol($ms.M̂.corner_sinds_proc,$ms.M̂.xyz,$vxl_min,$vxl_max,$shapes)

sr2 = S_rvol(ms.M̂.corner_sinds_proc,ms.M̂.xyz,vxl_min,vxl_max,shapes2)
@btime S_rvol($ms.M̂.corner_sinds_proc,$ms.M̂.xyz,$vxl_min,$vxl_max,$shapes2)

corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc)

S_rvol(shapes;ms)
@btime S_rvol($shapes;ms=$ms)
@btime S_rvol(shapes2;ms)

const εᵥ = SMatrix{3,3}(1.,0.,0.,0.,1.,0.,0.,0.,1.)
fεs = map(m->fε(m)[1],mats)
λs = 0.5:0.1:1.6
ωs = 1 ./ λs
εs = [vcat([SMatrix{3,3}(fep(lm)) for fep in fεs],[εᵥ,]) for lm in λs]
minds= matinds(shapes2)

epsm = εₛ(εs[1],ms.M̂.corner_sinds_proc,minds,sr1)
@btime εₛ($εs[1],$ms.M̂.corner_sinds_proc,$minds,$sr1)

εₛ11 = [ee[1,1] for ee in epsm][:,:,1]
εₛ22 = [ee[2,2] for ee in epsm][:,:,1]
εₛ12 = [ee[1,2] for ee in epsm][:,:,1]



geom = Geometry(shapes2)

##
# check that materials/shapes lists and cross-reference index lists work by adding a few shapes
bx1 = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
				[0. , 0.1],            	# c: center
				[2.8, 0.4 ],	# r: "radii" (half span of each axis)
				SMatrix{2,2}(1.,0.,0.,1.),	    		        # axes: box axes
				MgO_LiNbO₃,					        # data: any type, data associated with box shape
			)
bx2 = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
				[-0.5 , 0.4],            	# c: center
				[0.8, 0.2 ],	# r: "radii" (half span of each axis)
				SMatrix{2,2}(1.,0.,0.,1.),	    		        # axes: box axes
				SiO₂,					        # data: any type, data associated with box shape
			)
bx3 = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
				[0.5 , 0.4],            	# c: center
				[0.2, 0.2 ],	# r: "radii" (half span of each axis)
				SMatrix{2,2}(1.,0.,0.,1.),	    		        # axes: box axes
				Si₃N₄,					        # data: any type, data associated with box shape
			)
shapes3 = vcat(shapes2, [bx1,bx2,bx3])
mats3 = materials(shapes3)
sinds2minds3 = map(s->findfirst(m->isequal(s.data,m), mats3),shapes3)


##
struct Material{T}
	ε::SMatrix{3,3,T,9}
end
n(mat::Material) = sqrt.(diag(mat.ε))
n(mat::Material,axind::Int) = sqrt(mat.ε[axind,axind])
ng(mat::Material) = ng_sym.(n.(mat))
ng(mat::Material,axind::Int) = ng_sym(n(mat,axind))
gvd(mat::Material) = gvd_sym.(n.(mat))
gvd(mat::Material,axind::Int) = gvd_sym(n(mat,axind))

using ModelingToolkit
pₑ_MgO_LiNbO₃ = (
    a₁ = 5.756,
    a₂ = 0.0983,
    a₃ = 0.202,
    a₄ = 189.32,
    a₅ = 12.52,
    a₆ = 1.32e-2,
    b₁ = 2.86e-6,
    b₂ = 4.7e-8,
    b₃ = 6.113e-8,
    b₄ = 1.516e-4,
    T₀ = 24.5,      # reference temperature in [Deg C]
)
pₒ_MgO_LiNbO₃ = (
    a₁ = 5.653,
    a₂ = 0.1185,
    a₃ = 0.2091,
    a₄ = 89.61,
    a₅ = 10.85,
    a₆ = 1.97e-2,
    b₁ = 7.941e-7,
    b₂ = 3.134e-8,
    b₃ = -4.641e-9,
    b₄ = -2.188e-6,
    T₀ = 24.5,      # reference temperature in [Deg C]
)
function n²_MgO_LiNbO₃_sym(λ, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    f = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    λ² = λ^2
    a₁ + b₁*f + (a₂ + b₂*f) / (λ² - (a₃ + b₃*f)^2) + (a₄ + b₄*f) / (λ² - a₅^2) - a₆*λ²
end
@variables λ, T
nₑ²_MgO_LiNbO₃_λT_sym = n²_MgO_LiNbO₃_sym(λ, T; pₑ_MgO_LiNbO₃...)
nₑ²_MgO_LiNbO₃_sym = substitute(nₑ²_MgO_LiNbO₃_λT_sym,[T=>pₑ_MgO_LiNbO₃.T₀])
nₒ²_MgO_LiNbO₃_λT_sym = n²_MgO_LiNbO₃_sym(λ, T; pₒ_MgO_LiNbO₃...)
nₒ²_MgO_LiNbO₃_sym = substitute(nₒ²_MgO_LiNbO₃_λT_sym,[T=>pₒ_MgO_LiNbO₃.T₀])
ε_MgO_LiNbO₃_λT_sym = Diagonal( [ nₑ²_MgO_LiNbO₃_λT_sym, nₒ²_MgO_LiNbO₃_λT_sym, nₒ²_MgO_LiNbO₃_λT_sym ] )
ε_MgO_LiNbO₃_sym = Diagonal( [ nₑ²_MgO_LiNbO₃_sym, nₒ²_MgO_LiNbO₃_sym, nₒ²_MgO_LiNbO₃_sym ] )
LN = Material(SMatrix{3,3}(ε_MgO_LiNbO₃_sym))

function materials(shapes::Vector{Shape{N,N²,D,T}}) where {N,N²,D,T}
	unique!(getfield.(shapes,:data))
end

materials2(shapes) = unique!(getfield.(shapes,:data))

struct Geometry3{N,N²,D,T}
	shapes::Vector{Shape{N,N²,D,T}}
end

wg1 = Geometry3(rwg(p))
shapes3 = vcat(shapes,shapes,shapes,shapes,shapes)
wg3 = Geometry3(shapes3)
##
xy = [ SVector(ms.M̂.x[i],ms.M̂.y[j]) for i=1:Ny,j=1:Nx ]
xyc = [SVector{2}(ms.M̂.xc[i],ms.M̂.yc[j]) for i=1:(Nx+1),j=1:(Ny+1)]

update_corner_sinds!(ms.M̂.corner_sinds,shapes,xy,xyc)
@btime update_corner_sinds!($ms.M̂.corner_sinds,$shapes,$xy,$xyc)

update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc)
@btime update_corner_sinds!($ms.M̂.corner_sinds,$shapes,$ms.M̂.xyz,$ms.M̂.xyzc)

proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc)
@btime proc_corner_sinds!($ms.M̂.corner_sinds,$ms.M̂.corner_sinds_proc)


function τ_trans(ε::AbstractMatrix{T}) where T<:Real
    return @inbounds SMatrix{3,3,T,9}(
        -1/ε[1,1],      ε[2,1]/ε[1,1],                  ε[3,1]/ε[1,1],
        ε[1,2]/ε[1,1],  ε[2,2] - ε[2,1]*ε[1,2]/ε[1,1],  ε[3,2] - ε[3,1]*ε[1,2]/ε[1,1],
        ε[1,3]/ε[1,1],  ε[2,3] - ε[2,1]*ε[1,3]/ε[1,1],  ε[3,3] - ε[3,1]*ε[1,3]/ε[1,1]
    )
end

function τ⁻¹_trans(τ::AbstractMatrix{T}) where T<:Real
    return @inbounds SMatrix{3,3,T,9}(
        -1/τ[1,1],          -τ[2,1]/τ[1,1],                 -τ[3,1]/τ[1,1],
        -τ[1,2]/τ[1,1],     τ[2,2] - τ[2,1]*τ[1,2]/τ[1,1],  τ[3,2] - τ[3,1]*τ[1,2]/τ[1,1],
        -τ[1,3]/τ[1,1],     τ[2,3] - τ[2,1]*τ[1,3]/τ[1,1],  τ[3,3]- τ[3,1]*τ[1,3]/τ[1,1]
    )
end

function avg_param(ε_fg, ε_bg, n12, rvol1)
	n = n12 / norm(n12)
	# n = normalize(n12) #n12 / norm(n12) #sqrt(sum2(abs2,n12))
    # Pick a vector that is not along n.
    h = any(iszero.(n)) ? n × normalize(iszero.(n)) :  n × SVector(1., 0. , 0.)
	v = n × h
    # Create a local Cartesian coordinate system.
    S = [n h v]  # unitary
    τ1 = τ_trans(transpose(S) * ε_fg * S)  # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans(transpose(S) * ε_bg * S)  # express param2 in S coordinates, and apply τ transform
    τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)  # volume-weighted average
    return SMatrix{3,3}(S * τ⁻¹_trans(τavg) * transpose(S))  # apply τ⁻¹ and transform back to global coordinates
end

# alleq(itr) = length(itr)==0 || all( ==(itr[1]), itr)
get_ε(shapes,ind) = ind>lastindex(shapes) ? SMatrix{3,3}(1.,0.,0.,0.,1.,0.,0.,0.,1.) : shapes[ind].data
V3(v) = isequal(length(v),3) ? v : vcat(v,zeros(3-length(v)))

function n_rvol(shape,xyz,vxl_min,vxl_max)
	r₀,n⃗ = surfpt_nearby(xyz, shape)
	rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
	return V3(n⃗),rvol
end

function _smooth(shapes,sinds_proc,xyz,vxl_min,vxl_max)
	iszero(sinds_proc[2]) && return get_ε(shapes,sinds_proc[1])
	iszero(sinds_proc[3]) && return avg_param(	shapes[sinds_proc[1]].data,
												get_ε(shapes,sinds_proc[2]),
												n_rvol(shapes[sinds_proc[1]],xyz,vxl_min,vxl_max)...
												)
	return mapreduce(i->get_ε(shapes,i),+,sinds_proc) / 8
end

function smooth(shapes,sinds_proc,xyz,xyzc)
	vxl_min = @view xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
	vxl_max = @view xyzc[min(2,end):end,min(2,end):end,min(2,end):end]
	f(sp,x,vn,vp) = let s=shapes
		_smooth(s,sp,x,vn,vp)
	end
	map(f,sinds_proc,xyz,vxl_min,vxl_max)
end

# smooth(shapes,ms::ModeSolver) = smooth(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)

function smooth(shapes;ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)
	HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()}}(
		reshape(
			reinterpret(
				reshape,
				Float64,
				smooth(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc),
				),
			(3,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz),
		)
	)
end

function smoothinv(shapes,sinds_proc,xyz,xyzc)
	vxl_min = @view xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
	vxl_max = @view xyzc[min(2,end):end,min(2,end):end,min(2,end):end]
	f(sp,x,vn,vp) = let s=shapes
		inv(_smooth(s,sp,x,vn,vp))
	end
	map(f,sinds_proc,xyz,vxl_min,vxl_max)
end

function smoothinv(shapes;ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)
	HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()}}(
		reshape(
			reinterpret(
				reshape,
				Float64,
				smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc),
				),
			(3,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz),
		)
	)
end

using StaticArrays: Dynamic
function epsi(shapes::Vector{<:Shape};ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	vxl_min = @view ms.M̂.xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
	vxl_max = @view ms.M̂.xyzc[min(2,end):end,min(2,end):end,min(2,end):end]
	f(sp,x,vn,vp) = let s=shapes
		inv(_smooth(s,sp,x,vn,vp))
	end
	eibuf = Zygote.Buffer(bounds(shapes[1])[1],3,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)
	# eibuf = Buffer(bounds(shapes[1])[1],3,3,Nx,Ny,Nz)
    for ix=1:ms.M̂.Nx,iy=1:ms.M̂.Ny,iz=1:ms.M̂.Nz
		# eps = εₛ(shapes,Zygote.dropgrad(tree),Zygote.dropgrad(g.x[i]),Zygote.dropgrad(g.y[j]),Zygote.dropgrad(g.δx),Zygote.dropgrad(g.δy))
		# eps = εₛ(shapes,x[i],y[j];tree,δx,δy)
		# epsi = inv(eps) # inv( (eps' + eps) / 2) # Hermitian(inv(eps))  # inv(Hermitian(eps)) #   # inv(eps)
        # eibuf[:,:,i,j,kk] = epsi #(epsi' + epsi) / 2
		eibuf[:,:,ix,iy,iz] = f(ms.M̂.corner_sinds_proc[ix,iy,iz],ms.M̂.xyz[ix,iy,iz],vxl_min[ix,iy,iz],vxl_max[ix,iy,iz])
    end
    # return HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},T,5,5,Array{T,5}}( real(copy(eibuf)) )
	return HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()}}( real(copy(eibuf)) )
end

function epsi2(shapes::Vector{<:Shape};ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)
end

function epsi3(shapes::Vector{<:Shape};ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	reshape(
		reinterpret(
			reshape,
			Float64,
			smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc),
			),
		(3,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz),
	)
end

##
epsi(shapes;ms)
@btime epsi($shapes;ms=$ms) #64.106 ms (613646 allocations: 27.62 MiB)

epsi2(shapes;ms)
@btime epsi2($shapes;ms=$ms) # 2.991 ms (105718 allocations: 6.86 MiB)

epsi3(shapes;ms)
@btime epsi3($shapes;ms=$ms) # 3.246 ms (105721 allocations: 6.86 MiB)

smooth(shapes,ms.M̂.corner_sinds_proc,xy,xyc)
@btime smooth($shapes,$ms.M̂.corner_sinds_proc,$xy,$xyc)

smooth(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)
@btime smooth($shapes,$ms.M̂.corner_sinds_proc,$ms.M̂.xyz,$ms.M̂.xyzc)

smooth(shapes,ms)
@btime smooth($shapes,$ms)

es = smooth(shapes,ms)

# smooth2(shapes,ms)
# @btime smooth2($shapes,$ms)

# Compare with old smoothing function
tree = tree(shapes)
es_old = [SMatrix{3,3}(εₛ(shapes,ms.M̂.x[xind],ms.M̂.y[yind];tree,δx=ms.M̂.δx,δy=ms.M̂.δy)) for xind=1:Nx,yind=1:Ny]
@assert all(es_old .≈ es[:,:,1])
@btime [SMatrix{3,3}(εₛ($shapes,$ms.M̂.x[xind],$ms.M̂.y[yind];tree,δx=ms.M̂.δx,δy=ms.M̂.δy)) for xind=1:Nx,yind=1:Ny]
# 296.386 ms (1724616 allocations: 75.20 MiB)

using ChainRules, Zygote, ForwardDiff, FiniteDifferences
f1(x) = sum(sum(smooth(rwg(x);ms)))/(128*128)
f1(p)
f2(x) = sum(sum(smoothinv(rwg(x);ms)))/(128*128)
f2(p)
Zygote.gradient(f1,p)[1]
ForwardDiff.gradient(f1,p)
FiniteDifferences.grad(central_fdm(3,1),f1,p)

println("######  btimes for f1, using regular map:")
println("f1:")
@btime f1($p)
println("FowardDiff:")
@btime ForwardDiff.gradient($f1,$p)
println("Zygote:")
@btime FiniteDifferences.grad(central_fdm(3,1),$f1,$p)
println("FiniteDifferences:")
@btime Zygote.gradient($f1,$p)[1]

println("######  btimes for f2, using pmap:")
println("f2:")
@btime f2($p)
println("FowardDiff:")
@btime ForwardDiff.gradient($f2,$p)
println("Zygote:")
@btime FiniteDifferences.grad(central_fdm(3,1),$f2,$p)
println("FiniteDifferences:")
@btime Zygote.gradient($f2,$p)[1]
# ######  btimes for f1, using regular map:
# FowardDiff:
# 8.280 ms (57116 allocations: 14.59 MiB)
# Zygote:
# 67.081 ms (1147885 allocations: 147.68 MiB)
# FiniteDifferences:
# 678.245 ms (3484021 allocations: 171.39 MiB)
# ######  btimes for f2, using pmap:
# FowardDiff:
# 8.781 ms (57116 allocations: 14.59 MiB)
# Zygote:
# 69.119 ms (1147885 allocations: 147.68 MiB)
# FiniteDifferences:
# 630.052 ms (3484021 allocations: 171.39 MiB)

##
vxl_min2 = @view xyc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
vxl_max2 = @view xyc[min(2,end):end,min(2,end):end,min(2,end):end]
fsm(sp,x,vn,vp) = let s=shapes
	_smooth(s,sp,x,vn,vp)
end
##
# corner condition where simple averaging should occur: CartesianIndex(44, 54)

I = CartesianIndex(44,54,1)
_smooth(shapes,ms.M̂.corner_sinds_proc[I],xy[I],vxl_min2[I],vxl_max2[I])
_smooth(shapes,ms.M̂.corner_sinds_proc[I],ms.M̂.xyz[I],vxl_min2[I],vxl_max2[I])

##

function avg_param2(xy,sinds)
        r₀,nout = surfpt_nearby(xy, shapes[sinds[0]])

end

shapes = rwg(p)
tree = KDTree(shapes)
n_shapes = length(shapes)

# gridpoint positions
x = ( ( Δx / Nx ) .* (0:(Nx-1))) .- Δx/2.
y = ( ( Δy / Ny ) .* (0:(Ny-1))) .- Δy/2.
xy = [ SVector(x[i],y[j]) for i=1:Ny,j=1:Nx ]
# corner positions
xc = ( ( Δx / Nx ) .* (0:Nx) ) .- ( Δx/2. * ( 1 + 1. / Nx ) )
yc = ( ( Δy / Ny ) .* (0:Ny) ) .- ( Δy/2. * ( 1 + 1. / Ny ) )
xyc = [SVector{2}(xc[i],yc[j]) for i=1:(Nx+1),j=1:(Ny+1)]
sc = Array{Int}(undef,size(xyc))
sc_ext = Array{NTuple{4,Int}}(undef,size(xy))
sc .= [(a = findfirst(isequal(findfirst(SVector(xyc[i,j]),tree)),shapes); isnothing(a) ? (n_shapes+1) : a ) for i=1:(Nx+1),j=(1:Ny+1)]
sc_ext .= [ (unq = unique!( [sc[i,j], sc[1+1,j], sc[i+1,j+1], sc[i,j+1]] ); n_unq=length(unq); n_unq==1 ? (unq[1],0,0,0) : ( n_unq==2 ?  (minimum(unq),maximum(unq),0,0)  : ( sc[i,j],  sc[i+1,j],  sc[i+1,j+1],  sc[i,j+1] ) ) )  for i=1:Nx,j=1:Ny ]

sc

128 * 128



##
# gridpoint positions
x = ( ( Δx / Nx ) .* (0:(Nx-1))) .- Δx/2.
y = ( ( Δy / Ny ) .* (0:(Ny-1))) .- Δy/2.
z = ( ( Δz / Nz ) .* (0:(Nz-1))) .- Δz/2.
xy = [ SVector(x[i],y[j]) for i=1:Ny,j=1:Nx ]
xyz = [ SVector{3}(x[i],y[j],z[k]) for i=1:Ny,j=1:Nx,k=1:Nz ]
# corner positions
xc = ( ( Δx / Nx ) .* (0:Nx) ) .- ( Δx/2. * ( 1 + 1. / Nx ) )
yc = ( ( Δy / Ny ) .* (0:Ny) ) .- ( Δy/2. * ( 1 + 1. / Ny ) )
zc = ( ( Δz / Nz ) .* (0:Nz) ) .- ( Δz/2. * ( 1 + 1. / Nz ) )
xyc = [SVector{2}(xc[i],yc[j]) for i=1:(Nx+1),j=1:(Ny+1)]
xyzc = [SVector{3}(xc[i],yc[j],zc[k]) for i=1:(Nx+1),j=1:(Ny+1),k=1:(Nz+1)]
# arrays for shape index data

corner_sinds2 = zeros(Int, Nx+1,Ny+1)
corner_sinds_proc2 = fill((0,0,0,0), Nx,Ny)
corner_sinds3 = zeros(Int, Nx+1,Ny+1,Nz+1)
corner_sinds_proc3 = fill((0,0,0,0,0,0,0,0), Nx,Ny,Nz)

# update_corner_sinds!(corner_sinds2,corner_sinds_proc2,shapes,xy,xyc)
# @btime update_corner_sinds!($corner_sinds2,$corner_sinds_proc2,$shapes,$xy,$xyc)

update_corner_sinds4!(corner_sinds2,corner_sinds_proc2,shapes,xy,xyc)
@btime update_corner_sinds4!($corner_sinds2,$corner_sinds_proc2,$shapes,$xy,$xyc)

##
function update_corner_sinds!(corner_sinds,corner_sinds_proc,shapes::AbstractVector{<:GeometryPrimitives.Shape{2}},xyz,xyzc)
	n_shapes = length(shapes)
	tree = KDTree(shapes)
	a = 0
	unq = [0,0]
	for I ∈ eachindex(xyzc)
		a = findfirst(isequal(findfirst(xyzc[I],tree)),shapes)
		corner_sinds[I] = isnothing(a) ? (n_shapes+1) : a
	end
	for I ∈ CartesianIndices(xyz)
		unq = [	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0)],
					corner_sinds[I+CartesianIndex(0,1)],
					corner_sinds[I+CartesianIndex(1,1)],
		  		]
		# unq = unique!( unq )
		unique!( unq )
		a = length(unq)
		corner_sinds_proc[I] = a==1 ? (unq[1],0,0,0) :
			( a==2 ?  (minimum(unq),maximum(unq),0,0)  :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0)],
					corner_sinds[I+CartesianIndex(0,1)],
					corner_sinds[I+CartesianIndex(1,1)],
				)
		)
	end
end

in3(x::SVector{2,<:Real}, s::GeometryPrimitives.Polygon) = all(sum(s.n .* (x' .- s.v), dims=Val(2)) .≤ 0)

function in3(x::SVector{N,<:Real}, b::GeometryPrimitives.Box{N}) where {N}
    d = b.p * (x - b.c)
    for i = 1:N
        abs(d[i]) > b.r[i] && return false  # boundary is considered inside
    end
    return true
end

function f_cinds2(shapes::Vector{S}) where {S<:Shape{N}} where N #::Int
	function f_cinds2_inner(p)
		let s=shapes # pairs(shapes)
			# x-> something(ff4(x,s),lp1) #::Int
			@inbounds for (i, a) in pairs(s)
				in3(p,a) && return i
			end
			return lastindex(s)+1
		end
	end
end

function update_corner_sinds4!(corner_sinds,corner_sinds_proc,shapes,xyz,xyzc)
	# a = 0
	unq = [0,0]
	function f_cinds(p::SVector{N,T}) where {N,T} #::Int
		let s=shapes # pairs(shapes)
			# x-> something(ff4(x,s),lp1) #::Int
			for (i, a) in pairs(s)
		        in3(p,a) && return i
				# true && return i
		    end
		    return lastindex(s)+1
		end
	end
	# corner_sinds .= f_cinds.(xyzc)
	# map!(corner_sinds,xyzc) do p
	# 	let s=shapes
	# 		for (i, a) in pairs(s)
	# 			in3(p,a) && return i
	# 		end
	# 		return lastindex(s)+1
	# 	end
	# end
	@inbounds for I in eachindex(xyzc)
		corner_sinds[I] = f_cinds(xyzc[I]) #::Int
	end
	for I ∈ CartesianIndices(xyz)
	 	unq = [		corner_sinds[I],
								corner_sinds[I+CartesianIndex(1,0)],
								corner_sinds[I+CartesianIndex(0,1)],
								corner_sinds[I+CartesianIndex(1,1)]
			  				]
		# unq = unique!( unq )
		unique!( unq )
		a = length(unq)
		println("f0")
		corner_sinds_proc[I] = isone(a) ? (unq[1],0,0,0) :
			( a===2 ?  (minimum(unq),maximum(unq),0,0)  :
				( corner_sinds[I],
							corner_sinds[I+CartesianIndex(1,0)],
							corner_sinds[I+CartesianIndex(0,1)],
							corner_sinds[I+CartesianIndex(1,1)]
						)
		)
	end
end




function update_corner_sinds!(corner_sinds,corner_sinds_proc,shapes::AbstractVector{<:GeometryPrimitives.Shape{3}},xyz,xyzc)
	n_shapes = length(shapes)
	tree = KDTree(shapes)
	a = 0
	unq = [0,0]
	for I ∈ eachindex(xyzc)
		a = findfirst(isequal(findfirst(xyzc[I],tree)),shapes)
		corner_sinds[I] = isnothing(a) ? (n_shapes+1) : a
	end
	for I ∈ eachindex(xyz)
		unq .= [	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0,0)],
					corner_sinds[I+CartesianIndex(0,1,0)],
					corner_sinds[I+CartesianIndex(1,1,0)],
					corner_sinds[I+CartesianIndex(0,0,1)],
					corner_sinds[I+CartesianIndex(1,0,1)],
					corner_sinds[I+CartesianIndex(0,1,1)],
					corner_sinds[I+CartesianIndex(1,1,1)],
		  		]
		unq = unique!( unq )
		a = length(unq)
		corner_sinds_proc[I] = a==1 ? (unq[1],0,0,0,0,0,0,0) :
			( a==2 ?  (minimum(unq),maximum(unq),0,0,0,0,0,0)  :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0,0)],
					corner_sinds[I+CartesianIndex(0,1,0)],
					corner_sinds[I+CartesianIndex(1,1,0)],
					corner_sinds[I+CartesianIndex(0,0,1)],
					corner_sinds[I+CartesianIndex(1,0,1)],
					corner_sinds[I+CartesianIndex(0,1,1)],
					corner_sinds[I+CartesianIndex(1,1,1)],
				)
		)
	end
end

function ff2(p::SVector{N}, s::Vector{S}) where {N,S<:Shape{N,N²,D,T}} where {N²,D,T<:Real}
    # for i in eachindex(s)
    #     b::Tuple{SVector{2,T}, SVector{2,T}} = bounds(s[i])
    #     # if all(b[1] .< p .< b[2]) && p ∈ s[i]  # check if p is within bounding box is faster
	# 	if in(p, s[i])  # check if p is within bounding box is faster
    #         return s[i]
    #     end
    # end
	for ss in s
        # b::Tuple{SVector{2,T}, SVector{2,T}} = bounds(ss)
        # if all(b[1] .< p .< b[2]) && p ∈ s[i]  # check if p is within bounding box is faster
		if in(p,ss)  # check if p is within bounding box is faster
            return ss
        end
    end
	# return s[1]
	return nothing
end

function ff3(p::SVector{N}, s::Vector{S}) where {N,S<:Shape{N,N²,D,T}} where {N²,D,T<:Real}
	pin = let p = p
		x->in(p,x)
	end
	findfirst(pin,s)
	# findfirst(x->in(p,x),s)
end

ff4(p,s) = let p=p
	findfirst(x->in(p,x),s)
end



ff5(p::SVector{2,T},s) where T<:Real = findfirst(x->in(p,x),s)

function ff6(p,s)
	let p=p
		y = findfirst(x->in(p,x),s)
		!isnothing(y) ? y : length(s)+1
	end
end

function ff2(p::SVector{N}, kd::KDTree{N}) where {N}
    if isempty(kd.s)
        if p[kd.ix] ≤ kd.x
            return ff2(p, kd.left)
        else
            return ff2(p, kd.right)
        end
    else
        return ff2(p, kd.s)
    end
end

function ff2(p::SVector{N}, s::Vector{S}, sbg::S) where {N,S<:Shape{N}}
    @inbounds for i in eachindex(s)
        @inbounds b::Tuple{SVector{2}, SVector{2}} = bounds(s[i])
        @inbounds if all(b[1] .< p .< b[2]) && p ∈ s[i]  # check if p is within bounding box is faster
            @inbounds return s[i]
        end
    end
    return sbg
end

function ff2(p::SVector{N}, kd::KDTree{N}, sbg::Shape{N}) where {N}
    @inbounds if isempty(kd.s)
        @inbounds if p[kd.ix] ≤ kd.x
            return ff2(p, kd.left, sbg)
        else
            return ff2(p, kd.right, sbg)
        end
    else
        return ff2(p, kd.s, sbg)
    end
end
##
# function foo1(ω,geom::Geometry,grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	minds = geom.material_inds
# 	es = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fεs),[εᵥ,])
# 	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# 	ei_new = εₛ⁻¹(es,eis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function foo2(ω,pp::Vector{<:Real},grid::Grid)
# 	geom = rwg(pp)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	minds = geom.material_inds
# 	es = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fεs),[εᵥ,])
# 	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# 	ei_new = εₛ⁻¹(es,eis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function foo3(ω,pp::Vector{<:Real},grid::Grid)
# 	geom = rwg(pp)
# 	# Srvol = S_rvol(geom,grid)
# 	# ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	# minds = geom.material_inds
# 	es = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fεs),[εᵥ,])
# 	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# 	# ei_new = εₛ⁻¹(es,eis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# 	sum(sum(eis))
# end
#
# function foo4(ω,pp::Vector{<:Real},grid::Grid)
# 	geom = rwg(pp)
# 	Srvol = S_rvol(geom,grid)
# 	# ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	# minds = geom.material_inds
# 	# es = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fεs),[εᵥ,])
# 	# eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# 	# ei_new = εₛ⁻¹(es,eis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# 	sum(sum([srv[1] for srv in Srvol]))
# end
#
#
#
# function _V32(v::SVector{2,T})::SVector{3,T} where T<:Real
# 	return vcat(v,0.0)
# end
#
# _V32(v::SVector{3}) = v
#
# make_KDTree(shapes::AbstractVector{<:Shape}) = (tree = @ignore (KDTree(shapes)); tree)::KDTree
#
#
# function foo5(ω,pp::Vector{<:Real},grid::Grid)
# 	geom = rwg(pp)
# 	xyz::Matrix{SVector{3, Float64}} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc::Matrix{SVector{3, Float64}} = Zygote.@ignore(x⃗c(grid))
# 	ps::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(geom.shapes,grid))
#
# 	# xyz = x⃗(grid)		# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	# xyzc = x⃗c(grid)
# 	# ps = proc_sinds(geom.shapes,grid)
# 	vxlmin = @view xyzc[1:max((end-1),1),1:max((end-1),1)]
# 	vxlmax = @view xyzc[min(2,end):end,min(2,end):end]
# 	# # Srvol = Zygote.forwarddiff(p) do p
# 	# # 	geom = rwg(pp)
# 	# # 	# f(sp,xx,vn,vp) = let s=geom.shapes
# 	# # 	map(ps,xyz,vxlmin,vxlmax) do sp,xx,vn,vp
# 	# # 		# _S_rvol(sp,x,vn,vp,s)
# 	# # 		if iszero(sp[2])
# 	# # 			return (SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.), 0.)
# 	# # 		elseif iszero(sp[3])
# 	# # 			r₀,n⃗ = surfpt_nearby(_V32(xx), geom.shapes[sp[1]])
# 	# # 			rvol = volfrac((vn,vp),n⃗,r₀)
# 	# # 			return normcart(_V32(n⃗)), rvol # normcart(n⃗), rvol #
# 	# # 		else
# 	# # 			return (SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.), 0.)  # naive averaging to be used
# 	# # 		end
# 	# # 	end
# 	# # 	# map(f,ps,xyz,vxlmin,vxlmax)
# 	# # 	# [f(ps[i],xyz[i],vxlmin[i],vxlmax[i]) for i in eachindex(ps)]
# 	# # end
# 	Srvol = map(ps,xyz,vxlmin,vxlmax) do sp,xx,vn,vp
#
# 		if iszero(sp[2])
# 			return (SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.), 0.)
# 		elseif iszero(sp[3])
# 			r₀,n⃗ = surfpt_nearby(xx, geom.shapes[sp[1]])
# 			rvol = volfrac((vn,vp),n⃗,r₀)
# 			return normcart(n⃗), rvol # normcart(n⃗), rvol #
# 		else
# 			return (SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.), 0.)  # naive averaging to be used
# 		end
# 	end
# 	sum(sum([srv[1] for srv in Srvol]))
# end
#
# function foo6(ω,pp::Vector{T},grid::Grid) where T<:Real
# 	xyz::Matrix{SVector{3, Float64}} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc::Matrix{SVector{3, Float64}} = Zygote.@ignore(x⃗c(grid))
#
# 	vxlmin = @view xyzc[1:max((end-1),1),1:max((end-1),1)]
# 	vxlmax = @view xyzc[min(2,end):end,min(2,end):end]
#
# 	# Srvol::Matrix{Tuple{SMatrix{3, 3, T, 9}, T}} = Zygote.forwarddiff(p) do p
# 	# 	shapes::Vector{Shape2{Matrix{T},T}} = rwg2(pp)
# 	# 	ps::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(shapes,grid))
# 	# 	# f(sp,xx,vn,vp) = let s=shapes
# 	# 	map(ps,xyz,vxlmin,vxlmax) do sp,xx,vn,vp
# 	# 		# _S_rvol(sp,x,vn,vp,s)
# 	# 		if iszero(sp[2])
# 	# 			return (SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.), 0.)
# 	# 		elseif iszero(sp[3])
# 	# 			r₀,n⃗ = surfpt_nearby(xx, shapes[sp[1]])
# 	# 			rvol = volfrac((vn,vp),n⃗,r₀)
# 	# 			return normcart(n⃗), rvol # normcart(n⃗), rvol #
# 	# 		else
# 	# 			return (SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.), 0.)  # naive averaging to be used
# 	# 		end
# 	# 	end
# 	# 	# map(f,ps,xyz,vxlmin,vxlmax)
# 	# 	# [f(ps[i],xyz[i],vxlmin[i],vxlmax[i]) for i in eachindex(ps)]
# 	# end
#
# 	shapes::Vector{Shape2{Matrix{T},T}} = rwg2(pp)
# 	ps::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(shapes,grid))
#
# 	Srvol::Matrix{Tuple{SMatrix{3, 3, T, 9}, T}} = map(ps,xyz,vxlmin,vxlmax) do sp,xx,vn,vp
#
# 		if iszero(sp[2])
# 			return (SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.), 0.)
# 		elseif iszero(sp[3])
# 			r₀,n⃗ = surfpt_nearby(xx, shapes[sp[1]])
# 			rvol = volfrac((vn,vp),n⃗,r₀)
# 			return normcart(n⃗), rvol # normcart(n⃗), rvol #
# 		else
# 			return (SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.), 0.)  # naive averaging to be used
# 		end
# 	end
#
# 	sum(sum([srv[1] for srv in Srvol]))
# end
#
#
#
# function sidx1(xx::SVector{N,T},shapes::AbstractArray{S},def_idx::TI) where {N,T,S,TI<:Int}
# 	idx0 = findfirst(ss->in(xx,ss),shapes)
# 	idx::TI = isnothing(idx0) ? def_idx : idx0
# end
#
# function sidx1(xx::SVector{N,T},tree::KDTree,def_idx::TI) where {N,T,TI<:Int}
# 	idx0 = findfirst(xx,tree)
# 	idx::TI = isnothing(idx0) ? def_idx : idx0
# end
#
# function f_sidx1(shapes::AbstractArray{S},def_idx::TI) where {S,TI<:Int}
# 	ff = let shapes=shapes, def_idx=def_idx
# 		xx->sidx1(xx,shapes,def_idx)
# 	end
# 	return ff
# end
#
# function sidx3(xx::SVector{N,T},shapes::AbstractArray{S}) where {N,T,S}
# 	xin(ss) = let xx=xx
# 		in(xx,ss)
# 	end
# 	idx0 = findfirst(xin,shapes)
# 	idx::Int64 = isnothing(idx0) ? length(shapes)+1 : idx0
# end
#
# # function cs1(corner_sinds,shapes::Vector{S},xyz,xyzc::AbstractArray{T}) where {S<:GeometryPrimitives.Shape{2},T<:SVector{N}} where N
# # 	ps = pairs(shapes)
# # 	lsp1 = length(shapes) + 1
# # 	map!(corner_sinds,xyzc) do p
# # 		let ps=ps, lsp1=lsp1
# # 			for (i, a) in ps #pairs(s)
# # 				in(p::T,a::S)::Bool && return i
# # 			end
# # 			return lsp1
# # 		end
# # 	end
# # end
# #
# # function ps2(cs::AbstractArray{Int,2},xyz)
# # 	unq = [0,0,0,0]
# # 	sinds = zeros(Int64,(4,size(xyz)...)) #zeros(eltype(first(corner_sinds)),size(corner_sinds).-1)
# # 	@inbounds for I ∈ CartesianIndices(xyz)
# # 	 	unq = [		cs[I],
# # 					cs[I+CartesianIndex(1,0)],
# # 					cs[I+CartesianIndex(0,1)],
# # 					cs[I+CartesianIndex(1,1)]
# # 			  ]
# # 		unique!( sort!(unq) )
# # 		sinds[1:length(unq),I] = unq
# # 	end
# # 	return sinds
# # end
#
# function foo9(ω::T1,p::AbstractVector{T2},fname::Symbol,f_geom::F,grid::Grid) where {T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	om_p = vcat(ω,p)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		Srvol = S_rvol(geom,grid)
# 		ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 		minds = geom.material_inds
# 		es = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),getfield(geom,fname)),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 		return flat(εₛ(es,dropgrad(ps),dropgrad(minds),Srvol))  # new spatially smoothed ε tensor array
# 	end
# 	# return arr_flat
# 	return parent(parent(arr_flat))
# end
