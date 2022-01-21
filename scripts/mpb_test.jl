##
using Distributed
pids = addprocs(6)
@show wp = CachingPool(workers()) #default_worker_pool()
##
@everywhere begin
	using OptiMode
    using ProgressMeter
    using Rotations: RotY, MRP
    using StaticArrays
    using LinearAlgebra
    using RuntimeGeneratedFunctions
    using HDF5
    using Printf
    RuntimeGeneratedFunctions.init(@__MODULE__)
    LNx = rotate(MgO_LiNbO₃,Matrix(MRP(RotY(π/2))),name=:LiNbO₃_X);
    LNxN = NumMat(LNx;expr_module=@__MODULE__());
    SiO₂N = NumMat(SiO₂;expr_module=@__MODULE__());
    Si₃N₄N = NumMat(Si₃N₄;expr_module=@__MODULE__());
    ## geometry fn. (`rwg`) parameters (`ps`) and cost function (`fs`) values at each optimization step/epoch
    ωs_opt5 = collect(range(0.6,0.7,length=20))
    λs_opt5 = inv.(ωs_opt5)
    ps_opt5 = [[1.676716762547048, 0.43474974558777724, 0.7058578266295201, 0.20717224874415013],
                    [1.644502183820398, 0.9925531919718392, 0.7526681375676043, 0.19466632142136261],
                    [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],
                    [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],
                    [1.4077890254553842, 0.6689819948730713, 0.9011365392777256, 3.6177729235559175e-5],
                    [1.4077343282288142, 0.6689827559319353, 0.9011640749055386, 1.1273307247269671e-6],]

    include(joinpath(homedir(),"github","OptiMode","scripts","mpb.jl"))
    ωs_in   = range(1.0,1.1,11) |> collect
    epidx = 3 # iteration index
    p      = ps_opt5[epidx]
    Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
    num_bands   =   2
    test_path = joinpath(homedir(),"data","OptiMode","mpb_testp")
    band_min    =   1
    band_max    =   num_bands
    geom_fn(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNxN,SiO₂N,Δx,Δy)
    grid = Grid(Δx,Δy,Nx,Ny)
    k_dir=[0.,0.,1.]
    nω = length(ωs_in)
end

# using OptiMode
# using CairoMakie
using GLMakie
using FFTW
using Tullio
using Rotations: RotY, MRP
using ForwardDiff
# using StaticArrays
# using EllipsisNotation
# using LinearAlgebra
# using RuntimeGeneratedFunctions
# using HDF5
using Colors
import Colors: JULIA_LOGO_COLORS
using ChainRules
using FiniteDiff
using ForwardDiff
using FiniteDifferences
using Zygote
# using Printf
logocolors = JULIA_LOGO_COLORS
# RuntimeGeneratedFunctions.init(@__MODULE__)
# LNx = rotate(MgO_LiNbO₃,Matrix(MRP(RotY(π/2))),name=:LiNbO₃_X);
# LNxN = NumMat(LNx;expr_module=@__MODULE__());
# SiO₂N = NumMat(SiO₂;expr_module=@__MODULE__());

## geometry fn. (`rwg`) parameters (`ps`) and cost function (`fs`) values at each optimization step/epoch

ωs_opt5 = collect(range(0.6,0.7,length=20))
λs_opt5 = inv.(ωs_opt5)
ps_opt5 = [[1.676716762547048, 0.43474974558777724, 0.7058578266295201, 0.20717224874415013],
                   [1.644502183820398, 0.9925531919718392, 0.7526681375676043, 0.19466632142136261],
                   [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],
                   [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],
                   [1.4077890254553842, 0.6689819948730713, 0.9011365392777256, 3.6177729235559175e-5],
                   [1.4077343282288142, 0.6689827559319353, 0.9011640749055386, 1.1273307247269671e-6],]
fs_opt5 = [5.744935e-02, 1.521949e-02, 5.953687e-03, 5.747107e-03, 4.163141e-03, 4.161747e-03,]
data_dir = "/home/dodd/data"
its_fname = "shg_opt5_its.h5"
its_fpath = joinpath(data_dir,its_fname)
ω = ωs_opt5
nω = length(ω)

## function defs

gradRM(fn,in) 			= 	Zygote.gradient(fn,in)[1]
gradFM(fn,in) 			= 	ForwardDiff.gradient(fn,in)
gradFD(fn,in;n=3)		=	FiniteDifferences.grad(central_fdm(n,1),fn,in)[1]
gradFD2(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_gradient(fn,in;relstep=rs)

derivRM(fn,in) 			= 	Zygote.gradient(fn,in)[1]
derivFM(fn,in) 			= 	ForwardDiff.derivative(fn,in)
derivFD(fn,in;n=3)		=	FiniteDifferences.grad(central_fdm(n,1),fn,in)[1]
derivFD2(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_derivative(fn,in;relstep=rs)

x̂ = SVector(1.,0.,0.)
ŷ = SVector(0.,1.,0.)
ẑ = SVector(0.,0.,1.)


"""
partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, 
x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).
"""
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNxN,SiO₂N,Δx,Δy)

include("mpb.jl")
include("mpb_debug_utils.jl")


##
nω                  =   15
λ_min               =   1.4
λ_max               =   1.6
w                   =   1.225   # etched SN ridge width: 1.225um ± 25nm
t_core              =   0.17    # SiN ridge thickness: 170nm ± 10nm with 30-40nm of exposed SiO2 (HSQ) on top of unetched ridge
t_slab              =   0.20    # LN slab thickness:   200nm ± 10nm

mat_core            =   Si₃N₄N
mat_slab            =   LNxN
mat_subs            =   SiO₂N

Δx,Δy,Δz,Nx,Ny,Nz   =   8.0, 4.0, 1.0, 128, 128, 1;
edge_gap            =   0.5
num_bands           =   4

ds_dir              =   "SNwg_LNslab"
filename_prefix     =   "test2"


ωs_fund             =   range(inv(λ_max),inv(λ_min),nω)
ωs_shg              =   2. .* ωs_fund
ωs_in               =   vcat(ωs_fund,ωs_shg)
ps                  =   [w, t_core, t_slab]
ds_path             =   joinpath(homedir(),"data","OptiMode",ds_dir)
λs_fund             =   inv.(ωs_fund)
λs_shg              =   inv.(ωs_shg)

band_min            =   1
band_max            =   num_bands
geom_fn(x)          =   ridge_wg_slab_loaded(x[1],x[2],0.,x[3],edge_gap,mat_core,mat_slab,mat_subs,Δx,Δy)
grid                =   Grid(Δx,Δy,Nx,Ny)
k_dir               =   [0.,0.,1.]

##
ks, evecs = find_k(ωs_in,ps,geom_fn,grid;num_bands=num_bands,data_path=ds_path,filename_prefix=filename_prefix)  # mode solve for a vector of input frequencies

##

eps =  [copy(smooth(oo,ps,:fεs,false,geom_fn,grid).data) for oo in ωs_in]
epsi =  [copy(smooth(oo,ps,:fεs,true,geom_fn,grid).data) for oo in ωs_in]
deps_dom = [ ForwardDiff.derivative(oo->copy(getproperty(smooth(oo,p,(:fεs,:fεs),[false,true],geom_fn,grid,kottke_smoothing)[1],:data)),om) for om in ωs_in ]
mags_mns = [mag_mn(kk,grid) for kk in ks]
Es = [-1im * ε⁻¹_dot(fft(kx_tc(evecs[omidx,bndidx],mags_mns[omidx,bndidx][2],mags_mns[omidx,bndidx][1]),(2:3)),epsi[omidx]) for omidx=1:length(ωs_in), bndidx=1:num_bands]
Enorms =  [ EE[argmax(abs2.(EE))] for EE in Es ]
Es = Es ./ Enorms
ngs = [ group_index(ks[fidx,bidx],evecs[fidx,bidx],ωs_in[fidx],epsi[fidx],deps_dom[fidx],grid) for fidx=1:length(ωs_in),bidx=1:num_bands ]
neffs = ks ./ repeat(ωs_in,1,num_bands) 

neffs_fund  =   @view neffs[1:nω,:]
ngs_fund    =   @view ngs[1:nω,:]
Es_fund     =   @view Es[1:nω,:]
eps_fund    =   @view eps[1:nω]

neffs_shg   =   @view neffs[(nω+1):(2*nω),:]
ngs_shg     =   @view ngs[(nω+1):(2*nω),:]
Es_shg      =   @view Es[(nω+1):(2*nω),:]
eps_shg     =   @view eps[(nω+1):(2*nω)]

bidx_TE00_fund  =   [ 1 for oo in ωs_fund ]
bidx_TE00_shg   =   [ 1 for oo in ωs_fund ]
Δn_TE00         =   [ (neffs_shg[fidx,bidx_TE00_shg[fidx]] - neffs_fund[fidx,bidx_TE00_fund[fidx]]) for fidx=1:length(ωs_fund) ]
Λs_TE00         =   λs_shg ./ Δn_TE00


##
omidx   =   1
bndidx  =   1
axidx   =   1
cmap_Ex =   :diverging_bkr_55_10_c35_n256
cmap_nx =   :viridis
clr_fund    =   logocolors[:red]
clr_shg     =   logocolors[:blue]
clr_Λ       =   logocolors[:green]
labels          =   ["nx @ ω","Ex @ ω","Ex @ 2ω"] #label.*label_base

xs = x(grid)
ys = y(grid)
xlim =  -2.0,   2.0     # Tuple(extrema(xs)) 
ylim =  -0.8,   0.8     # Tuple(extrema(ys)) 

fig             =   Figure()
ax_neffs        =   fig[1,1] = Axis(fig)
ax_ngs          =   fig[2,1] = Axis(fig)
ax_Λs           =   fig[3,1] = Axis(fig)

ax_nxFund       =   fig[1,2] = Axis(fig)
ax_ExFund       =   fig[2,2] = Axis(fig)
ax_ExSHG        =   fig[3,2] = Axis(fig)
# cbax11  =   fig[1,1] = Axis(fig)

sls_neffs_fund  =   [scatterlines!(ax_neffs,λs_fund,neffs_fund[:,bndidx],color=clr_fund) for bndidx=1:num_bands ]
sls_neffs_shg   =   [scatterlines!(ax_neffs,λs_fund,neffs_shg[:,bndidx],color=clr_shg) for bndidx=1:num_bands ]
sls_ngs_fund    =   [scatterlines!(ax_ngs,λs_fund,ngs_fund[:,bndidx],color=clr_fund) for bndidx=1:num_bands ]
sls_ngs_shg     =   [scatterlines!(ax_ngs,λs_fund,ngs_shg[:,bndidx],color=clr_shg) for bndidx=1:num_bands ]
sl_Λs           =   scatterlines!(ax_Λs,λs_fund,Λs_TE00,color=clr_Λ) 

magmax_nxFund   =   @views maximum(abs,sqrt.(real(eps_fund[omidx][axidx,axind,:,:])))
magmax_ExFund   =   @views maximum(abs,Es_fund[omidx,bndidx])
magmax_ExSHG    =   @views maximum(abs,Es_shg[omidx,bndidx])

hm_nxFund   =   heatmap!(
    ax_nxFund,
    xs,
    ys,
    sqrt.(real(eps_fund[omidx][axidx,axind,:,:])),
    colormap=cmap_nx,label=labels[1],
    colorrange=(1.0,magmax_nxFund),
)
hm_ExFund   =   heatmap!(
    ax_ExFund,
    xs,
    ys,
    real(Es_fund[omidx,bndidx][axidx,:,:]),
    colormap=cmap_Ex,label=labels[2],
    colorrange=(-magmax_ExFund,magmax_ExFund),
)
hm_ExSHG    =   heatmap!(
    ax_ExSHG,
    xs,
    ys,
    -real(Es_shg[omidx,bndidx][axidx,:,:]),
    colormap=cmap_Ex,label=labels[3],
    colorrange=(-magmax_ExSHG,magmax_ExSHG),
)

ax_spatial = (ax_nxFund,ax_ExFund,ax_ExSHG)
for axx in ax_spatial
    axx.xlabel= "x (μm)"
    axx.ylabel= "y (μm)"
    xlims!(axx,xlim)
    ylims!(axx,ylim)
    # hidedecorations!(axx)
    axx.aspect=DataAspect()
end
linkaxes!(ax_spatial...)

fig 

##
Λ0_jank = 5.1794 #01 #5.1201
L_jank = 3e3 # 1cm in μm
##
nx = sqrt.(getindex.(inv.(ms_jank.M̂.ε⁻¹),1,1))
function Eperp_max(E)
    Eperp = view(E,1:2,:,:,:)
    maximum(abs,Eperp,dims=1:3)[1,1,1,:]
end
𝓐(n,ng,E) = inv.( n .* ng .* Eperp_max(E).^2)
AF_jank = 𝓐(nF_jank, ngF_jank, EF_jank) # inv.(nF_jank .* ngF_jank)
AS_jank = 𝓐(nS_jank, ngS_jank, ES_jank) # inv.(nS_jank .* ngS_jank)
ÊF_jank = [EF_jank[:,:,:,i] * sqrt(AF_jank[i] * nF_jank[i] * ngF_jank[i]) for i=1:nω_jank]
ÊS_jank = [ES_jank[:,:,:,i] * sqrt(AS_jank[i] * nS_jank[i] * ngS_jank[i]) for i=1:nω_jank]
𝓐₁₂₃_jank = ( AS_jank .* AF_jank.^2  ).^(1.0/3.0)
𝓞_jank = [real(sum( conj.(ÊS_jank[ind]) .* ÊF_jank[ind].^2 )) / 𝓐₁₂₃_jank[ind] * δ(grid) for ind=1:length(ωs_jank)] #
𝓞_jank_rel = 𝓞_jank/maximum(𝓞_jank)

χ⁽²⁾_LNx = χ⁽²⁾_fn(LNx)
χ⁽²⁾xxx_jank = [χ⁽²⁾_LNx(ll,ll,ll/2)[1,1,1] for ll in λs_jank]
χ⁽²⁾xxx_rel_jank = abs.(χ⁽²⁾xxx_jank) / maximum(abs.(χ⁽²⁾xxx_jank))

##

# ωs_in   = ωs_opt5
# ps      = ps_opt5
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# num_bands   =   4
# epidx = 3 # iteration index
# opt_path = joinpath(homedir(),"data","OptiMode","opt5_mpb")
# epoch_path = joinpath(opt_path,(@sprintf "epoch%03i" epidx))


ωs_in   = range(1.0,1.1,11) |> collect
ps      = ps_opt5
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
num_bands   =   2
epidx = 3 # iteration index
opt_path = joinpath(homedir(),"data","OptiMode","mpb_test")
epoch_path = opt_path

band_min    =   1
band_max    =   num_bands
geom_fn(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNxN,SiO₂N,Δx,Δy)
grid = Grid(Δx,Δy,Nx,Ny)
k_dir=[0.,0.,1.]
nω = length(ωs_in)


## test that I can load Julia-computed dielectric data into MPB modesolvers, and compare with MPB-smoothed data for equivalent geometry data
test_path = joinpath(homedir(),"data","OptiMode","mpb_testp")
cd(test_path)
f_ind = 1
pp = ps_opt5[3]
geom = rwg(pp)
feps_epsi_pp(oo) = copy.(getproperty.(smooth(oo,pp,(:fεs,:fεs),[false,true],geom_fn,grid,kottke_smoothing),(:data,)))
fdeps_epsi_dom_pp(oo) = ForwardDiff.derivative(feps_epsi_pp,oo)

feps_pp(oo) = feps_epsi_pp(oo)[1]
fepsi_pp(oo) = feps_epsi_pp(oo)[2]
fdeps_dom_pp(oo) = ForwardDiff.derivative(feps_pp,oo)
fdepsi_dom_pp(oo) = ForwardDiff.derivative(fepsi_pp,oo)

# feps_pp(oo) = copy(smooth(oo,pp,:fεs,false,geom_fn,grid).data)
# fdeps_dom_pp(oo) = ForwardDiff.derivative(feps_pp,oo)
# fepsi_pp(oo) = copy(smooth(oo,pp,:fεs,true,geom_fn,grid).data)
# fdepsi_dom_pp(oo) = ForwardDiff.derivative(fepsi_pp,oo)

fnng_nngi_pp(oo) = copy.(getproperty.(smooth(oo,pp,(:fnn̂gs,:fnn̂gs),[false,true],geom_fn,grid,kottke_smoothing),(:data,)))
fdnng_nngi_dom_pp(oo) = ForwardDiff.derivative(fnng_nngi_pp,oo)

fnng_pp(oo) = fnng_nngi_pp(oo)[1]
fnngi_pp(oo) = fnng_nngi_pp(oo)[2]

##
om = ωs_in[f_ind]
# eps, epsi, nng, nngi = smooth(om,pp,(:fεs,:fεs,:fnn̂gs,:fnn̂gs),[false,true,false,true],geom_fn,grid,kottke_smoothing)
# eps2_epsi2 =  smooth(ωs_in,pp,(:fεs,:fεs),[false,true],geom_fn,grid)
eps =  [copy(smooth(oo,pp,:fεs,false,geom_fn,grid).data) for oo in ωs_in]
epsi =  [copy(smooth(oo,pp,:fεs,true,geom_fn,grid).data) for oo in ωs_in]
# eps2 = smooth(ωs_in,pp,:fεs,false,[1. 0. 0.; 0. 1. 0. ; 0. 0. 1.],rwg,grid,kottke_smoothing)
eps1, epsi1 = feps_epsi_pp(om)
deps_dom1, depsi_dom1 = fdeps_epsi_dom_pp(om)

ms_eps_mpb = ms = ms_mpb(;
    geometry_lattice = lat_mpb(grid),
    geometry = mpGeom(geom;λ=inv(om)),
    resolution = res_mpb(grid),
    filename_prefix="",
    num_bands,
    mesh_size=3,
)
ms.output_epsilon()
eps_mpb_source_fname = DEFAULT_MPB_EPSPATH #epspath
eps_mpb_target_fname = @sprintf "eps_mpb.f%02i.h5" f_ind
cp(eps_mpb_source_fname,eps_mpb_target_fname;force=false)
eps_mpb, epsi_mpb, epsa_mpb = load_epsilon(eps_mpb_target_fname)

# plot_compare_epsilon(eps,eps_mpb,grid)

# save and reload epsilon data in MPB format to check that saved and loaded data match
eps_in_fname = @sprintf "eps_in.f%02i.h5" f_ind
save_epsilon(eps_in_fname,copy(eps.data))
eps_in_path = joinpath(test_path,eps_in_fname) # python code doesn't run in current dir, needs full path
eps_load1, epsi_load1, epsa_load1 = load_epsilon(eps_in_fname)
@assert eps_load1 ≈ eps
@assert epsi_load1 ≈ epsi

ms_eps_pyload = ms_mpb(ε[1],grid;filename_prefix="pyloadtest",num_bands=4)
ms_eps_pyload.output_epsilon()
eps_pyload, epsi_pyload, epsa_pyload = load_epsilon("pyloadtest-epsilon.h5")
@assert eps_pyload ≈ ε[1]
@assert epsi_pyload ≈ εinv[1]

evecs_eps_pyload = ms_eps_pyload.get_eigenvectors(1,4)
evec1_eps_pyload = ms_eps_pyload.get_eigenvectors(1,1)
evec1_eps_pyload[:,:,1] ≈ evecs_eps_pyload[:,:,1]

##
ms1 = ms_mpb(ε[1],grid;filename_prefix="ms1",num_bands=4)
A1 = zeros(ComplexF64,(128*128,2,ms1.num_bands))
kmags1 = ms1.find_k(
    mp.NO_PARITY,
    ωs_in[5],                    # ω at which to solve for k
    1,                 # band_min (find k(ω) for bands
    ms1.num_bands,                 # band_max  band_min:band_max)
    mp.Vector3(0,0,1),     # k direction to search
    ms1.tolerance,             # fractional k error tolerance
    2.0 * ωs_in[1],              # kmag_guess, |k| estimate
    1.0 * ωs_in[1],                # kmag_min (find k in range
    2.4 * ωs_in[1],               # kmag_max  kmag_min:kmag_max)
    # py"return_evec"(A1),
    py"return_and_save_evecs"(A1),
    # py"ret_ev4"(A1),
)
##
# plot_compare_epsilon(eps,eps_pyload,grid)
##
pp = ps_opt5[3]
ε =  [copy(smooth(oo,pp,:fεs,false,geom_fn,grid).data) for oo in ωs_in]
εinv =  [copy(smooth(oo,pp,:fεs,true,geom_fn,grid).data) for oo in ωs_in]
deps_dom = [ ForwardDiff.derivative(oo->copy(getproperty(smooth(oo,p,(:fεs,:fεs),[false,true],geom_fn,grid,kottke_smoothing)[1],:data)),om) for om in ωs_in ]

k1,evec1 = find_k(ωs_in[1],ε[1],grid,data_path=test_path,filename_prefix="singletest")      # single frequency mode solve
@assert evec1 ≈ [load_evec_arr( joinpath(test_path,("evecs." * "singletest" * (@sprintf ".b%02i.h5" bidx))), bidx, grid) for bidx=1:2]     # saved and returned mode fields eigenvectors match
ks1,evecs1 = find_k(ωs_in,ε,grid;worker_pool=wp,data_path=test_path,filename_prefix="spectest")  # mode solve for a vector of input frequencies
ks1r,evecs1r = find_k(ωs_in,ε,grid;worker_pool=wp,data_path=test_path,filename_prefix="spectest") # load pre-computed data when files already exist
evecs1 ≈ evecs1r # saved and returned mode fields eigenvectors match
ks2,evecs2 = find_k(ωs_in,pp,geom_fn,grid;data_path=test_path,filename_prefix="fktest") # method with geometry function and parameters input instead of dielectric data
# ε, εinv = copy.(getproperty.(smooth(ωs_in[1],pp,(:fεs,:fεs),[false,true],geom_fn,grid,kottke_smoothing),(:data,)))
ng1 = group_index(ωs_in,pp,geom_fn,grid;data_path=test_path,filename_prefix="ngtest")
ng2 = group_index(ωs_in,ps_opt5[2],geom_fn,grid;data_path=test_path,filename_prefix="ngtest2")

##

feps_epsi_pp(oo) = smooth(oo,pp,(:fεs,:fεs),[false,true],geom_fn,grid,kottke_smoothing)
fdeps_epsi_dom_pp(oo) = ForwardDiff.derivative(feps_epsi_pp,oo)
feps_pp(oo) = copy(getproperty(feps_epsi_pp(oo)[1],:data))
fepsi_pp(oo) = copy(getproperty(feps_epsi_pp(oo)[2],:data))
fdeps_dom_pp(oo) = ForwardDiff.derivative(feps_pp,oo)
fdepsi_dom_pp(oo) = ForwardDiff.derivative(fepsi_pp,oo)

## 
fidx = 11
bidx = 1
M = HelmholtzMap(SVector(0.,0.,ks1[fidx,bidx]), εinv[fidx], grid)
ev = vec(evecs1[fidx,bidx])
Mev = M * ev

@show om = ωs_in[fidx]
@show omsq = ωs_in[fidx]^2
@show kk = ks1[fidx,bidx]
@show neff = kk/om

mag, mn = mag_mn(kk,grid)
ee = feps_pp(om)
eei = fepsi_pp(om)
deedom = fdeps_dom_pp(om)


@show ev_norm = real(dot(ev,ev))
@show M_expect1 = real(dot(ev,Mev))
@show M_expect2 = real(dot(ev,M,ev))
@show M_expect3 = HMH(ev,εinv[fidx],mag,mn)
@show ng = om / HMₖH(vec(ev),eei,mag,mn) * (1-(om/2)*HMH(ev, _dot( eei, deedom, eei ),mag,mn))
@show ng_prev = ng1[fidx,bidx]

ng_val, ng_pb = Zygote.pullback(om,kk,ev) do om, kk, ev
    mag1, mn1 = mag_mn(kk,grid)
    eei1 = fepsi_pp(om)
    deedom1 = fdeps_dom_pp(om)
    return om / ( HMₖH(vec(ev),eei1,mag1,mn1) * (1.0-(om/2.0)*HMH(ev, _dot( eei1, deedom1, eei1 ),mag1,mn1)) )
end

om_bar, kk_bar, ev_bar = ng_pb(1)

##
lm = eig_adjt(M,omsq,ev,omsq_bar,ev_bar)

# X1 = randn(ComplexF64,size(M̂,1),2) #/N(grid)
# res1 = LOBPCG(M̂,X1,I,HelmholtzPreconditioner(M̂),1e-10,200;display_progress=true)

##


bidx = 1
fidx = 1

eps = feps_pp.(ωs_in)
epsi = fepsi_pp.(ωs_in)
deps_dom = fdeps_dom_pp.(ωs_in)

om = ωs_in[fidx]
evec = evecs1[fidx,bidx]
k = ks1[fidx,bidx]
ε = feps_pp(ωs_in[fidx])
ε⁻¹ = fepsi_pp(ωs_in[fidx])
∂ε∂ω = fdeps_dom_pp(ωs_in[fidx])
ε⁻¹_∂ε∂ω_ε⁻¹ = _dot( ε⁻¹, ∂ε∂ω, ε⁻¹ )
mag,mn = mag_mn(k,grid)
norm_fact = inv(sqrt(δ(grid) * N(grid))) # inv(sqrt(δ(grid) * N(grid)) * om)

nng = fnng_pp(om)
nng_err = ∂ε∂ω .- nng
@show maximum(abs.(nng_err))

D = fft(kx_tc(evec,mn,mag),(2:3)) * norm_fact
E = ε⁻¹_dot(D,ε⁻¹)

real(_expect(ε⁻¹_∂ε∂ω_ε⁻¹,D)) * δ(grid)
real(_expect(∂ε∂ω,E)) * δ(grid)
real(_expect(ε,E)) * δ(grid)
sum(abs2,E) *δ(grid)

om / HMₖH(vec(evec),ε⁻¹,mag,mn)
om / real( dot(evec, kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(evec,mn), (2:3) ), real(ε⁻¹)), (2:3)),mn,mag) ) )

-HMH(evec,ε⁻¹,mag,mn)

om / HMₖH(vec(evec),ε⁻¹,mag,mn) * (1-(om/2)*HMH(evec, _dot( ε⁻¹, ∂ε∂ω, ε⁻¹ ),mag,mn))

sum(abs2,evec)

ng1 = group_index(k,evec,om,ε⁻¹,∂ε∂ω,grid)
ngs1 = group_index(ks1,evecs1,ωs_in,epsi,deps_dom,grid)

ngs1 = map(ks1,evecs1,ωs_in,epsi,deps_dom) do k,evec,om,ε⁻¹,∂ε∂ω,grid
    group_index(k,evec,om,ε⁻¹,∂ε∂ω,grid)
end
##
# function group_index(k::Real,evec,om,ε⁻¹,∂ε∂ω,grid)
#     mag,mn = mag_mn(k,grid)
#     om / HMₖH(vec(evec),ε⁻¹,mag,mn) * (1-(om/2)*HMH(evec, _dot( ε⁻¹, ∂ε∂ω, ε⁻¹ ),mag,mn))
# end

# function group_index(ks::AbstractArray,evecs,om,ε⁻¹,∂ε∂ω,grid)
#     [ group_index(ks[fidx,bidx],evecs[fidx,bidx],om[fidx],ε⁻¹[fidx],∂ε∂ω[fidx],grid) for fidx=1:nω,bidx=1:num_bands ]
# end
##  

ks_test,Hvs_test = find_k(om,copy(eps.data),grid;data_path=test_path,num_bands=4)

ks_test,Hvs_test = find_k(ωs_in,eps3,grid;data_path=test_path,num_bands=2)

ω, kdir, neffs, ngs, band_idx, x_frac, y_frac, z_frac = parse_findk_logs(1:nω)
ngs = transpose(hcat(ngs...))
neffs = transpose(hcat(neffs...))
evecs   =   [load_evec_arr( joinpath(test_path, (@sprintf "evecs.f%02i.b%02i.h5" fidx bidx)), bidx, grid) for bidx=band_min:band_max, fidx=1:nω]
epss   =   [load_epsilon( joinpath(test_path, (@sprintf "eps.f%02i.h5" fidx )))[1] for fidx=1:nω]
Es      =   [load_field(joinpath(test_path, (@sprintf "e.f%02i.b%02i.h5" fidx bidx))) for bidx=band_min:band_max, fidx=1:nω]
Hs      =   [load_field(joinpath(test_path, (@sprintf "h.f%02i.b%02i.h5" fidx bidx))) for bidx=band_min:band_max, fidx=1:nω]
ks = neffs .* ω
ngs2 = [(ω[i] / HMₖH(vec(evecs[ib,i]),epsi3[i],mag_mn(ks[i,ib],grid)...)) for i=1:11,ib=1:2]
dx3 =  δx(grid) * δy(grid)
ngs3 = [ real(_expect(eps3[fidx],Es[bidx,fidx]) + sum(abs2,Hs[bidx,fidx])) * dx3 / (2*real(sum(_cross(Es[bidx,fidx],Hs[bidx,fidx])[3,:,:])) * dx3) for fidx=1:11,bidx=1:2]
ngs4 = [ inv( real(sum(_cross(Es[bidx,fidx],Hs[bidx,fidx])[3,:,:])) * dx3 )  for fidx=1:11,bidx=1:2]

dom = 1e-4
pp

dfp_path = joinpath(test_path,"dfp")
# mkpath(dfp_path)
# find_k(ωs_in.+dom,pp,rwg,grid;data_path=dfp_path,num_bands=2)
ω_dfp, kdir_dfp, neffs_dfp, ngs_dfp, band_idx_dfp, x_frac_dfp, y_frac_dfp, z_frac_dfp = parse_findk_logs(1:nω;data_path=dfp_path)
ks_dfp = ω_dfp .* neffs_dfp
ks_dfp = transpose(hcat(ks_dfp...))
ngs_dfp = transpose(hcat(ngs_dfp...))
neffs_dfp = transpose(hcat(neffs_dfp...))


dfn_path = joinpath(test_path,"dfn")
# mkpath(dfn_path)
# find_k(ωs_in.-dom,pp,rwg,grid;data_path=dfn_path,num_bands=2)
ω_dfn, kdir_dfn, neffs_dfn, ngs_dfn, band_idx_dfn, x_frac_dfn, y_frac_dfn, z_frac_dfn = parse_findk_logs(1:nω;data_path=dfn_path)
ks_dfn = ω_dfn .* neffs_dfn
ks_dfn = transpose(hcat(ks_dfn...))
ngs_dfn = transpose(hcat(ngs_dfn...))
neffs_dfn = transpose(hcat(neffs_dfn...))

deps_dom_FD = [(load_epsilon( joinpath(dfp_path, (@sprintf "eps.f%02i.h5" fidx )))[1] .- load_epsilon( joinpath(dfn_path, (@sprintf "eps.f%02i.h5" fidx )))[1]) * inv(2*dom) for fidx=1:nω]

@show deps_abs_max = maximum(abs.(deps_dom_FD[4]))

nondisp_H_energy_density = [ sum(abs2,Hs[bidx,fidx])*dx3 for fidx=1:nω, bidx=1:num_bands ]
nondisp_E_energy_density = [ real(_expect(epss[fidx],Es[bidx,fidx]))*dx3 for fidx=1:nω, bidx=1:num_bands ]
disp_energy_density_FD = nondisp_E_energy_density .+ [ 0.5*real(_expect(deps_dom_FD[fidx],Es[bidx,fidx]))*dx3 for fidx=1:nω, bidx=1:num_bands ]
ngs2_nondisp = [(-ω[fidx] / HMₖH(vec(evecs[bidx,fidx]),epsi3[fidx],mag_mn(ks[fidx,bidx],grid)...)) for fidx=1:nω, bidx=1:num_bands ]

ngs2_disp = [( -ω[fidx] * ( 1.0 + 0.5 * ω[fidx] * real(_expect(deps_dom_FD[fidx],Es[bidx,fidx]))*dx3 ) / HMₖH(vec(evecs[bidx,fidx]),epsi3[fidx],mag_mn(ks[fidx,bidx],grid)...)) for fidx=1:nω, bidx=1:num_bands ]
ngs3_disp = [ real(_expect(eps3[fidx],Es[bidx,fidx]) + sum(abs2,Hs[bidx,fidx])) * dx3 / (2*real(sum(_cross(Es[bidx,fidx],Hs[bidx,fidx])[3,:,:])) * dx3) for fidx=1:11,bidx=1:2]
ngs_FD = (ks_dfp .- ks_dfn) / (2*dom)
@show ngs_err = ngs_FD - ngs







##
# mkdir(epoch_path)
cd(epoch_path)
# kmags = find_k(ωs_in,geom_fn(ps[epidx]),grid;data_path=epoch_path,num_bands)

##
# load data from single spectrum
ω, kdir, neffs, ngs, band_idx, x_frac, y_frac, z_frac = parse_findk_logs(1:nω);

# generate (mag, m, n) data for each solution
kvals   =   neffs .* ω
ks      =   [ kvals[fidx] .* (k_dir,)  for fidx=1:nω]

# load fields
# evecs   =   [load_evecs(joinpath(epoch_path, (@sprintf "evecs.f%02i.b%02i.h5" fidx bidx)))[:,:,bidx] for bidx=band_min:band_max, fidx=1:nω]
evecs   =   [load_evec_arr( joinpath(epoch_path, (@sprintf "evecs.f%02i.b%02i.h5" fidx bidx)), bidx, grid) for bidx=band_min:band_max, fidx=1:nω]
Es      =   [load_field(joinpath(epoch_path, (@sprintf "e.f%02i.b%02i.h5" fidx bidx))) for bidx=band_min:band_max, fidx=1:nω]
Hs      =   [load_field(joinpath(epoch_path, (@sprintf "h.f%02i.b%02i.h5" fidx bidx))) for bidx=band_min:band_max, fidx=1:nω]
eps_epsi_epsave =   [load_epsilon(joinpath(epoch_path, (@sprintf "eps.f%02i.h5" fidx))) for fidx=1:nω]
eps, epsi, epsave    =   ( [ getindex(a,ii) for a in eps_epsi_epsave ] for ii=1:3) |> collect


## calculate E-fields from evecs and compare with loaded data
fidx = 5
bidx = 1
dx3 =  δx(grid) * δy(grid)
dx3_2 = Dx(grid) * Dy(grid) / N(grid)
evec    = evecs[bidx,fidx]
eeps   = eps[fidx]
eepsi   = epsi[fidx]
EE  = Es[bidx,fidx]
HH  = Hs[bidx,fidx]
om = ω[fidx] 
kk = ks[fidx][bidx][3]
kvec = SVector(ks[fidx][bidx]...)
mag, mn = mag_mn(kvec,grid)
norm_fact = inv(sqrt(δ(grid) * N(grid)) * om)
Hv = copy(evec) * norm_fact
##
sum(abs2,evec)
sum(abs2,Hv) 

ng = ngs[fidx][bidx]
ng1 = real(_expect(eeps,EE) + sum(abs2,HH)) * dx3 / (2*real(sum(_cross(EE,HH)[3,:,:])) * dx3)
@show ng1/ng  

real(sum(_cross(EE,HH)[3,:,:])) * dx3
- HMₖH(evec,eepsi,mag,mn) / om


om / HMₖH(evec,eepsi,mag2,mn2)

m = view(mn,:,1,:,:) |> copy
n = view(mn,:,2,:,:) |> copy

om / HMₖH(evec,eepsi,mag,m,n)

ng2 
ng2 / ng1

HMH(evec,eepsi,mag,mn) / om^2

# plot_compare_fields(
#     EE,
#     ε⁻¹_dot(fft(kx_tc(Hv,mn,mag),(2:3)),eepsi),
#     grid,
# )

# plot_compare_fields(
#     HH,
#     fft(tc(Hv,mn),(2:3)) * om,
#     grid,
# )

# plot_compare_fields(
#     HH,
#     fft(tc(kx_ct(ifft(EE,(2:3)),mn,mag),mm,nn),(2:3)) / om,
#     grid,
# )

## check that ForwardDiff is correctly differentiating dielectric tensor w.r.t ω
om = ωs_in[1]
dom = 1e-8

deps_dom_FD = ( feps_pp(om + dom/2) - feps_pp(om - dom/2) ) / dom
deps_dom = fdeps_dom_pp(om)

deps_dom_err = deps_dom_FD - deps_dom

@show maximum(abs.(deps_dom_err))



