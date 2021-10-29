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


function plot_compare_fields(F1,F2,grid;xlim=extrema(x(grid)),ylim=extrema(y(grid)),cmap=:diverging_bkr_55_10_c35_n256,)
	fig = Figure()
	@assert isequal(size(F1),size(F2))
    # ND1 = size(F1)[1]
    
    labels1r = ["real(F1_1)","real(F1_2)","real(F1_3)"]
    labels1i = ["imag(F1_1)","imag(F1_2)","imag(F1_3)"]
    labels2r = ["real(F2_1)","real(F2_2)","real(F2_3)"]
    labels2i = ["imag(F2_1)","imag(F2_2)","imag(F2_3)"]
    labelsdr = ["real((F1-F2)_1)","real((F1-F2)_2)","real((F1-F2)_3)"]
    labelsdi = ["imag((F1-F2)_1)","imag((F1-F2)_2)","imag((F1-F2)_3)"]

    ax11r = fig[1,1] = Axis(fig,backgroundcolor=:transparent,title=labels1r[1])
    ax12r = fig[1,2] = Axis(fig,backgroundcolor=:transparent,title=labels1r[2])
    ax13r = fig[1,3] = Axis(fig,backgroundcolor=:transparent,title=labels1r[3])
    ax11i = fig[2,1] = Axis(fig,backgroundcolor=:transparent,title=labels1i[1])
    ax12i = fig[2,2] = Axis(fig,backgroundcolor=:transparent,title=labels1i[2])
    ax13i = fig[2,3] = Axis(fig,backgroundcolor=:transparent,title=labels1i[3])

    ax21r = fig[3,1] = Axis(fig,backgroundcolor=:transparent,title=labels2r[1])
    ax22r = fig[3,2] = Axis(fig,backgroundcolor=:transparent,title=labels2r[2])
    ax23r = fig[3,3] = Axis(fig,backgroundcolor=:transparent,title=labels2r[3])
    ax21i = fig[4,1] = Axis(fig,backgroundcolor=:transparent,title=labels2i[1])
    ax22i = fig[4,2] = Axis(fig,backgroundcolor=:transparent,title=labels2i[2])
    ax23i = fig[4,3] = Axis(fig,backgroundcolor=:transparent,title=labels2i[3])

    axd1r = fig[5,1] = Axis(fig,backgroundcolor=:transparent,title=labelsdr[1])
    axd2r = fig[5,2] = Axis(fig,backgroundcolor=:transparent,title=labelsdr[2])
    axd3r = fig[5,3] = Axis(fig,backgroundcolor=:transparent,title=labelsdr[3])
    axd1i = fig[6,1] = Axis(fig,backgroundcolor=:transparent,title=labelsdi[1])
    axd2i = fig[6,2] = Axis(fig,backgroundcolor=:transparent,title=labelsdi[2])
    axd3i = fig[6,3] = Axis(fig,backgroundcolor=:transparent,title=labelsdi[3])

    ax1r = [ax11r,ax12r,ax13r]
    ax1i = [ax11i,ax12i,ax13i]
    ax2r = [ax21r,ax22r,ax23r]
    ax2i = [ax21i,ax22i,ax23i]
    axdr = [axd1r,axd2r,axd3r]
    axdi = [axd1i,axd2i,axd3i]

    Fd = F1 .- F2
    magmax1 = max(maximum(abs,real(F1)),maximum(abs,imag(F1)))
    magmax2 = max(maximum(abs,real(F2)),maximum(abs,imag(F2)))
    magmaxd = max(maximum(abs,real(Fd)),maximum(abs,imag(Fd)))
    x_um = x(grid)
    y_um = y(grid)
    hms1r = [heatmap!(ax1r[didx],x_um,y_um,real(F1[didx,:,:]);colorrange=(-magmax1,magmax1)) for didx=1:3]
    hms1i = [heatmap!(ax1i[didx],x_um,y_um,imag(F1[didx,:,:]);colorrange=(-magmax1,magmax1)) for didx=1:3]
    hms2r = [heatmap!(ax2r[didx],x_um,y_um,real(F2[didx,:,:]);colorrange=(-magmax2,magmax2)) for didx=1:3]
    hms2i = [heatmap!(ax2i[didx],x_um,y_um,imag(F2[didx,:,:]);colorrange=(-magmax2,magmax2)) for didx=1:3]
    hmsdr = [heatmap!(axdr[didx],x_um,y_um,real(Fd[didx,:,:]);colorrange=(-magmaxd,magmaxd)) for didx=1:3]
    hmsdi = [heatmap!(axdi[didx],x_um,y_um,imag(Fd[didx,:,:]);colorrange=(-magmaxd,magmaxd)) for didx=1:3]

    cb1r = Colorbar(fig[1,4], hms1r[1],  width=20 )
    cb1i = Colorbar(fig[2,4], hms1i[1],  width=20 )
    cb2r = Colorbar(fig[3,4], hms2r[1],  width=20 )
    cb2i = Colorbar(fig[4,4], hms2i[1],  width=20 )
    cbdr = Colorbar(fig[5,4], hmsdr[1],  width=20 )
    cbdi = Colorbar(fig[6,4], hmsdi[1],  width=20 )


    ax = (ax1r...,ax1i...,ax2r...,ax2i...,axdr...,axdi...)
    for axx in ax
        # axx.xlabel= "x [μm]"
        # xlims!(axx,xlim)
        # ylims!(axx,ylim)
        hidedecorations!(axx)
        axx.aspect=DataAspect()

    end
    # linkaxes!(ax...)

	fig
end

function plot_compare_epsilon(F1,F2,grid;xlim=extrema(x(grid)),ylim=extrema(y(grid)),cmap=:diverging_bkr_55_10_c35_n256,)
	fig = Figure()
	@assert isequal(size(F1),size(F2))
    # ND1 = size(F1)[1]
    offdiag_inds = [(1,2),(1,3),(2,3)]
    labels1r = ["eps1_11","eps1_22","eps1_33"]
    labels1i = ["eps1_12","eps1_13","eps1_23"]
    labels2r = ["eps2_11","eps2_22","eps2_33"]
    labels2i = ["eps2_12","eps2_13","eps2_23"]
    labelsdr = ["(eps1-eps2)_11","(eps1-eps2)_22","(eps1-eps2)_33"]
    labelsdi = ["(eps1-eps2)_12","(eps1-eps2)_13","(eps1-eps2)_23"]

    ax11r = fig[1,1] = Axis(fig,backgroundcolor=:transparent,title=labels1r[1])
    ax12r = fig[1,3] = Axis(fig,backgroundcolor=:transparent,title=labels1r[2])
    ax13r = fig[1,5] = Axis(fig,backgroundcolor=:transparent,title=labels1r[3])
    ax11i = fig[2,1] = Axis(fig,backgroundcolor=:transparent,title=labels1i[1])
    ax12i = fig[2,3] = Axis(fig,backgroundcolor=:transparent,title=labels1i[2])
    ax13i = fig[2,5] = Axis(fig,backgroundcolor=:transparent,title=labels1i[3])

    ax21r = fig[3,1] = Axis(fig,backgroundcolor=:transparent,title=labels2r[1])
    ax22r = fig[3,3] = Axis(fig,backgroundcolor=:transparent,title=labels2r[2])
    ax23r = fig[3,5] = Axis(fig,backgroundcolor=:transparent,title=labels2r[3])
    ax21i = fig[4,1] = Axis(fig,backgroundcolor=:transparent,title=labels2i[1])
    ax22i = fig[4,3] = Axis(fig,backgroundcolor=:transparent,title=labels2i[2])
    ax23i = fig[4,5] = Axis(fig,backgroundcolor=:transparent,title=labels2i[3])

    axd1r = fig[5,1] = Axis(fig,backgroundcolor=:transparent,title=labelsdr[1])
    axd2r = fig[5,3] = Axis(fig,backgroundcolor=:transparent,title=labelsdr[2])
    axd3r = fig[5,5] = Axis(fig,backgroundcolor=:transparent,title=labelsdr[3])
    axd1i = fig[6,1] = Axis(fig,backgroundcolor=:transparent,title=labelsdi[1])
    axd2i = fig[6,3] = Axis(fig,backgroundcolor=:transparent,title=labelsdi[2])
    axd3i = fig[6,5] = Axis(fig,backgroundcolor=:transparent,title=labelsdi[3])

    ax1r = [ax11r,ax12r,ax13r]
    ax1i = [ax11i,ax12i,ax13i]
    ax2r = [ax21r,ax22r,ax23r]
    ax2i = [ax21i,ax22i,ax23i]
    axdr = [axd1r,axd2r,axd3r]
    axdi = [axd1i,axd2i,axd3i]

    Fd = F1 .- F2

    magmax1 = [max(maximum(abs,view(F1,ii,ii,:,:)),1e-15) for ii=1:3]
    magmax2 = [max(maximum(abs,view(F2,ii,ii,:,:)),1e-15) for ii=1:3]
    magmaxd = [max(maximum(abs,view(Fd,ii,ii,:,:)),1e-15) for ii=1:3]

    magmax1o = [max(maximum(abs,view(F1,offdiag_inds[ii][1],offdiag_inds[ii][2],:,:)),1e-15) for ii=1:3]
    magmax2o = [max(maximum(abs,view(F2,offdiag_inds[ii][1],offdiag_inds[ii][2],:,:)),1e-15) for ii=1:3]
    magmaxdo = [max(maximum(abs,view(Fd,offdiag_inds[ii][1],offdiag_inds[ii][2],:,:)),1e-15) for ii=1:3]

    x_um = x(grid)
    y_um = y(grid)
    hms1r = [heatmap!(ax1r[didx],x_um,y_um,view(F1,didx,didx,:,:);colorrange=(-magmax1[didx],magmax1[didx])) for didx=1:3]
    hms1i = [heatmap!(ax1i[didx],x_um,y_um,view(F1,offdiag_inds[didx][1],offdiag_inds[didx][2],:,:);colorrange=(-magmax1o[didx],magmax1o[didx])) for didx=1:3]
    hms2r = [heatmap!(ax2r[didx],x_um,y_um,view(F2,didx,didx,:,:);colorrange=(-magmax2[didx],magmax2[didx])) for didx=1:3]
    hms2i = [heatmap!(ax2i[didx],x_um,y_um,view(F2,offdiag_inds[didx][1],offdiag_inds[didx][2],:,:);colorrange=(-magmax2o[didx],magmax2o[didx])) for didx=1:3]
    hmsdr = [heatmap!(axdr[didx],x_um,y_um,view(Fd,didx,didx,:,:);colorrange=(-magmaxd[didx],magmaxd[didx])) for didx=1:3]
    hmsdi = [heatmap!(axdi[didx],x_um,y_um,view(Fd,offdiag_inds[didx][1],offdiag_inds[didx][2],:,:);colorrange=(-magmaxdo[didx],magmaxdo[didx])) for didx=1:3]

    cb11r = [Colorbar(fig[1,2*ii], hms1r[ii],  width=20 ) for ii=1:3]
    cb11i = [Colorbar(fig[2,2*ii], hms1i[ii],  width=20 ) for ii=1:3]
    cb21r = [Colorbar(fig[3,2*ii], hms2r[ii],  width=20 ) for ii=1:3]
    cb21i = [Colorbar(fig[4,2*ii], hms2i[ii],  width=20 ) for ii=1:3]
    cbd1r = [Colorbar(fig[5,2*ii], hmsdr[ii],  width=20 ) for ii=1:3]
    cbd1i = [Colorbar(fig[6,2*ii], hmsdi[ii],  width=20 ) for ii=1:3]


    ax = (ax1r...,ax1i...,ax2r...,ax2i...,axdr...,axdi...)
    for axx in ax
        # axx.xlabel= "x [μm]"
        xlims!(axx,xlim)
        ylims!(axx,ylim)
        hidedecorations!(axx)
        axx.aspect=DataAspect()

    end
    linkaxes!(ax...)

	fig
end


"""
partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, 
x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).
"""
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNxN,SiO₂N,Δx,Δy)

include("mpb.jl")

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
function group_index(k::Real,evec,om,ε⁻¹,∂ε∂ω,grid)
    mag,mn = mag_mn(k,grid)
    om / HMₖH(vec(evec),ε⁻¹,mag,mn) * (1-(om/2)*HMH(evec, _dot( ε⁻¹, ∂ε∂ω, ε⁻¹ ),mag,mn))
end

function group_index(ks::AbstractArray,evecs,om,ε⁻¹,∂ε∂ω,grid)
    [ group_index(ks[fidx,bidx],evecs[fidx,bidx],om[fidx],ε⁻¹[fidx],∂ε∂ω[fidx],grid) for fidx=1:nω,bidx=1:num_bands ]
end
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




function find_k_dom(ω::AbstractVector,p::AbstractVector,geom_fn::TF,grid::Grid{ND};dom=1e-4,k_dir=[0.,0.,1.], num_bands=2,band_min=1,band_max=num_bands,filename_prefix="",
    band_func=(mpb.fix_efield_phase, mpb.output_efield, mpb.output_hfield,py"output_dfield_energy", py"output_evecs", mpb.display_group_velocities),
    parity=mp.NO_PARITY,n_guess_factor=0.9,data_path=pwd(),return_to_inital_dir=true,allow_overwrite=false,logpath=DEFAULT_MPB_LOGPATH,kwargs...) where {ND,TF<:Function}

    ks,evecs = find_k(ω,p,geom_fn,grid;data_path,num_bands,k_dir,band_min,band_max,band_func,parity,n_guess_factor,logpath,filename_prefix,return_to_inital_dir,allow_overwrite,kwargs...)

    dfp_path = joinpath(data_path,"dfp")
    mkpath(dfp_path)
    find_k(ω.+dom,p,geom_fn,grid;data_path=dfp_path,num_bands,k_dir,band_min,band_max,band_func,parity,n_guess_factor,logpath,filename_prefix,return_to_inital_dir,allow_overwrite,kwargs...)

    dfn_path = joinpath(data_path,"dfn")
    mkpath(dfn_path)
    find_k(ω.-dom,p,geom_fn,grid;data_path=dfn_path,num_bands,k_dir,band_min,band_max,band_func,parity,n_guess_factor,logpath,filename_prefix,return_to_inital_dir,allow_overwrite,kwargs...)

    return ks,evecs
end

ks_test,Hvs_test = find_k_dom(ωs_in,pp,rwg,grid;data_path=test_path,num_bands=2)

dom = 1e-4


function proc_find_k_dom(nω,dom;data_path=pwd())
    ω, kdir, neffs, ngs, band_idx, x_frac, y_frac, z_frac = parse_findk_logs(1:nω;data_path)
    ks = ω .* neffs
    ks = transpose(hcat(ks...))
    ngs = transpose(hcat(ngs...))
    neffs = transpose(hcat(neffs...))
    evecs   =   [load_evec_arr( joinpath(data_path, (@sprintf "evecs.f%02i.b%02i.h5" fidx bidx)), bidx, grid) for bidx=band_min:band_max, fidx=1:nω]
    epsis   =   [load_epsilon( joinpath(data_path, (@sprintf "eps.f%02i.h5" fidx )))[2] for fidx=1:nω]
    # epss,epsis   =   (eps_epsi = [load_epsilon( joinpath(data_path, (@sprintf "eps.f%02i.h5" fidx )))[1:2] for fidx=1:nω]; (getindex.(eps_epsi,(1,)),getindex.(eps_epsi,(2,))))
    Es      =   [load_field(joinpath(data_path, (@sprintf "e.f%02i.b%02i.h5" fidx bidx))) for bidx=band_min:band_max, fidx=1:nω]

    dfp_path = joinpath(data_path,"dfp")
    ω_dfp, kdir_dfp, neffs_dfp, ngs_dfp, band_idx_dfp, x_frac_dfp, y_frac_dfp, z_frac_dfp = parse_findk_logs(1:nω;data_path=dfp_path)
    ks_dfp = ω_dfp .* neffs_dfp
    ks_dfp = transpose(hcat(ks_dfp...))
    ngs_dfp = transpose(hcat(ngs_dfp...))
    neffs_dfp = transpose(hcat(neffs_dfp...))
    
    dfn_path = joinpath(data_path,"dfn")
    ω_dfn, kdir_dfn, neffs_dfn, ngs_dfn, band_idx_dfn, x_frac_dfn, y_frac_dfn, z_frac_dfn = parse_findk_logs(1:nω;data_path=dfn_path)
    ks_dfn = ω_dfn .* neffs_dfn
    ks_dfn = transpose(hcat(ks_dfn...))
    ngs_dfn = transpose(hcat(ngs_dfn...))
    neffs_dfn = transpose(hcat(neffs_dfn...))
    
    deps_dom_FD = [(load_epsilon( joinpath(dfp_path, (@sprintf "eps.f%02i.h5" fidx )))[1] .- load_epsilon( joinpath(dfn_path, (@sprintf "eps.f%02i.h5" fidx )))[1]) * inv(2*dom) for fidx=1:nω]
    ngs_disp = [( -ω[fidx] * ( 1.0 + 0.5 * ω[fidx] * real(_expect(deps_dom_FD[fidx],Es[bidx,fidx]))*dx3 ) / HMₖH(vec(evecs[bidx,fidx]),epsis[fidx],mag_mn(ks[fidx,bidx],grid)...)) for fidx=1:nω, bidx=1:num_bands ]
    dk_dom_FD = (ks_dfp .- ks_dfn) / (2*dom)
    @show ngs_err = dk_dom_FD - ngs_disp

    d2k_dom2_FD = (ks_dfp .- 2*ks .+ ks_dfn) / (2*dom)^2

    return neffs, ngs_disp, dk_dom_FD, d2k_dom2_FD
end

function find_k_dom_dp(ω::AbstractVector,p::AbstractVector,geom_fn::TF,grid::Grid{ND};dp=1e-4*ones(length(p)),dom=1e-4,k_dir=[0.,0.,1.], num_bands=2,band_min=1,band_max=num_bands,filename_prefix="",
    band_func=(mpb.fix_efield_phase, mpb.output_efield, mpb.output_hfield,py"output_dfield_energy", py"output_evecs", mpb.display_group_velocities),
    parity=mp.NO_PARITY,n_guess_factor=0.9,data_path=pwd(),return_to_inital_dir=true,allow_overwrite=false,logpath=DEFAULT_MPB_LOGPATH,kwargs...) where {ND,TF<:Function}

    ks,evecs = find_k_dom(ω,p,geom_fn,grid;dom,data_path,num_bands,k_dir,band_min,band_max,band_func,parity,n_guess_factor,logpath,filename_prefix,return_to_inital_dir,allow_overwrite,kwargs...)
    num_p = length(p)
    for pidx=1:num_p
        ddpp = [ (isequal(ii,pidx) ? dp[ii] : 0.0) for ii=1:num_p ]
        dpp_path = joinpath(data_path,(@sprintf "dpp%02i" pidx))
        mkpath(dpp_path)
        find_k_dom(ω,p.+ddpp,geom_fn,grid;dom,data_path=dpp_path,num_bands,k_dir,band_min,band_max,band_func,parity,n_guess_factor,logpath,filename_prefix,return_to_inital_dir,allow_overwrite,kwargs...)

        dpn_path = joinpath(data_path,(@sprintf "dpn%02i" pidx))
        mkpath(dpn_path)
        find_k_dom(ω,p.-ddpp,geom_fn,grid;dom,data_path=dpn_path,num_bands,k_dir,band_min,band_max,band_func,parity,n_guess_factor,logpath,filename_prefix,return_to_inital_dir,allow_overwrite,kwargs...)
    end
    return ks,evecs
end

function find_k_dom_dp2(ω::AbstractVector,p::AbstractVector,geom_fn::TF,grid::Grid{ND};dp=1e-4*ones(length(p)),dom=1e-4,k_dir=[0.,0.,1.], num_bands=2,band_min=1,band_max=num_bands,filename_prefix="",
    band_func=(mpb.fix_efield_phase, mpb.output_efield, mpb.output_hfield,py"output_dfield_energy", py"output_evecs", mpb.display_group_velocities),
    parity=mp.NO_PARITY,n_guess_factor=0.9,data_path=pwd(),return_to_inital_dir=true,allow_overwrite=false,logpath=DEFAULT_MPB_LOGPATH,kwargs...) where {ND,TF<:Function}

    # ks,evecs = find_k_dom(ω,p,geom_fn,grid;dom,data_path,num_bands,k_dir,band_min,band_max,band_func,parity,n_guess_factor,logpath,filename_prefix,return_to_inital_dir,allow_overwrite,kwargs...)
    num_p = length(p)
    for pidx=1:num_p
        ddpp = [ (isequal(ii,pidx) ? dp[ii] : 0.0) for ii=1:num_p ]
        dpp_path = joinpath(data_path,(@sprintf "dpp%02i" pidx))
        mkpath(dpp_path)
        find_k_dom(ω,p.+ddpp,geom_fn,grid;dom,data_path=dpp_path,num_bands,k_dir,band_min,band_max,band_func,parity,n_guess_factor,logpath,filename_prefix,return_to_inital_dir,allow_overwrite,kwargs...)

        dpn_path = joinpath(data_path,(@sprintf "dpn%02i" pidx))
        mkpath(dpn_path)
        find_k_dom(ω,p.-ddpp,geom_fn,grid;dom,data_path=dpn_path,num_bands,k_dir,band_min,band_max,band_func,parity,n_guess_factor,logpath,filename_prefix,return_to_inital_dir,allow_overwrite,kwargs...)
    end
    return (3.1,2.2)
end


function proc_find_k_dom_dp(nω,dom,dp;data_path=pwd())
    num_p = length(dp)
    neff, ng, ngFD, gvdFD = proc_find_k_dom(nω,dom;data_path=data_path)
    dneff_dp = zeros((size(neff)...,num_p))
    dng_dp = zeros((size(ng)...,num_p))
    dngFD_dp = zeros((size(ngFD)...,num_p))
    dgvdFD_dp = zeros((size(gvdFD)...,num_p))
    for pidx=1:num_p
        dpp_path = joinpath(data_path,(@sprintf "dpp%02i" pidx))
        dpn_path = joinpath(data_path,(@sprintf "dpn%02i" pidx))
        neff_dpp, ng_dpp, ngFD_dpp, gvdFD_dpp = proc_find_k_dom(nω,dom;data_path=dpp_path)
        neff_dpn, ng_dpn, ngFD_dpn, gvdFD_dpn = proc_find_k_dom(nω,dom;data_path=dpn_path)
        dneff_dp[:,:,pidx] = (neff_dpp .- neff_dpn) ./ (2*dp[pidx])
        dng_dp[:,:,pidx] = (ng_dpp .- ng_dpn) ./ (2*dp[pidx])
        dngFD_dp[:,:,pidx] = (ngFD_dpp .- ngFD_dpn) ./ (2*dp[pidx])
        dgvdFD_dp[:,:,pidx] = (gvdFD_dpp .- gvdFD_dpn) ./ (2*dp[pidx])
    end
    return neff, ng, ngFD, gvdFD, dneff_dp, dng_dp, dngFD_dp, dgvdFD_dp
end

dom = 1e-5
dp = 1e-4 * ones(length(pp))
# ks_test,evecs_test = find_k_dom_dp(ωs_in,pp,rwg,grid;dp,dom,data_path=test_path,num_bands=2)
# find_k_dom_dp2(ωs_in,pp,rwg,grid;dp,dom,data_path=test_path,num_bands=2)

neffs, ngs, dk_dom_FD, d2k_dom2_FD = proc_find_k_dom(nω,dom;data_path=test_path);
neff, ng, ngFD, gvdFD, dneff_dp, dng_dp, dngFD_dp, dgvdFD_dp = proc_find_k_dom_dp(nω,dom,dp;data_path=test_path)

##
fig = Figure()
ax11 = fig[1,1] = Axis(fig)
ax12 = fig[1,2] = Axis(fig)
ax13 = fig[1,3] = Axis(fig)
ax21 = fig[2,1] = Axis(fig)
ax22 = fig[2,2] = Axis(fig)
ax23 = fig[2,3] = Axis(fig)

sls_neff = [scatterlines!(ax11, ω, view(neff,:,ii)) for ii=1:num_bands]

sls_ng = [scatterlines!(ax12, ω, view(ng,:,ii)) for ii=1:num_bands]
sls_ngFD = [scatter!(ax12, ω, view(ngFD,:,ii)) for ii=1:num_bands]

sls_gvdFD = [scatterlines!(ax13, ω, view(gvdFD,:,ii)) for ii=1:num_bands]

sls_dneffdp = [scatterlines!(ax21, ω, view(dneff_dp,:,ii,pidx)) for ii=1:num_bands,pidx=1:length(dp)]

sls_dngdp = [scatterlines!(ax22, ω, view(dng_dp,:,ii,pidx)) for ii=1:num_bands,pidx=1:length(dp)]
# sls_dngFDdp = [scatter!(ax22, ω, view(dngFD_dp,:,ii,pidx)) for ii=1:num_bands,pidx=1:length(dp)]

sls_dgvdFDdp = [scatterlines!(ax23, ω, view(dgvdFD_dp,:,ii,pidx)) for ii=1:num_bands,pidx=1:length(dp)]

fig


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



## wrapping c source code in `mpb/src` and library ("user interface") code in `mpb/mpb`
"""
code from `/github/mpb/src/util/sphere-quad.c`, now compiled to `/github/mpb/src/util/sphere-quad.so`
"""

"""
#define NQUAD3 50 /* use 50-point quadrature formula by default */

/**********************************************************************/
#define K_PI 3.141592653589793238462643383279502884197
#define NQUAD2 12
/**********************************************************************/

double sqr(double x) { return x * x; }

double dist2(double x1, double y1, double z1,
	     double x2, double y2, double z2)
    {
    return sqr(x1-x2) + sqr(y1-y2) + sqr(z1-z2);
    }

double min2(double a, double b) { return a < b ? a : b; }

/* sort the array to maximize the spacing of each point with the
   previous points */
void sort_by_distance(int n, double x[], double y[], double z[], double w[])
    {
        ...
    }
"""

mpb_dist = joinpath(homedir(),"github","mpb")
mpb_src = joinpath(mpb_dist,"src")
mpb_lib = joinpath(mpb_dist,"mpb")
mpb_src_util = joinpath(mpb_src,"util")
mpb_src_matrices = joinpath(mpb_src,"matrices")
mpb_src_matrixio = joinpath(mpb_src,"matrixio")
mpb_src_maxwell = joinpath(mpb_src,"maxwell")

const mpb_sphere_quad = joinpath(mpb_src_util,"sphere-quad")
const mpb_matrices_lib = joinpath(mpb_src_matrices,"matrices")

#

function sphrqd_dist2(x1,y1,z1,x2,y2,z2)::Float64
    return ccall((:dist2,mpb_sphere_quad),Cdouble,(Float64,Float64,Float64,Float64,Float64,Float64),x1,y1,z1,x2,y2,z2)
end

function sphrqd_dist2(v1,v2)::Float64
    return ccall((:dist2,mpb_sphere_quad),Cdouble,(Float64,Float64,Float64,Float64,Float64,Float64),v1[1],v1[2],v1[3],v2[1],v2[2],v2[3])
end

get_NQUAD2() = unsafe_load(cglobal((:NQUAD2,mpb_sphere_quad)))

x1,y1,z1,x2,y2,z2 = rand(Float64,6)

d1 = dist21(x1,y1,z1,x2,y2,z2)

dist21(rand(3),rand(3))


"""
typedef struct {
    int N, localN, Nstart, allocN;
    int c;
    int n, p, alloc_p;
    scalar *data;
} evectmatrix;

evectmatrix create_evectmatrix(int N, int c, int p,
			       int localN, int Nstart, int allocN)
{
     evectmatrix X;
 
     CHECK(localN <= N && allocN >= localN && Nstart < N,
	   "invalid N arguments");
    
     X.N = N;
     X.localN = localN;
     X.Nstart = Nstart;
     X.allocN = allocN;
     X.c = c;
     
     X.n = localN * c;
     X.alloc_p = X.p = p;
     
     if (allocN > 0) {
	  CHK_MALLOC(X.data, scalar, allocN * c * p);
     }
     else
	  X.data = NULL;

     return X;
}

void destroy_evectmatrix(evectmatrix X)
{
     free(X.data);
}


"""

"""
typedef struct {
    int p, alloc_p;
    scalar *data;
} sqmatrix;
"""

mutable struct evectmatrix
end

function evectmatrix_alloc(N::Integer, c::Integer, p::Integer, localN::Integer, Nstart::Integer, allocN::Integer)
    output_ptr = ccall(
        (:create_evectmatrix, mpb_matrices_lib),    # name of C function and library
        Ptr{evectmatrix},                           # output type
        (Cint,Cint,Cint,Cint,Cint,Cint),            # tuple of input types
        (N,c,p,localN,Nstart,allocN),               # name of Julia variable to pass in
    )
end

typedef struct {
    int p, alloc_p;
    scalar *data;
} sqmatrix;