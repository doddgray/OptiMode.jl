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
    # frequencies and parameters from "opt 5" LN rwg optimization, used in IPC conference submission
    ω = collect(range(0.6,0.7,length=20))
    λ = inv.(ω)
    p = [[1.676716762547048, 0.43474974558777724, 0.7058578266295201, 0.20717224874415013],
                    [1.644502183820398, 0.9925531919718392, 0.7526681375676043, 0.19466632142136261],
                    [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],
                    [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],
                    [1.4077890254553842, 0.6689819948730713, 0.9011365392777256, 3.6177729235559175e-5],
                    [1.4077343282288142, 0.6689827559319353, 0.9011640749055386, 1.1273307247269671e-6],]
    cost_fn_vals = [5.744935e-02, 1.521949e-02, 5.953687e-03, 5.747107e-03, 4.163141e-03, 4.161747e-03,] # cost fn is sum of squared 1ω/2ω group index mismatches
    include(joinpath(homedir(),"github","OptiMode","scripts","mpb.jl"))
    Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
    num_bands   =   2
    band_min    =   1
    band_max    =   num_bands
    geom_fn(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNxN,SiO₂N,Δx,Δy)
    grid = Grid(Δx,Δy,Nx,Ny)
    k_dir=[0.,0.,1.]
    data_path = joinpath(homedir(),"data","OptiMode","LNsweep")
    nω = length(ω)

    geom_warmup = geom_fn(p[1])
    eps_warmup = copy(smooth(ω[1],p[1],:fεs,false,geom_fn,grid).data) 
end

dom = 1e-4
dp = [1e-4 for pp in p[1]]

ps = [[ww, tt, 0.9, 1e-3] for ww=1.3:0.1:2.2, tt=0.5:0.08:0.9 ]
nw,nt = size(ps)
prfx = [ (@sprintf "w%02it%02i"  windx tindx) for windx=1:nw, tindx=1:nt]
cd(data_path)
##
ks,evecs = [ find_k_dom_dp(vcat(ω,2*ω),ps[windx,tindx],geom_fn,grid;dom,dp,data_path,filename_prefix=prfx[windx,tindx],num_bands) for windx=1:nw, tindx=1:nt]