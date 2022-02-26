"""
Plot a pair of (complex) vector fields in space for comparison 
"""
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

"""
Plot a pair of spatially varying dielectric tensors for comparison 
"""
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
functions to calculate neff, ng sensitivities to frequency and geometry parameters over a range of parameter values by finite difference 
"""
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

function proc_find_k_dom(nω,dom;data_path=pwd(),pfx="")
    ω, kdir, neffs, ngs, band_idx, x_frac, y_frac, z_frac = parse_findk_logs(1:nω;data_path,pfx)
    ks = ω .* neffs
    ks = transpose(hcat(ks...))
    ngs = transpose(hcat(ngs...))
    neffs = transpose(hcat(neffs...))
    
    # evecs   =   [load_evec_arr( joinpath(data_path, (@sprintf "evecs.f%02i.b%02i.h5" fidx bidx)), bidx, grid) for bidx=band_min:band_max, fidx=1:nω]
    # epsis   =   [load_epsilon( joinpath(data_path, (@sprintf "eps.f%02i.h5" fidx )))[2] for fidx=1:nω]
    # # epss,epsis   =   (eps_epsi = [load_epsilon( joinpath(data_path, (@sprintf "eps.f%02i.h5" fidx )))[1:2] for fidx=1:nω]; (getindex.(eps_epsi,(1,)),getindex.(eps_epsi,(2,))))
    # Es      =   [load_field(joinpath(data_path, (@sprintf "e.f%02i.b%02i.h5" fidx bidx))) for bidx=band_min:band_max, fidx=1:nω]

    evecs   =   [load_evec_arr( joinpath(data_path, (rstrip(join(["evecs",pfx],'.'),'.')*(@sprintf ".f%02i.b%02i.h5" fidx bidx))), bidx, grid) for bidx=band_min:band_max, fidx=1:nω]
    epsis   =   [load_epsilon( joinpath(data_path, (rstrip(join(["eps",pfx],'.'),'.')*(@sprintf ".f%02i.h5" fidx ))))[2] for fidx=1:nω]
    # epss,epsis   =   (eps_epsi = [load_epsilon( joinpath(data_path, (@sprintf "eps.f%02i.h5" fidx )))[1:2] for fidx=1:nω]; (getindex.(eps_epsi,(1,)),getindex.(eps_epsi,(2,))))
    Es      =   [load_field(joinpath(data_path, (rstrip(join(["e",pfx],'.'),'.')*(@sprintf ".f%02i.b%02i.h5" fidx bidx)))) for bidx=band_min:band_max, fidx=1:nω]

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
    
    # deps_dom_FD = [(load_epsilon( joinpath(dfp_path, (@sprintf "eps.f%02i.h5" fidx )))[1] .- load_epsilon( joinpath(dfn_path, (@sprintf "eps.f%02i.h5" fidx )))[1]) * inv(2*dom) for fidx=1:nω]
    deps_dom_FD = [(load_epsilon( joinpath(dfp_path, (rstrip(join(["eps",pfx],'.'),'.')*(@sprintf ".f%02i.h5" fidx ))))[1] .- load_epsilon( joinpath(dfn_path, (rstrip(join(["eps",pfx],'.'),'.')*(@sprintf ".f%02i.h5" fidx ))))[1]) * inv(2*dom) for fidx=1:nω]
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

function proc_find_k_dom_dp(nω,dom,dp;data_path=pwd(),pfx="")
    num_p = length(dp)
    neff, ng, ngFD, gvdFD = proc_find_k_dom(nω,dom;data_path=data_path,pfx)
    dneff_dp = zeros((size(neff)...,num_p))
    dng_dp = zeros((size(ng)...,num_p))
    dngFD_dp = zeros((size(ngFD)...,num_p))
    dgvdFD_dp = zeros((size(gvdFD)...,num_p))
    for pidx=1:num_p
        dpp_path = joinpath(data_path,(@sprintf "dpp%02i" pidx))
        dpn_path = joinpath(data_path,(@sprintf "dpn%02i" pidx))
        neff_dpp, ng_dpp, ngFD_dpp, gvdFD_dpp = proc_find_k_dom(nω,dom;data_path=dpp_path,pfx)
        neff_dpn, ng_dpn, ngFD_dpn, gvdFD_dpn = proc_find_k_dom(nω,dom;data_path=dpn_path,pfx)
        dneff_dp[:,:,pidx] = (neff_dpp .- neff_dpn) ./ (2*dp[pidx])
        dng_dp[:,:,pidx] = (ng_dpp .- ng_dpn) ./ (2*dp[pidx])
        dngFD_dp[:,:,pidx] = (ngFD_dpp .- ngFD_dpn) ./ (2*dp[pidx])
        dgvdFD_dp[:,:,pidx] = (gvdFD_dpp .- gvdFD_dpn) ./ (2*dp[pidx])
    end
    return neff, ng, ngFD, gvdFD, dneff_dp, dng_dp, dngFD_dp, dgvdFD_dp
end


# ks_test,Hvs_test = find_k_dom(ωs_in,pp,rwg,grid;data_path=test_path,num_bands=2)
# dom = 1e-4

dom = 1e-5
dp = 1e-4 * ones(length(pp))
# ks_test,evecs_test = find_k_dom_dp(ωs_in,pp,rwg,grid;dp,dom,data_path=test_path,num_bands=2)
# find_k_dom_dp2(ωs_in,pp,rwg,grid;dp,dom,data_path=test_path,num_bands=2)

neffs, ngs, dk_dom_FD, d2k_dom2_FD = proc_find_k_dom(nω,dom;data_path=test_path);
neff, ng, ngFD, gvdFD, dneff_dp, dng_dp, dngFD_dp, dgvdFD_dp = proc_find_k_dom_dp(nω,dom,dp;data_path=test_path)
##

lnsweep_path = joinpath(homedir(),"data","OptiMode","LNsweep")
dom = 1e-4
dp = [1e-4 for pp in 1:4]
ps = [[ww, tt, 0.9, 1e-3] for ww=1.3:0.1:2.2, tt=0.5:0.08:0.9 ]
nw,nt = size(ps)
prfx = [ (@sprintf "w%02it%02i"  windx tindx) for windx=1:nw, tindx=1:nt]
cd(lnsweep_path)
# ks,evecs = [ find_k_dom_dp(vcat(ω,2*ω),ps[windx,tindx],geom_fn,grid;dom,dp,lnsweep_path,filename_prefix=prfx[windx,tindx],num_bands) for windx=1:nw, tindx=1:nt]

prfx[1,1]
neff, ng, ngFD, gvdFD, dneff_dp, dng_dp, dngFD_dp, dgvdFD_dp = proc_find_k_dom_dp(nω,dom,dp;data_path=lnsweep_path,pfx=prfx[1,1])

logpath11 = "/home/dodd/data/OptiMode/LNsweep/w01t01.f01.log"
k_lines, gv_lines, d_energy_lines = findlines(logpath11,["kvals","velocity","D-energy"])

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