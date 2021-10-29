using OptiMode
using CairoMakie
using FFTW
using Tullio
using Rotations: RotY, MRP
using ForwardDiff
using StaticArrays
using LinearAlgebra
using RuntimeGeneratedFunctions
using HDF5
using Colors
import Colors: JULIA_LOGO_COLORS
logocolors = JULIA_LOGO_COLORS
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
fs_opt5 = [5.744935e-02, 1.521949e-02, 5.953687e-03, 5.747107e-03, 4.163141e-03, 4.161747e-03,]
data_dir = "/home/dodd/data"
its_fname = "shg_opt5_its.h5"
its_fpath = joinpath(data_dir,its_fname)
ω = ωs_opt5
nω = length(ω)

# geometry function and grid used for optimization in IPC 2021 abstract figure
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
grid = Grid(Δx,Δy,Nx,Ny)
omind = 10
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNxN,SiO₂N,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).
f_eps(om,pp) = smooth(om,pp,(:fεs,:fεs),[false,true],rwg,grid,kottke_smoothing)[1]
E_dot_D_disp(E,eps,deps_dom) = @views [real( dot(E[:,ix,iy,omind],eps[:,:,ix,iy]+deps_dom[:,:,ix,iy],E[:,ix,iy,omind]) ) for ix=1:Nx,iy=1:Ny] # E_dot_D = E1 ⋅ ε ⋅ E1
E_dot_D_nondisp(E,eps,deps_dom) = @views [real( dot(E[:,ix,iy,omind],eps[:,:,ix,iy],E[:,ix,iy,omind]) ) for ix=1:Nx,iy=1:Ny] # E_dot_D = E1 ⋅ ε ⋅ E1
W(om,E,eps,deps_dom)= @views sum( [real( dot(E[:,ix,iy],eps[:,:,ix,iy],E[:,ix,iy]) ) for ix=1:Nx,iy=1:Ny] + om/2. * [real( dot(E[:,ix,iy],deps_dom[:,:,ix,iy],E[:,ix,iy]) ) for ix=1:Nx,iy=1:Ny])


##
@show λ = inv(ω[omind]) # [μm] wavelength used to generate dielectric data 
xlims = -2,2 # [μm]
ylims = -0.6,0.6 # [μm]
savedir = "/home/dodd/google-drive/Documents/ORISE/OptiMode/schematics and figures/"
fname_prefix = "opt5"
res = (600,200)
nx_cmap = :cividis
cmap_I = :magma
frame_color = HSLA(0,0,1,0.5) # white, semi-transparent

# plot and save geometry for each step
for (i,pp) in enumerate(ps_opt5)
    fig = Figure(resolution=res,backgroundcolor=:transparent)
    ax = fig[1,1] = Axis(
        fig,
        backgroundcolor=:transparent,
        xlims = xlims,
        ylims = ylims,
        xlabel="x (μm)",
        ylabel="y (μm)",
    )
    plot_shapes(rwg(pp),ax,mat_legend=false)
    xlims!(ax,xlims)
    ylims!(ax,ylims)
    ax.aspect = DataAspect()
    save(joinpath(savedir,fname_prefix*"_geom_step$i.png"), fig)
    save(joinpath(savedir,fname_prefix*"_geom_step$i.svg"), fig)
    hidedecorations!(ax)
    save(joinpath(savedir,fname_prefix*"_geom_NoDecs_step$i.png"), fig)
    save(joinpath(savedir,fname_prefix*"_geom_NoDecs_step$i.svg"), fig)
end
fig
# plot and save nx (index along x) and E-field intensity heatmaps for each step
for (i,pp) in enumerate(ps_opt5)
    # plot and save smoothed nx (x index)
    eps,epsi = smooth(ω[omind],pp,(:fεs,:fεs),[false,true],rwg,grid,kottke_smoothing)
    deps_dom = ForwardDiff.derivative(oo->f_eps(oo,pp),ω[omind])
    nx = sqrt.(eps.data[1,1,:,:])
    fig = Figure(resolution=res,backgroundcolor=:transparent)
    ax = fig[1,1] = Axis(
        fig,
        backgroundcolor=:transparent,
        xlims = xlims,
        ylims = ylims,
        xlabel="x (μm)",
        ylabel="y (μm)",
    )
    xs = x(grid)
    ys = y(grid)
    hm = heatmap!(ax,xs,ys,nx;colormap=nx_cmap)
    xlims!(ax,xlims)
    ylims!(ax,ylims)
    ax.aspect = DataAspect()
    save(joinpath(savedir,fname_prefix*"_nx_step$i.png"), fig)
    # save(joinpath(savedir,fname_prefix*"_nx_step$i.svg"), fig)
    hidedecorations!(ax)
    save(joinpath(savedir,fname_prefix*"_nx_NoDecs_step$i.png"), fig)
    # save(joinpath(savedir,fname_prefix*"_nx_NoDecs_step$i.svg"), fig)

    # plot and save Intensity with overlaid shape frames
    E1, E2, n1, n2, ng1, ng2 = h5open(its_fpath, "r") do file
        E1 = read(file, "$i/EF")
        n1 = read(file, "$i/nF")
        ng1 = read(file, "$i/ngF")
        E2 = read(file, "$i/ES")
        n2 = read(file, "$i/nS")
        ng2 = read(file, "$i/ngS")
        return E1, E2, n1, n2, ng1, ng2
    end
    E_dot_D = @views [real( dot(E1[:,ix,iy,omind],eps[:,:,ix,iy],E1[:,ix,iy,omind]) ) for ix=1:Nx,iy=1:Ny] # E_dot_D = E1 ⋅ ε ⋅ E1
    fig = Figure(resolution=res,backgroundcolor=:transparent)
    ax = fig[1,1] = Axis(
        fig,
        backgroundcolor=:transparent,
        xlims = xlims,
        ylims = ylims,
        xlabel="x (μm)",
        ylabel="y (μm)",
    )
    hmE_dot_D = heatmap!(ax,x(grid),y(grid),I1;colormap=cmap_I)
    plot_shapes(rwg(pp),ax,mat_legend=false,frame_only=true,strokecolor=frame_color) 
    xlims!(ax,xlims)
    ylims!(ax,ylims)
    ax.aspect = DataAspect()
    save(joinpath(savedir,fname_prefix*"_Esq_step$i.png"), fig)
    # save(joinpath(savedir,fname_prefix*"_Esq_step$i.svg"), fig)
    hidedecorations!(ax)
    save(joinpath(savedir,fname_prefix*"_Esq_NoDecs_step$i.png"), fig)
    # save(joinpath(savedir,fname_prefix*"_Esq_NoDecs_step$i.svg"), fig)

end
fig
##
savedir = "/home/dodd/google-drive/Documents/ORISE/OptiMode/schematics and figures/"
fname_prefix = "opt5"
res = (400,300)
xlims_disp = 1.42,1.675 # [μm]
ylims_disp = 2.25,2.32 # [μm]
λs_opt5 = inv.(ωs_opt5)
ipc_theme = Theme(fontsize=18)
set_theme!(ipc_theme)
clr1 = logocolors[:red]
clr2 = logocolors[:blue]
# labels = ["wₜₒₚ [μm]", "tₗₙ [μm]", "etch frac", "sw. angle [rad]"]
# i = 2
for (i,pp) in enumerate(ps_opt5)
    n1, n2, ng1, ng2 = h5open(its_fpath, "r") do file
        n1 = read(file, "$i/nF")
        ng1 = read(file, "$i/ngF")
        n2 = read(file, "$i/nS")
        ng2 = read(file, "$i/ngS")
        return n1, n2, ng1, ng2
    end;
    fig = Figure(resolution=res,backgroundcolor=:transparent) #:transparent)
    ax = fig[1,1] = Axis(
        fig,
        backgroundcolor=:transparent,
        xlims = xlims_disp,
        ylims = ylims_disp,
        xlabel="wavelength (μm)",
        ylabel="eff. group index",
    )

    # scatterlines!(ax,λs_opt5,n1,color="red")
    # scatterlines!(ax,λs_opt5,n2,color="blue")
    scatterlines!(ax,λs_opt5,ng1,color=clr1,marker=:rect,markercolor=clr1,linewidth=2,markersize=6)
    scatterlines!(ax,λs_opt5,ng2,color=clr2,marker=:rect,markercolor=clr2,linewidth=2,markersize=6)
    xlims!(ax,xlims_disp)
    ylims!(ax,ylims_disp)
    save(joinpath(savedir,fname_prefix*"_ngDisp_step$i.png"), fig)
    save(joinpath(savedir,fname_prefix*"_ngDisp_step$i.svg"), fig)
    hidedecorations!(ax)
    save(joinpath(savedir,fname_prefix*"_ngDisp_NoDecs_step$i.png"), fig)
    save(joinpath(savedir,fname_prefix*"_ngDisp_NoDecs_step$i.svg"), fig)
end
fig
##
omind = 10
i = 3
λ = inv(ωs_opt5[omind])
pp = ps_opt5[i]
eps,epsi = smooth(ω[omind],pp,(:fεs,:fεs),[false,true],rwg,grid,kottke_smoothing);
deps_dom = ForwardDiff.derivative(oo->f_eps(oo,pp),ω[omind]);
E1, E2, n1, n2, ng1, ng2 = h5open(its_fpath, "r") do file
    E1 = read(file, "$i/EF")
    n1 = read(file, "$i/nF")
    ng1 = read(file, "$i/ngF")
    E2 = read(file, "$i/ES")
    n2 = read(file, "$i/nS")
    ng2 = read(file, "$i/ngS")
    return E1, E2, n1, n2, ng1, ng2
end;
E = copy(E1[:,:,:,omind])
E = E / sqrt(abs(_expect(eps,E)))
k = n1[i]
mag, m⃗, n⃗ = mag_m_n(k,grid);
mn = copy(reshape(reinterpret(Float64,hcat.(m⃗,n⃗)),(3,2,Nx,Ny)));
# mns = vcat(reshape(flat(m⃗),1,3,Nx,Ny),reshape(flat(n⃗),1,3,Nx,Ny)); old mns shape
Hₜ = kx_ct( ifft( E, (2:3) ), mn, mag) * (-1im / ωs_opt5[omind]);

# `H`: real-space, cartesian basis H-field corresponding to E-field `E`
H = fft( tc(Hₜ,mn), (2:3) ) #* (-1im * om)
##
function grp_vel(om,k,E,eps,deps_dom)
    mag, m⃗, n⃗ = mag_m_n(k,grid)
    mn = copy(reshape(reinterpret(Float64,hcat.(m⃗,n⃗)),(3,2,Nx,Ny)))
    # `Hₜ`: reciprocal-space (plane-wave), transverse basis H-field corresponding to E-field `E`
    Hₜ = kx_ct( ifft( E, (2:3) ), mn, mag) * (-1im * om)
    # `H`: real-space, cartesian basis H-field corresponding to E-field `E`
    H = fft( tc(Hₜ,mn), (2:3) ) * (-1im * om)
    # energy density per unit length: W = ∫dA E⃗ ⋅ [ ( ε + ω * ∂ε/∂ω  ) ⋅ E⃗ ]
    # W = sum( [real( dot(E[:,ix,iy],eps[:,:,ix,iy],E[:,ix,iy]) ) for ix=1:Nx,iy=1:Ny] + om/2. * [real( dot(E[:,ix,iy],deps_dom[:,:,ix,iy],E[:,ix,iy]) ) for ix=1:Nx,iy=1:Ny])
    # W = abs(_expect(eps,E)) + om*abs(_expect(deps_dom,E))
    W = abs(_expect(eps+om*deps_dom,E)) # _expect output should be real, but real(_expect(...)) breaks type inference for the gradient 
    # integrated Poynting flux parallel to ẑ: P = ∫dA ( conj(E⃗) × H⃗ ) ⋅ ẑ
    P = 2*real(_sum_cross_z(conj(E),H))
    ng = W / P
    return ng
end
E1x,E1y,E1z = (view(E1,axind,:,:,omind) for axind=1:3)
E2x,E2y,E2z = (view(E2,axind,:,:,omind) for axind=1:3)
eps,epsi = smooth(ωs_opt5[omind],pp,(:fεs,:fεs),[false,true],rwg,grid,kottke_smoothing)
# D1 = _dot(eps,view(E1,:,:,:,i))
# I1 = reshape(sum(real.(conj.(view(E1,:,:,:,i)) .* D1),dims=1),(Nx,Ny))

# nng = eps + om * deps_dom
eps,epsi,nngi = smooth(ω[omind],pp,(:fεs,:fεs,:fnn̂gs),[false,true,true],rwg,grid,kottke_smoothing);
m = copy(flat(m⃗))
n = copy(flat(n⃗))
# ng_z(Hₜ,om,epsi,nng,mag,m,n)
mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3])))
mn = copy(reshape(reinterpret(Float64,hcat.(m⃗,n⃗)),(3,2,Nx,Ny)))
E = 1im * ε⁻¹_dot( fft( kx_tc(Hₜ,mn,mag), (2:3) ), real(epsi))
H = (-1im * om) * fft( tc(Hₜ,mn), (2:3) )
Htn = Hₜ/ sqrt(sum(abs2,Hₜ));
# WW = 0.5*(abs(_expect(eps+om*deps_dom,E)) + sum(abs2,H)) #real(dot(E,_dot(nng,E))) + ( om^2 * Nx * Ny )
WW = 0.5*(abs(_expect(eps,E)) + sum(abs2,H)) #real(dot(E,_dot(nng,E))) + ( om^2 * Nx * Ny )
@tullio P_z := conj(E)[1,ix,iy] * H[2,ix,iy] - conj(E)[2,ix,iy] * H[1,ix,iy]
@show ngz = WW / (2*real(P_z))
@show ngz_old = om / HMₖH(Htn,nngi,mag,mn)
@show ngz_old_nondisp = om / HMₖH(Htn,epsi,mag,mn)
@show ngz_load = ng1[i]

## resolve with mpb to check
include("mpb.jl")
omind = 10
i = 3
num_bands=4
om = ωs_opt5[omind]
geom = rwg(ps_opt5[i])
data_path = joinpath(homedir(),"data","OptiMode","opt5_mpb")
dx3 =  δx(grid) * δy(grid)

kvals = find_k(om,geom,grid;data_path,num_bands)
@show neffs_mpb = kvals ./ om
@show n1[omind]
# Hv = [load_evecs(joinpath(data_path,"eigenvectors_b0$ib.h5")) for ib=1:num_bands]
ib=2    # band index of TE₀₀ mode
k_mpb = kvals[ib]


eps_mpb_perm,epsi_mpb_perm,_ = load_epsilon(joinpath(data_path,"-epsilon.h5"))
eps_mpb = copy(eps_mpb_perm)
epsi_mpb = copy(epsi_mpb_perm)
eps_mpb = permutedims!(eps_mpb,eps_mpb_perm,(1,2,4,3))
epsi_mpb = permutedims!(epsi_mpb,epsi_mpb_perm,(1,2,4,3))
E_mpb = copy(permutedims(load_field(joinpath(data_path,"-e.k01.b0$ib.h5")),(1,3,2)))
H_mpb = copy(permutedims(load_field(joinpath(data_path,"-h.k01.b0$ib.h5")),(1,3,2)))
# check normalization
@assert real(_expect(eps_mpb,E_mpb))*dx3 ≈ 1.0
@assert sum(abs2,H_mpb)*dx3 ≈ 1.0
W_mpb_nondisp = 0.5*(abs(_expect(eps_mpb,E_mpb)) + sum(abs2,H_mpb)) * dx3
W_mpb = 0.5*(abs(_expect(eps_mpb+om*deps_dom,E_mpb)) + sum(abs2,H_mpb)) * dx3
P_mpb = real(_sum_cross_z(E_mpb,H_mpb)) * dx3
ng_nondisp_mpb = W_mpb_nondisp/P_mpb
ng_mpb =   W_mpb/P_mpb
##
ib=2
evecs_fname = "evecs.b0$ib.h5"
Hₜ_mpb = reshape(permutedims(load_evecs(joinpath(data_path,evecs_fname)),(3,2,1))[2,:,:],(2,128,128))
# Hₜ_mpb = reshape(permutedims(load_evecs(joinpath(data_path,evecs_fname))[:,:,ib],(2,1)),(2,128,128))
mag, m⃗, n⃗ = mag_m_n(k_mpb,grid);
mn = copy(reshape(reinterpret(Float64,hcat.(m⃗,n⃗)),(3,2,Nx,Ny)));
Hₜ_mpb2 = -kx_ct( ifft( E_mpb, (2:3) ), mn, mag) * sqrt( (pi/om)^3 / 2 )
@show Hₜ_mpb[argmax(abs2.(Hₜ_mpb))]
@show Hₜ_mpb2[argmax(abs2.(Hₜ_mpb2))]
@show maximum(abs,Hₜ_mpb)/ maximum(abs,Hₜ_mpb2)
@show sum(abs2,Hₜ_mpb)
@show sum_Ht2 = sum(abs2,Hₜ_mpb2) #* 2*Nx*Ny*dx3
##
omega, kdir, neffs, ngs, band_idx, x_frac, y_frac, z_frac = parse_findk_log()
##
include("mpb.jl")
omind = 10
i = 3
om = ωs_opt5[omind]
geom = rwg(ps_opt5[i])
data_path = joinpath(homedir(),"data","OptiMode","opt5_mpb")
k_dir=[0.,0.,1.]
num_bands=4

band_func = (
    mpb.fix_efield_phase,
    mpb.output_efield,
    mpb.output_hfield,
    py"output_dfield_energy",
    py"output_evecs",
    mpb.display_group_velocities,
)
parity=mp.NO_PARITY
n_guess_factor=0.9
allow_overwrite=false
logpath=DEFAULT_MPB_LOGPATH
init_path = pwd()
cd(data_path)
λ = inv.(ω)
nω = length(ω)
n_min, n_max = extrema(( n_range(geom;λ=λ[1])...,n_range(geom;λ=λ[nω])...))
n_guess = n_guess_factor * n_max
evecs_fname = @sprintf "evecs.b%02i.h5" num_bands
kvals = zeros(Float64,(num_bands,nω))
ms = open(logpath,"w") do logfile
    return redirect_stdout(logfile) do
        ms = ms_mpb(; geometry_lattice = lat_mpb(grid),
            geometry = mpGeom(geom;λ=λ[1]),
            resolution = res_mpb(grid),
            dimensions = 2,
            filename_prefix="",
            num_bands,
        )
        ms.output_epsilon()
        k_init = ms.find_k(
            parity,         # parity (meep parity object)
            ω[1],                    # ω at which to solve for k
            1,                 # band_min (find k(ω) for bands
            ms.num_bands,                 # band_max  band_min:band_max)
            mp.Vector3(k_dir...),     # k direction to search
            ms.tolerance,             # fractional k error tolerance
            n_guess * ω[1],              # kmag_guess, |k| estimate
            n_min * ω[1],                # kmag_min (find k in range
            n_max * ω[1],               # kmag_max  kmag_min:kmag_max)
            band_func...
        )
        kvals[:,1] = k_init
        return ms
    end
end
copy_findk_single_data(1,1:ms.num_bands;logpath,allow_overwrite)
i = 3
kvi = open(logpath,"w") do logfile
    return redirect_stdout(logfile) do
        ms.geometry = mpGeom(geom,λ=λ[i])
        ms.init_params(parity,false)
        # ms.init_params(parity,evecs_fname)
        kvals[:,i] = ms.find_k(
            parity,         # parity (meep parity object)
            ω[i],                    # ω at which to solve for k
            1,                 # band_min (find k(ω) for bands
            ms.num_bands,                 # band_max  band_min:band_max)
            mp.Vector3(k_dir...),     # k direction to search
            ms.tolerance,             # fractional k error tolerance
            py"$kvals[:,$i-1].astype(list)",      # kmag_guess, |k| estimate
            n_min * ω[i],                # kmag_min (find k in range
            n_max * ω[i],               # kmag_max  kmag_min:kmag_max)
            band_func...
        )
        #     end
        # end
        
        # return ms
        return kvals[:,i]
    end
end
# copy_findk_single_data(i,1:ms.num_bands;logpath,allow_overwrite)
evecs1 = ms.get_eigenvectors(1,num_bands)
evecs_fname = @sprintf "evecs.f%02i.b%02i.h5" i num_bands
ms.load_eigenvectors(evecs_fname)
evecs2 = ms.get_eigenvectors(1,num_bands)
##
ks_opt5 = find_k(ωs_opt5,geom,grid;data_path,num_bands)

##
omega, kdir, neffs, ngs, band_idx, x_frac, y_frac, z_frac = parse_findk_logs(1:20)


##
using FiniteDiff
using ForwardDiff
using Zygote
gradFM(fn,in) 			= 	ForwardDiff.gradient(fn,in)
gradFD(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_gradient(fn,in;relstep=rs)
derivFM(fn,in) 			= 	ForwardDiff.gradient(fn,in)
derivFD(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_derivative(fn,in;relstep=rs)

function f_neff_mpb(oo;ib=2)
    kvals = find_k(oo,geom,grid;data_path,num_bands)
    neff_mpb = kvals[ib]/oo
    return neff_mpb
end

function f_k_mpb(oo;ib=2)
    kvals = find_k(oo,geom,grid;data_path,num_bands)
    return kvals[ib]
end

f_eps(omg) = smooth(omg,ps_opt5[i],(:fεs,:fεs),[false,true],rwg,grid,kottke_smoothing)[1]

function f_ng_mpb(oo;ib=2)
    # eps,epsi = smooth(ω[omind],pp,(:fεs,:fεs),[false,true],rwg,grid,kottke_smoothing);
    deps_dom = ForwardDiff.derivative(x->f_eps(x,pp),oo);
    kvals = find_k(oo,geom,grid;data_path,num_bands)
    # ib=2    # band index of TE₀₀ mode
    eps_mpb = permutedims(load_epsilon(joinpath(data_path,"-epsilon.h5"))[1],(1,2,4,3))
    # epsi_mpb = permutedims!(epsi_mpb,epsi_mpb_perm,(1,2,4,3))
    E_mpb = permutedims(load_field(joinpath(data_path,"-e.k01.b0$ib.h5")),(1,3,2))
    H_mpb = permutedims(load_field(joinpath(data_path,"-h.k01.b0$ib.h5")),(1,3,2))
    W_mpb = 0.5*(abs(_expect(eps_mpb+oo*deps_dom,E_mpb)) + sum(abs2,H_mpb)) * dx3
    P_mpb = real(_sum_cross_z(E_mpb,H_mpb)) * dx3
    ng_mpb =   W_mpb/P_mpb
    return ng_mpb
end

function f_ng_mpb(om,k,E,eps)
    # eps,epsi = smooth(ω[omind],pp,(:fεs,:fεs),[false,true],rwg,grid,kottke_smoothing);
    deps_dom = ForwardDiff.derivative(x->f_eps(x,pp),om);

    W = 0.5*(abs(_expect(eps+om*deps_dom,E)) + sum(abs2,H)) * dx3
    P = real(_sum_cross_z(E,H)) * dx3
    ng =   W/P
    return ng
end

@show dneff_dom_FD = derivFD(f_neff_mpb,om)
@show dk_dom_FD = derivFD(f_k_mpb,om,rs=1e-3)
@show dng_dom_FD = derivFD(f_ng_mpb,om,rs=1e-3)

##
E⃗_mpb = [SVector{3,ComplexF64}(E_mpb[:,ix,iy]...) for ix=1:Nx,iy=1:Ny]
H⃗_mpb = [SVector{3,ComplexF64}(H_mpb[:,ix,iy]...) for ix=1:Nx,iy=1:Ny]
ẑ = SVector(0.,0.,1.)
P_mpb2 = 2 * sum(real.(dot.(cross.(E⃗_mpb,H⃗_mpb),(ẑ,)))) * dx3



ms =  ms_mpb(geom,grid;λ=inv(om),num_bands)

##
using Tullio
# first-order (linear) vector-tensor-vector muliplication (three element dot product)
function _3dot(v₂::AbstractVector,χ::AbstractArray{T,2},v₁::AbstractVector) where T<:Real
	@tullio out := conj(v₂)[i] * χ[i,j] * v₁[j]
end

function _3dot(v₂::AbstractArray{Complex{T},3},χ::AbstractArray{T,4},v₁::AbstractArray{Complex{T},3}) where T<:Real
	@tullio out[ix,iy] := conj(v₂)[i,ix,iy] * χ[i,j,ix,iy] * v₁[j,ix,iy]
end

function _3dot(v₂::AbstractArray{Complex{T},4},χ::AbstractArray{T,5},v₁::AbstractArray{Complex{T},4}) where T<:Real
	@tullio out[ix,iy,iz] := conj(v₂)[i,ix,iy,iz] * χ[i,j,k,ix,iy,iz] * v₁[j,ix,iy,iz]
end

# expectation value/inner product of vector field over tensor 
function _expect(χ::AbstractArray{T,2},v₁::AbstractVector) where T<:Real
	@tullio out := conj(v₁)[i] * χ[i,j] * v₁[j]
end

function _expect(χ::AbstractArray{T,4},v₁::AbstractArray{Complex{T},3}) where T<:Real
	@tullio out := conj(v₁)[i,ix,iy] * χ[i,j,ix,iy] * v₁[j,ix,iy] 
end

function _expect(χ::AbstractArray{T,5},v₁::AbstractArray{Complex{T},4}) where T<:Real
	@tullio out := conj(v₁)[i,ix,iy,iz] * χ[i,j,k,ix,iy,iz] * v₁[j,ix,iy,iz]
end

##
fig = Figure(resolution=res,backgroundcolor=:transparent)
ax = fig[1,1] = Axis(
    fig,
    backgroundcolor=:transparent,
    xlims = xlims,
    ylims = ylims,
    xlabel="x (μm)",
    ylabel="y (μm)",
)
hmI1 = heatmap!(ax,x(grid),y(grid),I1;colormap=cmap_I)
plot_shapes(rwg(pp),ax,mat_legend=false,frame_only=true,strokecolor=frame_color) 
xlims!(ax,xlims)
ylims!(ax,ylims)
ax.aspect = DataAspect()
hidedecorations!(ax)
fig

## Code to generate paramter evolution fig. from abstract (for reference)
ps_opt5M = hcat(ps_opt5...)'
ws_opt5 = ps_opt5M[:,1]
ts_opt5 = ps_opt5M[:,2]
rpes_opt5 =  ps_opt5M[:,3]
swas_opt5 =  ps_opt5M[:,4]
ipc_theme = Theme(fontsize=16)
set_theme!(ipc_theme)
clrs = [logocolors[:blue],logocolors[:green],:black]
labels = ["wₜₒₚ [μm]", "tₗₙ [μm]", "etch frac", "sw. angle [rad]"]

fig = Figure(resolution=(340,310))
ax = fig[1,1] = Axis(fig,
    yaxisposition=:right,
    ygridvisible=false,
    yminorgridvisible=false,
    ylabel="parameter value",
)
axf = fig[1,1] = Axis(fig,xticks=1:6,xlabel="optimization step",ylabel = "cost function Σ (Δng)²",)
sls = [scatterlines!(ax,1:6,p;color=clr,markercolor=clr,marksersize=1.5,strokecolor=clr,label=lbl) for (p,lbl,clr) in zip([ws_opt5,ts_opt5,rpes_opt5,swas_opt5],labels,logocolors)]
slf = scatterlines!(axf,1:6,fs_opt5,color=:black,marker=:rect,markercolor=:black,linewidth=2,markersize=6)
trim!(fig.layout)
fig

# save("example_opt_params_v2.png", fig)
# save("example_opt_params_v2.svg", fig)