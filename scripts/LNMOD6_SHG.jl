using Distributed
pids = addprocs(6)
@show wp = CachingPool(workers()) #default_worker_pool()
# initialize worker processes
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
    include(joinpath(homedir(),"github","OptiMode","scripts","mpb.jl"))
end


using GLMakie
using FFTW
using Tullio
using Rotations: RotY, MRP
using ForwardDiff
using Colors
import Colors: JULIA_LOGO_COLORS
using ChainRules
using FiniteDiff
using ForwardDiff
using FiniteDifferences
using Zygote
# using Printf
logocolors = JULIA_LOGO_COLORS

# function defs

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

include("mpb.jl")


function E_relpower_xyz(eps::AbstractArray{T,4},E::AbstractArray{Complex{T},3}) where T<:Real
    return normalize( real( @tullio E_abspower_xyz[i] := conj(E)[i,ix,iy] * eps[i,j,ix,iy] * E[j,ix,iy] ))
end

function E_relpower_xyz(eps::AbstractArray{T,5},E::AbstractArray{Complex{T},4}) where T<:Real
    return normalize( real( @tullio E_abspower_xyz[i] := conj(E)[i,ix,iy,iz] * eps[i,j,ix,iy,iz] * E[j,ix,iy,iz] ))
end

"""
Return slices (views) of E-field component `component_idx` along each axis intersecting the point at Cartesian index `pos_idx`
"""
function Eslices(E,component_idx,pos_idx::CartesianIndex{2})
    return ( view(E,component_idx,:,pos_idx[2]), view(E,component_idx,pos_idx[1],:) ) 
end

"""
Return slices (views) of E-field component `component_idx` along each axis intersecting the point at Cartesian index `pos_idx`
"""
function Eslices(E,component_idx,pos_idx::CartesianIndex{3})
    return ( view(E,component_idx,:,pos_idx[2],pos_idx[3]), view(E,component_idx,pos_idx[1],:,pos_idx[3]), view(E,component_idx,pos_idx[1],pos_idx[2],:) ) 
end

"""
Count the number of E-field zero crossings along each grid axis intersecting the peak E-field intensity maximum for a given field component `component_idx`.
This assumes that the input E-field amplitude is normalized to and real at (phase=0) the peak amplitude component-wise amplitude. 
Slices of the specified E-field component are cropped at the points where the component magnitude drops below `rel_amp_min` to avoid counting 
zero-crossings in numerical noise where the E-field magnitude is negligible.

This is useful when filtering for a specific mode order (eg. TE₀₀ or TM₂₁) in the presence of mode-crossings.
"""
function count_E_nodes(E,eps,component_idx;rel_amp_min=0.1) 
    peak_idx = argmax(real(_3dot(E,eps,E)[component_idx,..]))
    E_slcs = Eslices(real(E),component_idx,peak_idx)
    node_counts = map(E_slcs) do E_slice
        min_idx = findfirst(x->(x>rel_amp_min),abs.(E_slice))
        max_idx = findlast(x->(x>rel_amp_min),abs.(E_slice))
        n_zero_xing = sum(abs.(diff(sign.(E_slice[min_idx:max_idx]))))
        return n_zero_xing
    end
    return node_counts
end

"""
utility function for debugging count_E_nodes function.
inspect output with something like: 
    slcs_inds = windowed_E_slices(E,eps,component_idx;rel_amp_min=0.1)
    ind_min_xslc, ind_max_xslc = slcs_inds[1][2:3]
    E_xslc = slcs_inds[1][1]
    scatterlines(x(grid)[ind_min_xslc:ind_max_xslc],real(E_xslc[ind_min_xslc:ind_max_xslc]))
"""
function windowed_E_slices(E,eps,component_idx;rel_amp_min=0.1) 
    peak_idx = argmax(real(_3dot(E,eps,E)[component_idx,..]))
    E_slcs = Eslices(real(E),component_idx,peak_idx)
    slices_and_window_inds = map(E_slcs) do E_slice
        min_idx = findfirst(x->(x>rel_amp_min),abs.(E_slice))
        max_idx = findlast(x->(x>rel_amp_min),abs.(E_slice))
        # n_zero_xing = sum(abs.(diff(sign.(E_slice[min_idx:max_idx]))))
        return E_slice, min_idx, max_idx
    end
    return slices_and_window_inds
end

function inspect_slcs_inds(slcs_inds;ax=nothing)
    if isnothing(ax)
        fig = Figure()
        ax = fig[1,1] = Axis(fig)
        ret = fig
    else
        ret = ax
    end
    map(slcs_inds,(x(grid),y(grid)),(logocolors[:red],logocolors[:blue])) do slc_ind, ax_coords, ln_clr
        E_slc = slc_ind[1]
        ind_min_slc, ind_max_slc = slc_ind[2:3]
        scatterlines!(ax,ax_coords[ind_min_slc:ind_max_slc],real(E_slc[ind_min_slc:ind_max_slc]),color=ln_clr)
    end
    return ret
end

function mode_idx(Es,eps;pol_idx=1,mode_order=(0,0),rel_amp_min=0.1)
    maxval, maxidx = findmax(Es) do EE
        EEpwr = E_relpower_xyz(eps,EE)
        EEpol_axind = argmax(EEpwr)
        correct_pol = Float64(isequal(EEpol_axind,pol_idx))
        correct_order = Float64(isequal(
            count_E_nodes(EE,eps,EEpol_axind;rel_amp_min),
            mode_order,
        ))
        return correct_pol * correct_order * EEpwr
    end
    return maxidx
end


TE_filter = (ms,ω²H)->E_relpower_xyz(ms,ω²H)[1]>0.7
TM_filter = (ms,ω²H)->E_relpower_xyz(ms,ω²H)[2]>0.7
oddX_filter = (ms,αX)->sum(abs2,𝓟x̄(ms.grid)*αX[2])>0.7
evenX_filter = (ms,αX)->sum(abs2,𝓟x(ms.grid)*αX[2])>0.7

function Eperp_max(E::AbstractArray{T,3}) where T
    Eperp = view(E,1:2,:,:)
    maximum(abs,Eperp)
end

function Eperp_max(E::AbstractArray{T,4}) where T
    Eperp = view(E,1:2,:,:,:)
    maximum(abs,Eperp)
end

𝓐(n,ng,E) = inv( n * ng * Eperp_max(E)^2)

chi2_def = SArray{Tuple{3,3,3}}(zeros(Float64,(3,3,3)))


function proc_modes_shg(oms,pp,kk_ev;pol_idx=1,mode_order=(0,0),rel_amp_min=0.1)
    ks, evecs = kk_ev
    
    eps =  [copy(smooth(oo,pp,:fεs,false,geom_fn,grid).data) for oo in oms]
    epsi =  [copy(smooth(oo,pp,:fεs,true,geom_fn,grid).data) for oo in oms]
    deps_dom = [ ForwardDiff.derivative(oo->copy(getproperty(smooth(oo,pp,(:fεs,:fεs),[false,true],geom_fn,grid,kottke_smoothing)[1],:data)),om) for om in oms ]
    mags_mns = [mag_mn(kk,grid) for kk in ks]
    Es = [-1im * ε⁻¹_dot(fft(kx_tc(evecs[omidx,bndidx],mags_mns[omidx,bndidx][2],mags_mns[omidx,bndidx][1]),(2:3)),epsi[omidx]) for omidx=1:length(oms), bndidx=1:num_bands]
    Enorms =  [ EE[argmax(abs2.(EE))] for EE in Es ]
    Es = Es ./ Enorms
    
    ngs = [ group_index(ks[fidx,bidx],evecs[fidx,bidx],oms[fidx],epsi[fidx],deps_dom[fidx],grid) for fidx=1:length(oms),bidx=1:num_bands ]
    neffs = ks ./ repeat(oms,1,num_bands) 

    bnd_inds = [mode_idx(Es[omidx,:],eps[omidx];pol_idx,mode_order,rel_amp_min) for omidx=1:length(oms)]

    nω = Int(length(oms) // 2)  # oms has all first and second harmonic frequencies, nω is just the number of first harmonic frequencies
    
    # neffs_fund  =   @view neffs[1:nω,:]
    # ngs_fund    =   @view ngs[1:nω,:]
    # Es_fund     =   @view Es[1:nω,:]
    # eps_fund    =   @view eps[1:nω]
    # bnd_inds_fund    =   @view bnd_inds[1:nω]

    # neffs_sh   =   @view neffs[(nω+1):(2*nω),:]
    # ngs_sh     =   @view ngs[(nω+1):(2*nω),:]
    # Es_sh      =   @view Es[(nω+1):(2*nω),:]
    # eps_sh     =   @view eps[(nω+1):(2*nω)]
    # bnd_inds_sh = @view bnd_inds[(nω+1):(2*nω)]

    # return neffs_fund, ngs_fund, Es_fund, eps_fund, bnd_inds_fund, neffs_sh, ngs_sh, Es_sh, eps_sh, bnd_inds_sh

    neffs_fund  =    [ neffs[idx,bnd_inds[idx]] for idx=1:nω]
    ngs_fund    =    [ ngs[idx,bnd_inds[idx]] for idx=1:nω]
    Es_fund     =    [ Es[idx,bnd_inds[idx]] for idx=1:nω]
    eps_fund    =    @view eps[1:nω]

    neffs_sh   =    [ neffs[idx,bnd_inds[idx]] for idx=(nω+1):(2*nω) ]
    ngs_sh     =    [ ngs[idx,bnd_inds[idx]] for idx=(nω+1):(2*nω) ]
    Es_sh      =    [ Es[idx,bnd_inds[idx]] for idx=(nω+1):(2*nω) ]
    eps_sh     =    @view eps[(nω+1):(2*nω)]
    
    λs_sh      =    inv.(ωs_in[(nω+1):(2*nω)])
    Λs         =    λs_sh ./ ( neffs_sh .- neffs_fund )

    As_fund = 𝓐.(neffs_fund, ngs_fund, Es_fund)
    As_sh = 𝓐.(neffs_sh, ngs_sh, Es_sh)
    Ês_fund = [Es_fund[i] * sqrt(As_fund[i] * neffs_fund[i] * ngs_fund[i]) for i=1:length(Es_fund)]
    Ês_sh = [Es_sh[i] * sqrt(As_sh[i] * neffs_sh[i] * ngs_sh[i]) for i=1:length(Es_sh)]
    𝓐₁₂₃ = ( As_sh .* As_fund.^2  ).^(1.0/3.0)
    χ⁽²⁾ = -1 .* smooth([(omm,omm,2omm) for omm in ωs_in[1:length(Es_fund)]],ps[1],:fχ⁽²⁾s,false,chi2_def,geom_fn,grid,volfrac_smoothing)
    χ⁽²⁾_rel = [abs.(chi2) / maximum(abs.(chi2)) for chi2 in χ⁽²⁾]
    χ⁽²⁾xxx	=	[ view(chi2,1,1,1,:,:)  for chi2 in χ⁽²⁾]
    χ⁽²⁾xxx_LN	=	[ chi2xxx[argmax(abs.(chi2xxx))] for chi2xxx in χ⁽²⁾xxx]
    # χ⁽²⁾xxx_rel	=	[ chi2xxx ./ chi2xxx_LN  for (chi2xxx,chi2xxx_LN) in zip(χ⁽²⁾xxx,χ⁽²⁾xxx_LN)]

    𝓞 = [ real( sum( dot( conj.(Ês_sh[ind]), _dot(χ⁽²⁾_rel[ind],Ês_fund[ind],Ês_fund[ind]) ) ) ) / 𝓐₁₂₃[ind] * δ(grid) for ind=1:length(Es_sh)] #
    𝓞_rel = 𝓞/maximum(𝓞)

    χ⁽²⁾xxx_LN_cmV⁻¹ = χ⁽²⁾xxx_LN .* 1e-10
    deff_cmV⁻¹ = χ⁽²⁾xxx_LN_cmV⁻¹.* (2/π)
    𝓐₁₂₃_cm² = 𝓐₁₂₃ .* 1e-8
    c₀_cm = 3e10
    ε₀_Fcm⁻¹ = 8.854e-14 # F/cm, ε₀	= 8.854e-12 F/m
    λ_cm = inv.(ωs_fund) .* 1e-4
    om_radHz = 2π .* (c₀_cm ./ λ_cm)
    κ_sqrtW⁻¹cm⁻¹ = ( sqrt(2) * om_radHz .* deff_cmV⁻¹ .* 𝓞) ./ sqrt.( c₀_cm^3 * ε₀_Fcm⁻¹ .* neffs_sh .* neffs_fund.^2 .* 𝓐₁₂₃_cm²)
    κ²_W⁻¹cm⁻² = abs2.(κ_sqrtW⁻¹cm⁻¹)
    κ²_pctW⁻¹cm⁻² = 100.0 .* κ²_W⁻¹cm⁻²
    # calculate phase matching bandwidth based on group index mismatch
    Δng         =       ngs_sh .- ngs_fund
    Δf_Hz_cm    =       c₀_cm ./ ( 4.0 .* abs.(Δng) )   # Phase matching bandwidth around first harmonic in Hz⋅cm (or in Hz of a 1cm waveguide)
    f_Hz        =       c₀_cm ./ λ_cm                   # fundamental frequencies in Hz
    ∂λ_∂f       =       -c₀_cm ./ (f_Hz.^2)       # partial derivative of each fundatmental vacuum wavelength w.r.t. frequency
    nm_per_cm   =       1e7
    Δλ_nm_cm    =       abs.( ∂λ_∂f .*  Δf_Hz_cm .* nm_per_cm )  # Phase matching bandwidth around first harmonic in nm⋅cm (or in nm of a 1cm waveguide)

    return neffs_fund, ngs_fund, Es_fund, eps_fund, neffs_sh, ngs_sh, Es_sh, eps_sh, Λs, κ²_pctW⁻¹cm⁻², Δλ_nm_cm
end

function scatterlines_err!(ax,x,y,y_lw,y_up;color=logocolors[:red],fill_alpha=0.2,linewidth=2,markersize=2,label=nothing)
    sl  =   scatterlines!(
        ax,
        x,
        y,
        color=color,
        linewidth=linewidth,
		markersize=markersize,
		markercolor=color,
		strokecolor=color,
        label=label,
    )
    bnd =   band!(
        ax,
        x,
        y_up,
        y_lw,
        color=(color,fill_alpha),
    )
    return sl,bnd
end

function scatterlines_err(x,y,y_lw,y_up;color=logocolors[:red],fill_alpha=0.2,linewidth=2,markersize=2,label=nothing)
    fig = Figure()
    ax = fig[1,1] = Axis(fig)
    sl  =   scatterlines!(
        ax,
        x,
        y,
        color=color,
        linewidth=linewidth,
		markersize=markersize,
		markercolor=color,
		strokecolor=color,
        label=label,
    )
    bnd =   band!(
        ax,
        x,
        y_up,
        y_lw,
        color=(color,fill_alpha),
    )
    return fig
end


##

# ## sw1 parameters (best estimates)
# @everywhere begin
#     nω                  =   15
#     λ_min               =   1.4
#     λ_max               =   1.6
#     w                   =   1.225   # etched SN ridge width: 1.225um ± 25nm
#     t_core              =   0.17    # SiN ridge thickness: 170nm ± 10nm with 30-40nm of exposed SiO2 (HSQ) on top of unetched ridge
#     t_slab              =   0.20    # LN slab thickness:   200nm ± 10nm
#     δw                  =   0.025   # etched SN ridge width: 1.225um ± 25nm
#     δt_core             =   0.01    # SiN ridge thickness: 170nm ± 10nm with 30-40nm of exposed SiO2 (HSQ) on top of unetched ridge
#     δt_slab             =   0.01    # LN slab thickness:   200nm ± 10nm

#     mat_core            =   Si₃N₄N
#     mat_slab            =   LNxN
#     mat_subs            =   SiO₂N

#     Δx,Δy,Δz,Nx,Ny,Nz   =   8.0, 4.0, 1.0, 128, 128, 1;
#     edge_gap            =   0.5
#     num_bands           =   2

#     ds_dir              =   "SNwg_LNslab"
#     filename_prefix     =   ("sw1c", "sw1p", "sw1n") # need to rename "test2" to "sw1c"


#     ωs_fund             =   range(inv(λ_max),inv(λ_min),nω)
#     ωs_shg              =   2. .* ωs_fund
#     ωs_in               =   vcat(ωs_fund,ωs_shg)
#     ps                  =   ([w, t_core, t_slab], [w+δw, t_core+δt_core, t_slab+δt_slab], [w-δw, t_core-δt_core, t_slab-δt_slab])
#     ds_path             =   joinpath(homedir(),"data","OptiMode",ds_dir)
#     λs_fund             =   inv.(ωs_fund)
#     λs_shg              =   inv.(ωs_shg)

#     band_min            =   1
#     band_max            =   num_bands
#     geom_fn(x)          =   ridge_wg_slab_loaded(x[1],x[2],0.,x[3],edge_gap,mat_core,mat_slab,mat_subs,Δx,Δy)
#     grid                =   Grid(Δx,Δy,Nx,Ny)
#     k_dir               =   [0.,0.,1.]
# end
## sw1: solve for modes using best-guess parameters as well as parameters +/- uncertainties that give the smallest and largest waveguides

# ks_evecs = [find_k(ωs_in,ps[pidx],geom_fn,grid;num_bands=num_bands,data_path=ds_path,filename_prefix=filename_prefix[pidx]) for pidx=1:3]  # mode solve for a vector of input frequencies

## sw2 parameters (best estimates)

# @everywhere begin
#     nω                  =   20
#     λ_min               =   1.0
#     λ_max               =   1.6
#     w                   =   1.225   # etched SN ridge width: 1.225um ± 25nm
#     t_core              =   0.17    # SiN ridge thickness: 170nm ± 10nm with 30-40nm of exposed SiO2 (HSQ) on top of unetched ridge
#     t_slab              =   0.20    # LN slab thickness:   200nm ± 10nm
#     δw                  =   0.025   # etched SN ridge width: 1.225um ± 25nm
#     δt_core             =   0.01    # SiN ridge thickness: 170nm ± 10nm with 30-40nm of exposed SiO2 (HSQ) on top of unetched ridge
#     δt_slab             =   0.01    # LN slab thickness:   200nm ± 10nm
#     t_mask              =   0.03    # remaining HSQ etch mask thickness: 30nm 

#     mat_core            =   Si₃N₄N
#     mat_slab            =   LNxN
#     mat_subs            =   SiO₂N
#     mat_mask            =   SiO₂N

#     Δx,Δy,Δz,Nx,Ny,Nz   =   12.0, 4.0, 1.0, 256, 128, 1;
#     edge_gap            =   0.5
#     num_bands           =   8

#     ds_dir              =   "SNwg_LNslab"
#     filename_prefix     =   ("sw2c", "sw2p", "sw2n")


#     ωs_fund             =   range(inv(λ_max),inv(λ_min),nω)
#     ωs_shg              =   2. .* ωs_fund
#     ωs_in               =   vcat(ωs_fund,ωs_shg)
#     ps                  =   ([w, t_core, t_slab], [w+δw, t_core+δt_core, t_slab+δt_slab], [w-δw, t_core-δt_core, t_slab-δt_slab])
#     ds_path             =   joinpath(homedir(),"data","OptiMode",ds_dir)
#     λs_fund             =   inv.(ωs_fund)
#     λs_shg              =   inv.(ωs_shg)
#     λs_sh               =   inv.(ωs_shg)

#     band_min            =   1
#     band_max            =   num_bands
#     # geom_fn(x)          =   ridge_wg_slab_loaded(x[1],x[2],0.,x[3],edge_gap,mat_core,mat_slab,mat_subs,Δx,Δy)
#     grid                =   Grid(Δx,Δy,Nx,Ny)
#     k_dir               =   [0.,0.,1.]

#     function LNMOD6_geom(wₜₒₚ::Real,t_core::Real,t_slab::Real;θ=0.0,t_mask,edge_gap,mat_core,mat_slab,mat_subs,mat_mask,Δx,Δy) #::Geometry{2}
#         t_subs = (Δy -t_core - edge_gap )/2. - t_slab
#         c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
#         c_slab_y = -Δy/2. + edge_gap/2. + t_subs + t_slab/2.
#         c_mask_y = -Δy/2. + edge_gap/2. + t_subs + t_slab + t_core + t_mask/2.
#         wt_half = wₜₒₚ / 2
#         wb_half = wt_half + ( t_core * tan(θ) )
#         tc_half = t_core / 2
#         verts = SMatrix{4,2}(   wt_half,     -wt_half,     -wb_half,    wb_half, tc_half,     tc_half,    -tc_half,      -tc_half )
#         core = GeometryPrimitives.Polygon(					                        # Instantiate 2D polygon, here a trapazoid
#                         # SMatrix{4,2}(verts),			                            # v: polygon vertices in counter-clockwise order
#                         verts,
#                         mat_core,					                                    # data: any type, data associated with box shape
#                     )
#         ax = [      1.     0.
#                     0.     1.      ]
#         b_mask = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
#                         [0. , c_mask_y],           	# c: center
#                         [wₜₒₚ, t_mask],	            # r: "radii" (half span of each axis)
#                         ax,	    		        	# axes: box axes
#                         mat_mask,					# data: any type, data associated with box shape
#                     )
#         b_slab = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
#                     [0. , c_slab_y],           	# c: center
#                     [Δx - edge_gap, t_slab ],	# r: "radii" (half span of each axis)
#                     ax,	    		        	# axes: box axes
#                     mat_slab,					 # data: any type, data associated with box shape
#                 )
#         b_subs = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
#                         [0. , c_subs_y],           	# c: center
#                         [Δx - edge_gap, t_subs ],	# r: "radii" (half span of each axis)
#                         ax,	    		        	# axes: box axes
#                         mat_subs,					 # data: any type, data associated with box shape
#                     )
#         return Geometry([core,b_slab,b_subs,b_mask])
#     end
#     geom_fn(x)          =   LNMOD6_geom(x[1],x[2],x[3];θ=0.0,t_mask=t_mask,edge_gap=edge_gap,mat_core=mat_core,mat_slab=mat_slab,mat_subs=mat_subs,mat_mask=mat_mask,Δx=Δx,Δy=Δy)
# end

## sw3 parameters (best estimates)

@everywhere begin
    nω                  =   20      # number of frequencies (wavelengths) solved for each harmonic
    λ_min               =   1.0     # minimum first vacuum wavelength (μm)
    λ_max               =   1.6     # minimum first vacuum wavelength (μm)
    w                   =   1.225   # etched SN ridge width: 1.225um ± 25nm
    t_core              =   0.17    # SiN ridge thickness: 170nm ± 10nm with 30-40nm of exposed SiO2 (HSQ) on top of unetched ridge
    t_slab              =   0.20    # LN slab thickness:   200nm ± 10nm
    δw                  =   0.025   # etched SN ridge width: 1.225um ± 25nm
    δt_core             =   0.01    # SiN ridge thickness: 170nm ± 10nm with 30-40nm of exposed SiO2 (HSQ) on top of unetched ridge
    δt_slab             =   0.01    # LN slab thickness:   200nm ± 10nm
    t_mask              =   0.03    # remaining HSQ etch mask thickness: 30nm 

    mat_core            =   Si₃N₄N
    mat_slab            =   LNxN
    mat_subs            =   SiO₂N
    mat_mask            =   SiO₂N

    Δx,Δy,Δz,Nx,Ny,Nz   =   12.0, 4.0, 1.0, 512, 256, 1;
    edge_gap            =   0.5
    num_bands           =   8

    ds_dir              =   "SNwg_LNslab"
    filename_prefix     =   ("sw3c", "sw3p", "sw3n")


    ωs_fund             =   range(inv(λ_max),inv(λ_min),nω)
    ωs_shg              =   2. .* ωs_fund
    ωs_in               =   vcat(ωs_fund,ωs_shg)
    ps                  =   ([w, t_core, t_slab], [w+δw, t_core+δt_core, t_slab+δt_slab], [w-δw, t_core-δt_core, t_slab-δt_slab])
    ds_path             =   joinpath(homedir(),"data","OptiMode",ds_dir)
    λs_fund             =   inv.(ωs_fund)
    λs_shg              =   inv.(ωs_shg)
    λs_sh               =   inv.(ωs_shg)

    band_min            =   1
    band_max            =   num_bands
    # geom_fn(x)          =   ridge_wg_slab_loaded(x[1],x[2],0.,x[3],edge_gap,mat_core,mat_slab,mat_subs,Δx,Δy)
    grid                =   Grid(Δx,Δy,Nx,Ny)
    k_dir               =   [0.,0.,1.]

    function LNMOD6_geom(wₜₒₚ::Real,t_core::Real,t_slab::Real;θ=0.0,t_mask,edge_gap,mat_core,mat_slab,mat_subs,mat_mask,Δx,Δy) #::Geometry{2}
        t_subs = (Δy -t_core - edge_gap )/2. - t_slab
        c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
        c_slab_y = -Δy/2. + edge_gap/2. + t_subs + t_slab/2.
        c_mask_y = -Δy/2. + edge_gap/2. + t_subs + t_slab + t_core + t_mask/2.
        wt_half = wₜₒₚ / 2
        wb_half = wt_half + ( t_core * tan(θ) )
        tc_half = t_core / 2
        verts = SMatrix{4,2}(   wt_half,     -wt_half,     -wb_half,    wb_half, tc_half,     tc_half,    -tc_half,      -tc_half )
        core = GeometryPrimitives.Polygon(					                        # Instantiate 2D polygon, here a trapazoid
                        # SMatrix{4,2}(verts),			                            # v: polygon vertices in counter-clockwise order
                        verts,
                        mat_core,					                                    # data: any type, data associated with box shape
                    )
        ax = [      1.     0.
                    0.     1.      ]
        b_mask = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
                        [0. , c_mask_y],           	# c: center
                        [wₜₒₚ, t_mask],	            # r: "radii" (half span of each axis)
                        ax,	    		        	# axes: box axes
                        mat_mask,					# data: any type, data associated with box shape
                    )
        b_slab = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
                    [0. , c_slab_y],           	# c: center
                    [Δx - edge_gap, t_slab ],	# r: "radii" (half span of each axis)
                    ax,	    		        	# axes: box axes
                    mat_slab,					 # data: any type, data associated with box shape
                )
        b_subs = GeometryPrimitives.Box(			# Instantiate N-D box, here N=2 (rectangle)
                        [0. , c_subs_y],           	# c: center
                        [Δx - edge_gap, t_subs ],	# r: "radii" (half span of each axis)
                        ax,	    		        	# axes: box axes
                        mat_subs,					 # data: any type, data associated with box shape
                    )
        return Geometry([core,b_slab,b_subs,b_mask])
    end
    geom_fn(x)          =   LNMOD6_geom(x[1],x[2],x[3];θ=0.0,t_mask=t_mask,edge_gap=edge_gap,mat_core=mat_core,mat_slab=mat_slab,mat_subs=mat_subs,mat_mask=mat_mask,Δx=Δx,Δy=Δy)
end

## solve for or load modes using each of parameters
filename_prefix     =   ("sw3c", "sw3p", "sw3n")
ps                  =   ([w, t_core, t_slab], [w+δw, t_core+δt_core, t_slab+δt_slab], [w-δw, t_core-δt_core, t_slab-δt_slab])
ks_evecs = [find_k(ωs_in,ps[pidx],geom_fn,grid;num_bands=num_bands,data_path=ds_path,filename_prefix=filename_prefix[pidx]) for pidx=1:3] 
##
p_sw3c, p_sw3p, p_sw3n = ps
ds_sw3c, ds_sw3p, ds_sw3n = (proc_modes_shg(ωs_in,ps[idx],ks_evecs[idx]) for idx=1:3)
ks_sw3c, evecs_sw3c = ks_evecs[1]
ks_sw3p, evecs_sw3p = ks_evecs[2]
ks_sw3n, evecs_sw3n = ks_evecs[3]
neffs_fund_sw3c, ngs_fund_sw3c, Es_fund_sw3c, eps_fund_sw3c, neffs_sh_sw3c, ngs_sh_sw3c, Es_sh_sw3c, eps_sh_sw3c, Λs_sw3c, κ²_pctW⁻¹cm⁻²_sw3c, Δλ_nm_cm_sw3c = ds_sw3c
neffs_fund_sw3p, ngs_fund_sw3p, Es_fund_sw3p, eps_fund_sw3p, neffs_sh_sw3p, ngs_sh_sw3p, Es_sh_sw3p, eps_sh_sw3p, Λs_sw3p, κ²_pctW⁻¹cm⁻²_sw3p, Δλ_nm_cm_sw3p = ds_sw3p
neffs_fund_sw3n, ngs_fund_sw3n, Es_fund_sw3n, eps_fund_sw3n, neffs_sh_sw3n, ngs_sh_sw3n, Es_sh_sw3n, eps_sh_sw3n, Λs_sw3n, κ²_pctW⁻¹cm⁻²_sw3n, Δλ_nm_cm_sw3n = ds_sw3n
# rename appropriate data for plotting
neffs_fund, ngs_fund, Es_fund, eps_fund, neffs_sh, ngs_sh, Es_sh, eps_sh, Λs, κ²_pctW⁻¹cm⁻², Δλ_nm_cm = ds_sw3c
neffs_fund_up, ngs_fund_up, Es_fund_up, eps_fund_up, neffs_sh_up, ngs_sh_up, Es_sh_up, eps_sh_up, Λs_up, κ²_pctW⁻¹cm⁻²_up, Δλ_nm_cm_up = ds_sw3p
neffs_fund_lw, ngs_fund_lw, Es_fund_lw, eps_fund_lw, neffs_sh_lw, ngs_sh_lw, Es_sh_lw, eps_sh_lw, Λs_lw, κ²_pctW⁻¹cm⁻²_lw, Δλ_nm_cm_lw = ds_sw3n

##
# p_sw2c, p_sw2p, p_sw2n = ps
# ds_sw2c, ds_sw2p, ds_sw2n = (proc_modes_shg(ωs_in,ps[idx],ks_evecs[idx]) for idx=1:3)
# ks_sw2c, evecs_sw2c = ks_evecs[1]
# ks_sw2p, evecs_sw2p = ks_evecs[2]
# ks_sw2n, evecs_sw2n = ks_evecs[3]
# neffs_fund_sw2c, ngs_fund_sw2c, Es_fund_sw2c, eps_fund_sw2c, neffs_sh_sw2c, ngs_sh_sw2c, Es_sh_sw2c, eps_sh_sw2c, Λs_sw2c, κ²_pctW⁻¹cm⁻²_sw2c, Δλ_nm_cm_sw2c = ds_sw2c
# neffs_fund_sw2p, ngs_fund_sw2p, Es_fund_sw2p, eps_fund_sw2p, neffs_sh_sw2p, ngs_sh_sw2p, Es_sh_sw2p, eps_sh_sw2p, Λs_sw2p, κ²_pctW⁻¹cm⁻²_sw2p, Δλ_nm_cm_sw2p = ds_sw2p
# neffs_fund_sw2n, ngs_fund_sw2n, Es_fund_sw2n, eps_fund_sw2n, neffs_sh_sw2n, ngs_sh_sw2n, Es_sh_sw2n, eps_sh_sw2n, Λs_sw2n, κ²_pctW⁻¹cm⁻²_sw2n, Δλ_nm_cm_sw2n = ds_sw2n
# # rename appropriate data for plotting
# neffs_fund, ngs_fund, Es_fund, eps_fund, neffs_sh, ngs_sh, Es_sh, eps_sh, Λs, κ²_pctW⁻¹cm⁻², Δλ_nm_cm = ds_sw2c
# neffs_fund_up, ngs_fund_up, Es_fund_up, eps_fund_up, neffs_sh_up, ngs_sh_up, Es_sh_up, eps_sh_up, Λs_up, κ²_pctW⁻¹cm⁻²_up, Δλ_nm_cm_up = ds_sw2p
# neffs_fund_lw, ngs_fund_lw, Es_fund_lw, eps_fund_lw, neffs_sh_lw, ngs_sh_lw, Es_sh_lw, eps_sh_lw, Λs_lw, κ²_pctW⁻¹cm⁻²_lw, Δλ_nm_cm_lw = ds_sw2n

##
omidx   =   20
bndidx  =   1
axidx   =   1
cmap_Ex =   :diverging_bkr_55_10_c35_n256
cmap_nx =   :viridis
clr_fund    =   logocolors[:red]
clr_sh     =   logocolors[:blue]
clr_Λ       =   logocolors[:green]
clr_κ²     =   logocolors[:purple]
clr_Δλ     =   :black
labels          =   ["nx @ ω","Ex @ ω","Ex @ 2ω"] #label.*label_base

xs = x(grid)
ys = y(grid)
xlim =  -2.0,   2.0     # Tuple(extrema(xs)) 
ylim =  -0.8,   0.8     # Tuple(extrema(ys)) 

fig             =   Figure()
ax_neffs        =   fig[1,1] = Axis(fig)
ax_Λs           =   fig[2,1] = Axis(fig)
ax_κ²s          =   fig[3,1] = Axis(fig)
ax_Δλs          =   fig[2,1] = Axis(fig, yticklabelcolor=clr_Δλ, ytickcolor=clr_Δλ, backgroundcolor=:transparent)
# hidexdecorations!(ax_Δλs)
# hidespines!(ax_Δλs)
# yaxis_right!(ax_Δλs)
ax_Δλs.yaxisposition = :right
ax_Δλs.yticklabelalign = (:left, :center)
ax_Δλs.xticklabelsvisible = false
ax_Δλs.xticklabelsvisible = false
ax_Δλs.xlabelvisible = false
linkxaxes!(ax_Λs, ax_Δλs)

ax_nxFund       =   fig[1,2] = Axis(fig)
ax_ExFund       =   fig[2,2] = Axis(fig)
ax_ExSHG        =   fig[3,2] = Axis(fig)
# cbax11  =   fig[1,1] = Axis(fig)

sl_neffs_fund,bnd_neffs_fund    =   scatterlines_err!(ax_neffs,λs_fund,neffs_fund,neffs_fund_lw,neffs_fund_up;color=clr_fund,fill_alpha=0.2,linewidth=2,markersize=4,label=nothing)
sl_neffs_sh,bnd_neffs_sh        =   scatterlines_err!(ax_neffs,λs_sh,neffs_sh,neffs_sh_lw,neffs_sh_up;color=clr_sh,fill_alpha=0.2,linewidth=2,markersize=4,label=nothing)
sl_ngs_fund,bnd_ngs_fund        =   scatterlines_err!(ax_neffs,λs_fund,ngs_fund,ngs_fund_lw,ngs_fund_up;color=clr_fund,fill_alpha=0.2,linewidth=2,markersize=4,label=nothing)
sl_ngs_sh,bnd_ngs_sh            =   scatterlines_err!(ax_neffs,λs_sh,ngs_sh,ngs_sh_lw,ngs_sh_up;color=clr_sh,fill_alpha=0.2,linewidth=2,markersize=4,label=nothing)
sl_Λs,bnd_Λs                    =   scatterlines_err!(ax_Λs,λs_fund,Λs,Λs_lw,Λs_up;color=clr_Λ,fill_alpha=0.2,linewidth=2,markersize=4,label=nothing)
sl_Δλs,bnd_Δλs                  =   scatterlines_err!(ax_Δλs,λs_fund,Δλ_nm_cm,Δλ_nm_cm_lw,Δλ_nm_cm_up;color=clr_Δλ,fill_alpha=0.2,linewidth=2,markersize=4,label=nothing)
sl_κ²s,bnd_κ²s                  =   scatterlines_err!(ax_κ²s,λs_fund,κ²_pctW⁻¹cm⁻²,κ²_pctW⁻¹cm⁻²_lw,κ²_pctW⁻¹cm⁻²_up;color=clr_κ²,fill_alpha=0.2,linewidth=2,markersize=4,label=nothing)

magmax_nxFund   =   @views maximum(abs,sqrt.(real(eps_fund[omidx][axidx,axidx,:,:])))
magmax_ExFund   =   @views maximum(abs,Es_fund[omidx])
magmax_ExSHG    =   @views maximum(abs,Es_sh[omidx])

hm_nxFund   =   heatmap!(
    ax_nxFund,
    xs,
    ys,
    sqrt.(real(eps_fund[omidx][axidx,axidx,:,:])),
    colormap=cmap_nx,label=labels[1],
    colorrange=(1.0,magmax_nxFund),
)
hm_ExFund   =   heatmap!(
    ax_ExFund,
    xs,
    ys,
    real(Es_fund[omidx][axidx,:,:]),
    colormap=cmap_Ex,label=labels[2],
    colorrange=(-magmax_ExFund,magmax_ExFund),
)
hm_ExSHG    =   heatmap!(
    ax_ExSHG,
    xs,
    ys,
    -real(Es_sh[omidx][axidx,:,:]),
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

LNMOD6_QPM_meas = [ (1.52,3.0) ]
LNMOD6_poling_periods = [ 2.8, 3.0, 3.3, 2.9, 3.15 ]

fig = Figure()
ax_Λs           =   fig[1,1] = Axis(fig,
    xlabel="fundamental wavelength (μm)",
    ylabel="poling period (μm)"
)
sl_Λs,bnd_Λs                    =   scatterlines_err!(ax_Λs,λs_fund,Λs,Λs_lw,Λs_up;color=clr_Λ,fill_alpha=0.2,linewidth=2,markersize=4,label=nothing)

hlines!(ax_Λs, LNMOD6_poling_periods, color = :black, linestyle=:dash)
sc_qpm_meas = scatter!(ax_Λs,LNMOD6_QPM_meas)

fig 


##
using Interpolations
λ_min_LWMF, λ_max_LWMF = 1.47,1.58
LNMOD6_QPM_meas = [ (1.52,3.0) ]
LNMOD6_poling_periods = [ 2.8, 3.0, 3.3, 2.9, 3.15 ]

λs_fund,Λs,Λs_lw,Λs_up
Λs_itp = 2:0.01:3.5 #|> collect
λvsΛ_itp = LinearInterpolation(reverse(Λs), reverse(λs_fund), extrapolation_bc=Line())
λvsΛup_itp = LinearInterpolation(reverse(Λs_up), reverse(λs_fund), extrapolation_bc=Line())
λvsΛlw_itp = LinearInterpolation(reverse(Λs_lw), reverse(λs_fund), extrapolation_bc=Line())

λs_fund_itp = λvsΛ_itp(Λs_itp)
λs_fund_itp_up = λvsΛup_itp(Λs_itp)
λs_fund_itp_lw = λvsΛlw_itp(Λs_itp)
##

fig = Figure()
ax_Λs           =   fig[1,1] = Axis(fig,
    ylabel="fundamental wavelength (μm)",
    xlabel="poling period (μm)",
    title="LNMOD6 SHG Phase Matching",
    # xminorticksvisible = true,
    # yminorticksvisible = true,
    # xminorgridvisible = true,
    yminorgridvisible = true,
    yticks=collect(0.9:0.1:2.2),
    yminorticks=collect(0.95:0.1:2.15),
    # ylim = [0.9,2.2]
)
bnd_tuning = band!(ax_Λs,[extrema(Λs_itp)...],[(λ_min_LWMF, λ_min_LWMF)...],[(λ_max_LWMF, λ_max_LWMF)...],color=(:magenta,0.2))
sl_Λs,bnd_Λs = scatterlines_err!(ax_Λs,Λs_itp,λs_fund_itp,λs_fund_itp_lw,λs_fund_itp_up;color=clr_Λ,fill_alpha=0.2,linewidth=2,markersize=4,label=nothing)
vlines!(ax_Λs, LNMOD6_poling_periods, color = :black, linestyle=:dash)
sc_qpm_meas = scatter!(ax_Λs,reverse.(LNMOD6_QPM_meas))

ylims!(ax_Λs,(0.9,2.2))

fig 


##
# fig = Figure()
# ax = fig[1,1] = Axis(fig)
# scatterlines!(ax,λs_fund,Λs_sw2c,color="black")
# scatterlines!(ax,λs_fund,Λs_sw2p,color=logocolors[:red])
# scatterlines!(ax,λs_fund,Λs_sw2n,color=logocolors[:blue])
# fig
##
# bidx_TE00 = [mode_idx(Es[omidx,:],eps[omidx];pol_idx=1,mode_order=(0,0),rel_amp_min=0.1) for omidx=1:nω]
# Epols = [E_relpower_xyz(eps[omidx],Es[omidx,bidx]) for omidx=1:length(oms), bidx=1:num_bands]
# Epols_fund  = [ [relpwr[a] for relpwr in [E_relpower_xyz(eps_fund[omidx],Es_fund[omidx,bidx]) for omidx=1:nω, bidx=1:num_bands] ] for a=1:3 ]
# Es = Es_fund
# eps = eps_fund

# Es = Es_sh
# eps = eps_sh


# Epwr    = [E_relpower_xyz(eps[omidx],Es[omidx,bidx]) for omidx=1:nω, bidx=1:num_bands]
# Epwr_x, Epwr_y, Epwr_z = @views (getindex.(Epwr,(a,)) for a=1:3)
# Epol_axind = map(argmax,Epwr)
# Epeakinds = [ argmax(real(_3dot(Es[omidx,bidx],eps[omidx],Es[omidx,bidx])[Epol_axind[omidx,bidx],..])) for omidx=1:nω, bidx=1:num_bands]
# node_counts = [ count_E_nodes(Es[omidx,bidx],eps[omidx],Epol_axind[omidx,bidx];rel_amp_min=0.1)  for omidx=1:nω, bidx=1:num_bands]

# slices_wdw_inds = [ windowed_E_slices(Es[omidx,bidx],eps[omidx],Epol_axind[omidx,bidx];rel_amp_min=0.1)  for omidx=1:nω, bidx=1:num_bands]


# function mode_idx(Es,eps;pol_idx=1,mode_order=(0,0),rel_amp_min=0.1)
#     maxval, maxidx = findmax(Es) do EE
#         EEpwr = E_relpower_xyz(eps,EE)
#         EEpol_axind = argmax(EEpwr)
#         correct_pol = Float64(isequal(EEpol_axind,pol_idx))
#         correct_order = Float64(isequal(
#             count_E_nodes(EE,eps,EEpol_axind;rel_amp_min),
#             mode_order,
#         ))
#         return correct_pol * correct_order * EEpwr
#     end
#     return maxidx
# end
# ##
# bidx_TE00 = [mode_idx(Es[omidx,:],eps[omidx];pol_idx=1,mode_order=(0,0),rel_amp_min=0.1) for omidx=1:nω]

##

##
# inspect_slcs_inds(slices_wdw_inds[20,3])


##



# bidx_TE00_fund  =   [ 1 for oo in ωs_fund ]
# bidx_TE00_sh   =   [ 1 for oo in ωs_fund ]

# ##
# Λ0_jank = 5.1794 #01 #5.1201
# L_jank = 3e3 # 1cm in μm
# ##
# nx = sqrt.(getindex.(inv.(ms_jank.M̂.ε⁻¹),1,1))
# function Eperp_max(E)
#     Eperp = view(E,1:2,:,:,:)
#     maximum(abs,Eperp,dims=1:3)[1,1,1,:]
# end
# 𝓐(n,ng,E) = inv.( n .* ng .* Eperp_max(E).^2)
# AF_jank = 𝓐(nF_jank, ngF_jank, EF_jank) # inv.(nF_jank .* ngF_jank)
# AS_jank = 𝓐(nS_jank, ngS_jank, ES_jank) # inv.(nS_jank .* ngS_jank)
# ÊF_jank = [EF_jank[:,:,:,i] * sqrt(AF_jank[i] * nF_jank[i] * ngF_jank[i]) for i=1:nω_jank]
# ÊS_jank = [ES_jank[:,:,:,i] * sqrt(AS_jank[i] * nS_jank[i] * ngS_jank[i]) for i=1:nω_jank]
# 𝓐₁₂₃_jank = ( AS_jank .* AF_jank.^2  ).^(1.0/3.0)
# 𝓞_jank = [real(sum( conj.(ÊS_jank[ind]) .* ÊF_jank[ind].^2 )) / 𝓐₁₂₃_jank[ind] * δ(grid) for ind=1:length(ωs_jank)] #
# 𝓞_jank_rel = 𝓞_jank/maximum(𝓞_jank)

# χ⁽²⁾_LNx = χ⁽²⁾_fn(LNx)
# χ⁽²⁾xxx_jank = [χ⁽²⁾_LNx(ll,ll,ll/2)[1,1,1] for ll in λs_jank]
# χ⁽²⁾xxx_rel_jank = abs.(χ⁽²⁾xxx_jank) / maximum(abs.(χ⁽²⁾xxx_jank))
