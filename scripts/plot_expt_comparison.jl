# using GLMakie, AbstractPlotting, Interpolations, FFTW
# using GLMakie: lines, lines!, heatmap, heatmap!
using Revise
using OptiMode
using LinearAlgebra
using StaticArrays
using Rotations
using CairoMakie
using AbstractPlotting
using Interpolations
using FFTW
using AbstractPlotting: lines, lines!, heatmap, heatmap!
using AbstractPlotting.GeometryBasics
using Colors
import Colors: JULIA_LOGO_COLORS
logocolors = JULIA_LOGO_COLORS
using OhMyREPL
using Crayons.Box       # for color printing
using Zygote: @ignore, dropgrad
using StaticArrays: Dynamic
using IterativeSolvers: bicgstabl
using Rotations: RotY, MRP
using HDF5
# using ColorSchemes
# using ColorSchemeTools
using PyCall
cplot = pyimport("cplot")

# noto_sans = "../assets/NotoSans-Regular.ttf"
# noto_sans_bold = "../assets/NotoSans-Bold.ttf"

LNx = rotate(MgO_LiNbO₃,Matrix(MRP(RotY(π/2))),name=:LiNbO₃_X)
AD_style = BOLD*BLUE_FG #NEGATIVE*BOLD*BLUE_FG      # defined in Crayons.Box
FD_style = BOLD*RED_FG
MAN_style = BOLD*GREEN_FG

AD_style_N = NEGATIVE*BOLD*BLUE_FG #NEGATIVE*BOLD*BLUE_FG      # defined in Crayons.Box
FD_style_N = NEGATIVE*BOLD*RED_FG
MAN_style_N = NEGATIVE*BOLD*GREEN_FG

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
grid = Grid(Δx,Δy,Nx,Ny)
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).


"""
Takes an array of complex number and converts it to an array of [r, g, b],
where phase gives hue and saturaton/value are given by the absolute value.
Especially for use with imshow for complex plots.
"""
function complex_to_rgb(X; alpha=1.0, colorspace="cam16")
    return [RGB(x...) for x in cplot.get_srgb1.(X;alpha,colorspace)]
end

function E_relpower_xyz(ms::ModeSolver{ND,T},ω²H) where {ND,T<:Real}
    E = 1im * ε⁻¹_dot( fft( kx_tc( reshape(ω²H[2],(2,size(ms.grid)...)),mn(ms),ms.M̂.mag), (2:1+ND) ), flat( ms.M̂.ε⁻¹ ))
    Es = reinterpret(reshape, SVector{3,Complex{T}},  E)
    Pₑ_xyz_rel = normalize([mapreduce((ee,epss)->(abs2(ee[a])*inv(epss)[a,a]),+,Es,ms.M̂.ε⁻¹) for a=1:3],1)
    return Pₑ_xyz_rel
end

TE_filter(threshold) = (ms,ω²H)->E_relpower_xyz(ms,ω²H)[1]>threshold
TM_filter(threshold) = (ms,ω²H)->E_relpower_xyz(ms,ω²H)[2]>threshold
oddX_filter(threshold) = (ms,αX)->sum(abs2,𝓟x̄(ms.grid)*αX[2])>threshold
evenX_filter(threshold) = (ms,αX)->sum(abs2,𝓟x(ms.grid)*αX[2])>threshold



##




# function count_zero_crossings_E(Hₜ,plot=False):
#     if s.getdata(mode,'Ex').real.max() > s.getdata(mode,'Ey').real.max():
#         E = s.getdata(mode,'Ex').squeeze().real
#         pol = 'TE'
#     else:
#         pol = 'TM'
#         E = s.getdata(mode,'Ey').squeeze().real
#     x = Q_(s.getdata(mode,'x').squeeze(),'m').to(u.um)
#     y = Q_(s.getdata(mode,'y').squeeze(),'m').to(u.um)
#     x_crop_mask = (x > -self.params['width'] / 2.) * (x < self.params['width'] / 2.)
#     y_crop_mask = (y > 0.*u.m) * (y < self.params['height'] )
#     E_crop = E[x_crop_mask,:][:,y_crop_mask]
#     x0ind,y0ind = np.unravel_index(np.argmax(np.abs(E_crop)),E_crop.shape)
#     n_zero_cross_x = (np.diff(np.sign(E_crop[:,y0ind])) != 0).sum() - (E_crop[:,y0ind] == 0).sum()
#     n_zero_cross_y = (np.diff(np.sign(E_crop[x0ind,:])) != 0).sum() - (E_crop[x0ind,:] == 0).sum()
#     if plot:
#         x_crop = x[x_crop_mask]
#         y_crop = y[y_crop_mask]
#         print(f'x0ind: {x0ind}')
#         print(f'x0: {x_crop[x0ind].m} μm')
#         print(f'y0ind: {y0ind}')
#         print(f'y0: {y_crop[y0ind].m} μm')
#         print('polarization: ' + pol)
#         print(f'n_zero_cross_x: {n_zero_cross_x}')
#         print(f'n_zero_cross_y: {n_zero_cross_y}')
#         plt.plot(x_crop,E_crop[:,y0ind],'C0')
#         plt.plot(y_crop,E_crop[x0ind,:],'C3')
# return n_zero_cross_x,n_zero_cross_y

##
function plot_ifg!(ax,ifg,nωs,Δωs,;n_fmb=1,color=:blue,label="log₁₀( Σ (Δng)² )")
    itrs = [x[1] for x in ifg] .+ length(ifg)*(n_fmb-1)
    fs = [x[2] for x in ifg]
    fs_norm = fs .* Δωs / nωs
    # gs = [x[3] for x in ifg]
    scatterlines!(ax,itrs,
		fs_norm,
		linewidth=2,
		markersize=2,
		color=color,
		markercolor=color,
		strokecolor=color,
        label=label,
		)
end

ifgs = [ ifg_fmb11, ifg_fmb21, ifg_fmb31, ifg_fmb41, vcat(ifg_fmb51, [x.+[3.,0.,0.] for x in ifg_fmb52]) ]
ifg_colors = [ :red,    :blue,      :green,     :purple,    :black]
ifg_labels = [ "opt1",  "opt2",     "opt3",     "opt4",     "opt5", ]
ifg_nfmbs = [ 1, 1, 1, 1, 1, 1 ]
ifg_nωs = [ length(ωs1), length(ωs2), length(ωs3), length(ωs4), length(ωs5),]
ifg_Δωs = -1*[ -(extrema(ωs1)...), -(extrema(ωs2)...), -(extrema(ωs3)...), -(extrema(ωs4)...), -(extrema(ωs5)...)]

fig_opts = Figure(resolution = (400,600))
ax_opts = fig_opts[1,1] = Axis(fig_opts)

[ plot_ifg!(ax_opts,
                ifgs[i],
                ifg_nωs[i],
                ifg_Δωs[i],
                n_fmb=ifg_nfmbs[i],
                label=ifg_labels[i],
                color=ifg_colors[i]) for i = 1:length(ifgs)]

# label, format
ax_opts.ylabel = "log₁₀( Σ (Δng)² )"
ax_opts.xlabel = "iteration #"
axislegend(ax_opts)
fig_opts
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

nω_jank = 20
ωs_jank = collect(range(1/2.25,1/1.95,length=nω_jank)) # collect(0.416:0.01:0.527)
λs_jank = 1 ./ ωs_jank

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
grid = Grid(Δx,Δy,Nx,Ny)

ms_jank = ModeSolver(kguess(1/2.25,rwg(p_jank)), rwg(p_jank), grid; nev=4)
# nFS_jank,ngFS_jank,gvdFS_jank,EFS_jank = solve_n(
#     ms_jank,
#     vcat(ωs_jank,2*ωs_jank),
#     rwg(p_jank);
#     f_filter=TE_filter(0.7),
# )
# nF_jank = nFS_jank[1:nω_jank]
# ngF_jank = ngFS_jank[1:nω_jank]
# gvdF_jank = gvdFS_jank[1:nω_jank]
# EF_jank = view(EFS_jank,:,:,:,1:nω_jank)
#
# nS_jank = nFS_jank[nω_jank+1:end]
# ngS_jank = ngFS_jank[nω_jank+1:end]
# gvdS_jank = gvdFS_jank[nω_jank+1:end]
# ES_jank = view(EFS_jank,:,:,:,(nω_jank+1):(2*nω_jank))

data_dir=joinpath(homedir(),"data")
fname = "compare_qpm_jank.h5"
grp_name="256"
fpath 	= joinpath(data_dir, fname)

# h5open(fpath, "cw") do fid			# "cw"== "read-write, create if not existing"
# 	g = create_group(fid,grp_name) #fid[grp_name]
#     g["nF"] = nF_jank
#     g["ngF"] = ngF_jank
#     g["gvdF"] = gvdF_jank
#     g["EF"] = EF_jank
#     g["nS"] = nS_jank
#     g["ngS"] = ngS_jank
#     g["gvdS"] = gvdS_jank
#     g["ES"] = ES_jank
# end

nF_jank,ngF_jank,gvdF_jank,EF_jank,nS_jank,ngS_jank,gvdS_jank,ES_jank = h5open(fpath, "r") do fid			# "cw"== "read-write, create if not existing"
	g = fid[grp_name] #fid[grp_name]
    nF_jank		=		read(g,"nF")
    ngF_jank	=		read(g,"ngF")
    gvdF_jank	=		read(g,"gvdF")
    EF_jank		=		read(g,"EF")
    nS_jank		=		read(g,"nS")
    ngS_jank	=		read(g,"ngS")
    gvdS_jank	=		read(g,"gvdS")
    ES_jank		=		read(g,"ES")
	nF_jank,ngF_jank,gvdF_jank,EF_jank,nS_jank,ngS_jank,gvdS_jank,ES_jank
end

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

# d₃₃ =   25.0    #   pm/V
# d₃₁ =   4.4     #   pm/V
# d₂₂ =   1.9     #   pm/V
#
# #          xx      yy       zz      zy      zx      xy
# deff = [    0.      0.      0.      0.      d₃₁     -d₂₂     #   x
#             -d₂₂    d₂₂     0.      d₃₁     0.      0.       #   y
#             d₃₁     d₃₁     d₃₃     0.      0.      0.   ]   #   z


# GLMakie.heatmap(abs2.(ES_jank[1][1,:,:]))
# image!(complex_to_rgb(EF_jank[1][1,:,:]))
## plot and save comparison subplots individually
ipc_theme = Theme(fontsize=16)
set_theme!(ipc_theme)
##
λs = λs_jank;nS=nS_jank;nF=nF_jank;ngF=ngF_jank;ngS=ngS_jank;Λ0=5.1794;L=3e3; #Λ0=Λ0_jank;L=L_jank;

fig = Figure(resolution = (400,200))

ax_ng = fig[1,1] = Axis(fig,
	ylabel = "group index",
	xlabel = "fundamental wavelength (μm)",
	xlabelpadding=3,
)
ax_Λ = fig[1,1] = Axis(fig,
	yaxisposition=:right,
	ygridvisible=false,
	yminorgridvisible=false,
	ylabel = "poling period Λ (μm)",
)

# rowgap!(gl1,Relative(0.02))
# fig[1,1] = gl1
linkxaxes!(ax_Λ,ax_ng,ax_qpm)
hidespines!(ax_Λ)
# hidexdecorations!(ax_ng,grid=false)
hidexdecorations!(ax_Λ)
ax_ng.ytickformat = "{:.3f}"
ax_Λ.ytickformat = "{:.3f}"
ωs = inv.(λs)
# sl_nF = scatterlines!(ax_ng,
# 	λs,
# 	nF;
# 	color=logocolors[:red],
# 	linewidth=2,
# 	label="n(ω)",
# 	markercolor=logocolors[:red],
# 	strokecolor=logocolors[:red],
# 	alpha=0.5,
# 	markersize=2,
# )
# sl_nS = scatterlines!(ax_ng,
# 	λs,
# 	nS;
# 	color=logocolors[:blue],
# 	linewidth=2,
# 	label="n(2ω)",
# 	markercolor=logocolors[:blue],
# 	strokecolor=logocolors[:blue],
# 	alpha=0.5,
# 	markersize=2,
# )
sl_ngF = scatterlines!(ax_ng,
	λs,
	ngF;
	color=logocolors[:red],
	linewidth=2,
	label="ng(ω)",
	markercolor=logocolors[:red],
	strokecolor=logocolors[:red],
	markersize=2,
)
sl_ngS = scatterlines!(ax_ng,
	λs,
	ngS;
	color=logocolors[:blue],
	linewidth=2,
	label="ng(2ω)",
	markercolor=logocolors[:blue],
	strokecolor=logocolors[:blue],
	markersize=2,
)
Δn = ( nS .- nF )
Λ = (λs ./ 2) ./ Δn
sl_LMng = scatterlines!(ax_Λ,
	λs,
	Λ;
	color=logocolors[:green],
	linewidth=2,
	label="poling period Λ",
	markercolor=logocolors[:green],
	strokecolor=logocolors[:green],
	markersize=2,
)

fig

##
save("compare_qpm_jank_disp_v1.png",fig)
save("compare_qpm_jank_disp_v1.svg",fig)

##
fig = Figure(resolution = (330,260))
ax_qpm = fig[1,1] = Axis(fig,
	yticks=[0.0,0.25,0.5,0.75,1.0],
	ylabel = "SHG efficiency",
	xlabel = "fundamental wavelength (μm)",
	xlabelpadding = 3,
)
ωs = inv.(λs)
Δk_qpm = ( 4π ./ λs) .* (  nS .-  nF ) .- (2π / Λ0)
Δk_qpm_itp = LinearInterpolation(ωs,Δk_qpm)
χ⁽²⁾xxx_rel_itp = LinearInterpolation(ωs,χ⁽²⁾xxx_rel_jank)
𝓞_rel_itp = LinearInterpolation(ωs,𝓞_jank_rel)
ωs_dense = collect(range(extrema(ωs)...,length=n_dense))
λs_dense = inv.(ωs_dense)
Δk_qpm_dense = Δk_qpm_itp.(ωs_dense)
χ⁽²⁾xxx_rel_dense = χ⁽²⁾xxx_rel_itp.(ωs_dense)
𝓞_rel_dense = 𝓞_rel_itp.(ωs_dense)
sinc2Δk_dense = (sinc.(Δk_qpm_dense * L / 2.0)).^2
SHG_trnsfr_dense = sinc2Δk_dense.*(χ⁽²⁾xxx_rel_dense.^2).*𝓞_rel_dense
SHG_trnsfr_dense = SHG_trnsfr_dense ./ maximum(SHG_trnsfr_dense)

l_shg_tot_jank = lines!(ax_qpm, λs_dense, SHG_trnsfr_dense; color=logocolors[:blue],linewidth=2,label="SHG efficiency\n(model)")
# l_qpm_theory_jank   =lines!(
#     ax_qpm,
#     λs_theory_jank,
#     SHGrel_theory_jank,
#     label="theory (authors)",
#     color=:pink,
#     linewidth=2,
# )
l_shg_expt_jank     =lines!(
    ax_qpm,
    λs_expt_jank,
    SHGrel_expt_jank,
    label="SHG efficiency\n(expt. [2])",
    color=:black,
    linewidth=2,
)
l_shg_disp_jank = lines!(ax_qpm, λs_dense, sinc2Δk_dense; color=logocolors[:purple],linewidth=2,label="QPM, Λ=5.18μm")
l_shg_overlap_jank     =lines!(
    ax_qpm,
    λs_jank,
    𝓞_jank_rel,
    label="overlap",
    color=:orange,
    linewidth=2,
)
l_shg_χ⁽²⁾_jank     =lines!(
    ax_qpm,
    λs_jank,
    χ⁽²⁾xxx_rel_jank.^2,
    label="|χ⁽²⁾|²",
    color=logocolors[:red],
    linewidth=2,
)
ylims!(ax_qpm,(-0.05,1.08))
# axislegend(ax_qpm,fontsize=12)
# axislegend(ax_ng,fontsize=12)
fig

##

save("compare_qpm_jank_shg_trnsfr_v1.png",fig)
save("compare_qpm_jank_shg_trnsfr_v1.svg",fig)

## nx spatial plot
fig = Figure()
cbr_lbl = "nₓ"
x_lims=(-1.8,1.8)
y_lims=(-0.8,0.8)
Eind=5
ax = fig[1,1] = Axis(fig,
	ylabel = "y (μm)",
	xlabel = "x (μm)",
	xlabelpadding = 1,
	ylabelpadding = 1,
)
xs = x(grid)
ys = y(grid)
hm = heatmap!(ax,xs,ys,nx;colormap=:cividis)
xlims!(ax,x_lims)
ylims!(ax,y_lims)
ax.aspect=DataAspect()
cbar = Colorbar(fig[1,2],hm, width=15, height=Relative(2/3), label=cbr_lbl,tickformat = "{:.1f}")
# hidedecorations!(ax)
# hidespines!(ax)

fig
##
# save("ln_rwg_nx_heatmap_v0_nohm.svg",fig)
# save("ln_rwg_nx_heatmap_v0_noax.png",fig)
save("ln_rwg_nx_heatmap_v0.png",fig)
save("ln_rwg_nx_heatmap_v0.svg",fig)

## nx zoom


fig = Figure()
cbr_lbl = "nₓ"
x_zoom=0.9
dx_zoom = 0.1
y_zoom = 0.25
dy_zoom = 0.1
# x_lims=(0.9,1.05)
# y_lims=(0.2,0.35)
x_lims=(x_zoom,x_zoom+dx_zoom)
y_lims=(y_zoom,y_zoom+dy_zoom)
ax = fig[1,1] = Axis(fig)
xs = x(grid)
ys = y(grid)
hm = heatmap!(ax,xs,ys,nx;colormap=:cividis)
xlims!(ax,x_lims)
ylims!(ax,y_lims)
ax.aspect=DataAspect()
# cbar = Colorbar(fig[1,2],hm, width=15, height=Relative(2/3), label=cbr_lbl,tickformat = "{:.1f}")
hidedecorations!(ax)
# hidespines!(ax)

fig
##
# save("ln_rwg_nx_zoom_v0_nohm.svg",fig)
# save("ln_rwg_nx_zoom_v0_noax.png",fig)
save("ln_rwg_nx_zoom_v0.png",fig)
save("ln_rwg_nx_zoom_v0.svg",fig)


##
# ax_E = [Axis(pos[1,1]) for pos in pos_E]
bb_dx,bb_dy = 70,80
bb1x,bb1y = 220,335
bb2x,bb2y = 200,410
ax_E = [Axis(fig, bbox = BBox(bb1x,bb1x+bb_dx,bb1y,bb1y+bb_dy)),
		Axis(fig, bbox = BBox(bb2x,bb2x+bb_dx,bb2y,bb2y+bb_dy)),
]

Ex = [ EF[1,:,:,Eind], ES[1,:,:,Eind] ]
cmaps_E = [:linear_ternary_red_0_50_c52_n256, :linear_ternary_blue_0_44_c57_n256]
labels_E = ["rel. |Eₓ|² @ ω","rel. |Eₓ|² @ 2ω"]
hm_E = [heatmap!(ax, xs, ys, abs2.(Ex[i]),colormap=cmaps_E[i],label=labels_E[i]) for (i, ax) in enumerate(ax_E)]
hm_nx = heatmap!(ax_nx,xs,ys,nx;colormap=:viridis)
[xlims!(ax,xlims) for ax in (ax_nx,ax_E...)]
[ylims!(ax,ylims) for ax in (ax_nx,ax_E...)]
# text!(ax_nx,"nₓ",position=(1.4,1.1),textsize=0.7,color=:white)
# text!(ax_E[1],"rel. |Eₓ|² (ω)",position=(-1.4,1.1),textsize=0.7,color=:white)
# text!(ax_E[2],"rel. |Eₓ|² (2ω)",position=(-1.7,1.1),textsize=0.7,color=:white)
# cbar_nx = Colorbar(pos_nx[1,2],hm_nx, width=20 ) #,label="nₓ")
# cbar_EF = fig[3,3] = Colorbar(fig,hm_E[1]) #,label="rel. |Eₓ|² @ ω")
# cbar_ES = fig[4,3] = Colorbar(fig,hm_E[2]) #,label="rel. |Eₓ|² @ 2ω")
# cbars_E = [Colorbar(pp[1, 2], hm,  width=20 ) for (pp,hm) in zip(pos_E, hm_E )]
# for cb in [cbar_nx,cbars_E...]
#     cb.width=30
#     cb.height=Relative(2/3)
# end
# label, format
hidedecorations!.(ax_E)
# hidexdecorations!(ax_E[1])
# ax_E[1].ylabel = "y [μm]"


ax_nx.aspect=DataAspect()
ax_E[1].aspect=DataAspect()
ax_E[2].aspect=DataAspect()



##
function plt_qpm_compare(λs,nF,nS,ngF,ngS,gvdF,gvdS,Λ0,L,EF,ES,nx,grid;n_dense=3000,res=(400,600),
	xlims=(-1.8,1.8),ylims=(-0.8,0.8),Eind=5)
    fig = Figure(;resolution=res)
	gl1 = GridLayout()
	gl2 = GridLayout()
	# ax_n = fig[1,1] = Axis(fig)
	ax_ng = gl1[1:4,1] = Axis(fig,
		ylabel = "group index",
	)
	ax_Λ = gl1[1:4,1] = Axis(fig,
		yaxisposition=:right,
		ygridvisible=false,
		yminorgridvisible=false,
		ylabel = "poling period [μm]",
	)

	# ax_gvd = fig[3,1] = Axis(fig)
    ax_qpm = gl1[5:8,1] = Axis(fig,
		yticks=[0.0,0.25,0.5,0.75,1.0],
		ylabel = "SHG transfer fn.",
		xlabel = "λ [μm]",
		xlabelpadding = 3,
	)
	ax_opts = gl2[2,1] = Axis(fig,xticks=1:6,
		yscale=AbstractPlotting.log10,
		yminorticksvisible = true,
	 	yminorgridvisible = true,
        yminorticks = IntervalsBetween(8),
		ylabel = "log₁₀( Σ (Δng)² )",
		xlabel = "optimization steps",
		xlabelpadding = 3,
		)

	ax_nx = gl2[1,1] = Axis(fig,
		ylabel = "y [μm]",
		xlabel = "x [μm]",
		xlabelpadding = 3,
	)
	# ax_nx = Axis(pos_nx[1:3,1])
	# ax_nx.ylabel = "y [μm]"
	# ax_nx.xlabel = "x [μm]"

	rowgap!(gl1,Relative(0.05))

	fig[1,1] = gl1
	fig[1,2] = gl2

	linkxaxes!(ax_Λ,ax_ng,ax_qpm)
	hidespines!(ax_Λ)

	hidexdecorations!(ax_ng,grid=false)
	hidexdecorations!(ax_Λ)


	[ax.ytickformat = "{:.2f}" for ax in (ax_ng,ax_Λ)]




	ifg_plot_inds = [2,3,5]
	[ plot_ifg!(ax_opts,
	                ifgs[i],
	                ifg_nωs[i],
	                ifg_Δωs[i],
	                n_fmb=ifg_nfmbs[i],
	                label=ifg_labels[i],
	                color=ifg_colors[i]) for i in ifg_plot_inds]

	# [ax.ytickformat = "{:.2f}" for ax in (ax_ng,ax_Λ)]
	ωs = inv.(λs)

	# sl_nF = scatterlines!(ax_n,
	# 	λs,
	# 	nF;
	# 	color=logocolors[:red],
	# 	linewidth=2,
	# 	label="n(ω)",
	# 	markercolor=logocolors[:red],
	# 	# strokecolor=color,
	# 	markersize=2,
	# )
    # sl_nS = scatterlines!(ax_n,
	# 	λs,
	# 	nS;
	# 	color=logocolors[:blue],
	# 	linewidth=2,
	# 	label="n(2ω)",
	# 	markercolor=logocolors[:blue],
	# 	# strokecolor=color,
	# 	markersize=2,
	# )
	sl_ngF = scatterlines!(ax_ng,
		λs,
		ngF;
		color=logocolors[:red],
		linewidth=2,
		label="ng(ω)",
		markercolor=logocolors[:red],
		# strokecolor=color,
		markersize=2,
	)

	sl_ngS = scatterlines!(ax_ng,
		λs,
		ngS;
		color=logocolors[:blue],
		linewidth=2,
		label="ng(2ω)",
		markercolor=logocolors[:blue],
		# strokecolor=color,
		markersize=2,
	)
	# axislegend(ax_n,position=:rt)
    # sl_gvd_F = scatterlines!(ax_gvd,
	# 	λs,
	# 	gvdF;
	# 	color=logocolors[:red],
	# 	linewidth=2,
	# 	label="gvd(ω)",
	# 	markercolor=logocolors[:red],
	# 	# strokecolor=color,
	# 	markersize=2,
	# )
    # sl_gvd_S = scatterlines!(ax_gvd,
	# 	λs,
	# 	gvdS;
	# 	color=logocolors[:blue],
	# 	linewidth=2,
	# 	label="gvd(2ω)",
	# 	markercolor=logocolors[:blue],
	# 	# strokecolor=color,
	# 	markersize=2,
	# )
    # axislegend(ax_gvd,position=:rt)

    Δn = ( nS .- nF )
    Λ = (λs ./ 2) ./ Δn

	sl_LMng = scatterlines!(ax_Λ,
		λs,
		Λ;
		color=logocolors[:green],
		linewidth=2,
		label="poling period [μm]",
		markercolor=logocolors[:green],
		strokecolor=logocolors[:green],
		markersize=2,
	)


    # lines!(ax_Λ,  λs, Λ; color=logocolors[:green],linewidth=2,label="poling period [μm]")
    # axislegend(ax_Λ,position=:rb)
    # plot!(ax_Λ,  λs, Λ; color=logocolors[:green],markersize=2)
	ωs = inv.(λs)
    Δk_qpm = ( 4π ./ λs) .* (  nS .-  nF ) .- (2π / Λ0)
    Δk_qpm_itp = LinearInterpolation(ωs,Δk_qpm)
	χ⁽²⁾xxx_rel_itp = LinearInterpolation(ωs,χ⁽²⁾xxx_rel_jank)
	𝓞_rel_itp = LinearInterpolation(ωs,𝓞_jank_rel)

    ωs_dense = collect(range(extrema(ωs)...,length=n_dense))
    λs_dense = inv.(ωs_dense)
    Δk_qpm_dense = Δk_qpm_itp.(ωs_dense)
	χ⁽²⁾xxx_rel_dense = χ⁽²⁾xxx_rel_itp.(ωs_dense)
	𝓞_rel_dense = 𝓞_rel_itp.(ωs_dense)
    sinc2Δk_dense = (sinc.(Δk_qpm_dense * L / 2.0)).^2
	SHG_trnsfr_dense = sinc2Δk_dense.*χ⁽²⁾xxx_rel_dense.*𝓞_rel_dense
	SHG_trnsfr_dense = SHG_trnsfr_dense ./ maximum(SHG_trnsfr_dense)
    l_shg_disp_jank = lines!(ax_qpm, λs_dense, sinc2Δk_dense; color=logocolors[:purple],linewidth=2,label="QPM disp.\npoling=$Λ0")
	l_shg_tot_jank = lines!(ax_qpm, λs_dense, SHG_trnsfr_dense; color=logocolors[:blue],linewidth=2,label="rel. SHG\npoling=$Λ0")

	# l_qpm_theory_jank   =lines!(
	#     ax_qpm,
	#     λs_theory_jank,
	#     SHGrel_theory_jank,
	#     label="theory (authors)",
	#     color=logocolors[:blue],
	#     linewidth=2,
	# )

	l_shg_expt_jank     =lines!(
	    ax_qpm,
	    λs_expt_jank,
	    SHGrel_expt_jank,
	    label="expt (authors)",
	    color=:black,
	    linewidth=2,
	)
	l_shg_overlap_jank     =lines!(
	    ax_qpm,
	    λs_jank,
	    𝓞_jank_rel,
	    label="overlap",
	    color=:orange,
	    linewidth=2,
	)
	l_shg_χ⁽²⁾_jank     =lines!(
	    ax_qpm,
	    λs_jank,
	    χ⁽²⁾xxx_rel_jank,
	    label="rel. LiNbO₃ χ⁽²⁾",
	    color=logocolors[:red],
	    linewidth=2,
	)
	# axislegend(ax_qpm)
	leg_shg = gl1[9,1] = Legend(fig,
		[l_shg_expt_jank,l_shg_tot_jank,l_shg_disp_jank,l_shg_overlap_jank,l_shg_χ⁽²⁾_jank],
		["expt. [2]","model","disp.","overlap","rel. χ⁽²⁾"],
		orientation = :horizontal,
		height=20,
		# tellwidth = false,
		# tellheight = true,
	)

    # Spatial plots #

	xs = x(grid)
    ys = y(grid)


	pos_E = [fig[i,j] for i=1:2,j=4:4]
	# ax_E = [Axis(pos[1,1]) for pos in pos_E]
	bb_dx,bb_dy = 70,80
	bb1x,bb1y = 220,335
	bb2x,bb2y = 200,410
	ax_E = [Axis(fig, bbox = BBox(bb1x,bb1x+bb_dx,bb1y,bb1y+bb_dy)),
			Axis(fig, bbox = BBox(bb2x,bb2x+bb_dx,bb2y,bb2y+bb_dy)),
	]

    Ex = [ EF[1,:,:,Eind], ES[1,:,:,Eind] ]
    cmaps_E = [:linear_ternary_red_0_50_c52_n256, :linear_ternary_blue_0_44_c57_n256]
    labels_E = ["rel. |Eₓ|² @ ω","rel. |Eₓ|² @ 2ω"]
    hm_E = [heatmap!(ax, xs, ys, abs2.(Ex[i]),colormap=cmaps_E[i],label=labels_E[i]) for (i, ax) in enumerate(ax_E)]
    hm_nx = heatmap!(ax_nx,xs,ys,nx;colormap=:viridis)
	[xlims!(ax,xlims) for ax in (ax_nx,ax_E...)]
	[ylims!(ax,ylims) for ax in (ax_nx,ax_E...)]
    # text!(ax_nx,"nₓ",position=(1.4,1.1),textsize=0.7,color=:white)
    # text!(ax_E[1],"rel. |Eₓ|² (ω)",position=(-1.4,1.1),textsize=0.7,color=:white)
    # text!(ax_E[2],"rel. |Eₓ|² (2ω)",position=(-1.7,1.1),textsize=0.7,color=:white)
    # cbar_nx = Colorbar(pos_nx[1,2],hm_nx, width=20 ) #,label="nₓ")
    # cbar_EF = fig[3,3] = Colorbar(fig,hm_E[1]) #,label="rel. |Eₓ|² @ ω")
    # cbar_ES = fig[4,3] = Colorbar(fig,hm_E[2]) #,label="rel. |Eₓ|² @ 2ω")
	# cbars_E = [Colorbar(pp[1, 2], hm,  width=20 ) for (pp,hm) in zip(pos_E, hm_E )]
    # for cb in [cbar_nx,cbars_E...]
    #     cb.width=30
    #     cb.height=Relative(2/3)
    # end
    # label, format
    hidedecorations!.(ax_E)
	# hidexdecorations!(ax_E[1])
    # ax_E[1].ylabel = "y [μm]"


	ax_nx.aspect=DataAspect()
    ax_E[1].aspect=DataAspect()
    ax_E[2].aspect=DataAspect()


	# trim!(fig.layout)
	# trim!(comp_sl)
    return fig
end

ipc_theme = Theme(fontsize=14)
set_theme!(ipc_theme)

fig_jank = plt_qpm_compare(inv.(ωs_jank),
	nF_jank,
	nS_jank,
	ngF_jank,
	ngS_jank,
	gvdF_jank,
	gvdS_jank,
	5.1794,
	L_jank,
	EF_jank,
	ES_jank,
	nx,
	grid,
	Eind=3,
)

##
save("qpm_compare_jank.png",fig_jank)
save("qpm_compare_jank.svg",fig_jank)
##
# ax_ng_jank = fig_jank[1,1]
# ax_ins1 =

fig_jank

## data from Mark's paper, grabbed with web plot digitizer
λs_SHGrel_theory_jank =   [     1.97904    0.05997
                                1.98101    0.05855
                                1.98298    0.05416
                                1.98495    0.04673
                                1.98692    0.03782
                                1.98889    0.02827
                                1.99086    0.01948
                                1.99274    0.01507
                                1.99802    0.01587
                                1.99999    0.02562
                                2.00169    0.03950
                                2.00330    0.05558
                                2.00449    0.07217
                                2.00546    0.08730
                                2.00635    0.10413
                                2.00725    0.12238
                                2.00805    0.13884
                                2.00877    0.15482
                                2.00949    0.17213
                                2.01020    0.18936
                                2.01083    0.20498
                                2.01146    0.22167
                                2.01217    0.24129
                                2.01289    0.26127
                                2.01361    0.28124
                                2.01438    0.30058
                                2.01509    0.32160
                                2.01580    0.34276
                                2.01639    0.36123
                                2.01712    0.38183
                                2.01776    0.40036
                                2.01826    0.41896
                                2.01871    0.43391
                                2.01939    0.45207
                                2.02011    0.47408
                                2.02082    0.49553
                                2.02151    0.51740
                                2.02222    0.53879
                                2.02294    0.56009
                                2.02365    0.58100
                                2.02437    0.60152
                                2.02509    0.62180
                                2.02580    0.64169
                                2.02660    0.66284
                                2.02722    0.68110
                                2.02793    0.69974
                                2.02865    0.71794
                                2.02937    0.73578
                                2.03008    0.75309
                                2.03080    0.76996
                                2.03152    0.78674
                                2.03231    0.80370
                                2.03305    0.82074
                                2.03393    0.83750
                                2.03483    0.85476
                                2.03572    0.87138
                                2.03671    0.88792
                                2.03769    0.90398
                                2.03879    0.92067
                                2.03993    0.93783
                                2.04128    0.95338
                                2.04289    0.96904
                                2.04477    0.98319
                                2.04674    0.99378
                                2.04871    0.99985
                                2.05068    1.00250
                                2.05265    1.00146
                                2.05462    0.99739
                                2.05659    0.99081
                                2.05856    0.98132
                                2.06053    0.96976
                                2.06250    0.95614
                                2.06447    0.94103
                                2.06635    0.92536
                                2.06805    0.91015
                                2.06966    0.89516
                                2.07128    0.87977
                                2.07289    0.86383
                                2.07450    0.84828
                                2.07611    0.83250
                                2.07772    0.81664
                                2.07934    0.80078
                                2.08086    0.78603
                                2.08238    0.77182
                                2.08399    0.75674
                                2.08560    0.74191
                                2.08722    0.72699
                                2.08892    0.71137
                                2.09062    0.69614
                                2.09232    0.68118
                                2.09411    0.66563
                                2.09590    0.65050
                                2.09778    0.63538
                                2.09975    0.61975
                                2.10172    0.60490
                                2.10369    0.59076
                                2.10566    0.57727
                                2.10763    0.56422
                                2.10960    0.55209
                                2.11157    0.54079
                                2.11354    0.52994
                                2.11551    0.51948
                                2.11748    0.50992
                                2.11945    0.50069
                                2.12142    0.49185
                                2.12339    0.48345
                                2.12536    0.47506
                                2.12733    0.46712
                                2.12930    0.45905
                                2.13128    0.45130
                                2.13325    0.44374
                                2.13522    0.43613
                                2.13719    0.42818
                                2.13916    0.42082
                                2.14113    0.41308
                                2.14310    0.40533
                                2.14507    0.39784
                                2.14704    0.39099
                                2.14901    0.38409
                                2.15098    0.37756
                                2.15295    0.37104
                                2.15492    0.36472
                                2.15689    0.35807
                                2.15886    0.35122
                                2.16083    0.34431
                                2.16280    0.33721
                                2.16477    0.32972
                                2.16674    0.32172
                                2.16871    0.31326
                                2.17068    0.30435
                                2.17265    0.29518
                                2.17462    0.28511
                                2.17659    0.27471
                                2.17856    0.26315
                                2.18053    0.25140
                                2.18250    0.23901
                                2.18447    0.22622
                                2.18644    0.21305
                                2.18841    0.19936
                                2.19038    0.18555
                                2.19235    0.17160
                                2.19432    0.15746
                                2.19629    0.14332
                                2.19826    0.12925
                                2.20023    0.11504
                                2.20220    0.10097
                                2.20417    0.08715
                                2.20614    0.07424
                                2.20811    0.06113
                                2.21008    0.04822
                                2.21205    0.03640
                                2.21402    0.02633
                                2.21599    0.01845
                                2.21796    0.01509
                                2.21904    0.01393
                                2.22477    0.01503
                                2.22674    0.01761
                                2.22871    0.02329
                                2.23068    0.02956
                                2.23265    0.03556
                                2.23462    0.04021
                                2.23659    0.04215
                                2.23856    0.04176
                                2.24053    0.03789
                                2.24250    0.03091
                                2.24447    0.02213
                                2.24644    0.01580
                                2.24760    0.01429
                                2.25163    0.01488      ]
λs_SHGrel_expt_jank =   [       1.98470    0.01704
                                1.98627    0.01919
                                1.98728    0.02281
                                1.99029    0.03623
                                1.99125    0.04007
                                1.99229    0.04873
                                1.99319    0.05726
                                1.99417    0.06708
                                1.99516    0.07629
                                1.99596    0.08531
                                1.99677    0.09490
                                1.99775    0.10283
                                1.99865    0.11194
                                1.99937    0.12118
                                1.99999    0.13041
                                2.00080    0.13917
                                2.00178    0.14617
                                2.00250    0.15456
                                2.00313    0.16503
                                2.00393    0.17430
                                2.00474    0.18137
                                2.00562    0.19021
                                2.00662    0.19887
                                2.00734    0.21019
                                2.00796    0.22132
                                2.00868    0.22771
                                2.00975    0.23812
                                2.01083    0.24310
                                2.01643    0.32529
                                2.01728    0.33513
                                2.01781    0.34383
                                2.01853    0.35152
                                2.01925    0.35732
                                2.01969    0.36928
                                2.02023    0.38135
                                2.02104    0.39094
                                2.02184    0.40077
                                2.02234    0.41290
                                2.02256    0.42326
                                2.02283    0.43474
                                2.02310    0.44669
                                2.02372    0.45818
                                2.02444    0.46776
                                2.02498    0.47700
                                2.02543    0.48623
                                2.02596    0.49878
                                2.02641    0.50825
                                2.02686    0.51748
                                2.02740    0.52932
                                2.02793    0.53926
                                2.02847    0.54612
                                2.02910    0.55512
                                2.02972    0.56412
                                2.03026    0.57335
                                2.03071    0.58211
                                2.03116    0.59087
                                2.03169    0.60176
                                2.03232    0.61194
                                2.03304    0.62117
                                2.03366    0.63111
                                2.03393    0.63857
                                2.03438    0.65029
                                2.03465    0.65739
                                2.03510    0.66876
                                2.03519    0.67728
                                2.03555    0.68403
                                2.03581    0.69503
                                2.03608    0.70320
                                2.03653    0.71255
                                2.03680    0.71883
                                2.03725    0.72960
                                2.03778    0.73788
                                2.03832    0.74522
                                2.03886    0.75682
                                2.03940    0.77103
                                2.03966    0.77884
                                2.04011    0.79281
                                2.04056    0.80228
                                2.04101    0.81222
                                2.04155    0.82453
                                2.04199    0.83424
                                2.04244    0.84584
                                2.04298    0.86004
                                2.04343    0.87117
                                2.04387    0.88111
                                2.04441    0.89248
                                2.04495    0.90100
                                2.04575    0.90384
                                2.04656    0.89129
                                2.04701    0.88182
                                2.04746    0.87259
                                2.04826    0.86845
                                2.04907    0.87259
                                2.05157    0.88324
                                2.05234    0.89354
                                2.05301    0.90384
                                2.05346    0.91236
                                2.05578    0.93136
                                2.05629    0.94290
                                2.05659    0.95545
                                2.05668    0.96208
                                2.05704    0.97309
                                2.05954    0.99244
                                2.06044    0.99948
                                2.06107    0.99546
                                2.06125    0.98765
                                2.06160    0.97581
                                2.06250    0.94337
                                2.06304    0.93059
                                2.06366    0.92266
                                2.06456    0.91449
                                2.06563    0.91035
                                2.06662    0.90341
                                2.06895    0.88750
                                2.06948    0.87496
                                2.07020    0.86975
                                2.07223    0.85105
                                2.07244    0.83974
                                2.07271    0.82856
                                2.07298    0.81648
                                2.07325    0.80583
                                2.07396    0.79109
                                2.07495    0.78251
                                2.07584    0.77440
                                2.07647    0.76179
                                2.07710    0.75132
                                2.07790    0.74319
                                2.07862    0.73291
                                2.07934    0.72231
                                2.08005    0.71563
                                2.08050    0.70462
                                2.08122    0.69693
                                2.08229    0.70551
                                2.08265    0.70693
                                2.08337    0.69859
                                2.08381    0.68817
                                2.08417    0.67586
                                2.08444    0.66698
                                2.08507    0.65540
                                2.08578    0.64698
                                2.08614    0.63893
                                2.08632    0.63111
                                2.08668    0.62022
                                2.08722    0.60365
                                2.08784    0.59258
                                2.08865    0.58467
                                2.08954    0.57767
                                2.09035    0.57181
                                2.09062    0.55903
                                2.09089    0.54979
                                2.09116    0.54021
                                2.09134    0.52884
                                2.09160    0.51730
                                2.09237    0.50356
                                2.09250    0.50256
                                2.09348    0.50256
                                2.09411    0.49191
                                2.09456    0.48220
                                2.09528    0.47714
                                2.09599    0.48250
                                2.09671    0.47108
                                2.09702    0.46670
                                2.09796    0.47330
                                2.09868    0.46255
                                2.09913    0.45249
                                2.09957    0.44314
                                2.09993    0.43438
                                2.10020    0.42320
                                2.10047    0.41237
                                2.10110    0.40112
                                2.10208    0.39987
                                2.10271    0.39106
                                2.10316    0.37970
                                2.10369    0.36573
                                2.10450    0.35673
                                2.10531    0.34608
                                2.10611    0.33791
                                2.10683    0.34489
                                2.10754    0.35413
                                2.10835    0.34655
                                2.10898    0.33690
                                2.10969    0.33424
                                2.11041    0.34880
                                2.11131    0.35223
                                2.11256    0.34691
                                2.11372    0.34816
                                2.11471    0.33980
                                2.11551    0.34324
                                2.11605    0.35129
                                2.11659    0.36265
                                2.11731    0.36904
                                2.11793    0.35803
                                2.11850    0.34679
                                2.11901    0.34578
                                2.11999    0.35384
                                2.12080    0.36318
                                2.12169    0.37082
                                2.12241    0.36265
                                2.12268    0.34987
                                2.12295    0.33868
                                2.12357    0.33197
                                2.12474    0.33543
                                2.12545    0.32856
                                2.12581    0.31595
                                2.12638    0.30465
                                2.12649    0.29335
                                2.12760    0.30938
                                2.12832    0.29944
                                2.12886    0.28612
                                2.12957    0.28275
                                2.13011    0.29447
                                2.13038    0.30228
                                2.13074    0.31412
                                2.13154    0.32027
                                2.13235    0.30773
                                2.13289    0.29802
                                2.13351    0.28825
                                2.13441    0.28169
                                2.13522    0.28760
                                2.13571    0.30441
                                2.13584    0.31507
                                2.13620    0.32536
                                2.13674    0.33477
                                2.13728    0.32501
                                2.13794    0.31350
                                2.13907    0.31731
                                2.14014    0.31365
                                2.14095    0.31436
                                2.14122    0.30264
                                2.14148    0.28973
                                2.14220    0.28083
                                2.14265    0.29163
                                2.14292    0.31134
                                2.14274    0.29873
                                2.14338    0.32582
                                2.14408    0.31365
                                2.14435    0.29991
                                2.14453    0.28997
                                2.14533    0.27884
                                2.14641    0.28618
                                2.14748    0.26855
                                2.14820    0.26642
                                2.14896    0.25348
                                2.14883    0.24570
                                2.14977    0.26687
                                2.15089    0.26204
                                2.15187    0.26659
                                2.15252    0.26876
                                2.15330    0.26937
                                2.15375    0.27565
                                2.15449    0.28623
                                2.15438    0.29092
                                2.15516    0.27671
                                2.15554    0.26571
                                2.15590    0.25860
                                2.15662    0.25020
                                2.15742    0.24168
                                2.15823    0.23161
                                2.15930    0.23339
                                2.16011    0.22889
                                2.16065    0.21990
                                2.16145    0.21244
                                2.16253    0.20794
                                2.16342    0.20889
                                2.16432    0.21824
                                2.16541    0.21620
                                2.16647    0.21705
                                2.16727    0.21090
                                2.16781    0.20072
                                2.16835    0.19125
                                2.16880    0.18545
                                2.16924    0.17492
                                2.16987    0.16397
                                2.17077    0.15976
                                2.17157    0.15716
                                2.17238    0.14982
                                2.17345    0.15112
                                2.17453    0.14485
                                2.17560    0.14449
                                2.17659    0.14206
                                2.17757    0.13680
                                2.17856    0.13169
                                2.17936    0.12029
                                2.18026    0.11182
                                2.18135    0.11464
                                2.18223    0.11283
                                2.18295    0.10395
                                2.18384    0.09667
                                2.18492    0.09194
                                2.18599    0.08661
                                2.18707    0.08377
                                2.18812    0.07600
                                2.18921    0.07158
                                2.19029    0.06980
                                2.19136    0.06613
                                2.19246    0.06265
                                2.19346    0.05868
                                2.19459    0.05501
                                2.19566    0.05228
                                2.19683    0.04980
                                2.19799    0.04459
                                2.19908    0.04021
                                2.20014    0.03666
                                2.20121    0.03405
                                2.20229    0.03121
                                2.20338    0.02870
                                2.20444    0.02731
                                2.20551    0.02518
                                2.20659    0.02340
                                2.20766    0.02103
                                2.20874    0.02021
                                2.20892    0.00955
                                2.20981    0.01843
                                2.20972    0.00896
                                2.21089    0.01807
                                2.21196    0.01500
                                2.21312    0.01156
                                2.21405    0.00947
                                2.21518    0.00896
                                2.21581    0.00896      ]
λs_theory_jank      = λs_SHGrel_theory_jank[:,1]
SHGrel_theory_jank  = λs_SHGrel_theory_jank[:,2]
λs_expt_jank        = λs_SHGrel_expt_jank[:,1]
SHGrel_expt_jank    = λs_SHGrel_expt_jank[:,2]
