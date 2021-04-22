using GLMakie, AbstractPlotting, Interpolations, FFTW
using GLMakie: lines, lines!, heatmap, heatmap!
using AbstractPlotting.GeometryBasics
import Colors: JULIA_LOGO_COLORS
logocolors = JULIA_LOGO_COLORS
AbstractPlotting.convert_arguments(x::GeometryPrimitives.Polygon) = (GeometryBasics.Polygon([Point2f0(x.v[i,:]) for i=1:size(x.v)[1]]),) #(GeometryBasics.Polygon([Point2f0(x.v[i,:]) for i=1:size(x.v)[1]]),)
plottype(::GeometryPrimitives.Polygon) = Poly
AbstractPlotting.convert_arguments(x::GeometryPrimitives.Box) = (GeometryBasics.Rect2D((x.c-x.r)..., 2*x.r...),) #(GeometryBasics.Polygon(Point2f0.(coordinates(GeometryBasics.Rect2D((x.c-x.r)..., 2*x.r...)))),)
plottype(::GeometryPrimitives.Box) = Poly
AbstractPlotting.convert_arguments(P::Type{<:Poly}, x::Geometry) = (x.shapes...,)
plottype(::Geometry) = Poly
using Colors, ColorSchemes, ColorSchemeTools
using PyCall
cplot = pyimport("cplot")

# noto_sans = "../assets/NotoSans-Regular.ttf"
# noto_sans_bold = "../assets/NotoSans-Bold.ttf"

"""
Takes an array of complex number and converts it to an array of [r, g, b],
where phase gives hue and saturaton/value are given by the absolute value.
Especially for use with imshow for complex plots.
"""
function complex_to_rgb(X; alpha=1.0, colorspace="cam16")
    return [RGB(x...) for x in cplot.get_srgb1.(X;alpha,colorspace)]
end

##

Δx = 6.0
Δy = 4.0
rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).

function plt_rwg_phasematch(λs,nF,nS,ngF,ngS,Λ0,L,EF,ES,ms;ng_nodisp=false,n_dense=3000)
    fig = Figure()
    ax_n = fig[1,1] = Axis(fig)
    ax_ng = fig[2,1] = Axis(fig)
    ax_Λ = fig[3,1] = Axis(fig)
    ax_qpm = fig[4,1] = Axis(fig)
    ωs = inv.(λs)
    ln1 = lines!(ax_n,  λs, nF; color=logocolors[:red],linewidth=2,label="neff(ω)")
    ln2 = lines!(ax_n,  λs, nS; color=logocolors[:blue],linewidth=2,label="neff(2ω)")
    axislegend(ax_n,position=:rc)
    plot!(ax_n,  λs, nF; color=logocolors[:red],markersize=2)
    plot!(ax_n,  λs, nS; color=logocolors[:blue],markersize=2)

    lines!(ax_ng,  λs, ngF; color=logocolors[:red],linewidth=2,label="ng(ω)")
    lines!(ax_ng,  λs, ngS; color=logocolors[:blue],linewidth=2,label="ng(2ω)")
    axislegend(ax_ng,position=:rt)
    plot!(ax_ng,  λs, ngF; color=logocolors[:red],markersize=2)
    plot!(ax_ng,  λs, ngS; color=logocolors[:blue],markersize=2)
    if ng_nodisp
        lines!(ax_ng,  λs, ngF_old; color=logocolors[:red],linestyle=:dash,linewidth=2)
        lines!(ax_ng,  λs, ngS_old; color=logocolors[:blue],linestyle=:dash,linewidth=2)
        plot!(ax_ng,  λs, ngF_old; color=logocolors[:red],markersize=2)
        plot!(ax_ng,  λs, ngS_old; color=logocolors[:blue],markersize=2)
    end

    Δn = ( nS .- nF )
    # Δk = 4π .* ωs .* Δn
    Λ = (λs ./ 2) ./ Δn

    lines!(ax_Λ,  λs, Λ; color=logocolors[:green],linewidth=2,label="poling period [μm]")
    axislegend(ax_Λ,position=:rb)
    plot!(ax_Λ,  λs, Λ; color=logocolors[:green],markersize=2)

    Δk_qpm = ( 4π ./ λs) .* (  nS .-  nF ) .- (2π / Λ0)
    Δk_qpm_itp = LinearInterpolation(ωs,Δk_qpm)
    ωs_dense = collect(range(extrema(ωs)...,length=n_dense))
    λs_dense = inv.(ωs_dense)
    Δk_qpm_dense = Δk_qpm_itp.(ωs_dense)
    sinc2Δk_dense = (sinc.(Δk_qpm_dense * L / 2.0)).^2

    lines!(ax_qpm, λs_dense, sinc2Δk_dense; color=logocolors[:purple],linewidth=2,label="rel. SHG\npoling=$Λ0")
    axislegend(ax_qpm,position=:rt)

    # Spatial plots
    xs = x(ms.grid)
    ys = y(ms.grid)

    ax_nx = fig[2,2] = Axis(fig)
    ax_E = fig[3:4,2] = [ Axis(fig) for i=1:2] #, title = t) for t in [ "|Eₓ|² @ ω", "|Eₓ|² @ 2ω" ] ]
    nx = sqrt.(getindex.(inv.(ms.M̂.ε⁻¹),1,1))
    Ex = [ EF[1,:,:], ES[1,:,:] ]
    cmaps_E = [:linear_ternary_red_0_50_c52_n256, :linear_ternary_blue_0_44_c57_n256]
    labels_E = ["rel. |Eₓ|² @ ω","rel. |Eₓ|² @ 2ω"]
    heatmaps_E = [heatmap!(ax, xs, ys, abs2.(Ex[i]),colormap=cmaps_E[i],label=labels_E[i]) for (i, ax) in enumerate(ax_E)]
    hm_nx = heatmap!(ax_nx,xs,ys,nx;colormap=:viridis)
    text!(ax_nx,"nₓ",position=(1.4,1.1),textsize=0.7,color=:white)
    text!(ax_E[1],"rel. |Eₓ|² (ω)",position=(-1.4,1.1),textsize=0.7,color=:white)
    text!(ax_E[2],"rel. |Eₓ|² (2ω)",position=(-1.7,1.1),textsize=0.7,color=:white)
    cbar_nx = fig[2,3] = Colorbar(fig,hm_nx ) #,label="nₓ")
    cbar_EF = fig[3,3] = Colorbar(fig,heatmaps_E[1]) #,label="rel. |Eₓ|² @ ω")
    cbar_ES = fig[4,3] = Colorbar(fig,heatmaps_E[2]) #,label="rel. |Eₓ|² @ 2ω")
    for cb in [cbar_EF, cbar_ES, cbar_nx]
        cb.width=30
        cb.height=Relative(2/3)
    end
    # label, format
    hidexdecorations!(ax_E[1])
    hidexdecorations!(ax_n)
    hidexdecorations!(ax_ng)
    hidexdecorations!(ax_Λ)
    ax_E[1].ylabel = "y [μm]"
    ax_E[2].ylabel = "y [μm]"
    ax_E[2].xlabel = "x [μm]"
    ax_qpm.xlabel = "λ [μm]"
    ax_nx.aspect=DataAspect()
    ax_E[1].aspect=DataAspect()
    ax_E[2].aspect=DataAspect()

    return fig
end



##
fig_opts = Figure(resolution = (1200, 700))
axes_fsgs = fig_opts[1:2,1] = [Axis(fig_opts) for t in ["cost fn vals","grad norms"]]

ifgs = [ ifg_fmb11, ifg_fmb21, ifg_fmb31, ifg_fmb41, vcat(ifg_fmb51, [x.+[3.,0.,0.] for x in ifg_fmb52]) ]
ifg_colors = [ :red, :blue, :green, :purple, :black, :cyan ]
ifg_nfmbs = [ 1, 1, 1, 1, 1, 1 ]

[ plot_ifg!(axes_fsgs,ifgs[i],n_fmb=ifg_nfmbs[i],color=ifg_colors[i]) for i = 1:length(ifgs)]
# plot_ifg!(axes_fsgs,ifg_fmb51)
# plot_ifg!(axes_fsgs,ifg_fmb52,n_fmb=2)

# label, format
hidexdecorations!(axes_fsgs[1])
axes_fsgs[1].ylabel = "log₁₀ rel. cost fn\nf/f₀ (f = Σ (Δng)²)"
axes_fsgs[2].xlabel = "iterations"
axes_fsgs[2].ylabel = "log₁₀ rel. cost fn \ngradient mag |g⃗|/|g⃗₀|"

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

# λs_jank = collect(reverse(1.9:0.02:2.4))
# ωs_jank = 1 ./ λs_jank

ωs_jank = collect(range(0.416,0.527,step=0.01)) # collect(0.416:0.01:0.527)
λs_jank = 1 ./ ωs_jank

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
grid = Grid(Δx,Δy,Nx,Ny)
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 512, 512, 1;

ms_jank = ModeSolver(kguess(1/1.55,rwg_pe(p_jank)), rwg_pe(p_jank), grid; nev=1)

nF_jank,ngF_jank = solve_n(ms_jank,ωs_jank,rwg_pe(p_jank))
EF_jank = copy(E⃗(ms_jank;svecs=false)[eigind])
iEmagmaxF_jank = argmax(abs2.(EF_jank))
EmagmaxF_jank = EF_jank[iEmagmaxF_jank]
ErelF_jank = EF_jank ./ EmagmaxF_jank
ErelF_jank = ErelF_jank ./ maximum(abs.(ErelF_jank))

nS_jank,ngS_jank = solve_n(ms_jank,2*ωs_jank,rwg_pe(p_jank))
ES_jank = copy(E⃗(ms_jank;svecs=false)[eigind])
iEmagmaxS_jank = argmax(abs2.(ES_jank))
EmagmaxS_jank = ES_jank[iEmagmaxS_jank]
ErelS_jank = ES_jank ./ EmagmaxS_jank

Λ0_jank = 5.1201 #5.1201
L_jank = 3e3 # 1cm in μm
# _,ng_jankF_old = solve_n(ms,ωs_jank,rwg_pe(p_jank);ng_nodisp=true)
# _,ng_jankS_old = solve_n(ms,2*ωs_jank,rwg_pe(p_jank);ng_nodisp=true)
##
kF_jank,HF_jank = solve_k(ms_jank,ωs_jank,rwg_pe(p_jank))
kS_jank,HS_jank = solve_k(ms_jank,2*ωs_jank,rwg_pe(p_jank))

# kF_jank,HF_jank = solve_k(ωs_jank,rwg_pe(p_jank),grid)
# kS_jank,HS_jank = solve_k(2*ωs_jank,rwg_pe(p_jank),grid)
##
svecs = false
normalized = true
EF_jank = [E⃗(kF_jank[ind],HF_jank[ind,:,1],ωs_jank[ind],rwg_pe(p_jank),grid; svecs, normalized) for ind=1:length(ωs_jank)]
ES_jank = [E⃗(kS_jank[ind],HS_jank[ind,:,1],2*ωs_jank[ind],rwg_pe(p_jank),grid; svecs, normalized) for ind=1:length(ωs_jank)]
function Eperp_max(E)
    Eperp = E[1:2,:,:]
    imagmax = argmax(abs2.(Eperp))
    return abs(Eperp[imagmax])
end
𝓐(n,ng,E) = inv( n * ng * Eperp_max(E)^2)
AF_jank = 𝓐.(nF_jank, ngF_jank, EF_jank) # inv.(nF_jank .* ngF_jank)
AS_jank = 𝓐.(nS_jank, ngS_jank, ES_jank) # inv.(nS_jank .* ngS_jank)
ÊF_jank = EF_jank .* sqrt.(AF_jank .* nF_jank .* ngF_jank)
ÊS_jank = ES_jank .* sqrt.(AS_jank .* nS_jank .* ngS_jank)
𝓐₁₂₃_jank = ( AS_jank .* AF_jank.^2  ).^(1.0/3.0)
# 𝓞_jank = [sum( ( conj.(ES_jank[ind]) .* EF_jank[ind].^2 )  ./ 𝓐₁₂₃_jank[ind]  .* δ(grid)) for ind=1:length(ωs_jank)] #
𝓞_jank = [real(sum( conj.(ÊS_jank[ind]) .* ÊF_jank[ind].^2 )) / 𝓐₁₂₃_jank[ind] * δ(grid) for ind=1:length(ωs_jank)] #
maximum(abs2.(ES_jank[1]),dims=(2,3))


d₃₃ =   25.0    #   pm/V
d₃₁ =   4.4     #   pm/V
d₂₂ =   1.9     #   pm/V

#          xx      yy       zz      zy      zx      xy
deff = [    0.      0.      0.      0.      d₃₁     -d₂₂     #   x
            -d₂₂    d₂₂     0.      d₃₁     0.      0.       #   y
            d₃₁     d₃₁     d₃₃     0.      0.      0.   ]   #   z


GLMakie.heatmap(abs2.(ES_jank[1][1,:,:]))
image!(complex_to_rgb(EF_jank[1][1,:,:]))

fig_jank = plt_rwg_phasematch(λs_jank,nF_jank,nS_jank,ngF_jank,ngS_jank,Λ0_jank,L_jank,ErelF_jank,ErelS_jank,ms_jank)
fig_jank
