""" opt 1 : flattening ng ( I think ) """
ωs00 = [ 1.4285714285714286,
         1.3333333333333333,
         1.25,
         1.1764705882352942,
         1.1111111111111112,
         1.0526315789473684,
         1.0,
         0.9523809523809523,
         0.9090909090909091 ]


ifg_fmb001 = [   [0,     8.498314e-05,     6.594861e-04],
            [1,     1.454428e-05,     6.436278e-05],
            [2,     9.629194e-06,     1.519242e-05],
            [3,     9.413559e-06,     2.616302e-06],
            [4,     9.409294e-06,     1.897483e-07],
            [5,     9.409267e-06,     4.172450e-07],
            [6,     9.409176e-06,     2.768810e-07],
            [7,     9.408899e-06,     5.473291e-07],
            [8,     9.384243e-06,     3.170912e-06],
            [9,     9.373474e-06,     2.602562e-06],
            [10,    9.370656e-06,     1.740643e-06],
    ]

ifg_fmb002 = [    [0,     9.211430e-06,     1.730827e-06],
    [1,     9.210854e-06,     9.089252e-07],
    [2,     9.210443e-06,     7.891384e-07],
    [3,     9.210445e-06,     8.765670e-07],
    [4,     9.209873e-06,     2.727668e-07],
    [5,     9.209785e-06,     3.367705e-07],
    [6,     9.209783e-06,     1.117497e-06],
    [7,     9.209783e-06,     1.641101e-07],
    [8,     9.209783e-06,     4.363236e-07],
    [9,     9.209783e-06,     3.558985e-07],
    [10,    9.209783e-06,     3.591572e-07],
    ]

x_fmb001 = [ [1.7, 0.7, 0.2243994752564138],
            [1.7222247638281714, 1.1284276222760623, 0.23961042451866174],
            [1.71504171568567, 1.0193577435257075, 0.23382600465803224],
            [1.7135719021732212, 0.9973925130806568, 0.23268484704229064],
            [1.7133502360445376, 0.9942087788196526, 0.23254607153151022],
            [1.7131016116594016, 0.9946405934445752, 0.23289711152390843],
            [1.7130017961257242, 0.9943479415361147, 0.2329369748922578],
            [1.7115416726550623, 0.9936108304399704, 0.23420490389766824],
            [1.5968021469518514, 0.9934582861328269, 0.3435855180133476],
            [1.5027032121131607, 0.9966340024568505, 0.433766156254365],
            [1.484843243367148, 0.9982116766701578, 0.4509252644953674],]

x_fmb002 = [ [1.484843243367148, 0.9982116766701578, 0.4509252644953674],
            [1.4850740472917792, 1.0014066726233688, 0.45061462483159415],
            [1.486473089259767, 0.9997031788826629, 0.4411853916739758],
            [1.486472361899325, 0.9997024485532594, 0.4411760639931854],
            [1.4975730114085066, 1.0006481861213496, 0.4316610242954845],
            [1.5003888052504515, 1.0006730508628525, 0.42941796131760473],
            [1.500431423173811, 1.0006720359324792, 0.42938519391612834],
            [1.500431423173811, 1.0006720359324792, 0.4293851939161282],
            [1.5004314231738107, 1.0006720359324788, 0.4293851939161289],
            [1.5004314231738105, 1.0006720359324783, 0.4293851939161294],
            [1.5004314231738105, 1.0006720359324783, 0.4293851939161294],]

 """ opt 0 : minimizing <ng²> without material disp in ng """

ωs0 = [   0.625,
         0.6289308176100629,
         0.6329113924050632,
         0.6369426751592356,
         0.641025641025641,
         0.6451612903225806,
         0.6493506493506493,
         0.6535947712418301,
         0.6578947368421053,
         0.6622516556291391,
         0.6666666666666666,
         0.6711409395973155,
         0.6756756756756757,
         0.6802721088435374,
         0.684931506849315,
         0.6896551724137931,
         0.6944444444444444,
         0.6993006993006994,
         0.7042253521126761,
         0.7092198581560284,
         0.7142857142857143,
    ]

ifg_fmb01 = [   [0,     6.131435e-03,     2.147646e-02],
                [1,     1.834491e-03,     9.487007e-03],
                [2,     7.125945e-05,     3.282646e-03],
                [3,     6.007662e-05,     2.053307e-03],
                [4,     5.153350e-05,     6.574411e-04],    ]

ifg_fmb02 =  [  [0,     3.119590e-05,     6.720872e-04],
                [1,     3.063802e-05,     5.404484e-04],
                [2,     2.960157e-05,     2.652610e-04],
                [3,     2.951907e-05,     1.981138e-04],
                [4,     2.870806e-05,     3.363244e-03],    ]

x_fmb01 =       [[1.7, 0.7, 0.2243994752564138],
                [1.7975026277763393, 0.594324514363967, 0.30403130784474214],
                [1.881847401756922, 0.5424118417300735, 0.36728030475652684],
                [1.883760187134629, 0.5391388169651558, 0.36871766585300764],
                [1.8862240428283517, 0.5332345375383438, 0.37066043731861265],]

x_fmb02 =       [[1.8862240428283517, 0.5332345375383438, 0.37066043731861265],
                [1.886172701939964, 0.5323243054043917, 0.3706325060782424],
                [1.8826171663681932, 0.5303377349423279, 0.36836585858129717],
                [1.8813456693984658, 0.5304900000534588, 0.3678895801579516],
                [1.8011508036004562, 0.5200089884193593, 0.3417385069329951],]

 """ opt 1-4 : minimizing <ng²> with material disp in ng """

 λs1 = reverse(1.45:0.02:1.65)
 ωs1 = 1 ./ λs1

 ifg_fmb11 = [   [0,     5.744935e-02,     5.697679e-01],
                 [1,     1.521949e-02,     3.004023e-02],
                 [2,     5.953687e-03,     1.956094e-02],   ]

 ifg_fmb12 =  [  [0,     5.747107e-03,     1.953595e-02],
                 [1,     4.163141e-03,     1.651796e-02],
                 [2,     4.161747e-03,     1.644100e-02],    ]

 x_fmb11 =       [  [1.676716762547048, 0.43474974558777724, 0.7058578266295201, 0.20717224874415013],
                    [1.644502183820398, 0.9925531919718392, 0.7526681375676043, 0.19466632142136261],
                    [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425], ]

 x_fmb12 =       [  [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],
                    [1.4077890254553842, 0.6689819948730713, 0.9011365392777256, 3.6177729235559175e-5],
                    [1.4077343282288142, 0.6689827559319353, 0.9011640749055386, 1.1273307247269671e-6], ]

# ωs_opt2 = collect(0.58:0.01:0.72)
# λs_opt2 = inv.(ωs_opt2)
λs2 = reverse(1.45:0.02:1.65)
ωs2 = 1 ./ λs2

ifg_fmb21 =       [[0,     4.196073e-03,     5.849341e-03],
                   [1,     1.651162e-04,     2.571512e-04],
                   [2,     1.643483e-04,     1.358986e-04],
                   [3,     1.642274e-04,     1.805282e-04],
                   [4,     1.274098e-04,     7.147653e-04],
                   [5,     1.220156e-04,     1.015985e-03],]

x_fmb21 =         [[1.3953487636667194, 1.696287815825243, 0.02471676311871973, 0.4470011635893353],
                   [1.392267259916656, 1.3076499287011645, 0.47197580870703526, 0.44681155059650324],
                   [1.392208904313119, 1.3036411608732554, 0.4765150523416733, 0.4467823535695984],
                   [1.3919349080100405, 1.3011821291920767, 0.47766534304145514, 0.4465331633208667],
                   [1.3377131874659247, 1.1120841800675583, 0.34036484899217845, 0.39512281240837],
                   [1.3268125214499205, 1.0710776245823201, 0.3162223761376952, 0.38481443313198],]
# p_opt2 = [1.3268125214499205, 1.0710776245823201, 0.3162223761376952, 0.38481443313198]

λs3 = reverse(1.45:0.02:1.65)
ωs3 = 1 ./ λs3

ifg_fmb31 =       [[0,     2.794650e-03,     7.826270e-03],
                  [1,     1.053719e-04,     3.193261e-03],
                  [2,     8.657558e-05,     1.252427e-03],
                  [3,     7.779650e-05,     1.303005e-04],
                  [4,     7.751930e-05,     2.498477e-04],
                  [5,     5.519917e-05,     5.054679e-04],]

x_fmb31 =         [[0.4858236499543672, 1.457695458268576, 0.034171494389373835, 0.5989537685691309],
                   [0.4854969613872562, 1.2102665810468323, 0.4269287896155209, 0.5989294113226804],
                   [0.4853224522172848, 1.2274472438261546, 0.3988325399680015, 0.5987270250244889],
                   [0.48527391339696485, 1.2207005032492637, 0.40905160488333975, 0.5986960144111407],
                   [0.4837994770827205, 1.2195349139084013, 0.40650867240366934, 0.5972939277851915],
                   [0.4530632837741969, 1.156600872107478, 0.4143888525596142, 0.5678757306129366],]

λs4 = reverse(1.45:0.02:1.65)
ωs4 = 1 ./ λs4

ifg_fmb41 =       [[0,     1.162767e-03,     5.674435e-03],
                  [1,     5.193007e-04,     2.615410e-03],
                  [2,     2.275973e-05,     2.936016e-03],
                  [3,     1.303306e-05,     2.236529e-03],
                  [4,     2.101732e-06,     1.264689e-04],
                  [5,     2.061630e-06,     4.083142e-05],]

x_fmb41 =         [[0.6382558898028023, 1.1437215614013543, 0.07316373932835196, 0.24018205347692223],
                   [0.6370301632361384, 1.0504104425087728, 0.22829139185770422, 0.23996507910397993],
                   [0.6363301562157482, 0.9296839628709647, 0.42781800284642346, 0.23993118097118646],
                   [0.6363405184696823, 0.9275076869288639, 0.43148329800378926, 0.23994504920751902],
                   [0.6361894223771479, 0.9222135985572288, 0.4400993066019841, 0.23987162124817837],
                   [0.636143209603401, 0.9218203566788058, 0.440673406907343, 0.2398469659583548],]


ωs5 = collect(range(0.6,0.7,length=10))
λs5 = inv.(ωs5)

ifg_fmb51 =       [[0,     5.744935e-02,     5.697679e-01],
                  [1,     1.521949e-02,     3.004023e-02],
                  [2,     5.953687e-03,     1.956094e-02],]

ifg_fmb52 =       [[0,     5.747107e-03,     1.953595e-02],
                  [1,     4.163141e-03,     1.651796e-02],
                  [2,     4.161747e-03,     1.644100e-02],]

x_fmb51 =         [[1.676716762547048, 0.43474974558777724, 0.7058578266295201, 0.20717224874415013],
                   [1.644502183820398, 0.9925531919718392, 0.7526681375676043, 0.19466632142136261],
                   [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],]

x_fmb52 =         [[1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],
                   [1.4077890254553842, 0.6689819948730713, 0.9011365392777256, 3.6177729235559175e-5],
                   [1.4077343282288142, 0.6689827559319353, 0.9011640749055386, 1.1273307247269671e-6],]

##
using GLMakie, AbstractPlotting, Interpolations, FFTW
using GLMakie: lines, lines!, heatmap, heatmap!, image, image!
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
function plot_ifg!(ax,ifg;n_fmb=1,color=:blue)
    itrs = [x[1] for x in ifg] .+ length(ifg)*(n_fmb-1)
    fs = [x[2] for x in ifg]
    log10fs_rel = log10.(fs./fs[1])
    gs = [x[3] for x in ifg]
    log10gs_rel = log10.(gs./gs[1])
    lines!(ax[1],itrs,log10fs_rel,linewidth=2;color=color)
    lines!(ax[2],itrs,log10gs_rel,linewidth=2;color=color)
    plot!(ax[1],itrs,log10fs_rel;color=color)
    plot!(ax[2],itrs,log10gs_rel;color=color)
end

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


# save("rwg_qpm_jank.png", fig_jank)
## swedish 1550->780 result
# parameters used by Swedish group (Gallo) to get broadband phase-matching in
# x-cut partially etched, unclad MgO:LiNbO₃-on-SiO₂ ridge waveguides:
# Fergestad and Gallo, "Ultrabroadband second harmonic generation at telecom wavelengths in lithium niobate waveguides"
# Integrated Photonics Research, Silicon and Nanophotonics (pp. ITu4A-13). OSA 2020
p_sw = [
    0.7,        # 700 nm top width of angle sidewall ridge
    0.6,        # 600 nm MgO:LiNbO₃ ridge thickness
    5. / 6.,    # etch fraction (they say etch depth of 500 nm, full thickness 600 nm)
    0.349,      # 20° sidewall angle in radians (they call it 70° , in our params 0° is vertical)
]

λs_sw = collect(reverse(1.4:0.01:1.6))

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy)

ms_sw = ModeSolver(kguess(1/1.55,rwg_pe(p_sw)), rwg_pe(p_sw), gr; nev=2)
ωs_sw = 1 ./ λs_sw
nF_sw,ngF_sw = solve_n(ms,ωs_sw,rwg_pe(p_sw))
EF_sw = copy(E⃗(ms_sw;svecs=false)[1])
iEmagmaxF_sw = argmax(abs2.(EF_sw))
EmagmaxF_sw = E[iEmagmaxF_sw]
ErelF_sw = EF_sw ./ EmagmaxF_sw
ErelF_sw = ErelF_sw ./ maximum(abs.(ErelF_sw))

nF_sw,ngF_sw = n_swF,ng_swF

nS_sw,ngS_sw = solve_n(ms_sw,2*ωs_sw,rwg_pe(p_sw))

ES_sw = copy(E⃗(ms_sw;svecs=false)[1])
iEmagmaxS_sw = argmax(abs2.(ES_sw))
EmagmaxS_sw = E[iEmagmaxS_sw]
ErelS_sw = ES_sw ./ EmagmaxS_sw
ErelS_sw = ErelS_sw ./ maximum(abs.(ErelS_sw))
# _,ng_swF_old = solve_n(ms,ωs_sw,rwg_pe(p_sw);ng_nodisp=true)
# _,ng_swS_old = solve_n(ms,2*ωs_sw,rwg_pe(p_sw);ng_nodisp=true)
# k1,H1 = solve_k(ms,2*ωs_sw[end],rwg_pe(p_sw))
# Ex1 = E⃗x(ms); Ey1 = E⃗y(ms)
# k2,H2 = solve_k(ms,2*ωs_sw[end],rwg_pe(p_sw))
# Ex2 = E⃗x(ms); Ey2 = E⃗y(ms)
Λ0_sw = 2.8545
L_sw = 1e3

fig_sw = plt_rwg_phasematch(λs_sw,nF_sw,nS_sw,ngF_sw,ngS_sw,Λ0_sw,L_sw,ErelF_sw,ErelS_sw,ms_sw)
fig_sw



##
p_opt5 = [1.4077343282288142, 0.6689827559319353, 0.9011640749055386, 1.1273307247269671e-6]

ωs_opt5 = collect(range(0.6,0.7,length=10))
λs_opt5 = inv.(ωs_opt5)
# λs_opt5 = collect(reverse(1.4:0.01:1.6))
# ωs_opt5 = 1 ./ λs_opt5
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy)

ms_opt5 = ModeSolver(kguess(1/1.55,rwg_pe(p_opt5)), rwg_pe(p_opt5), gr; nev=2)


nF_opt5,ngF_opt5 = solve_n(ms_opt5,ωs_opt5,rwg_pe(p_opt5))
EF_opt5 = copy(E⃗(ms_opt5;svecs=false)[1])
iEmagmaxF_opt5 = argmax(abs2.(EF_opt5))
EmagmaxF_opt5 = EF_opt5[iEmagmaxF_opt5]
ErelF_opt5 = EF_opt5 ./ EmagmaxF_opt5
# ErelF_opt5 = ErelF_opt5 ./ maximum(abs.(ErelF_opt5))

nS_opt5,ngS_opt5 = solve_n(ms_opt5,2*ωs_opt5,rwg_pe(p_opt5))
ES_opt5 = copy(E⃗(ms_opt5;svecs=false)[2])
iEmagmaxS_opt5 = argmax(abs2.(ES_opt5))
EmagmaxS_opt5 = ES_opt5[iEmagmaxS_opt5]
ErelS_opt5 = ES_opt5 ./ EmagmaxS_opt5
# ErelF_opt5 = ErelF_opt5 ./ maximum(abs.(ErelF_opt5))

Λ0_opt5 = 3.961 # 128x128
# Λ0_opt5 = 2.86275 # 256x256
L_opt5 = 1e3 # 1cm in μm
fig_opt5 = plt_rwg_phasematch(λs_opt5,nF_opt5,nS_opt5,ngF_opt5,ngS_opt5,Λ0_opt5,L_opt5,ErelF_opt5,ErelS_opt5,ms_opt5)
fig_opt5

# _,ng_opt5F_old = solve_n(ms,ωs_opt5,rwg_pe(p_opt5);ng_nodisp=true)
# _,ng_opt5S_old = solve_n(ms,2*ωs_opt5,rwg_pe(p_opt5);ng_nodisp=true)
# k1,H1 = solve_k(ms,2*ωs_opt5[end],rwg_pe(p_opt5))
# Ex1 = E⃗x(ms); Ey1 = E⃗y(ms)
# k2,H2 = solve_k(ms,2*ωs_opt5[end],rwg_pe(p_opt5))
# Ex2 = E⃗x(ms); Ey2 = E⃗y(ms)


##
p_opt1 = [0.629290654535625, 0.7142246705802344, 0.7658459012111655, 0.12536671438304348]
ωs_opt1 = collect(range(0.6,0.75,length=25))
λs_opt1 = inv.(ωs_opt1)
# λs_opt1 = collect(reverse(1.4:0.01:1.6))
# ωs_opt1 = 1 ./ λs_opt1
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy)

ms_opt1 = ModeSolver(kguess(1/1.55,rwg_pe(p_opt1)), rwg_pe(p_opt1), gr; nev=2)
ε⁻¹_opt1 = εₛ⁻¹(first(ωs_opt1),rwg_pe(p_opt1);ms=ms_opt1)
update_ε⁻¹(ms_opt1,ε⁻¹_opt1)
nF_opt1,ngF_opt1 = solve_n(ms_opt1,ωs_opt1,rwg_pe(p_opt1))
EF_opt1 = copy(E⃗(ms_opt1;svecs=false)[1])
iEmagmaxF_opt1 = argmax(abs2.(EF_opt1))
EmagmaxF_opt1 = EF_opt1[iEmagmaxF_opt1]
ErelF_opt1 = EF_opt1 ./ EmagmaxF_opt1
# ErelF_opt1 = ErelF_opt1 ./ maximum(abs.(ErelF_opt1))

nS_opt1,ngS_opt1 = solve_n(ms_opt1,2*ωs_opt1,rwg_pe(p_opt1))
ES_opt1 = copy(E⃗(ms_opt1;svecs=false)[2])
iEmagmaxS_opt1 = argmax(abs2.(ES_opt1))
EmagmaxS_opt1 = ES_opt1[iEmagmaxS_opt1]
ErelS_opt1 = ES_opt1 ./ EmagmaxS_opt1
# ErelF_opt1 = ErelF_opt1 ./ maximum(abs.(ErelF_opt1))

Λ0_opt1 = 2.6561 # 128x128
# Λ0_opt1 = 2.86275 # 256x256
L_opt1 = 1e3 # 1cm in μm
fig_opt1 = plt_rwg_phasematch(λs_opt1,nF_opt1,nS_opt1,ngF_opt1,ngS_opt1,Λ0_opt1,L_opt1,ErelF_opt1,ErelS_opt1,ms_opt1)
fig_opt1

##
p_opt3 = [0.62147291,0.71479611,0.767645,0.11996]
ωs_opt3 = collect(range(0.6,0.75,length=25))
λs_opt3 = inv.(ωs_opt3)
# λs_opt3 = collect(reverse(1.4:0.01:1.6))
# ωs_opt3 = 1 ./ λs_opt3
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy)

ms_opt3 = ModeSolver(kguess(1/1.55,rwg_pe(p_opt3)), rwg_pe(p_opt3), gr; nev=2)
ε⁻¹_opt3 = εₛ⁻¹(first(ωs_opt3),rwg_pe(p_opt3);ms=ms_opt3)
update_ε⁻¹(ms_opt3,ε⁻¹_opt3)
nF_opt3,ngF_opt3 = solve_n(ms_opt3,ωs_opt3,rwg_pe(p_opt3))
EF_opt3 = copy(E⃗(ms_opt3;svecs=false)[1])
iEmagmaxF_opt3 = argmax(abs2.(EF_opt3))
EmagmaxF_opt3 = EF_opt3[iEmagmaxF_opt3]
ErelF_opt3 = EF_opt3 ./ EmagmaxF_opt3
# ErelF_opt3 = ErelF_opt3 ./ maximum(abs.(ErelF_opt3))

nS_opt3,ngS_opt3 = solve_n(ms_opt3,2*ωs_opt3,rwg_pe(p_opt3))
ES_opt3 = copy(E⃗(ms_opt3;svecs=false)[2])
iEmagmaxS_opt3 = argmax(abs2.(ES_opt3))
EmagmaxS_opt3 = ES_opt3[iEmagmaxS_opt3]
ErelS_opt3 = ES_opt3 ./ EmagmaxS_opt3
# ErelF_opt3 = ErelF_opt3 ./ maximum(abs.(ErelF_opt3))

# Λ0_opt3 = 2.86275 # 256x256
Λ0_opt3 = 2.6182 # 128x128
L_opt3 = 3e3 # 1cm in μm
fig_opt3
fig_opt3 = plt_rwg_phasematch(λs_opt3,nF_opt3,nS_opt3,ngF_opt3,ngS_opt3,Λ0_opt3,L_opt3,ErelF_opt3,ErelS_opt3,ms_opt3)
##
fig = Figure()
ax_n = fig[1,1] = Axis(fig)
ax_ng = fig[2,1] = Axis(fig)
ax_Λ = fig[3,1] = Axis(fig)
ax_qpm = fig[4,1] = Axis(fig)

lines!(ax_n, λs_opt5, n_opt5F; color=logocolors[:red],linewidth=2)
lines!(ax_n, λs_opt5, n_opt5S; color=logocolors[:blue],linewidth=2)
plot!(ax_n, λs_opt5, n_opt5F; color=logocolors[:red],markersize=2)
plot!(ax_n, λs_opt5, n_opt5S; color=logocolors[:blue],markersize=2)

lines!(ax_ng, λs_opt5, ng_opt5F; color=logocolors[:red],linewidth=2)
lines!(ax_ng, λs_opt5, ng_opt5S; color=logocolors[:blue],linewidth=2)
plot!(ax_ng, λs_opt5, ng_opt5F; color=logocolors[:red],markersize=2)
plot!(ax_ng, λs_opt5, ng_opt5S; color=logocolors[:blue],markersize=2)

# lines!(ax_ng, λs_opt5, ng_opt5F_old; color=logocolors[:red],linewidth=2,linestyle=:dash)
# lines!(ax_ng, λs_opt5, ng_opt5S_old; color=logocolors[:blue],linewidth=2,linestyle=:dash)
# plot!(ax_ng, λs_opt5, ng_opt5F_old; color=logocolors[:red],markersize=2)
# plot!(ax_ng, λs_opt5, ng_opt5S_old; color=logocolors[:blue],markersize=2)

# Δk_opt5 = ( 4π ./ λs_opt5 ) .* ( n_opt5S .- n_opt5F )
# Λ_opt5 = 2π ./ Δk_opt5
Λ_opt5 = ( λs_opt5 ./ 2 ) ./ ( n_opt5S .- n_opt5F )

lines!(ax_Λ, λs_opt5, Λ_opt5; color=logocolors[:green],linewidth=2)
plot!(ax_Λ, λs_opt5, Λ_opt5; color=logocolors[:green],markersize=2)

Λ0_opt5 = 3.961 # 128x128
# Λ0_opt5 = 2.86275 # 256x256
L_opt5 = 1e3 # 1cm in μm
Δk_qpm_opt5 = ( 4π ./ λs_opt5 ) .* ( n_opt5S .- n_opt5F ) .- (2π / Λ0_opt5)

Δk_qpm_opt5_itp = LinearInterpolation(ωs_opt5,Δk_qpm_opt5)
ωs_opt5_dense = collect(range(extrema(ωs_opt5)...,length=3000))
λs_opt5_dense = inv.(ωs_opt5_dense)
Δk_qpm_opt5_dense = Δk_qpm_opt5_itp.(ωs_opt5_dense)
sinc2Δk_opt5_dense = (sinc.(Δk_qpm_opt5_dense * L_opt5 / 2.0)).^2
lines!(ax_qpm, λs_opt5_dense, sinc2Δk_opt5_dense; color=logocolors[:purple],linewidth=2)

#fig

Ex_axes = fig[1:2, 2] = [Axis(fig, title = t) for t in ["|Eₓ₁|²","|Eₓ₂|²"]] #,"|Eₓ₃|²","|Eₓ₄|²"]]
Ey_axes = fig[3:4, 2] = [Axis(fig, title = t) for t in ["|Ey₁|²","|Ey₂|²"]] #,"|Ey₃|²","|Ey₄|²"]]
# Es = [Ex[1],Ey[1],Ex[2],Ey[2],Ex[3],Ey[3],Ex[4],Ey[4]]

Earr = [ε⁻¹_dot( fft( kx_tc( unflat(ms.H⃗; ms)[i],mn(ms),ms.M̂.mag), (2:3) ), copy(flat( ms.M̂.ε⁻¹ ))) for i=1:2]
Ex = [Earr[i][1,:,:] for i=1:2]
Ey = [Earr[i][2,:,:] for i=1:2]

heatmaps_x = [heatmap!(ax, abs2.(Ex[i])) for (i, ax) in enumerate(Ex_axes)]
heatmaps_y = [heatmap!(ax, abs2.(Ey[i])) for (i, ax) in enumerate(Ey_axes)]

fig

##
