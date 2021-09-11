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
function plot_ifg!(ax,ifg,nωs,Δωs,;n_fmb=1,color=:blue,label="log₁₀( Σ (Δng)² )")
    itrs = [x[1] for x in ifg] .+ length(ifg)*(n_fmb-1)
    fs = [x[2] for x in ifg]
    fs_norm = fs .* Δωs / nωs
    # gs = [x[3] for x in ifg]
    scatterlines!(ax,itrs,
		log10.(fs_norm),
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
Δx = 6.0
Δy = 4.0
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).



function plt_rwg_phasematch(λs,nF,nS,ngF,ngS,Λ0,L,EF,ES,ms;ng_nodisp=false,n_dense=3000)
    fig = Figure()
    ax_n = fig[1,1] = Axis(fig)
    ax_ng = fig[2,1] = Axis(fig)
    ax_Λ = fig[3,1] = Axis(fig)
    ax_qpm = fig[4,1] = Axis(fig)
    ωs = inv.(λs)
    ln1 = lines!(ax_n,  λs, nF; color=logocolors[:red],linewidth=2,label="neff(ω)")
    ln2 = lines!(ax_n,  λs, nS; color=logocolors[:blue],linewidth=2,label="neff(2ω)")
    # axislegend(ax_n,position=:rc)
    plot!(ax_n,  λs, nF; color=logocolors[:red],markersize=2)
    plot!(ax_n,  λs, nS; color=logocolors[:blue],markersize=2)

    lines!(ax_ng,  λs, ngF; color=logocolors[:red],linewidth=2,label="ng(ω)")
    lines!(ax_ng,  λs, ngS; color=logocolors[:blue],linewidth=2,label="ng(2ω)")
    # axislegend(ax_ng,position=:rt)
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
    # axislegend(ax_Λ,position=:rb)
    plot!(ax_Λ,  λs, Λ; color=logocolors[:green],markersize=2)

    Δk_qpm = ( 4π ./ λs) .* (  nS .-  nF ) .- (2π / Λ0)
    Δk_qpm_itp = LinearInterpolation(ωs,Δk_qpm)
    ωs_dense = collect(range(extrema(ωs)...,length=n_dense))
    λs_dense = inv.(ωs_dense)
    Δk_qpm_dense = Δk_qpm_itp.(ωs_dense)
    sinc2Δk_dense = (sinc.(Δk_qpm_dense * L / 2.0)).^2

    lines!(ax_qpm, λs_dense, sinc2Δk_dense; color=logocolors[:purple],linewidth=2,label="rel. SHG\npoling=$Λ0")
    # axislegend(ax_qpm,position=:rt)

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

ms_jank = ModeSolver(kguess(1/1.55,rwg(p_jank)), rwg(p_jank), grid; nev=1)

nF_jank,ngF_jank = solve_n(ms_jank,ωs_jank,rwg(p_jank))
EF_jank = copy(E⃗(ms_jank;svecs=false)[eigind])
iEmagmaxF_jank = argmax(abs2.(EF_jank))
EmagmaxF_jank = EF_jank[iEmagmaxF_jank]
ErelF_jank = EF_jank ./ EmagmaxF_jank
ErelF_jank = ErelF_jank ./ maximum(abs.(ErelF_jank))

nS_jank,ngS_jank = solve_n(ms_jank,2*ωs_jank,rwg(p_jank))
ES_jank = copy(E⃗(ms_jank;svecs=false)[eigind])
iEmagmaxS_jank = argmax(abs2.(ES_jank))
EmagmaxS_jank = ES_jank[iEmagmaxS_jank]
ErelS_jank = ES_jank ./ EmagmaxS_jank

Λ0_jank = 5.1201 #5.1201
L_jank = 3e3 # 1cm in μm
# _,ng_jankF_old = solve_n(ms,ωs_jank,rwg(p_jank);ng_nodisp=true)
# _,ng_jankS_old = solve_n(ms,2*ωs_jank,rwg(p_jank);ng_nodisp=true)
##
kF_jank,HF_jank = solve_k(ms_jank,ωs_jank,rwg(p_jank))
kS_jank,HS_jank = solve_k(ms_jank,2*ωs_jank,rwg(p_jank))

# kF_jank,HF_jank = solve_k(ωs_jank,rwg(p_jank),grid)
# kS_jank,HS_jank = solve_k(2*ωs_jank,rwg(p_jank),grid)
##
svecs = false
normalized = true
EF_jank = [E⃗(kF_jank[ind],HF_jank[ind,:,1],ωs_jank[ind],rwg(p_jank),grid; svecs, normalized) for ind=1:length(ωs_jank)]
ES_jank = [E⃗(kS_jank[ind],HS_jank[ind,:,1],2*ωs_jank[ind],rwg(p_jank),grid; svecs, normalized) for ind=1:length(ωs_jank)]
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
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy)

ms_sw = ModeSolver(kguess(1/1.55,rwg(p_sw)), rwg(p_sw), gr; nev=2)
ωs_sw = 1 ./ λs_sw
nF_sw,ngF_sw = solve_n(ms,ωs_sw,rwg(p_sw))
EF_sw = copy(E⃗(ms_sw;svecs=false)[1])
iEmagmaxF_sw = argmax(abs2.(EF_sw))
EmagmaxF_sw = E[iEmagmaxF_sw]
ErelF_sw = EF_sw ./ EmagmaxF_sw
ErelF_sw = ErelF_sw ./ maximum(abs.(ErelF_sw))

nF_sw,ngF_sw = n_swF,ng_swF

nS_sw,ngS_sw = solve_n(ms_sw,2*ωs_sw,rwg(p_sw))

ES_sw = copy(E⃗(ms_sw;svecs=false)[1])
iEmagmaxS_sw = argmax(abs2.(ES_sw))
EmagmaxS_sw = E[iEmagmaxS_sw]
ErelS_sw = ES_sw ./ EmagmaxS_sw
ErelS_sw = ErelS_sw ./ maximum(abs.(ErelS_sw))
# _,ng_swF_old = solve_n(ms,ωs_sw,rwg(p_sw);ng_nodisp=true)
# _,ng_swS_old = solve_n(ms,2*ωs_sw,rwg(p_sw);ng_nodisp=true)
# k1,H1 = solve_k(ms,2*ωs_sw[end],rwg(p_sw))
# Ex1 = E⃗x(ms); Ey1 = E⃗y(ms)
# k2,H2 = solve_k(ms,2*ωs_sw[end],rwg(p_sw))
# Ex2 = E⃗x(ms); Ey2 = E⃗y(ms)
Λ0_sw = 2.8545
L_sw = 1e3

fig_sw = plt_rwg_phasematch(λs_sw,nF_sw,nS_sw,ngF_sw,ngS_sw,Λ0_sw,L_sw,ErelF_sw,ErelS_sw,ms_sw)
fig_sw


##

function solve_SHG(p::Vector{<:Number},ωs,f_geom,grid::Grid;fname="",group=nothing,nev=2,f_filter=nothing,data_dir=joinpath(homedir(),"data"))
    @show fname = "shg_" * fname * ".h5"
    @show fpath 	= joinpath(data_dir, fname)
    λs = inv.(ωs)
    nFS,ngFS,gvdFS,EFS = solve_n(
        ModeSolver(
            kguess(ωs[1],f_geom(p)),
            f_geom(p),
            grid;
            nev,
        ),
        vcat(ωs,2*ωs),
        rwg(p);
        f_filter,
    )

    nω = length(ωs)
    nF = nFS[1:nω]
    ngF = ngFS[1:nω]
    gvdF = gvdFS[1:nω]
    EF = view(EFS,:,:,:,1:nω)

    nS = nFS[nω+1:(2*nω)]
    ngS = ngFS[nω+1:(2*nω)]
    gvdS = gvdFS[nω+1:(2*nω)]
    ES = view(EFS,:,:,:,(nω+1):(2*nω))

    if isnothing(group)
        h5open(fpath, "cw") do fid			# "cw"== "read-write, create if not existing"
            fid["nF"] = nF
            fid["ngF"] = ngF
            fid["gvdF"] = gvdF
            fid["EF"] = EF
            fid["nS"] = nS
            fid["ngS"] = ngS
            fid["gvdS"] = gvdS
            fid["ES"] = ES
        end
    else
        h5open(fpath, "cw") do fid			# "cw"== "read-write, create if not existing"
        	g = create_group(fid,group) #fid[grp_name]
            g["nF"] = nF
            g["ngF"] = ngF
            g["gvdF"] = gvdF
            g["EF"] = EF
            g["nS"] = nS
            g["ngS"] = ngS
            g["gvdS"] = gvdS
            g["ES"] = ES
        end
    end
    return fpath
end

function solve_SHG(ps::Vector{<:Vector},ωs,f_geom,grid::Grid;fname="",group=nothing,nev=2,f_filter=nothing,data_dir=joinpath(homedir(),"data"))
    if isnothing(group)
        groups=[string(i) for i=1:length(ps)]
    else
        groups = [group * string(i) for i=1:length(ps)]
    end
    fpaths = [ solve_SHG(ps[i],ωs,f_geom,grid;fname,group=groups[i],nev,f_filter,data_dir) for i=1:length(ps)]
end

##

ps_opt5 = [[1.676716762547048, 0.43474974558777724, 0.7058578266295201, 0.20717224874415013],
                   [1.644502183820398, 0.9925531919718392, 0.7526681375676043, 0.19466632142136261],
                   [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],
                   [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],
                   [1.4077890254553842, 0.6689819948730713, 0.9011365392777256, 3.6177729235559175e-5],
                   [1.4077343282288142, 0.6689827559319353, 0.9011640749055386, 1.1273307247269671e-6],]

fs_opt5 = [5.744935e-02, 1.521949e-02, 5.953687e-03, 5.747107e-03, 4.163141e-03, 4.161747e-03,]

ps_opt5M = hcat(ps_opt5...)'
ws_opt5 = ps_opt5M[:,1]
ts_opt5 = ps_opt5M[:,2]
rpes_opt5 =  ps_opt5M[:,3]
swas_opt5 =  ps_opt5M[:,4]

# ax_Λ = gl1[1:4,1] = Axis(fig,
#     yaxisposition=:right,
#     ygridvisible=false,
#     yminorgridvisible=false,
#     ylabel = "poling period [μm]",
# )
#
# # ax_gvd = fig[3,1] = Axis(fig)
# ax_qpm = gl1[5:8,1] = Axis(fig,
#     yticks=[0.0,0.25,0.5,0.75,1.0],
#     ylabel = "SHG transfer fn.",
#     xlabel = "λ [μm]",
#     xlabelpadding = 3,
# )
# ax_opts = gl2[2,1] = Axis(fig,xticks=1:6,
#     yscale=AbstractPlotting.log10,
#     yminorticksvisible = true,
#     yminorgridvisible = true,
#     yminorticks = IntervalsBetween(8),
#     ylabel = "log₁₀( Σ (Δng)² )",
#     xlabel = "optimization steps",
#     xlabelpadding = 3,
#     )

##
ipc_theme = Theme(fontsize=16)
set_theme!(ipc_theme)
clrs = [logocolors[:blue],logocolors[:green],:black]
# labels = ["1", "3", "5"]
labels = ["wₜₒₚ [μm]", "tₗₙ [μm]", "etch frac", "sw. angle [rad]"]

fig = Figure(resolution=(340,310))

ax = fig[1,1] = Axis(fig,
    yaxisposition=:right,
    ygridvisible=false,
    yminorgridvisible=false,
    ylabel="parameter value",
    # yscale=AbstractPlotting.log10,
)
axf = fig[1,1] = Axis(fig,xticks=1:6,xlabel="optimization step",ylabel = "cost function Σ (Δng)²",)
sls = [scatterlines!(ax,1:6,p;color=clr,markercolor=clr,marksersize=1.5,strokecolor=clr,label=lbl) for (p,lbl,clr) in zip([ws_opt5,ts_opt5,rpes_opt5,swas_opt5],labels,logocolors)]
slf = scatterlines!(axf,1:6,fs_opt5,color=:black,marker=:rect,markercolor=:black,linewidth=2,markersize=6)
# sls = [scatterlines!(ax,0:10,p;color=clr,markercolor=clr,marker=:utriangle,marksersize=2,strokecolor=clr,label=lbl) for (p,lbl,clr) in zip([ws_opt31,ts_opt31,rpes_opt31,swas_opt31],labels,logocolors)]

# lgnd = Legend(fig[1,2],vcat(sls...,slf),vcat(labels...,"Σ(Δng)² (obj. fn.)")) #,orientation=:horizontal)
# lgnd.width=30
# sls_11 = [scatterlines!(ax,0:5,p;color=clr,markercolor=clr,marker=:xcross,marksersize=2,strokecolor=clr,label=lbl) for (p,lbl,clr) in zip([ws_opt11,ts_opt11,rpes_opt11,swas_opt11],labels,logocolors)]

trim!(fig.layout)
fig

##
save("example_opt_params_v2.png", fig)
save("example_opt_params_v2.svg", fig)

##
data_dir=joinpath(homedir(),"data")
fname = "shg_opt5_its.h5"
fpath 	= joinpath(data_dir, fname)
ωs_opt5 = collect(range(0.6,0.7,length=20))
λs_opt5 = inv.(ωs_opt5)
ωs = ωs_opt5
nω = length(ωs)


nFs,ngFs,nSs,ngSs = h5open(fpath, "r") do fid			# "cw"== "read-write, create if not existing"
    nFs = zeros(nω,6)
    nSs = zeros(nω,6)
    ngFs = zeros(nω,6)
    ngSs = zeros(nω,6)
    for i=1:6
        	g = fid[string(i)] #fid[grp_name]
            nFs[:,i]		=		read(g,"nF")
            ngFs[:,i]	=		read(g,"ngF")
            nSs[:,i]		=		read(g,"nS")
            ngSs[:,i]	=		read(g,"ngS")
    end
	nFs,ngFs,nSs,ngSs
end


##
# nω = 20
# nSs = nS_opt5
# nFs = nF_opt5
# ngSs = ngS_opt5
# ngFs = ngF_opt5
Λs = zeros(nω,6)
Λ0s = [
    3.32,
    6.14,
    4.125,
    4.132,
    4.033,
    4.033
]
n_dense = 3000
L = 1e3
sinc2Δks = zeros(3000,6)
λs = inv.(ωs)
for i=1:6
    Δn = ( nSs[:,i] .- nFs[:,i] )
    Λs[:,i] = (λs ./ 2) ./ Δn
    Δk_qpm = ( 4π ./ λs) .* (  nSs[:,i] .-  nFs[:,i] ) .- (2π / Λ0s[i])
    Δk_qpm_itp = LinearInterpolation(ωs,Δk_qpm)
    ωs_dense = collect(range(extrema(ωs)...,length=n_dense))
    λs_dense = inv.(ωs_dense)
    Δk_qpm_dense = Δk_qpm_itp.(ωs_dense)
    sinc2Δks[:,i] = (sinc.(Δk_qpm_dense * L / 2.0)).^2
end
##
colors= [:red,:blue,:green,:magenta,:cyan,:orange,:black]
ωs_dense = collect(range(extrema(ωs)...,length=n_dense))
λs_dense = inv.(ωs_dense)
fig = Figure()
ax1 = fig[1,1] = Axis(fig,xlabel="λ [μm]",ylabel="SHG transfer function",xlims=(1.5,1.67))
# ax2 = fig[2,1] = Axis(fig)
cg = cgrad(:Blues_7,categorical=true)
# cb = Colorbar(fig[1, 2], width = 25, limits = (0, 6),
#     colormap = cgrad(:Blues_6,categorical=true), label="Opt. step.")
ls1 = [lines!(ax1,λs_dense,sinc2Δks[:,i],color=cg.colors[i+1],linewidth=2) for i=1:6]
text_labels = ["1ˢᵗ","2ⁿᵈ","3ʳᵈ","4ᵗʰ","5ᵗʰ","6ᵗʰ"]
text_pos = [Point2f0(xy) for xy in [(1.6,0.3),(1.65,0.3),(1.51,0.3),(1.58,0.6),(1.56,0.8),(1.64,0.7)]]
txts = [text!(ax1,text_labels[i],position=text_pos[i],color=cg.colors[i+1]) for i=1:6]
# ls2 = [lines!(ax2,λs,Λs[:,i],color=colors[i]) for i=1:6]
xlims!(ax1,(1.5,1.67))
fig
##
save("example_opt_spectra_v1.png", fig)
save("example_opt_spectra_v1.svg", fig)

##

colors= [:red,:blue,:green,:magenta,:cyan,:orange,:black]
ωs_dense = collect(range(extrema(ωs)...,length=n_dense))
λs_dense = inv.(ωs_dense)
fig = Figure(resolution=(400,260))
ax1 = fig[1,1] = Axis(fig,xlabel="λ [μm]",ylabel="SHG transfer function",xlims=(1.5,1.67),) #yticks=[0,1])
# ax2 = fig[2,1] = Axis(fig)
cg = cgrad(:Blues_7,categorical=true)
colors_fill = [RGBA(clr.r,clr.g,clr.b,0.4) for clr in cg.colors]
# cb = Colorbar(fig[1, 2], width = 25, limits = (0, 6),
#     colormap = cgrad(:Blues_6,categorical=true), label="Opt. step.")
offsets = [1.0*i for i=0:5] |> reverse
ls1 = [lines!(ax1,λs_dense,sinc2Δks[:,i].+offsets[i],color=cg.colors[i+1],linewidth=2) for i=1:6]
fbs1 = [band!(λs_dense,zeros(length(λs_dense)).+offsets[i],sinc2Δks[:,i].+offsets[i],color=colors_fill[i+1], transparency=true)  for i=1:6]
text_labels = ["1ˢᵗ","2ⁿᵈ","3ʳᵈ","4ᵗʰ","5ᵗʰ","6ᵗʰ"]
text_pos = [Point2f0(xy) for xy in [(1.6,0.3),(1.65,0.3),(1.51,0.3),(1.58,0.6),(1.56,0.8),(1.64,0.7)]]
# txts = [text!(ax1,text_labels[i],position=text_pos[i],color=cg.colors[i+1]) for i=1:6]
# ls2 = [lines!(ax2,λs,Λs[:,i],color=colors[i]) for i=1:6]
xlims!(ax1,(1.5,1.666))
ylims!(ax1,(-0.05,6.05))
fig

##
save("example_opt_spectra_v2.png", fig)
save("example_opt_spectra_v2.svg", fig)

##



ipc_theme = Theme(fontsize=14)
set_theme!(ipc_theme)
fig = Figure(resolution=(800,260))


##################### Jankowski work comparison ##################
ax_qpm = fig[1,1:3] = Axis(fig,
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
#######################end Jankowski work comparison ####################

clrs = [logocolors[:blue],logocolors[:green],:black]
# labels = ["1", "3", "5"]
labels = ["wₜₒₚ [μm]", "tₗₙ [μm]", "etch frac", "sw. angle [rad]"]

# fig = Figure(resolution=(340,310))
# ww2 = Relative(1/2)
ax = fig[1,4:5] = Axis(fig,
    yaxisposition=:right,
    ygridvisible=false,
    yminorgridvisible=false,
    ylabel="parameter value",
    # width=ww2,
    # yscale=AbstractPlotting.log10,
)
axf = fig[1,4:5] = Axis(fig,xticks=1:6,xlabel="optimization step",ylabel = "cost function Σ (Δng)²")
sls = [scatterlines!(ax,1:6,p;color=clr,markercolor=clr,markersize=5,strokecolor=clr,label=lbl) for (p,lbl,clr) in zip([ws_opt5,ts_opt5,rpes_opt5,swas_opt5],labels,logocolors)]
slf = scatterlines!(axf,1:6,fs_opt5,color=:black,marker=:rect,markercolor=:black,linewidth=2,markersize=6)
# sls = [scatterlines!(ax,0:10,p;color=clr,markercolor=clr,marker=:utriangle,marksersize=2,strokecolor=clr,label=lbl) for (p,lbl,clr) in zip([ws_opt31,ts_opt31,rpes_opt31,swas_opt31],labels,logocolors)]

# lgnd = Legend(fig[1,2],vcat(sls...,slf),vcat(labels...,"Σ(Δng)² (obj. fn.)")) #,orientation=:horizontal)
# lgnd.width=30
# sls_11 = [scatterlines!(ax,0:5,p;color=clr,markercolor=clr,marker=:xcross,marksersize=2,strokecolor=clr,label=lbl) for (p,lbl,clr) in zip([ws_opt11,ts_opt11,rpes_opt11,swas_opt11],labels,logocolors)]

trim!(fig.layout)
###
ωs = ωs_opt5
ωs_dense = collect(range(extrema(ωs)...,length=n_dense))
λs_dense = inv.(ωs_dense)

ax2 = fig[1,6:8] = Axis(fig,xlabel="λ [μm]",ylabel="SHG transfer function",xlims=(1.5,1.67),) #yticks=[0,1])
# ax2 = fig[2,1] = Axis(fig)
cg = cgrad(:Blues_7,categorical=true)
colors_fill = [RGBA(clr.r,clr.g,clr.b,0.4) for clr in cg.colors]
# cb = Colorbar(fig[1, 2], width = 25, limits = (0, 6),
#     colormap = cgrad(:Blues_6,categorical=true), label="Opt. step.")
offsets = [1.0*i for i=0:5] |> reverse
ls1 = [lines!(ax2,λs_dense,sinc2Δks[:,i].+offsets[i],color=cg.colors[i+1],linewidth=2) for i=1:6]
fbs1 = [band!(λs_dense,zeros(length(λs_dense)).+offsets[i],sinc2Δks[:,i].+offsets[i],color=colors_fill[i+1], transparency=true)  for i=1:6]
text_labels = ["1ˢᵗ","2ⁿᵈ","3ʳᵈ","4ᵗʰ","5ᵗʰ","6ᵗʰ"]
text_pos = [Point2f0(xy) for xy in [(1.6,0.3),(1.65,0.3),(1.51,0.3),(1.58,0.6),(1.56,0.8),(1.64,0.7)]]
# txts = [text!(ax2,text_labels[i],position=text_pos[i],color=cg.colors[i+1]) for i=1:6]
# ls2 = [lines!(ax2,λs,Λs[:,i],color=colors[i]) for i=1:6]
xlims!(ax2,(1.5,1.666))
ylims!(ax2,(-0.05,6.05))
fig



fig

##
save("gryphonfig5_alt.png", fig)
save("gryphonfig5_alt.svg", fig)
##
solve_SHG(ps_opt5,
    ωs_opt5,
    rwg,
    Grid(Δx,Δy,Nx,Ny);
    fname="opt5_its",
    group=nothing,
    nev=4,
    f_filter=TE_filter(0.6),
    )
##
ωs_opt1 = collect(range(0.6,0.75,length=25))
λs_opt1 = inv.(ωs_opt1)
ps_opt1 = [ [0.7334970625238433, 0.8159109163565503, 0.37490689234678776, 0.2663147502490992],
            [0.9569226171707059, 0.8855595438747766, 0.9385187240745907, 0.3728323974710411],
            [0.9098864374815023, 0.8685667303188279, 0.8726245107734417, 0.3342403783465787],
            [0.8198687779485274, 0.8308793849071482, 0.7094683855581461, 0.26820939345092315],
            [0.8198687779485274, 0.8308793849071482, 0.7094683855581461, 0.26820939345092315],
            [0.78373842149792, 0.8124145419900792, 0.7355140743205213, 0.23799879822410952],
            [0.7439842575843102, 0.7837018214522238, 0.7002141464334269, 0.20841286119371594],
            [0.7221826911399528, 0.7697286796656979, 0.6980442698956598, 0.19144661826806325],
            [0.7182481316537885, 0.7673982028648796, 0.6983821786792992, 0.18830330084011704],
            [0.7182481316537885, 0.7673982028648796, 0.6983821786792992, 0.18830330084011704],
            [0.7144395565812798, 0.7670623565480135, 0.7133091045570443, 0.1849917948156943],
            [0.6394634795622367, 0.7240976673679469, 0.7477342566612786, 0.13285149688836012],
            [0.6447075386382123, 0.7253846661981084, 0.7506522977431904, 0.13637751200391168],
            [0.629290654535625, 0.7142246705802344, 0.7658459012111655, 0.12536671438304348],
            [0.629290654535625, 0.7142246705802344, 0.7658459012111655, 0.12536671438304348]]

fs_opt1 = [ 2.954875e-02,
            3.607659e-03,
            3.265571e-03,
            2.394051e-03,
            2.172938e-03,
            2.021996e-03,
            1.383538e-03,
            1.163760e-03,
            1.135939e-03,
            1.135697e-03,
            1.026789e-03,
            2.448063e-04,
            2.117484e-04,
            1.085242e-04,
            1.085239e-04,   ]

ps_opt1M = hcat(ps_opt1...)'
ws_opt1 = ps_opt1M[:,1]
ts_opt1 = ps_opt1M[:,2]
rpes_opt1 =  ps_opt1M[:,3]
swas_opt1 =  ps_opt1M[:,4]

p_lower_opt1 = [0.4, 0.3, 0., 0.]
p_upper_opt1 = [2., 2., 1., π/4.]

ws_rel_opt1 = (ws_opt1.-p_lower_opt1[1])./(p_upper_opt1[1].-p_lower_opt1[1])
ts_rel_opt1 = (ts_opt1.-p_lower_opt1[2])./(p_upper_opt1[2].-p_lower_opt1[2])
rpes_rel_opt1 = (rpes_opt1.-p_lower_opt1[3])./(p_upper_opt1[3].-p_lower_opt1[3])
swas_rel_opt1 = (swas_opt1.-p_lower_opt1[4])./(p_upper_opt1[4].-p_lower_opt1[4])

##

# solve_SHG(ps_opt1,
#     ωs_opt1,
#     rwg,
#     Grid(6.0,4.0,128,128);
#     fname="opt1_its",
#     group=nothing,
#     nev=4,
#     f_filter=TE_filter(0.6),
#     )


##

clrs = [logocolors[:blue],logocolors[:green],:black]
# labels = ["1", "3", "5"]
labels = ["wₜₒₚ [μm]", "tₗₙ [μm]", "etch frac", "sw. angle [rad]"]

fig = Figure(resolution=(400,300))
ax = fig[1,1] = Axis(fig,xticks=[5,10,15],xlabel="optimization step",ylabel="parameter value")
axf = fig[1,1] = Axis(fig,
    yaxisposition=:right,
    ygridvisible=false,
    yminorgridvisible=false,
    ylabel = "Σ (Δng)²",
    # yscale=AbstractPlotting.log10,

)

# sls = [scatterlines!(ax,1:15,p;color=clr,markercolor=clr,markersize=4,strokecolor=clr,label=lbl) for (p,lbl,clr) in zip([ws_opt1,ts_opt1,rpes_opt1,swas_opt1],labels,logocolors)]
sls = [scatterlines!(ax,1:15,p;color=clr,markercolor=clr,markersize=4,strokecolor=clr,label=lbl) for (p,lbl,clr) in zip([ws_rel_opt1,ts_rel_opt1,rpes_rel_opt1,swas_rel_opt1],labels,logocolors)]

slf = scatterlines!(axf,1:15,fs_opt1,color=:black,marker=:rect,markercolor=:black,linewidth=2,markersize=6)
# sls = [scatterlines!(ax,0:10,p;color=clr,markercolor=clr,marker=:utriangle,marksersize=2,strokecolor=clr,label=lbl) for (p,lbl,clr) in zip([ws_opt31,ts_opt31,rpes_opt31,swas_opt31],labels,logocolors)]

# lgnd = Legend(fig[1,2],vcat(sls...,slf),vcat(labels...,"Σ(Δng)² (obj. fn.)")) #,orientation=:horizontal)
# lgnd.width=30
# sls_11 = [scatterlines!(ax,0:5,p;color=clr,markercolor=clr,marker=:xcross,marksersize=2,strokecolor=clr,label=lbl) for (p,lbl,clr) in zip([ws_opt11,ts_opt11,rpes_opt11,swas_opt11],labels,logocolors)]

xlims!(ax,[0.5,15.5])
xlims!(axf,[0.5,15.5])
trim!(fig.layout)
fig



##

data_dir=joinpath(homedir(),"data")
fname = "shg_opt1_its.h5"
fpath 	= joinpath(data_dir, fname)
# ωs_opt1 = collect(range(0.6,0.7,length=))
ωs_opt1 = collect(range(0.6,0.75,length=16))
λs_opt1 = inv.(ωs_opt1)
ωs = ωs_opt1
nω = length(ωs)

n_steps = 8
nFs,ngFs,nSs,ngSs = h5open(fpath, "r") do fid			# "cw"== "read-write, create if not existing"
    nFs = zeros(nω,n_steps)
    nSs = zeros(nω,n_steps)
    ngFs = zeros(nω,n_steps)
    ngSs = zeros(nω,n_steps)
    for i=1:n_steps
        	g = fid[string(i)] #fid[grp_name]
            nFs[:,i]		=		read(g,"nF")
            ngFs[:,i]	=		read(g,"ngF")
            nSs[:,i]		=		read(g,"nS")
            ngSs[:,i]	=		read(g,"ngS")
    end
	nFs,ngFs,nSs,ngSs
end

##

Λs = zeros(nω,n_steps)
Λ0s = [
    4.55,
    4.63,
    4.375,
    3.97,
    3.97,
    3.715,
    3.383,
    4.033,
    4.033,
    4.033,
]
n_dense = 3000
L = 1e3
sinc2Δks = zeros(3000,n_steps)
λs = inv.(ωs)
for i=1:n_steps
    Δn = ( nSs[:,i] .- nFs[:,i] )
    Λs[:,i] = (λs ./ 2) ./ Δn
    Δk_qpm = ( 4π ./ λs) .* (  nSs[:,i] .-  nFs[:,i] ) .- (2π / Λ0s[i])
    Δk_qpm_itp = LinearInterpolation(ωs,Δk_qpm)
    ωs_dense = collect(range(extrema(ωs)...,length=n_dense))
    λs_dense = inv.(ωs_dense)
    Δk_qpm_dense = Δk_qpm_itp.(ωs_dense)
    sinc2Δks[:,i] = (sinc.(Δk_qpm_dense * L / 2.0)).^2
end

##
colors= [:red,:blue,:green,:magenta,:cyan,:orange,:black]
ωs_dense = collect(range(extrema(ωs)...,length=n_dense))
λs_dense = inv.(ωs_dense)
fig = Figure()
ax1 = fig[1,1] = Axis(fig,xlabel="λ [μm]",ylabel="SHG transfer fn.") #,xlims=(1.5,1.67))
# ax2 = fig[2,1] = Axis(fig)
cg = cgrad(:Blues_9,categorical=true)
colors_fill = [RGBA(clr.r,clr.g,clr.b,0.4) for clr in cg.colors]
# cb = Colorbar(fig[1, 2], width = 25, limits = (0, 6),
#     colormap = cgrad(:Blues_6,categorical=true), label="Opt. step.")
offsets = [1.0*i for i=0:n_steps-1] |> reverse
ls1 = [lines!(ax1,λs_dense,sinc2Δks[:,i].+offsets[i],color=cg.colors[i+1],linewidth=2) for i=1:n_steps]
fbs1 = [band!(λs_dense,zeros(length(λs_dense)).+offsets[i],sinc2Δks[:,i].+offsets[i],color=colors_fill[i+1], transparency=true)  for i=1:n_steps]
text_labels = ["1ˢᵗ","2ⁿᵈ","3ʳᵈ","4ᵗʰ","5ᵗʰ","6ᵗʰ"]
text_pos = [Point2f0(xy) for xy in [(1.6,0.3),(1.65,0.3),(1.51,0.3),(1.58,0.6),(1.56,0.8),(1.64,0.7)]]
# txts = [text!(ax1,text_labels[i],position=text_pos[i],color=cg.colors[i+1]) for i=1:6]
# ls2 = [lines!(ax2,λs,Λs[:,i],color=colors[i]) for i=1:6]
# xlims!(ax1,(1.5,1.67))
fig




##


ps_opt31 =     [[0.629290654535625, 0.7142246705802344, 0.7658459012111655, 0.12536671438304348],
            [0.6283201962381307, 0.7132504824781828, 0.764045851409942, 0.12484492708004957],
   [0.6262064467363574, 0.7127950938473154, 0.7647345374158281, 0.12339323983458993],
    [0.6213759330223262, 0.7126452148572809, 0.7680087600052793, 0.12001043923861131],
    [0.6232805242024614, 0.7140814804188901, 0.7669124563551152, 0.12117549825751972],
     [0.6230710825610506, 0.7141516026813306, 0.7670320839398395, 0.12102003698418824],
      [0.6213923071189889, 0.7147006954627761, 0.7675802589445816, 0.11991632812708061],
      [0.6213923071189889, 0.7147006954627761, 0.7675802589445816, 0.11991632812708061],
     [0.6214729092237913, 0.7147961071499247, 0.7676450008308302, 0.11995964726861383],
      [0.6214729092237913, 0.7147961071499247, 0.7676450008308302, 0.11995964726861383],
     [0.6214729092638253, 0.7147961071816575, 0.7676450008257523, 0.11995964728751145],]

# ps_opt31 =         [[0.4858236499543672, 1.457695458268576, 0.034171494389373835, 0.5989537685691309],
#                    [0.4854969613872562, 1.2102665810468323, 0.4269287896155209, 0.5989294113226804],
#                    [0.4853224522172848, 1.2274472438261546, 0.3988325399680015, 0.5987270250244889],
#                    [0.48527391339696485, 1.2207005032492637, 0.40905160488333975, 0.5986960144111407],
#                    [0.4837994770827205, 1.2195349139084013, 0.40650867240366934, 0.5972939277851915],
#                    [0.4530632837741969, 1.156600872107478, 0.4143888525596142, 0.5678757306129366],]

ps_opt31M = hcat(ps_opt31...)'
ws_opt31 = ps_opt31M[:,1]
ts_opt31 = ps_opt31M[:,2]
rpes_opt31 =  ps_opt31M[:,3]
swas_opt31 =  ps_opt31M[:,4]

ps_opt11 =               [  [ 1.676716762547048, 0.43474974558777724, 0.7058578266295201, 0.20717224874415013],
                   [1.644502183820398, 0.9925531919718392, 0.7526681375676043, 0.19466632142136261],
                   [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],
                   [1.4546456244430521, 0.6662419432874312, 0.8756311701451857, 0.030211630755089425],
                   [1.4077890254553842, 0.6689819948730713, 0.9011365392777256, 3.6177729235559175e-5],
                   [1.4077343282288142, 0.6689827559319353, 0.9011640749055386, 1.1273307247269671e-6], ]

ps_opt11M = hcat(ps_opt11...)'
ws_opt11 = ps_opt11M[:,1]
ts_opt11 = ps_opt11M[:,2]
rpes_opt11 =  ps_opt11M[:,3]
swas_opt11 =  ps_opt11M[:,4]

##
solve_SHG(ps_opt31,
    ωs_opt3,
    rwg,
    Grid(Δx,Δy,Nx,Ny);
    fname="opt31_its",
    group=nothing,
    nev=4,
    f_filter=TE_filter(0.6),
    )
##
p_opt5 = [1.4077343282288142, 0.6689827559319353, 0.9011640749055386, 1.1273307247269671e-6]

ωs_opt5 = collect(range(0.6,0.7,length=20))
λs_opt5 = inv.(ωs_opt5)
# λs_opt5 = collect(reverse(1.4:0.01:1.6))
# ωs_opt5 = 1 ./ λs_opt5
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy)

ms_opt5 = ModeSolver(kguess(1/1.55,rwg(p_opt5)), rwg(p_opt5), gr; nev=4)

# nFS_opt5,ngFS_opt5,gvdFS_opt5,EFS_opt5 = solve_n(
#     ms_opt5,
#     vcat(ωs_opt5,2*ωs_opt5),
#     rwg(p_opt5);
#     f_filter=TE_filter(0.6),
# )
#
# nω_opt5 = length(ωs_opt5)
# nF_opt5 = nFS_opt5[1:nω_opt5]
# ngF_opt5 = ngFS_opt5[1:nω_opt5]
# gvdF_opt5 = gvdFS_opt5[1:nω_opt5]
# EF_opt5 = view(EFS_opt5,:,:,:,1:nω_opt5)
#
# nS_opt5 = nFS_opt5[nω_opt5+1:end]
# ngS_opt5 = ngFS_opt5[nω_opt5+1:end]
# gvdS_opt5 = gvdFS_opt5[nω_opt5+1:end]
# ES_opt5 = view(EFS_opt5,:,:,:,(nω_opt5+1):(2*nω_opt5))

data_dir=joinpath(homedir(),"data")
fname = "qpm_opt5.h5"
grp_name="128"
fpath 	= joinpath(data_dir, fname)

# # fid 	= h5open(fpath, "cw")
# h5open(fpath, "cw") do fid			# "cw"== "read-write, create if not existing"
# 	g = create_group(fid,grp_name) #fid[grp_name]
#     g["nF"] = nF_opt5
#     g["ngF"] = ngF_opt5
#     g["gvdF"] = gvdF_opt5
#     g["EF"] = EF_opt5
#     g["nS"] = nS_opt5
#     g["ngS"] = ngS_opt5
#     g["gvdS"] = gvdS_opt5
#     g["ES"] = ES_opt5
# end

nF_opt5,ngF_opt5,gvdF_opt5,EF_opt5,nS_opt5,ngS_opt5,gvdS_opt5,ES_opt5 = h5open(fpath, "r") do fid			# "cw"== "read-write, create if not existing"
	g = fid[grp_name] #fid[grp_name]
    nF_opt5		=		read(g,"nF")
    ngF_opt5	=		read(g,"ngF")
    gvdF_opt5	=		read(g,"gvdF")
    EF_opt5		=		read(g,"EF")
    nS_opt5		=		read(g,"nS")
    ngS_opt5	=		read(g,"ngS")
    gvdS_opt5	=		read(g,"gvdS")
    ES_opt5		=		read(g,"ES")
	nF_opt5,ngF_opt5,gvdF_opt5,EF_opt5,nS_opt5,ngS_opt5,gvdS_opt5,ES_opt5
end
##
Λ0_opt5 = 4.034 # 128x128
# Λ0_opt5 = 2.86275 # 256x256
L_opt5 = 1e3 # 1cm in μm
Eind_opt5 = 5
fig_opt5 = plt_rwg_phasematch(λs_opt5,nF_opt5,nS_opt5,ngF_opt5,ngS_opt5,Λ0_opt5,L_opt5,EF_opt5[:,:,:,Eind_opt5],ES_opt5[:,:,:,Eind_opt5],ms_opt5)
fig_opt5


##
p_opt1 = [0.629290654535625, 0.7142246705802344, 0.7658459012111655, 0.12536671438304348]

ωs_opt1 = collect(range(0.6,0.75,length=25))
λs_opt1 = inv.(ωs_opt1)
# λs_opt1 = collect(reverse(1.4:0.01:1.6))
# ωs_opt1 = 1 ./ λs_opt1
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy)

ms_opt1 = ModeSolver(kguess(1/1.55,rwg(p_opt1)), rwg(p_opt1), gr; nev=4)

# nFS_opt1,ngFS_opt1,gvdFS_opt1,EFS_opt1 = solve_n(
#     ms_opt1,
#     vcat(ωs_opt1,2*ωs_opt1),
#     rwg(p_opt1);
#     f_filter=TE_filter(0.6),
# )
#
# nω_opt1 = length(ωs_opt1)
# nF_opt1 = nFS_opt1[1:nω_opt1]
# ngF_opt1 = ngFS_opt1[1:nω_opt1]
# gvdF_opt1 = gvdFS_opt1[1:nω_opt1]
# EF_opt1 = view(EFS_opt1,:,:,:,1:nω_opt1)
#
# nS_opt1 = nFS_opt1[nω_opt1+1:end]
# ngS_opt1 = ngFS_opt1[nω_opt1+1:end]
# gvdS_opt1 = gvdFS_opt1[nω_opt1+1:end]
# ES_opt1 = view(EFS_opt1,:,:,:,(nω_opt1+1):(2*nω_opt1))

data_dir=joinpath(homedir(),"data")
fname = "qpm_opt1.h5"
grp_name="128"
fpath 	= joinpath(data_dir, fname)
# # fid 	= h5open(fpath, "cw")
# h5open(fpath, "cw") do fid			# "cw"== "read-write, create if not existing"
# 	g = create_group(fid,grp_name) #fid[grp_name]
#     g["nF"] = nF_opt1
#     g["ngF"] = ngF_opt1
#     g["gvdF"] = gvdF_opt1
#     g["EF"] = EF_opt1
#     g["nS"] = nS_opt1
#     g["ngS"] = ngS_opt1
#     g["gvdS"] = gvdS_opt1
#     g["ES"] = ES_opt1
# end

nF_opt1,ngF_opt1,gvdF_opt1,EF_opt1,nS_opt1,ngS_opt1,gvdS_opt1,ES_opt1 = h5open(fpath, "r") do fid			# "cw"== "read-write, create if not existing"
	g = fid[grp_name] #fid[grp_name]
    nF_opt1		=		read(g,"nF")
    ngF_opt1	=		read(g,"ngF")
    gvdF_opt1	=		read(g,"gvdF")
    EF_opt1		=		read(g,"EF")
    nS_opt1		=		read(g,"nS")
    ngS_opt1	=		read(g,"ngS")
    gvdS_opt1	=		read(g,"gvdS")
    ES_opt1		=		read(g,"ES")
	nF_opt1,ngF_opt1,gvdF_opt1,EF_opt1,nS_opt1,ngS_opt1,gvdS_opt1,ES_opt1
end

##
Λ0_opt1 = 2.725 # 128x128
# Λ0_opt1 = 2.86275 # 256x256
L_opt1 = 1e3 # 1cm in μm
Eind_opt1 = 5
fig_opt1 = plt_rwg_phasematch(λs_opt1,nF_opt1,nS_opt1,ngF_opt1,ngS_opt1,Λ0_opt1,L_opt1,EF_opt1[:,:,:,Eind_opt1],ES_opt1[:,:,:,Eind_opt1],ms_opt1)
fig_opt1

##
p_opt3 = [0.62147291,0.71479611,0.767645,0.11996]
p_opt1 = [0.629290654535625, 0.7142246705802344, 0.7658459012111655, 0.12536671438304348]
ωs_opt3 = collect(range(0.6,0.75,length=25))
λs_opt3 = inv.(ωs_opt3)
# λs_opt3 = collect(reverse(1.4:0.01:1.6))
# ωs_opt3 = 1 ./ λs_opt3
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy)
ms_opt3 = ModeSolver(kguess(1/1.55,rwg(p_opt3)), rwg(p_opt3), gr; nev=4)

# nFS_opt3,ngFS_opt3,gvdFS_opt3,EFS_opt3 = solve_n(
#     ms_opt3,
#     vcat(ωs_opt3,2*ωs_opt3),
#     rwg(p_opt3);
#     f_filter=TE_filter(0.6),
# )
#
# nω_opt3 = length(ωs_opt3)
# nF_opt3 = nFS_opt3[1:nω_opt3]
# ngF_opt3 = ngFS_opt3[1:nω_opt3]
# gvdF_opt3 = gvdFS_opt3[1:nω_opt3]
# EF_opt3 = view(EFS_opt3,:,:,:,1:nω_opt3)
#
# nS_opt3 = nFS_opt3[nω_opt3+1:end]
# ngS_opt3 = ngFS_opt3[nω_opt3+1:end]
# gvdS_opt3 = gvdFS_opt3[nω_opt3+1:end]
# ES_opt3 = view(EFS_opt3,:,:,:,(nω_opt3+1):(2*nω_opt3))
#
data_dir=joinpath(homedir(),"data")
fname = "qpm_opt3.h5"
grp_name="128"
fpath 	= joinpath(data_dir, fname)
# # #fid 	= h5open(fpath, "cw")
# h5open(fpath, "cw") do fid			# "cw"== "read-write, create if not existing"
# 	g = create_group(fid,grp_name) #fid[grp_name]
#     g["nF"] = nF_opt3
#     g["ngF"] = ngF_opt3
#     g["gvdF"] = gvdF_opt3
#     g["EF"] = EF_opt3
#     g["nS"] = nS_opt3
#     g["ngS"] = ngS_opt3
#     g["gvdS"] = gvdS_opt3
#     g["ES"] = ES_opt3
# end
nF_opt3,ngF_opt3,gvdF_opt3,EF_opt3,nS_opt3,ngS_opt3,gvdS_opt3,ES_opt3 = h5open(fpath, "r") do fid			# "cw"== "read-write, create if not existing"
	g = fid[grp_name] #fid[grp_name]
    nF_opt3		=		read(g,"nF")
    ngF_opt3	=		read(g,"ngF")
    gvdF_opt3	=		read(g,"gvdF")
    EF_opt3		=		read(g,"EF")
    nS_opt3		=		read(g,"nS")
    ngS_opt3	=		read(g,"ngS")
    gvdS_opt3	=		read(g,"gvdS")
    ES_opt3		=		read(g,"ES")
	nF_opt3,ngF_opt3,gvdF_opt3,EF_opt3,nS_opt3,ngS_opt3,gvdS_opt3,ES_opt3
end
##
# Λ0_opt3 = 2.86275 # 256x256
Λ0_opt3 = 2.6912 # 128x128
L_opt3 = 3e3 # 1cm in μm
Eind_opt3 = 10
fig_opt3 = plt_rwg_phasematch(λs_opt3,nF_opt3,nS_opt3,ngF_opt3,ngS_opt3,Λ0_opt3,L_opt3,EF_opt3[:,:,:,Eind_opt3],ES_opt3[:,:,:,Eind_opt3],ms_opt3)


##
p_opt4 = [0.636143209603401, 0.9218203566788058, 0.440673406907343, 0.2398469659583548]
# ωs_opt4 =  collect(range(0.6,0.75,length=20)) # reverse(1.45:0.02:1.65)
ωs_opt4 =  collect(range(0.62,0.68,length=16))
λs_opt4 = inv.(ωs_opt4)
# λs_opt4 = collect(reverse(1.4:0.01:1.6))
# ωs_opt4 = 1 ./ λs_opt4
Δx,Δy,Δz,Nx,Ny,Nz = 10.0, 4.0, 1.0, 256, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy)
ms_opt4 = ModeSolver(kguess(0.62,rwg(p_opt4)), rwg(p_opt4), gr; nev=4)

# nFS_opt4,ngFS_opt4,gvdFS_opt4,EFS_opt4 = solve_n(
#     ms_opt4,
#     vcat(ωs_opt4,2*ωs_opt4),
#     rwg(p_opt4);
#     f_filter=TE_filter(0.6),
# )
#
# nω_opt4 = length(ωs_opt4)
# nF_opt4 = nFS_opt4[1:nω_opt4]
# ngF_opt4 = ngFS_opt4[1:nω_opt4]
# gvdF_opt4 = gvdFS_opt4[1:nω_opt4]
# EF_opt4 = view(EFS_opt4,:,:,:,1:nω_opt4)
#
# nS_opt4 = nFS_opt4[nω_opt4+1:end]
# ngS_opt4 = ngFS_opt4[nω_opt4+1:end]
# gvdS_opt4 = gvdFS_opt4[nω_opt4+1:end]
# ES_opt4 = view(EFS_opt4,:,:,:,(nω_opt4+1):(2*nω_opt4))
#
data_dir=joinpath(homedir(),"data")
fname = "qpm_opt4.h5"
grp_name="128"
fpath 	= joinpath(data_dir, fname)
#
# h5open(fpath, "cw") do fid			# "cw"== "read-write, create if not existing"
# 	g = create_group(fid,grp_name) #fid[grp_name]
#     g["nF"] = nF_opt4
#     g["ngF"] = ngF_opt4
#     g["gvdF"] = gvdF_opt4
#     g["EF"] = EF_opt4
#     g["nS"] = nS_opt4
#     g["ngS"] = ngS_opt4
#     g["gvdS"] = gvdS_opt4
#     g["ES"] = ES_opt4
# end

nF_opt4,ngF_opt4,gvdF_opt4,EF_opt4,nS_opt4,ngS_opt4,gvdS_opt4,ES_opt4 = h5open(fpath, "r") do fid			# "cw"== "read-write, create if not existing"
	g = fid[grp_name] #fid[grp_name]
    nF_opt4		=		read(g,"nF")
    ngF_opt4	=		read(g,"ngF")
    gvdF_opt4	=		read(g,"gvdF")
    EF_opt4		=		read(g,"EF")
    nS_opt4		=		read(g,"nS")
    ngS_opt4	=		read(g,"ngS")
    gvdS_opt4	=		read(g,"gvdS")
    ES_opt4		=		read(g,"ES")
	nF_opt4,ngF_opt4,gvdF_opt4,EF_opt4,nS_opt4,ngS_opt4,gvdS_opt4,ES_opt4
end
##

Λ0_opt4 = 4.703 # 128x128
# Λ0_opt4 = 2.86275 # 256x256
L_opt4 = 1e3 # 1cm in μm
Eind_opt4 = 7
fig_opt4 = plt_rwg_phasematch(λs_opt4,nF_opt4,nS_opt4,ngF_opt4,ngS_opt4,Λ0_opt4,L_opt4,EF_opt4[:,:,:,Eind_opt4],ES_opt4[:,:,:,Eind_opt4],ms_opt4)

##
# fig = Figure()
# ax_n = fig[1,1] = Axis(fig)
# ax_ng = fig[2,1] = Axis(fig)
# ax_Λ = fig[3,1] = Axis(fig)
# ax_qpm = fig[4,1] = Axis(fig)
#
# lines!(ax_n, λs_opt5, n_opt5F; color=logocolors[:red],linewidth=2)
# lines!(ax_n, λs_opt5, n_opt5S; color=logocolors[:blue],linewidth=2)
# plot!(ax_n, λs_opt5, n_opt5F; color=logocolors[:red],markersize=2)
# plot!(ax_n, λs_opt5, n_opt5S; color=logocolors[:blue],markersize=2)
#
# lines!(ax_ng, λs_opt5, ng_opt5F; color=logocolors[:red],linewidth=2)
# lines!(ax_ng, λs_opt5, ng_opt5S; color=logocolors[:blue],linewidth=2)
# plot!(ax_ng, λs_opt5, ng_opt5F; color=logocolors[:red],markersize=2)
# plot!(ax_ng, λs_opt5, ng_opt5S; color=logocolors[:blue],markersize=2)
#
# # lines!(ax_ng, λs_opt5, ng_opt5F_old; color=logocolors[:red],linewidth=2,linestyle=:dash)
# # lines!(ax_ng, λs_opt5, ng_opt5S_old; color=logocolors[:blue],linewidth=2,linestyle=:dash)
# # plot!(ax_ng, λs_opt5, ng_opt5F_old; color=logocolors[:red],markersize=2)
# # plot!(ax_ng, λs_opt5, ng_opt5S_old; color=logocolors[:blue],markersize=2)
#
# # Δk_opt5 = ( 4π ./ λs_opt5 ) .* ( n_opt5S .- n_opt5F )
# # Λ_opt5 = 2π ./ Δk_opt5
# Λ_opt5 = ( λs_opt5 ./ 2 ) ./ ( n_opt5S .- n_opt5F )
#
# lines!(ax_Λ, λs_opt5, Λ_opt5; color=logocolors[:green],linewidth=2)
# plot!(ax_Λ, λs_opt5, Λ_opt5; color=logocolors[:green],markersize=2)
#
# Λ0_opt5 = 3.961 # 128x128
# # Λ0_opt5 = 2.86275 # 256x256
# L_opt5 = 1e3 # 1cm in μm
# Δk_qpm_opt5 = ( 4π ./ λs_opt5 ) .* ( n_opt5S .- n_opt5F ) .- (2π / Λ0_opt5)
#
# Δk_qpm_opt5_itp = LinearInterpolation(ωs_opt5,Δk_qpm_opt5)
# ωs_opt5_dense = collect(range(extrema(ωs_opt5)...,length=3000))
# λs_opt5_dense = inv.(ωs_opt5_dense)
# Δk_qpm_opt5_dense = Δk_qpm_opt5_itp.(ωs_opt5_dense)
# sinc2Δk_opt5_dense = (sinc.(Δk_qpm_opt5_dense * L_opt5 / 2.0)).^2
# lines!(ax_qpm, λs_opt5_dense, sinc2Δk_opt5_dense; color=logocolors[:purple],linewidth=2)
#
# #fig
#
# Ex_axes = fig[1:2, 2] = [Axis(fig, title = t) for t in ["|Eₓ₁|²","|Eₓ₂|²"]] #,"|Eₓ₃|²","|Eₓ₄|²"]]
# Ey_axes = fig[3:4, 2] = [Axis(fig, title = t) for t in ["|Ey₁|²","|Ey₂|²"]] #,"|Ey₃|²","|Ey₄|²"]]
# # Es = [Ex[1],Ey[1],Ex[2],Ey[2],Ex[3],Ey[3],Ex[4],Ey[4]]
#
# Earr = [ε⁻¹_dot( fft( kx_tc( unflat(ms.H⃗; ms)[i],mn(ms),ms.M̂.mag), (2:3) ), copy(flat( ms.M̂.ε⁻¹ ))) for i=1:2]
# Ex = [Earr[i][1,:,:] for i=1:2]
# Ey = [Earr[i][2,:,:] for i=1:2]
#
# heatmaps_x = [heatmap!(ax, abs2.(Ex[i])) for (i, ax) in enumerate(Ex_axes)]
# heatmaps_y = [heatmap!(ax, abs2.(Ey[i])) for (i, ax) in enumerate(Ey_axes)]
#
# fig

##
