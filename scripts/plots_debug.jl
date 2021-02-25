# using CairoMakie, AbstractPlotting
using GLMakie, AbstractPlotting
using AbstractPlotting.GeometryBasics

# noto_sans = "../assets/NotoSans-Regular.ttf"
# noto_sans_bold = "../assets/NotoSans-Bold.ttf"

esm1N = Node(esm1)
grN = Node(gr)
##
fig = Figure(resolution = (1200, 700), backgroundcolor = RGBf0(0.98, 0.98, 0.98) ) # , font = noto_sans)
ax_shapes = fig[2, 1] = Axis(fig, title = "ε")
ax_eps = fig[2, 2] = Axis(fig, title = "ε")
# ax_Esq = fig[3,1]
linkaxes!(ax_shapes,ax_eps)
geom_sublayout = GridLayout()
fig[2,1:2] = geom_sublayout
geom_sublayout[:h] = [ax_shapes,ax_eps]


hm_eps = epsilonheatmap!(ax_eps,grN,esm1N)

poly!(ax_shapes,geom2.shapes[1],color=:red,strokecolor=:black,strokewidth=1)
poly!(ax_shapes,geom2.shapes[2],color=:blue,strokecolor=:black,strokewidth=1)


xlims!(ax_eps,extrema(x(grN[])))
ylims!(ax_eps,extrema(y(grN[])))
ax_eps.aspect = DataAspect()
ax_shapes.aspect[] = ax_eps.aspect[] #( grN[].Δx / grN[].Δy )

cbar = fig[1,2] = Colorbar(fig, hm_eps.plots[1], label = "ε₁₁", vertical=false, height=30)

fig
##


AbstractPlotting.convert_arguments(x::GeometryPrimitives.Polygon) = ([Point2f0(x.v[i,:]) for i=1:size(x.v)[1]],)
plottype(::GeometryPrimitives.Polygon) = Poly
AbstractPlotting.convert_arguments(x::GeometryPrimitives.Box) = (GeometryBasics.Rect2D((x.c-x.r)..., 2*x.r...),)
plottype(::GeometryPrimitives.Box) = Poly

AbstractPlotting.convert_arguments(P::Type{<:Poly}, x::Geometry) = (x.shapes...,)
plottype(::Geometry) = Poly

poly!(pgn1,color=:blue,strokecolor=:black,strokewidth=1)
poly!(rect1,color=:red,strokecolor=:black,strokewidth=1)

#(geom.shapes[1])



rect1 = geom.shapes[2])
R1 = Rect2D((rect1.c-rect1.r)..., 2*rect1.r...)
poly!(ax2,R1,color=:orange,strokecolor=:black,strokewidth=1)

poly(geom2.shapes[1])

# ax_eps.autolimitaspect = ( grN[].Δx / grN[].Δy )



# esm2 = copy(esm1)

esm1N[] = copy(esm2)
esm1N[] = [SMatrix{3,3}(rand(3,3)) for i=1:128,j=1:128]

fig

@recipe




@recipe(EpsilonHeatmap, gr, ε) do scene
    Attributes(
        colormap = :viridis,
    )
end

function AbstractPlotting.plot!(hm::EpsilonHeatmap{<:Tuple{<:Grid{2,<:Real},<:Matrix{<:SMatrix{3,3,<:Real,9}}}})
    gr = hm[:gr]
    ε = hm[:ε]
    xs = lift(x,gr)
    ys = lift(y,gr)
    xcs = lift(xc,gr)
    ycs = lift(yc,gr)
    xlm = extrema(xs[])
    ylm = extrema(ys[])
    zs = lift(ε) do ems
        [eemm[1,1] for eemm in ems ]
    end
    #Node{AbstractMatrix}(rand(Nx,Ny))


    function update_plot(gr,ε)
        zs[]
        # xs[] = lift(x,gr)[]
        # ys[] = lift(y,gr)[]
        # zs = lift(flat,hm[:ε])
        new_zs = [eemm[1,1] for eemm in ε ] # lift(ε) do ems
        #     [eemm[1,1] for eemm in ems ]
        # end
        zs[] = new_zs
    end
    AbstractPlotting.Observables.onany(update_plot, gr, ε)
    update_plot(gr[], ε[])
    heatmap!(hm,
        xs,
        ys,
        zs,
        # colormap=sc.colormap,
    )
    # for xx in xcs[]
    #     lines!(hm,[xx,xx],[ylm...],color= :white)
    # end
    # for yy in ycs[]
    #     lines!(hm,[xlm...],[yy,yy],color= :white)
    # end
    hm
end

@recipe(GeomPoly,geom) do scene
    Attributes(
        colormap = :viridis,
    )
end

function AbstractPlotting.plot!(p::GeomPoly{<:Tuple{<:Geometry}})
    geom = p[:geom]
    shapes = geom[].shapes
    bnds = bounds.shapes
    xmin = minimum([bb[1][1] for bb in bnds])
    ymin = minimum([bb[1][2] for bb in bnds])
    xmax = maximum([bb[2][1] for bb in bnds])
    xmax = maximum([bb[2][2] for bb in bnds])

    #Node{AbstractMatrix}(rand(Nx,Ny))

    function update_plot(gr,ε)
        zs[]
        # xs[] = lift(x,gr)[]
        # ys[] = lift(y,gr)[]
        # zs = lift(flat,hm[:ε])
        new_zs = [eemm[1,1] for eemm in ε ] # lift(ε) do ems
        #     [eemm[1,1] for eemm in ems ]
        # end
        zs[] = new_zs
    end
    AbstractPlotting.Observables.onany(update_plot, gr, ε)
    update_plot(gr[], ε[])
    heatmap!(hm,
        xs,
        ys,
        zs,
        # colormap=sc.colormap,
    )
    # for xx in xcs[]
    #     lines!(hm,[xx,xx],[ylm...],color= :white)
    # end
    # for yy in ycs[]
    #     lines!(hm,[xlm...],[yy,yy],color= :white)
    # end
    hm
end

pgn1 = geom.shapes[1]
pgn_pts = [Point2f0(pgn1.v[i,:]) for i=1:size(pgn1.v)[1]]
poly!(ax2,pgn_pts,color=:red,strokecolor=:black,strokewidth=1)

ax2.limits[] = ax1.limits[]

fig

rect1 = geom.shapes[2])
R1 = Rect2D((rect1.c-rect1.r)..., 2*rect1.r...)
poly!(ax2,R1,color=:orange,strokecolor=:black,strokewidth=1)

hm1 = myheatmap(gr,esm1)

hm2 = heatmap(x(gr),
    y(gr),
    [eemm[1,1] for eemm in esm1],
    overdraw=true,
    axis = ( xlims = extrema(x(gr)),
        ylims = extrema(y(gr)),
        aspect = DataAspect(),
        ygridcolor=:white,
        xgridcolor=:white,
        yminorgridcolor=:white,
        xminorgridcolor=:white,
        backgroundcolor=:transparent,
     )
    )

hm2.axis.backgroundcolor[] = :transparent

hm2.axis.ygridcolor=:red

xlims!(extrema(x(grN[])))
ylims!(extrema(y(grN[])))
hm2.axis.aspect = DataAspect()
hm2.axis.attributes.ygridlinestyle = :dash
hm2.axis.attributes.yminorticksvisible = true
# CairoMakie.heatmap!(ax1,esm1_11); fig
#
#
# CairoMakie.heatmap!(ax1,
# # CairoMakie.heatmap(
#     x(gr),
#     y(gr),
#     flat(esm1)[1,1,:,:],
# )
#; fig

#
# data1 = randn(50, 2) * [1 2.5; 2.5 1] .+ [10 10]
#
# line1 = lines!(ax1, 5..15, x -> x, color = :red, linewidth = 2)
# scat1 = scatter!(ax1, data1,
#     color = (:red, 0.3), markersize = 15px, marker = '■')
#
# # fig
#
# ax2, line2 = lines(fig[1, 2], 7..17, x -> -x + 26,
#     color = :blue, linewidth = 2,
#     axis = (title = "C0v!d",))
#
# # fig
#
# data2 = randn(50, 2) * [1 -2.5; -2.5 1] .+ [13 13]
#
# scat2 = scatter!(data2,
#     color = (:blue, 0.3), markersize = 15px, marker = '▲')
#
# # fig
#
# linkaxes!(ax1,ax2)
# hideydecorations!(ax2, grid = false)
#
# ax1.xlabel = "time (minutes)"
# ax2.xlabel = "time (years)"
# ax1.ylabel = "\$\$\$\$"
#
# leg = fig[1, end+1] = Legend(fig,
#     [line1, scat1, line2, scat2],
#     ["f(x) = x", "Data", "f(x) = -x + 26", "Data"])
#
# fig



##
using Plots: plot, plot!, heatmap, @layout, cgrad, grid, heatmap!
using LaTeXStrings
cmap_n=cgrad(:viridis)
cmap_e=cgrad(:plasma)

# p0 = [
#     1.7,                #   top ridge width         `w_top`         [μm]
#     0.7,                #   ridge thickness         `t_core`        [μm]
#     π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
# ]
# # p_lower = [0.4, 0.3, 0.]
# # p_upper = [2., 1.8, π/4.]
# p_opt0 = [1.665691811699148, 0.36154202879652847, 0.2010932097703251]
# #
# # optimized using ωs = collect(0.625:0.025:0.7) => λs0 = [1.6, 1.538, 1.481, 1.428]
# # plot with higher density of frequency points,
# ωs = collect(0.53:0.02:0.8); λs = 1. ./ ωs
# x = ms.M̂.x; y = ms.M̂.y;

p = [
                   1.7,                #   top ridge width         `w_top`         [μm]
                   0.7,                #   ridge thickness         `t_core`        [μm]
                   π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
                   2.4,                #   core index              `n_core`        [1]
                   1.4,                #   substrate index         `n_subs`        [1]
                   0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
               ];
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1
rwg(p) = ridge_wg(p[1],p[2],p[3],p[6],p[4],p[5],Δx,Δy)
ms = ModeSolver(1.45, rwg(p), Δx, Δy, Δz, Nx, Ny, Nz);
mg = MaxwellGrid(Δx,Δy,Nx,Ny)
ω = 1/1.55


##

ng = solve_n(ms,ω,rwg(p))
k,H = solve_k(ms,ω,rwg(p))
mn = vcat(reshape(ms.M̂.m,(1,3,Nx,Ny,Nz)),reshape(ms.M̂.n,(1,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)))
e = ε⁻¹_dot( fft( kx_tc(reshape(H,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),mn,ms.M̂.mag), (2:4) ), ms.M̂.ε⁻¹ )
enorm = e ./ e[argmax(abs2.(e))]
x = ms.M̂.x; y = ms.M̂.y;

ε⁻¹ = ms.M̂.ε⁻¹
ε = [ inv(ε⁻¹[:,:,ix,iy,1]) for ix=1:Nx, iy=1:Ny ]
n₁ = [ √ε[ix,iy,1][1,1] for ix=1:Nx, iy=1:Ny ]
hm_e = heatmap(
        x,
        y,
        real(enorm[1,:,:,1])',
        c=cmap_e, #cgrad(:cherry),
        aspect_ratio=:equal,
        legend=false,
        colorbar = true,
        clim = (0,1),
        xlabel = "x (μm)",
        ylabel = "y (μm)",
        colorbar_title = "|Eₓ|²",
    )
hm_n = heatmap(
    x,
    y,
    transpose(n₁),
    c=cmap_n, #cgrad(:cherry),
    aspect_ratio=:equal,
    legend=false,
    colorbar = true,
    clim = (1,2.5),
    xlabel = "x (μm)",
    ylabel = "y (μm)",
    colorbar_title = "nₓ",
    title = "optimized params",
)

##
nng = solve_n(ms,ωs,rwg(p))
k,H = solve_k(ms,ωs[14],rwg(p_opt0))
mn = vcat(reshape(ms.M̂.m,(1,3,Nx,Ny,Nz)),reshape(ms.M̂.n,(1,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)))
e = ε⁻¹_dot( fft( kx_tc(reshape(H,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),mn,ms.M̂.mag), (2:4) ), ms.M̂.ε⁻¹ )
enorm = e ./ e[argmax(abs2.(e))]
x = ms.M̂.x; y = ms.M̂.y;

ε⁻¹ = ms.M̂.ε⁻¹
ε = [ inv(ε⁻¹[:,:,ix,iy,1]) for ix=1:Nx, iy=1:Ny ]
n₁ = [ √ε[ix,iy,1][1,1] for ix=1:Nx, iy=1:Ny ]


nng0 = solve_n(ms,ωs,rwg(p0))
k0,H0 = solve_k(ms,ωs[14],rwg(p0))
mn0 = vcat(reshape(ms.M̂.m,(1,3,Nx,Ny,Nz)),reshape(ms.M̂.n,(1,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)))
e0 = ε⁻¹_dot( fft( kx_tc(reshape(H0,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),mn0,ms.M̂.mag), (2:4) ), ms.M̂.ε⁻¹ )
enorm0 = e0 ./ e0[argmax(abs2.(e0))]

ε⁻¹0 = ms.M̂.ε⁻¹
ε0 = [ inv(ε⁻¹0[:,:,ix,iy,1]) for ix=1:Nx, iy=1:Ny ]
n₁0 = [ √ε0[ix,iy,1][1,1] for ix=1:Nx, iy=1:Ny ]

##
pyplot()
ylim_ng = (2.485,2.545)

hm_e = heatmap(
        x,
        y,
        real(enorm[1,:,:,1])',
        c=cmap_e, #cgrad(:cherry),
        aspect_ratio=:equal,
        legend=false,
        colorbar = true,
        clim = (0,1),
        xlabel = "x (μm)",
        ylabel = "y (μm)",
        colorbar_title = "|Eₓ|²",
    )
hm_n = heatmap(
    x,
    y,
    transpose(n₁),
    c=cmap_n, #cgrad(:cherry),
    aspect_ratio=:equal,
    legend=false,
    colorbar = true,
    clim = (1,2.5),
    xlabel = "x (μm)",
    ylabel = "y (μm)",
    colorbar_title = "nₓ",
    title = "optimized params",
)
hm_n0 = heatmap(
        x,
        y,
        transpose(n₁0),
        c=cmap_n, #cgrad(:cherry),
        aspect_ratio=:equal,
        legend=false,
        colorbar = true,
        clim = (1,2.5),
        xlabel = "x (μm)",
        ylabel = "y (μm)",
        colorbar_title = "nₓ",
        title = "intial params",
    )
hm_e0 = heatmap(
    x,
    y,
    real(enorm0[1,:,:,1])',
    c=cmap_e, #cgrad(:cherry),
    aspect_ratio=:equal,
    legend=false,
    colorbar = true,
    clim = (0,1),
    xlabel = "x (μm)",
    ylabel = "y (μm)",
    colorbar_title = "|Eₓ|²",

)

plt_n = plot(
    λs,nng0[1],
    xlabel="λ (μm)",
    ylabel="effective index n",
    label="init. params",
    legend=:bottomleft,
    m=:dot,
    msize=2,
    )
plot!(plt_n,λs,
    nng[1],
    label="opt. params",
    m=:dot,
    msize=2,
    )
plt_ng = plot(λs,
    nng0[2],
    xlabel="λ (μm)",
    ylabel="group index ng",
    legend=false,
    m=:dot,
    msize=2,
    ylim=ylim_ng,
    )
plot!(plt_ng,
    λs,
    nng[2],
    m=:dot,
    msize=2,
    fillalpha=0.2,
    )
annot_str1 = latexstring("\$ \\mathcal{L}(\\vec{p}) = \$")
annot_str2 = latexstring("\$ \\langle n_g^2 \\rangle - \\langle n_g \\rangle^2 \$")
annot_str = "minimize\n" * annot_str1 * "\n" * annot_str2 * "\n here"
plot!(plt_ng,
    [1.428,1.6],
    [2.52,2.52],
    linecolor=nothing,
    fill_between=(ylim_ng[1],ylim_ng[2]),
    fillcolor=:blue,
    fillalpha=0.2,
    annotations = (1.51,2.53, annot_str)
    )

l = @layout [   a   b
                c   d
                e   f   ]
plot(plt_n,
        plt_ng,
        hm_n0,
        hm_n,
        hm_e0,
        hm_e,
        layout=l,
        size= (900,700),
        )