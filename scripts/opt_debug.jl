using LinearAlgebra, StaticArrays, HybridArrays, ArrayInterface, LoopVectorization, Tullio, ChainRules, Zygote, GeometryPrimitives, OptiMode, Optim
using Zygote: dropgrad
using Statistics: mean
# # p = [
# #     1.7,                #   top ridge width         `w_top`         [μm]
# #     0.7,                #   ridge thickness         `t_core`        [μm]
# #     π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
# #     # 2.4,                #   core index              `n_core`        [1]
# #     # 1.4,                #   substrate index         `n_subs`        [1]
# #     # 0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
# # ]
#
# p0 = [
#     1.7,                #   top ridge width         `w_top`         [μm]
#     0.7,                #   ridge thickness         `t_core`        [μm]
#     π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
# ]
#
p_lower = [0.4, 0.3, 0.]
p_upper = [2., 1.8, π/4.]
#
# Δx,Δy,Δz,Nx,Ny,Nz = 6., 4., 1., 128, 128, 1
# rwg(p) = ridge_wg(p[1],p[2],p[3],0.5,2.4,1.4,Δx,Δy)
# @show ωs = collect(0.625:0.025:0.7)
# ms = ModeSolver(1.45, rwg(p0), Δx, Δy, Δz, Nx, Ny, Nz)
# shapes = rwg(p0)


p = [
       1.7,                #   top ridge width         `w_top`         [μm]
       0.7,                #   ridge thickness         `t_core`        [μm]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
       # 0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
               ];
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg(x) = ridge_wg(x[1],x[2],x[3],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy)
geom = rwg(p)
ms = ModeSolver(1.45, geom, gr)
λs = 0.7:0.05:1.1
ωs = 1 ./ λs
n1,ng1 = solve_n(ms,ωs,rwg(p))
##

function var_ng(ωs,p)
    ngs = solve_n(dropgrad(ms),ωs,rwg(p))[2]
    mean( abs2.( ngs ) ) - abs2(mean(ngs))
end

# warmup
println("warmup function runs")
p0 = copy(p)
@show var_ng(ωs,p0)
@show vng0, vng0_pb = Zygote.pullback(x->var_ng(ωs,x),p0)
@show grad_vng0 = vng0_pb(1)

# define function that computes value and gradient of function `f` to be optimized
# according to https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/
function fg!(F,G,x)
    value, value_pb = Zygote.pullback(x) do x
       var_ng(ωs,x)
    end
    if G != nothing
        G .= value_pb(1)[1]
    end
    if F != nothing
        # F = value
        return value
    end
end


@show fg!(0.,[0.,0.,0.],p0)
##
opts =  Optim.Options(
                        outer_iterations = 4,
                        iterations = 1,
                        store_trace = true,
                        show_trace = true,
                        show_every = 1,
                        extended_trace = true,
                    )



inner_optimizer = Optim.BFGS() # GradientDescent() #

# results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))
res = optimize( Optim.only_fg!(fg!),
                p_lower,
                p_upper,
                p0,
                Fminbox(inner_optimizer),
                opts,
            )


##
# first optimization result after a few very inefficient steps:
# Δx,Δy,Δz,Nx,Ny,Nz = 6., 4., 1., 128, 128, 1
# rwg(p) = ridge_wg(p[1],p[2],p[3],0.5,2.4,1.4,Δx,Δy)
using Plots: plot, plot!, heatmap, @layout, cgrad, grid, heatmap!
using LaTeXStrings
cmap_n=cgrad(:viridis)
cmap_e=cgrad(:plasma)

p0 = [
    1.7,                #   top ridge width         `w_top`         [μm]
    0.7,                #   ridge thickness         `t_core`        [μm]
    π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
]
# p_lower = [0.4, 0.3, 0.]
# p_upper = [2., 1.8, π/4.]
p_opt0 = [1.665691811699148, 0.36154202879652847, 0.2010932097703251]
#
# optimized using ωs = collect(0.625:0.025:0.7) => λs0 = [1.6, 1.538, 1.481, 1.428]
# plot with higher density of frequency points,
ωs = collect(0.53:0.02:0.8); λs = 1. ./ ωs
# x = ms.M̂.x; y = ms.M̂.y;
mg = MaxwellGrid(Δx,Δy,Nx,Ny)

nng = solve_n(ms,ωs,rwg(p_opt0))
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



##
