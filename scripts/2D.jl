using Revise
using OptiMode
using LinearAlgebra
using Statistics
using ArrayInterface
using RecursiveArrayTools
using StaticArrays
using HybridArrays
using SparseArrays
using FFTW
using LinearMaps
using GeometryPrimitives
using BenchmarkTools
using ChainRules
using Zygote
using ForwardDiff
using FiniteDifferences
using FiniteDiff
using UnicodePlots
using OhMyREPL
using Crayons.Box       # for color printing
using Zygote: @ignore, dropgrad
using Setfield: @set
using StaticArrays: Dynamic
using IterativeSolvers: bicgstabl
using Rotations: RotY, MRP
using RuntimeGeneratedFunctions
using Tullio
using GLMakie
RuntimeGeneratedFunctions.init(@__MODULE__)


gradRM(fn,in) 			= 	Zygote.gradient(fn,in)[1]
gradFM(fn,in) 			= 	ForwardDiff.gradient(fn,in)
gradFD(fn,in;n=3)		=	FiniteDifferences.grad(central_fdm(n,1),fn,in)[1]
gradFD2(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_gradient(fn,in;relstep=rs)

derivRM(fn,in) 			= 	Zygote.gradient(fn,in)[1]
derivFM(fn,in) 			= 	ForwardDiff.gradient(fn,in)
derivFD(fn,in;n=3)		=	FiniteDifferences.grad(central_fdm(n,1),fn,in)[1]
derivFD2(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_derivative(fn,in;relstep=rs)

AD_style = BOLD*BLUE_FG #NEGATIVE*BOLD*BLUE_FG      # defined in Crayons.Box
FD_style = BOLD*RED_FG
MAN_style = BOLD*GREEN_FG
AD_style_N = NEGATIVE*BOLD*BLUE_FG #NEGATIVE*BOLD*BLUE_FG      # defined in Crayons.Box
FD_style_N = NEGATIVE*BOLD*RED_FG
MAN_style_N = NEGATIVE*BOLD*GREEN_FG

Δx,Δy,Nx,Ny = 6.0, 4.0, 128, 128;
grid = Grid(Δx,Δy,Nx,Ny)
LNx = rotate(MgO_LiNbO₃,Matrix(MRP(RotY(π/2))),name=:LiNbO₃_X);
LNxN = NumMat(LNx;expr_module=@__MODULE__())
SiO₂N = NumMat(SiO₂;expr_module=@__MODULE__())
Si₃N₄N = NumMat(Si₃N₄;expr_module=@__MODULE__())
AlOxN = NumMat(αAl₂O₃;expr_module=@__MODULE__())
##
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,Si₃N₄N,SiO₂N,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).
# p = [
#        0.8,                #   top ridge width         `w_top`         [μm]
#        0.4,                #   ridge thickness         `t_core`        [μm]
#        0.75, #0.5,                #   ridge thickness         `t_core`        [μm]
#        π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
#                ];

p = [
       1.2,                #   top ridge width         `w_top`         [μm]
       0.5,                #   ridge thickness         `t_core`        [μm]
       0.85, #0.5,                #   ridge thickness         `t_core`        [μm]
       0.,           #   ridge sidewall angle    `θ`             [radian]
               ];

ω = inv(0.85)
# rwg1 = rwg(p)
ε⁻¹ = smooth(ω,p,:fεs,true,rwg,grid)
kguess =  k_guess(ω,ε⁻¹)
k⃗ = SVector(0.0,0.0,kguess)
M̂ = HelmholtzMap(k⃗, ε⁻¹, grid)
ms = ModeSolver(kguess, ε⁻¹, grid; nev=2, maxiter=300, tol=1e-6)
k1,Hv1 = solve_k(ω,p,rwg,grid;nev=1,eigind=1);
##
geom_fn = rwg
eigind = 1
nev = 2
solve(ω,p,rwg,grid;kguess=kguess,nev=2,eigind=1,maxiter=300,tol=1e-6,f_filter=nothing)
ε,ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fεs,:fnn̂gs,:fnn̂gs),[false,true,false,true],geom_fn,grid))

smooth(ω,p,:fnn̂gs,true,geom_fn,grid)

##
using KrylovKit
using IterativeSolvers
using DFTK: LOBPCG
evals,evecs,convinfo = eigsolve(x->ms.M̂*x,randn(ComplexF64,size(ms.H⃗,1)),size(ms.H⃗,2),:SR;maxiter=3000,tol=1e-6,krylovdim=40,verbosity=2)
# resIS = lobpcg(ms.eigs_itr,true;not_zeros=false,maxiter=100,tol=1e-6)
resIS = lobpcg(ms.M̂,true,4;not_zeros=false,maxiter=3000,tol=1e-6,P=ms.P̂)
resDF = LOBPCG(ms.M̂,randn(ComplexF64,size(ms.H⃗)),I,ms.P̂,1e-6,300; display_progress=true)

@show evals |> real
@show resIS.λ |> real
@show resDF.λ |> real

##
ω,p,fnames,invert_fn,f_geom,grid,smoothing_fn = ω,p,(:fnĝvds,:fnn̂gs),[false,false],geom_fn,grid,volfrac_smoothing
n_p = length(p)
n_fns=length(fnames)
om_p = vcat(ω,p)
xyz = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
xyzc = Zygote.@ignore(x⃗c(grid))
vxlmin,vxlmax = vxl_minmax(xyzc)

arr_flatB = Zygote.Buffer(om_p,9,size(grid)...,n_fns)
arr_flat = Zygote.forwarddiff(om_p) do om_p
        geom = f_geom(om_p[2:n_p+1])
        shapes = getfield(geom,:shapes)
        om_inv = inv(first(om_p))
        mat_vals = mapreduce(ss->[ map(f->(mat=SMatrix{3,3}(f(om_inv)); 0.5*(mat+mat')),getfield(geom,ss))... ], hcat, fnames)
        mat_vals_inv = inv.(mat_vals)
        # calcinv = repeat([invert_fn...]',size(mat_vals,1))
        sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
        smoothed_vals_nested = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
                Tuple(smoothing_fn(sinds,shapes,geom.material_inds,mat_vals,mat_vals_inv,invert_fn,xx,vn,vp))
        end
        smoothed_vals = hcat( [map(x->getindex(x,i),smoothed_vals_nested) for i=1:n_fns]...)
        smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
        return smoothed_vals_rr  # new spatially smoothed ε tensor array
end
copyto!(arr_flatB,copy(arr_flat))
arr_flat_r = copy(arr_flatB)
Nx = size(grid,1)
Ny = size(grid,2)
# fn_arrs = [hybridize(view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,axes(grid),n),grid) for n=1:n_fns]
views = [view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,axes(grid),n) for n in range(n_fns)]
view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,axes(grid),1)
arr_flat_r |> size
reshape(arr_flat_r,3,3,size(grid)...,n_fns)
view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,axes(grid)...,1)
view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,UnitRange.(axes(grid)),1)
##
ωₜ = 1.19
n_guess = 2.1
ω²,H⃗ = solve_ω²(ms,ω*n_guess; nev=1, eigind=1, maxiter=1000, tol=1e-6)
Δω² = ω²[1] - ωₜ^2
∂ω²∂k = 2 * HMₖH(H⃗[:,1],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn)
∂ω²∂k = 2 * HMₖH(H⃗[:,1],ms.M̂.ε⁻¹,ms.M̂.mag,view(ms.M̂.mn,:,1,:,:),view(ms.M̂.mn,:,2,:,:))

##
eigind = 1
_solve_Δω²(ms,kguess,ω,eigind=1,maxiter=300,tol=1e-6)
ω²,Hv = solve_ω²(ms,kguess,eigind=1,maxiter=300,tol=1e-6)

f1 = kk -> solve_ω²(ms,kk,eigind=1,maxiter=300,tol=1e-6)[1]
@show omsq = f1(kguess)
@show ∂ω²∂k_fd = derivFD2(f1,kguess;rs=1e-2)

@show ∂ω²∂k = 2 * HMₖH(ms.H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn) / 128^2
m = view(ms.M̂.mn,:,1,:,:)
n = view(ms.M̂.mn,:,2,:,:)
mag = ms.M̂.mag
ε⁻¹ = ms.M̂.ε⁻¹
mn2 = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3])))
permutedims(mn2,(2,1,3,4)) ≈ ms.M̂.mn

HMₖH(H1,ε⁻¹,mag,m,n)
##


E1 = E⃗(ms,1)
H1 = H⃗(ms)[1]
E1s = reinterpret(reshape, SVector{3,Complex{Float64}},  E1)
H1s = reinterpret(reshape, SVector{3,Complex{Float64}},  H1)
S1s = real.( cross.( conj.(E1s), H1s) )
S1 = reshape( reinterpret( Float64,  S1s), (3,size(grid)...))



##


ms.M̂.mn |> size

Hv = copy(ms.H⃗[:,1])
H = reshape(Hv,(2,size(grid)...))

∂ω²∂k / dot(H,H) / 2

dot(H,H) / size(H)[1]
dot(Hv,ms.M̂,Hv) / size(Hv)[1]
dot(H, -kx_ct( ifft( ε⁻¹_dot( fft( kx_tc(H,ms.M̂.mn,ms.M̂.mag), (2:3) ), real(ε⁻¹)), (2:3)),ms.M̂.mn,ms.M̂.mag) ) / sum(abs2.(H))



kx_ct( ifft( ε⁻¹_dot( fft( kx_tc(H,ms.M̂.mn,ms.M̂.mag), (2:3) ), real(ε⁻¹)), (2:3)),ms.M̂.mn,ms.M̂.mag)
ifft( ε⁻¹_dot( fft( kx_tc(H,ms.M̂.mn,ms.M̂.mag), (2:3) ), real(ε⁻¹)), (2:3))
size(H1)

dot(H, -kx_ct2( ifft( ε⁻¹_dot( fft( kx_tc2(H,m,n,mag), (2:3) ), real(ε⁻¹)), (2:3)),m,n,mag) ) / sum(abs2.(H))
dot(H, -kx_ct2( ifft( ε⁻¹_dot( fft( kx_tc2(H,n,m,mag), (2:3) ), real(ε⁻¹)), (2:3)),n,m,mag) ) / sum(abs2.(H))


##
dc = copy(ms.M̂.d)
Hc = copy(H)
kx_tc!(dc,Hc,ms.M̂.mn,ms.M̂.mag)
d2 = kx_tc(H,ms.M̂.mn,ms.M̂.mag)
dc ≈ d2
compare_fields(d2,dc)
##
d1 = copy(ms.M̂.d)
e1 = copy(ms.M̂.e)
𝓕! = ms.M̂.𝓕!
𝓕⁻¹! = ms.M̂.𝓕⁻¹!
H1 = copy(H)
kx_tc!(d1,H1,ms.M̂.mn,ms.M̂.mag)
mul!(d1,𝓕!,d1);
eid!(e1,ε⁻¹,d1);
# mul!(e1,𝓕⁻¹!,e1);

e2 =  ε⁻¹_dot( fft( kx_tc(H,ms.M̂.mn,ms.M̂.mag), (2:3) ), real(ε⁻¹))
e1 ≈ e2
compare_fields(e1,e2)

##
d1 = copy(ms.M̂.d)
e1 = copy(ms.M̂.e)
𝓕! = ms.M̂.𝓕!
𝓕⁻¹! = ms.M̂.𝓕⁻¹!
H1 = copy(H)
kx_tc!(d1,H1,ms.M̂.mn,ms.M̂.mag)
mul!(d1,𝓕!,d1);
eid!(e1,ε⁻¹,d1);
mul!(e1,𝓕⁻¹!,e1);

e2 = bfft( ε⁻¹_dot( fft( kx_tc(H,ms.M̂.mn,ms.M̂.mag), (2:3) ), real(ε⁻¹)), (2:3))
e1 ≈ e2
compare_fields(e1,e2)



##
function compare_fields(f1,f2)
        print("field comparison\n")
        @show maximum(abs2.(f1))
        @show maximum(abs2.(f2))
        @show maximum(real(f1))
        @show minimum(real(f1))
        @show maximum(real(f2))
        @show minimum(real(f2))
        @show maximum(imag(f1))
        @show minimum(imag(f1))
        @show maximum(imag(f2))
        @show minimum(imag(f2))
        @show maximum(abs.(real(f1)))
        @show maximum(abs.(real(f2)))
        @show maximum(abs.(imag(f1)))
        @show maximum(abs.(imag(f2)))
        print("\n")
        return
end
##

## Makie example

using CairoMakie
using FileIO

noto_sans = assetpath("fonts", "NotoSans-Regular.ttf")
noto_sans_bold = assetpath("fonts", "NotoSans-Bold.ttf")

f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98),
    resolution = (1000, 700), font = noto_sans)

ga = f[1, 1] = GridLayout()
gb = f[2, 1] = GridLayout()
gcd = f[1:2, 2] = GridLayout()
gc = gcd[1, 1] = GridLayout()
gd = gcd[2, 1] = GridLayout()

# panel a
axtop = Axis(ga[1, 1])
axmain = Axis(ga[2, 1], xlabel = "before", ylabel = "after")
axright = Axis(ga[2, 2])

labels = ["treatment", "placebo", "control"]
data = randn(3, 100, 2) .+ [1, 3, 5]

for (label, col) in zip(labels, eachslice(data, dims = 1))
    scatter!(axmain, col, label = label)
    density!(axtop, col[:, 1])
    density!(axright, col[:, 2], direction = :y)
end

ylims!(axtop, low = 0)
xlims!(axright, low = 0)

leg = Legend(ga[1, 2], axmain)

hidedecorations!(axtop, grid = false)
hidedecorations!(axright, grid = false)
leg.tellheight = true

colgap!(ga, 10)
rowgap!(ga, 10)

Label(ga[1, 1:2, Top()], "Stimulus ratings", valign = :bottom,
    padding = (0, 0, 5, 0))

# panel b
xs = LinRange(0.5, 6, 50)
ys = LinRange(0.5, 6, 50)
data1 = [sin(x^1.5) * cos(y^0.5) for x in xs, y in ys] .+ 0.1 .* randn.()
data2 = [sin(x^0.8) * cos(y^1.5) for x in xs, y in ys] .+ 0.1 .* randn.()

ax1, hm = contourf(gb[1, 1], xs, ys, data1,
levels = 6)
ax1.title = "Histological analysis"
contour!(xs, ys, data1, levels = 5, color = :black)
hidexdecorations!(ax1)

_, hm2 = contourf(gb[2, 1], xs, ys, data2,
levels = 6)
contour!(xs, ys, data2, levels = 5, color = :black)
    
cb = Colorbar(gb[1:2, 2], hm, label = "cell group")
low, high = extrema(data1)
edges = range(low, high, length = 7)
centers = (edges[1:6] .+ edges[2:7]) .* 0.5
cb.ticks = (centers, string.(1:6))

cb.alignmode = Mixed(right = 0)
colgap!(gb, 10)
rowgap!(gb, 10)

# panel c
brain = load(assetpath("brain.stl"))

Axis3(gc[1, 1], title = "Brain activation")
m = mesh!(
    brain,
    color = [tri[1][2] for tri in brain for i in 1:3],
    colormap = Reverse(:magma),
)
Colorbar(gc[1, 2], m, label = "BOLD level")

# panel D

axs = [Axis(gd[row, col]) for row in 1:3, col in 1:2]
hidedecorations!.(axs, grid = false, label = false)

for row in 1:3, col in 1:2
    xrange = col == 1 ? (0:0.1:6pi) : (0:0.1:10pi)

    eeg = [sum(sin(pi * rand() + k * x) / k for k in 1:10)
        for x in xrange] .+ 0.1 .* randn.()

    lines!(axs[row, col], eeg, color = (:black, 0.5))
end

axs[3, 1].xlabel = "Day 1"
axs[3, 2].xlabel = "Day 2"
Label(gd[1, :, Top()], "EEG traces", valign = :bottom,
padding = (0, 0, 5, 0))
rowgap!(gd, 10)
colgap!(gd, 10)

for (i, label) in enumerate(["sleep", "awake", "test"])
        Box(gd[i, 3], color = :gray90)
        Label(gd[i, 3], label, rotation = pi/2, tellheight = false)
end
colgap!(gd, 2, 0)
n_day_1 = length(0:0.1:6pi)
n_day_2 = length(0:0.1:10pi)

colsize!(gd, 1, Auto(n_day_1))
colsize!(gd, 2, Auto(n_day_2))
for (label, layout) in zip(["A", "B", "C", "D"], [ga, gb, gc, gd])
Label(layout[1, 1, TopLeft()], label,
        textsize = 26,
        font = noto_sans_bold,
        padding = (0, 5, 5, 0),
        halign = :right)
end
colsize!(f.layout, 1, Auto(0.5))
rowsize!(gcd, 1, Auto(1.5))

f







