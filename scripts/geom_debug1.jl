using Revise
using OptiMode, GeometryPrimitives, ChainRules, Zygote, FiniteDifferences
using Zygote: @adjoint, ignore, dropgrad #,StaticArrays,FFTW
##
w           =   1.7
t_core      =   0.7
edge_gap    =   0.5               # μm
n_core      =   2.4
n_subs      =   1.4
λ           =   1.55                  # μm
Δx          =   6.                    # μm
Δy          =   4.                    # μm
Δz          =   1.
Nx          =   64
Ny          =   64
Nz          =   1
ω           =   1 / λ
##
function test_shapes2(p::Float64)
    ε₁, ε₂, ε₃ = test_εs(1.42,2.2,2.5)
    ax = [  1.  0.
            0.  1.  ]
    b = Box(					# Instantiate N-D box, here N=2 (rectangle)
        [0,0],					# c: center
        [3.0, 3.0],				# r: "radii" (half span of each axis)
        ax,         			# axes: box axes
        ε₁,						# data: any type, data associated with box shape
        )

    s = Sphere(					# Instantiate N-D sphere, here N=2 (circle)
        [0.,0.],					# c: center
        p,						# r: "radii" (half span of each axis)
        ε₂,						# data: any type, data associated with circle shape
        )

    t = regpoly(				# triangle::Polygon using regpoly factory method
        3,						# k: number of vertices
        p, #0.8,					# r: distance from center to vertices
        π/2,					# θ: angle of first vertex
        [0,0],					# c: center
        ε₃,						# data: any type, data associated with triangle
        )

    return [ t, s, b ]
end
using Plots, BenchmarkTools
g = MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
s1 = ridge_wg(w,t_core,edge_gap,n_core,n_subs,Δx,Δy)
s2 = circ_wg(0.7,0.2,edge_gap,n_core,n_subs,Δx,Δy)
s3 = test_shapes2(1.05)
plot_ε(εₛ(s2,g),g.x,g.y)
# plot_ε(εₛ⁻¹(s,g),g.x,g.y)

plot_ε(εₛ(s1,g),g.x,g.y)
plot_ε(εₛ⁻¹(s,g),g.x,g.y)

H,k = solve_k(ω,s1::Vector{<:GeometryPrimitives.Shape},g::MaxwellGrid)
plot_d⃗(H,k,g)




using ChainRulesCore
ChainRulesCore.refresh_rules()
Zygote.refresh()

################################################################################
#                                                                              #
#                            Ridge Waveguide Tests                             #
#                                                                              #
################################################################################
ts = collect(0.4:0.04:0.9)
ws = collect(1.2:0.05:2.0)
ωs = collect(0.645*(0.9:0.02:1.1))

function solve_rwg(om,ww,tt_core,eedge_gap,nn_core,nn_subs,Dx,Dy,Dz,nx,ny,nz)
    solve_k(om,ridge_wg(ww,tt_core,eedge_gap,nn_core,nn_subs,dropgrad(Dx),dropgrad(Dy)),dropgrad(Dx),dropgrad(Dy),dropgrad(Dz),dropgrad(nx),dropgrad(ny),dropgrad(nz))
end

using BenchmarkTools
plot_d⃗(solve_cwg(1/1.55,0.7,0.2,edge_gap,1.7,1.5,Δx,Δy,Δz,Nx,Ny,Nz)...,g)
plot_d⃗(solve_rwg(1/1.55,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)...,g)


@btime solve_cwg(1/1.55,0.7,0.2,$edge_gap,$n_core,$n_subs,$Δx,$Δy,$Δz,$Nx,$Ny,$Nz)
# 357.346 ms (569288 allocations: 96.75 MiB) for 1 value, 64×48 spatial grid
# 3.121 s (2935840 allocations: 686.20 MiB) for 1 value, 128×128 spatial grid

@benchmark solve_rwg(1/1.55,$w,$t_core,$edge_gap,$n_core,$n_subs,$Δx,$Δy,$Δz,$Nx,$Ny,$Nz)
# 58.9 ms (169302 allocations: 21.02 MiB) for 1 ω values, 32 x 32 spatial grid
# 731.258 ms (1780443 allocations: 252.59 MiB) for 1 ω values, 128 x 96 spatial grid
# 3.232 s (3276403 allocations: 1.16 GiB)   # for 11 ω values, 128 x 96 spatial grid
# 175.092 ms (456693 allocations: 62.81 MiB) for 1 value, 64×48 spatial grid
# 1.001 s (2362625 allocations: 322.87 MiB) for 1 value, 128×128 spatial grid

function ng_rwg(om,ww,tt_core,eedge_gap,nn_core,nn_subs,Dx,Dy,Dz,nx,ny,nz)
    solve_n(om,ridge_wg(ww,tt_core,eedge_gap,nn_core,nn_subs,dropgrad(Dx),dropgrad(Dy)),dropgrad(Dx),dropgrad(Dy),dropgrad(Dz),dropgrad(nx),dropgrad(ny),dropgrad(nz))[2]
end

function n_rwg(om,ww,tt_core,eedge_gap,nn_core,nn_subs,Dx,Dy,Dz,nx,ny,nz)
    solve_n(om,ridge_wg(ww,tt_core,eedge_gap,nn_core,nn_subs,dropgrad(Dx),dropgrad(Dy)),dropgrad(Dx),dropgrad(Dy),dropgrad(Dz),dropgrad(nx),dropgrad(ny),dropgrad(nz))[1]
end

function n_ng_rwg(om,ww,tt_core,eedge_gap,nn_core,nn_subs,Dx,Dy,Dz,nx,ny,nz)
    solve_n(om,ridge_wg(ww,tt_core,eedge_gap,nn_core,nn_subs,dropgrad(Dx),dropgrad(Dy)),dropgrad(Dx),dropgrad(Dy),dropgrad(Dz),dropgrad(nx),dropgrad(ny),dropgrad(nz))
end

# ng_rwg(1/1.55,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
# ng_rwg(collect(0.645*(0.9:0.02:1.1)),w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)

# @btime ng_rwg(0.645,$w,$t_core,$edge_gap,$n_core,$n_subs,$Δx,$Δy,$Δz,$Nx,$Ny,$Nz)
# 50.866 ms (173716 allocations: 22.07 MiB) for 1 frequency, 32×32 spatial grid
# 198.690 ms (630899 allocations: 75.48 MiB) for 1 frequency, 64×64 spatial grid
# 956.631 ms (2527945 allocations: 322.88 MiB) for 1 frequency, 128×128 spatial grid
# 9.661 s (9835867 allocations: 1.23 GiB) for 1 frequency, 256×256 spatial grid

# @btime gradient(ng_rwg,0.645,$w,$t_core,$edge_gap,$n_core,$n_subs,$Δx,$Δy,$Δz,$Nx,$Ny,$Nz)
# 476.851 ms (1772737 allocations: 118.51 MiB) for 1 frequency, 32×32 spatial grid
# 2.087 s (6093797 allocations: 499.66 MiB) for 1 frequency, 64×64 spatial grid
# 9.447 s (22503426 allocations: 2.55 GiB) for 1 frequency, 128×128 spatial grid
# 60.865 s (86090986 allocations: 15.07 GiB) for 1 frequency, 256×256 spatial grid

n_ng_rwg(ωs,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
ns_ω, ngs_ω = n_ng_rwg(ωs,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
∂ns_ω = [gradient(n_rwg,om,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)[1:6] for om in ωs]
∂ngs_ω = [gradient(ng_rwg,om,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)[1:6] for om in ωs]
# @btime n_ng_rwg($ωs,$w,$t_core,$edge_gap,$n_core,$n_subs,$Δx,$Δy,$Δz,$Nx,$Ny,$Nz)
# 998.123 ms (2063797 allocations: 482.05 MiB) for 11 frequencies, 64×64 spatial grid

n_ng_w = [n_ng_rwg(ω,ww,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz) for ww in ws]
n_ng_t = [n_ng_rwg(ω,w,tt,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz) for tt in ts]
n_w = [ nng[1] for nng in n_ng_w]
ng_w = [ nng[2] for nng in n_ng_w]
n_t = [ nng[1] for nng in n_ng_t]
ng_t = [ nng[2] for nng in n_ng_t]
∂ns_w = [gradient(n_rwg,ω,ww,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)[1:6] for ww in ws]
∂ns_t = [gradient(n_rwg,ω,w,tt,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)[1:6] for tt in ts]
∂ngs_w = [gradient(ng_rwg,ω,ww,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)[1:6] for ww in ws]
∂ngs_t = [gradient(ng_rwg,ω,w,tt,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)[1:6] for tt in ts]

∂n∂ω_AD = [ dnng[1] for dnng in ∂ns_ω ]
∂n∂w_AD = [ dnng[2] for dnng in ∂ns_w ]
∂n∂t_AD = [ dnng[3] for dnng in ∂ns_t ]

∂ng∂ω_AD = [ dnng[1] for dnng in ∂ngs_ω ]
∂ng∂w_AD = [ dnng[2] for dnng in ∂ngs_w ]
∂ng∂t_AD = [ dnng[3] for dnng in ∂ngs_t ]

∂n_ng∂ω_FD(om) = central_fdm(5, 1)(x->[n_ng_rwg(x,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)...],om)
∂n_ng∂w_FD(ww) = central_fdm(5, 1)(x->[n_ng_rwg(ω,x,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)...],ww)
∂n_ng∂t_FD(tt) = central_fdm(5, 1)(x->[n_ng_rwg(ω,w,x,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)...],tt)

∂nng∂ω_FD = hcat([∂n_ng∂ω_FD(om) for om in ωs]...)
∂nng∂t_FD = hcat([∂n_ng∂t_FD(tt) for tt in ts]...)
∂nng∂w_FD = hcat([∂n_ng∂w_FD(ww) for ww in ws]...)
∂n∂ω_FD = ∂nng∂ω_FD[1,:]
∂ng∂ω_FD = ∂nng∂ω_FD[2,:]
∂n∂t_FD = ∂nng∂t_FD[1,:]
∂ng∂t_FD = ∂nng∂t_FD[2,:]
∂n∂w_FD = ∂nng∂w_FD[1,:]
∂ng∂w_FD = ∂nng∂w_FD[2,:]

p_∂n∂ω = plot(ωs,∂n∂ω_FD,label="∂n∂ω,FD")
plot!(p_∂n∂ω,ωs,real(∂n∂ω_AD),label="∂n∂ω,AD")

p_∂n∂w = plot(ws,∂n∂w_FD,label="∂n∂w,FD")
plot!(p_∂n∂w,ws,real(∂n∂w_AD),label="∂n∂w,AD")


p_∂ng∂ω = plot(ωs,∂ng∂ω_FD,label="∂ng∂ω,FD")
plot!(p_∂ng∂ω,ωs,real(∂ng∂ω_AD),label="∂ng∂ω,AD")

p_∂ng∂t = plot(ts,∂ng∂t_FD,label="∂ng∂t,FD")
plot!(p_∂ng∂t,ts,∂ng∂t_AD/1e4,label="∂ng∂t,AD")

p_∂ng∂w = plot(ws,∂ng∂w_FD,label="∂ng∂w,FD")
plot!(p_∂ng∂w,ws,∂ng∂w_AD/1e4,label="∂ng∂w,AD")


################################################################################
#                                                                              #
#                            Circle Waveguide Tests                            #
#                                                                              #
################################################################################
t = 0.2
w = 0.7
ts = collect(0.4:0.04:0.9)
ws = collect(0.6:0.02:0.8)
ωs = collect(0.645*(0.9:0.02:1.1))
# ε

ε_pt_w_cwg(ww) = εₛ(circ_wg(ww,t,edge_gap,n_core,n_subs,Δx,Δy),g)[1,1,40,40,1]
∂ε_pt_w_cwg_AD = [gradient(ε_pt_w_cwg,ww) for ww in ws]
∂ε_pt_w_cwg_FD
plot(ws,ε_w_cwg)



function solve_cwg(om,ww,tt_core,eedge_gap,nn_core,nn_subs,Dx,Dy,Dz,nx,ny,nz)
    solve_k(om,circ_wg(ww,tt_core,eedge_gap,nn_core,nn_subs,dropgrad(Dx),dropgrad(Dy)),dropgrad(Dx),dropgrad(Dy),dropgrad(Dz),dropgrad(nx),dropgrad(ny),dropgrad(nz))
end

function ng_cwg(om,ww,tt_core,eedge_gap,nn_core,nn_subs,Dx,Dy,Dz,nx,ny,nz)
    solve_n(om,circ_wg(ww,tt_core,eedge_gap,nn_core,nn_subs,dropgrad(Dx),dropgrad(Dy)),dropgrad(Dx),dropgrad(Dy),dropgrad(Dz),dropgrad(nx),dropgrad(ny),dropgrad(nz))[2]
end

function n_cwg(om,ww,tt_core,eedge_gap,nn_core,nn_subs,Dx,Dy,Dz,nx,ny,nz)
    solve_n(om,circ_wg(ww,tt_core,eedge_gap,nn_core,nn_subs,dropgrad(Dx),dropgrad(Dy)),dropgrad(Dx),dropgrad(Dy),dropgrad(Dz),dropgrad(nx),dropgrad(ny),dropgrad(nz))[1]
end

function n_ng_cwg(om,ww,tt_core,eedge_gap,nn_core,nn_subs,Dx,Dy,Dz,nx,ny,nz)
    solve_n(om,circ_wg(ww,tt_core,eedge_gap,nn_core,nn_subs,dropgrad(Dx),dropgrad(Dy)),dropgrad(Dx),dropgrad(Dy),dropgrad(Dz),dropgrad(nx),dropgrad(ny),dropgrad(nz))
end

# ng_cwg(1/1.55,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
# ng_cwg(collect(0.645*(0.9:0.02:1.1)),w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)

# @btime ng_cwg(0.645,$w,$t_core,$edge_gap,$n_core,$n_subs,$Δx,$Δy,$Δz,$Nx,$Ny,$Nz)
# 50.866 ms (173716 allocations: 22.07 MiB) for 1 frequency, 32×32 spatial grid
# 198.690 ms (630899 allocations: 75.48 MiB) for 1 frequency, 64×64 spatial grid
# 956.631 ms (2527945 allocations: 322.88 MiB) for 1 frequency, 128×128 spatial grid
# 9.661 s (9835867 allocations: 1.23 GiB) for 1 frequency, 256×256 spatial grid

# @btime gradient(ng_cwg,0.645,$w,$t_core,$edge_gap,$n_core,$n_subs,$Δx,$Δy,$Δz,$Nx,$Ny,$Nz)
# 476.851 ms (1772737 allocations: 118.51 MiB) for 1 frequency, 32×32 spatial grid
# 2.087 s (6093797 allocations: 499.66 MiB) for 1 frequency, 64×64 spatial grid
# 9.447 s (22503426 allocations: 2.55 GiB) for 1 frequency, 128×128 spatial grid
# 60.865 s (86090986 allocations: 15.07 GiB) for 1 frequency, 256×256 spatial grid

n_ng_cwg(ωs,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
ns_ω, ngs_ω = n_ng_cwg(ωs,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
∂ns_ω = [gradient(n_cwg,om,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)[1:6] for om in ωs]
∂ngs_ω = [gradient(ng_cwg,om,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)[1:6] for om in ωs]
# @btime n_ng_cwg($ωs,$w,$t_core,$edge_gap,$n_core,$n_subs,$Δx,$Δy,$Δz,$Nx,$Ny,$Nz)
# 998.123 ms (2063797 allocations: 482.05 MiB) for 11 frequencies, 64×64 spatial grid

n_ng_w = [n_ng_cwg(ω,ww,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz) for ww in ws]
n_ng_t = [n_ng_cwg(ω,w,tt,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz) for tt in ts]
n_w = [ nng[1] for nng in n_ng_w]
ng_w = [ nng[2] for nng in n_ng_w]
n_t = [ nng[1] for nng in n_ng_t]
ng_t = [ nng[2] for nng in n_ng_t]
∂ns_w = [gradient(n_cwg,ω,ww,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)[1:6] for ww in ws]
∂ns_t = [gradient(n_cwg,ω,w,tt,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)[1:6] for tt in ts]
∂ngs_w = [gradient(ng_cwg,ω,ww,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)[1:6] for ww in ws]
∂ngs_t = [gradient(ng_cwg,ω,w,tt,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)[1:6] for tt in ts]

∂n∂ω_AD = [ dnng[1] for dnng in ∂ns_ω ]
∂n∂w_AD = [ dnng[2] for dnng in ∂ns_w ]
∂n∂t_AD = [ dnng[3] for dnng in ∂ns_t ]

∂ng∂ω_AD = [ dnng[1] for dnng in ∂ngs_ω ]
∂ng∂w_AD = [ dnng[2] for dnng in ∂ngs_w ]
∂ng∂t_AD = [ dnng[3] for dnng in ∂ngs_t ]

∂n_ng∂ω_FD(om) = central_fdm(5, 1)(x->[n_ng_cwg(x,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)...],om)
∂n_ng∂w_FD(ww) = central_fdm(5, 1)(x->[n_ng_cwg(ω,x,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)...],ww)
∂n_ng∂t_FD(tt) = central_fdm(5, 1)(x->[n_ng_cwg(ω,w,x,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)...],tt)

∂nng∂ω_FD = hcat([∂n_ng∂ω_FD(om) for om in ωs]...)
∂nng∂t_FD = hcat([∂n_ng∂t_FD(tt) for tt in ts]...)
∂nng∂w_FD = hcat([∂n_ng∂w_FD(ww) for ww in ws]...)
∂n∂ω_FD = ∂nng∂ω_FD[1,:]
∂ng∂ω_FD = ∂nng∂ω_FD[2,:]
∂n∂t_FD = ∂nng∂t_FD[1,:]
∂ng∂t_FD = ∂nng∂t_FD[2,:]
∂n∂w_FD = ∂nng∂w_FD[1,:]
∂ng∂w_FD = ∂nng∂w_FD[2,:]

p_∂n∂ω = plot(ωs,∂n∂ω_FD,label="∂n∂ω,FD")
plot!(p_∂n∂ω,ωs,real(∂n∂ω_AD),label="∂n∂ω,AD")

N = 64 * 64 * 2

p_∂n∂w = plot(ws,∂n∂w_FD,label="∂n∂w,FD")
plot!(p_∂n∂w,ws,real(∂n∂w_AD/N),label="∂n∂w,AD")

p_∂n∂t = plot(ts,∂n∂t_FD,label="∂n∂t,FD")
plot!(p_∂n∂t,ts,real(∂n∂t_AD/N),label="∂n∂t,AD")

p_∂ng∂ω = plot(ωs,∂ng∂ω_FD,label="∂ng∂ω,FD")
plot!(p_∂ng∂ω,ωs,real(∂ng∂ω_AD),label="∂ng∂ω,AD")

p_∂ng∂t = plot(ts,∂ng∂t_FD,label="∂ng∂t,FD")
plot!(p_∂ng∂t,ts,∂ng∂t_AD/1e4,label="∂ng∂t,AD")

p_∂ng∂w = plot(ws,∂ng∂w_FD,label="∂ng∂w,FD")
plot!(p_∂ng∂w,ws,∂ng∂w_AD/1e4,label="∂ng∂w,AD")




################################################################################
################################################################################
ng_rwg(collect(ωs,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
grads_ng_rwg = [gradient(ng_rwg,om,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz) for om in ωs]



ng_rwg_soln, ng_rwg_pb = Zygote.pullback(ng_rwg,1/1.55,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
∂ng = ng_rwg_pb(1)
∂ng = gradient(ng_rwg,1/1.55,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
∂ng[1:6]
Zygote.gradient(ng_rwg,collect(0.645*(0.9:0.02:1.1)),w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)

gradient(n_rwg,1/1.55,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)

ωs = 1/1.55 * collect(0.9:0.02:1.1)
loo(ωs,w,t_core,edge_gap,n_core,n_subs)
@btime loo($ωs,$w,$t_core,$edge_gap,$n_core,$n_subs)


Zygote.gradient(loo,1/1.55,w,t_core,edge_gap,n_core,n_subs)

# plot_ε(hoo(1/1.55,w,t_core,edge_gap,n_core,n_subs,g),g.x,g.y)
# H,k = hoo(1/1.55,w,t_core,edge_gap,n_core,n_subs,g)
# plot_d⃗(H,k,g)



sum(eps_hoo)

function goo(w,Δx,Δy,Δz,Nx,Ny,Nz)
        g = MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
        sum(εₛ⁻¹(test_shapes2(w),g))
end
foo2(w,Δx,Δy,Δz,Nx,Ny,Nz)
Zygote.gradient(foo2,w,Δx,Δy,Δz,Nx,Ny,Nz)

ε⁻¹ = [ε⁻¹_SA[ix,iy,iz][a,b] for a=1:3,b=1:3,ix=1:Nx,iy=1:Ny,iz=1:Nz]
H,k = solve_k(ω,ε⁻¹,Δx,Δy,Δz)


compare_fields(d,d,g)


g = MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
tree = KDTree(s)
# ε_sm_inv = copy(reshape( [inv(εₛ(shapes,tree,g.x[i],g.y[j],g.δx,g.δy)) for i=1:g.Nx,j=1:g.Ny], (g.Nx,g.Ny,1)) )
# ε_sm_inv = [inv(εₛ(shapes,tree,g.x[ix],g.y[iy],g.δx,g.δy))[a,b] for a=1:3,b=1:3,ix=1:g.Nx,iy=1:g.Ny,iz=1:g.Nz]
ε_sm_inv = [inv(εₛ(s,tree,g.x[ix],g.y[iy],g.δx,g.δy))[a,b] for a=1:3,b=1:3,ix=1:g.Nx,iy=1:g.Ny,iz=1:g.Nz]

@tullio einv[a,b]






##
using LinearAlgebra
ε = rand(3,3)
[   -1/ε[1,1]                  ε[1,2]/ε[1,1]                                 ε[1,3]/ε[1,1]
    ε[2,1]/ε[1,1]              ε[2, ]- ε[2,1]*ε[1,2]/ε[1,1]                  ε[2,3] - ε[2,1]*ε[1,3]/ε[1,1]
    ε[3,1]/ε[1,1]              ε[3,2] - ε[3,1]*ε[1,2]/ε[1,1]                 ε[3,3] - ε[3,1]*ε[1,3]/ε[1,1]       ]

a⃗ = SVector(1., 2., 3.)
b⃗ = SVector(4.,5.,6.)
normalize(a⃗×b⃗)

gradient(collect(1.:1.:6.)...) do a,b,c,d,e,f
    a⃗ = SVector(a, b, c)
    b⃗ = SVector(d, e, f)
    normalize(a⃗×b⃗)[1]
end

const Float = typeof(0.0)  # use Float = Float128 for quadruple precision in the future
const CFloat = Complex{Float}
const SVec3 = SVector{3}
const SVec3Complex = SVec3{CFloat}
const SVec3Number = SVec3{<:Number}
const SMat3Complex = SMatrix{3,3,CFloat,9}
const SVec2Float = SVec2{Float}
const SVec3Float = SVec3{Float}

ix,iy,iz = 6,9,1
tree = KDTree(s)
εₛ(s,tree,g.x[ix],g.y[iy],g.δx,g.δy)
[(println("$ix,$iy,$iz");inv(εₛ(s,tree,g.x[ix],g.y[iy],g.δx,g.δy))[a,b]) for a=1:3,b=1:3,ix=1:g.Nx,iy=1:g.Ny,iz=1:g.Nz]

# Equation (4) of the paper by Kottke et al.
# @inline function τ_trans(ε₁₁, ε₂₁, ε₃₁, ε₁₂, ε₂₂, ε₃₂, ε₁₃, ε₂₃, ε₃₃)
#     return SMat3Complex(
#         -1/ε₁₁, ε₂₁/ε₁₁, ε₃₁/ε₁₁,
#         ε₁₂/ε₁₁, ε₂₂ - ε₂₁*ε₁₂/ε₁₁, ε₃₂ - ε₃₁*ε₁₂/ε₁₁,
#         ε₁₃/ε₁₁, ε₂₃ - ε₂₁*ε₁₃/ε₁₁, ε₃₃ - ε₃₁*ε₁₃/ε₁₁
#     )
#     # t = zeros(3,3)
#     # t[2:3, 2:3] = s[2:3, 2:3]
#     # s11 = s[1,1]
#     # s[1,1] = -1
#     #
#     # t = t - (s(:,1) * s(1,:)) ./ s11
#     # return t
# end

A = [   11  21  31
        12  22  32
        13  23  33 ]

B = [   11  12  13
        21  22  23
        31  32  33 ]

function ftest(x1,x2,x3,x4,x5,x6,x7,x8,x9)
    println("x1: $x1")
    println("x2: $x2")
    println("x3: $x3")
    println("x4: $x4")
    println("x5: $x5")
    println("x6: $x6")
    println("x7: $x7")
    println("x8: $x8")
    println("x9: $x9")
end

ftest(A...)
# x1: 11
# x2: 12
# x3: 13
# x4: 21
# x5: 22
# x6: 23
# x7: 31
# x8: 32
# x9: 33

ftest(B...)
# x1: 11
# x2: 21
# x3: 31
# x4: 12
# x5: 22
# x6: 32
# x7: 13
# x8: 23
# x9: 33

ftest(vec(A)...)
ftest(vec(B)...)

ins = randn(9)
real(Array(τ_trans(ins...)))   ≈ τ_trans2(ins...)




τ_trans2(ins...)

Array(τ⁻¹_trans(collect(1:9)))   ≈ τ⁻¹_trans2(collect(1:9)...)

τ⁻¹_trans2(collect(1:9)...)

n12 = [1.; 0.; 0.]
n = n12 / sqrt(sum2(abs2,n12))
# Pick a vector that is not along n.
if any(n .== 0.)
    h_temp2 = Float64.(n .== 0.)
    println("n: $n")
    println("h_temp2: $h_temp2")
else
    h_temp2 = [1.; 1.; 1.]
end
h_temp2 = [1.; 1.; 1.]
# h_temp = [0., 0., 1.]
# Create two vectors that are normal to n and normal to each other.
# h_temp2 = n×[0.;0.;1.]
h = h_temp2 / sqrt(sum2(abs2,h_temp2))
v_temp = n×h
v = v_temp / sqrt(sum2(abs2,v_temp))
# Create a local Cartesian coordinate system.
S = [n h v]  # unitary



@inline function τ_trans3(εin)
    ε = εin
    [   -1/ε[1,1]                  ε[1,2]/ε[1,1]                                 ε[1,3]/ε[1,1]
        ε[2,1]/ε[1,1]              ε[2, ]- ε[2,1]*ε[1,2]/ε[1,1]                  ε[2,3] - ε[2,1]*ε[1,3]/ε[1,1]
        ε[3,1]/ε[1,1]              ε[3,2] - ε[3,1]*ε[1,2]/ε[1,1]                 ε[3,3] - ε[3,1]*ε[1,3]/ε[1,1]       ]
end

# Equation (23) of the paper by Kottke et al.
@inline function τ⁻¹_trans3(τin)
    τ = τin
    [   -1/τ[1,1]       -τ[1,2]/τ[1,1]                       -τ[1,3]/τ[1,1]
        -τ[2,1]/τ[1,1]    τ[2,2] - τ[2,1]*τ[1,2]/τ[1,1]       τ[2,3] - τ[2,1]*τ[1,3]/τ[1,1]
        -τ[3,1]/τ[1,1]    τ[3,2] - τ[3,1]*τ[1,2]/τ[1,1]       τ[3,3] - τ[3,1]*τ[1,3]/τ[1,1]      ]
end

param1 = ε_tensor(3.5)
param2 = ε_tensor(1.4)
rvol1 = 0.3
  # unitary
τ1 = τ_trans3(transpose(S) * param1 * S)  # express param1 in S coordinates, and apply τ transform
τ2 = τ_trans3(transpose(S) * param2 * S)  # express param2 in S coordinates, and apply τ transform
τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)  # volume
S * τ⁻¹_trans(τavg) * transpose(S)

@inline function τ_trans(ε₁₁, ε₂₁, ε₃₁, ε₁₂, ε₂₂, ε₃₂, ε₁₃, ε₂₃, ε₃₃)
    [   -1/ε₁₁               ε₁₂/ε₁₁                            ε₁₃/ε₁₁
        ε₂₁/ε₁₁              ε₂₂ - ε₂₁*ε₁₂/ε₁₁                  ε₂₃ - ε₂₁*ε₁₃/ε₁₁
        ε₃₁/ε₁₁              ε₃₂ - ε₃₁*ε₁₂/ε₁₁                  ε₃₃ - ε₃₁*ε₁₃/ε₁₁       ]
end

# Equation (23) of the paper by Kottke et al.
@inline function τ⁻¹_trans(τ₁₁, τ₂₁, τ₃₁, τ₁₂, τ₂₂, τ₃₂, τ₁₃, τ₂₃, τ₃₃)
    [   -1/τ₁₁      -τ₁₂/τ₁₁                -τ₁₃/τ₁₁
        -τ₂₁/τ₁₁    τ₂₂ - τ₂₁*τ₁₂/τ₁₁       τ₂₃ - τ₂₁*τ₁₃/τ₁₁
        -τ₃₁/τ₁₁    τ₃₂ - τ₃₁*τ₁₂/τ₁₁       τ₃₃ - τ₃₁*τ₁₃/τ₁₁      ]
end

function avg_param(param1, param2, n12, rvol1)
    n = n12 / sqrt(sum2(abs2,n12))
    # Pick a vector that is not along n.
    # if any(n .== 0)
    # 	h_temp = (n .== 0)
    # else
    # 	h_temp = [1., 0., 0.]
    # end
    # h_temp = [0., 0., 1.]
    # Create two vectors that are normal to n and normal to each other.
    h_temp2 = n×[0.;0.;1.]
    h = h_temp2 / sqrt(sum2(abs2,h_temp2))
    v_temp = n×h
    v = v_temp / sqrt(sum2(abs2,v_temp))
    # Create a local Cartesian coordinate system.
    S = [n h v]  # unitary
    τ1 = τ_trans(transpose(S) * param1 * S)  # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans(transpose(S) * param2 * S)  # express param2 in S coordinates, and apply τ transform
    τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)  # volume-weighted average
    return S * τ⁻¹_trans(τavg) * transpose(S)  # apply τ⁻¹ and transform back to global coordinates
end

ε_tensor(3.5)



gradient(3.5,1.4,1.,2.,3.,0.3) do n1,n2,a,b,c,r
    abs2(sum(avg_param(ε_tensor(n1),ε_tensor(n2),[a;b;c],r)))
end

function fgrad(p1,p2,p3,p4,p5,p6)
    gradient(p1,p2,p3,p4,p5,p6) do n1,n2,a,b,c,r
        abs2(sum(kottke_avg_param2(ε_tensor(n1),ε_tensor(n2),[a;b;c],r)))
    end
end

function f(n1,n2,a,b,c,r)
    # abs2(sum(kottke_avg_param2(ε_tensor(n1),ε_tensor(n2),[a,b,c],r)))
    kottke_avg_param2(ε_tensor(n1),ε_tensor(n2),[a,b,c],r)
end

@inline function τ_trans2(ε₁₁, ε₂₁, ε₃₁, ε₁₂, ε₂₂, ε₃₂, ε₁₃, ε₂₃, ε₃₃)
    [   -1/ε₁₁               ε₁₂/ε₁₁                            ε₁₃/ε₁₁
        ε₂₁/ε₁₁              ε₂₂ - ε₂₁*ε₁₂/ε₁₁                  ε₂₃ - ε₂₁*ε₁₃/ε₁₁
        ε₃₁/ε₁₁              ε₃₂ - ε₃₁*ε₁₂/ε₁₁                  ε₃₃ - ε₃₁*ε₁₃/ε₁₁       ]
end

# Equation (23) of the paper by Kottke et al.
@inline function τ⁻¹_trans2(τ₁₁, τ₂₁, τ₃₁, τ₁₂, τ₂₂, τ₃₂, τ₁₃, τ₂₃, τ₃₃)
    [   -1/τ₁₁      -τ₁₂/τ₁₁                -τ₁₃/τ₁₁
        -τ₂₁/τ₁₁    τ₂₂ - τ₂₁*τ₁₂/τ₁₁       τ₂₃ - τ₂₁*τ₁₃/τ₁₁
        -τ₃₁/τ₁₁    τ₃₂ - τ₃₁*τ₁₂/τ₁₁       τ₃₃ - τ₃₁*τ₁₃/τ₁₁      ]
end

function kottke_avg_param2(param1, param2, n12, rvol1)
    n = n12 / sqrt(sum2(abs2,n12))
    # Pick a vector that is not along n.
    # if any(n .== 0)
    # 	h_temp = (n .== 0)
    # else
    # 	h_temp = [1., 0., 0.]
    # end
    # h_temp = [0., 0., 1.]
    # Create two vectors that are normal to n and normal to each other.
    h_temp2 = n×[0.;0.;1.]
    h = h_temp2 / sqrt(sum2(abs2,h_temp2))
    v_temp = n×h
    v = v_temp / sqrt(sum2(abs2,v_temp))
    # Create a local Cartesian coordinate system.
    S = [n h v]  # unitary
    τ1 = τ_trans((transpose(S) * param1 * S)...)  # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans((transpose(S) * param2 * S)...)  # express param2 in S coordinates, and apply τ transform
    τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)  # volume-weighted average
    return S * τ⁻¹_trans(τavg...) * transpose(S)  # apply τ⁻¹ and transform back to global coordinates
end

f(3.5,1.4,1.,2.,3.,0.3)
fgrad(3.5,1.4,1.,2.,3.,0.3)
@btime f(3.5,1.4,1.,2.,3.,0.3)
@btime fgrad(3.5,1.4,1.,2.,3.,0.3)

##
tree = KDTree(s)
eps_bg = εₛ(s,g)[1,1,1]


x,y = g.x[50], g.y[50]
x1,y1 = x+g.δx/2.,y+g.δy/2
x2,y2 = x+g.δx/2.,y-g.δy/2
x3,y3 = x-g.δx/2.,y-g.δy/2
x4,y4 = x-g.δx/2.,y+g.δy/2

using MaxwellFDM: kottke_avg_param

vxl = (SVector{2,Float64}(x3,y3), SVector{2,Float64}(x1,y1))
rvol = volfrac(vxl,nout,r₀)
r₀,nout = surfpt_nearby([x, y], s[2])
e0 = kottke_avg_param(
        SHM3(s[2].data),
        SHM3(eps_bg),
        SVector{3,Float64}(nout[1],nout[2],0),
        rvol,)

@tullio eps[a,b,ix,iy,iz] :=

Array{Float64}(real.(e0))

ε_sm_inv = [inv(εₛ(s,tree,g.x[ix],g.y[iy],g.δx,g.δy))[a,b] for a=1:3,b=1:3,ix=1:g.Nx,iy=1:g.Ny,iz=1:g.Nz]

using Revise, StaticArrays, GeometryPrimitives, Zygote
##
Zygote.@adjoint (T::Type{<:SArray})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
Zygote.@adjoint (T::Type{<:SArray})(x::AbstractArray) = T(x), dv -> (nothing, dv)
Zygote.@adjoint (T::Type{<:SMatrix})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
Zygote.@adjoint (T::Type{<:SMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
Zygote.@adjoint (T::Type{<:SVector})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
Zygote.@adjoint (T::Type{<:SVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)


Zygote.@adjoint enumerate(xs) = enumerate(xs), diys -> (map(last, diys),)
_ndims(::Base.HasShape{d}) where {d} = d
_ndims(x) = Base.IteratorSize(x) isa Base.HasShape ? _ndims(Base.IteratorSize(x)) : 1
Zygote.@adjoint function Iterators.product(xs...)
                    d = 1
                    Iterators.product(xs...), dy -> ntuple(length(xs)) do n
                        nd = _ndims(xs[n])
                        dims = ntuple(i -> i<d ? i : i+nd, ndims(dy)-nd)
                        d += nd
                        func = sum(y->y[n], dy; dims=dims)
                        ax = axes(xs[n])
                        reshape(func, ax)
                    end
                end


function sum2(op,arr)
    return sum(op,arr)
end

function sum2adj( Δ, op, arr )
    n = length(arr)
    g = x->Δ*Zygote.gradient(op,x)[1]
    return ( nothing, map(g,arr))
end

Zygote.@adjoint function sum2(op,arr)
    return sum2(op,arr),Δ->sum2adj(Δ,op,arr)
end
##
Zygote.refresh()

b = Box([0.,0.],[1.,2.]) # ([center],[size along axes])

sum(abs2,sum.(bounds(b)))
bounds(b)
gradient(0.5,0.3,1.0,2.0) do c1,c2,r1,r2
        sum(abs2,sum.(bounds(Box([c1,c2],[r1,r2]))))
end



gradient(0.5,0.6,1.4,2.8) do c1,c2,r1,r2
        sum(abs2,sum.(bounds2(Box([c1,c2],[r1,r2]))))
end

signmatrix(b::Box{1}) = SMatrix{1,1}(1)
signmatrix(b::Box{2}) = SMatrix{2,2}(1,1, -1,1)
signmatrix(b::Box{3}) = SMatrix{3,4}(1,1,1, -1,1,1, 1,-1,1, 1,1,-1)

function bounds2(b::Box)
    A = inv(b.p) .* b.r'
    # m = maximum(abs.(A * signmatrix(b)), dims=2)[:,1] # extrema of all 2^N corners of the box
    m = maximum(abs.(Array((inv(b.p) .* b.r') * signmatrix(b))), dims=2)[:,1] # extrema of all 2^N corners of the box
    return (b.c-m,b.c+m)
end

bounds(b)
bounds2(b)
bounds(b) == bounds2(b)
A = inv(b.p) .* b.r'

abs.(A * signmatrix(b))

x1 = maximum(abs.((inv(b.p) .* b.r') * signmatrix(b)), dims=2)[:,1]

SVector{2,Float64}(maximum(abs.(Array((inv(b.p) .* b.r') * signmatrix(b))), dims=2)[:,1])
x2 = maximum(abs.( Array((inv(b.p) .* b.r') * signmatrix(b)) ), dims=2)
