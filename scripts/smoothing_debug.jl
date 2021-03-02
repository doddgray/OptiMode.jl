using Revise
using OptiMode
using LinearAlgebra, Statistics, StaticArrays, HybridArrays, GeometryPrimitives, BenchmarkTools
using ChainRules, Zygote, ForwardDiff, FiniteDifferences
p = [
       1.7,                #   top ridge width         `w_top`         [μm]
       0.7,                #   ridge thickness         `t_core`        [μm]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
       # 0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
               ];
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg(x) = ridge_wg(x[1],x[2],x[3],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy) # fourth param is gap between modeled object and sidewalls
geom = rwg(p)
ms = ModeSolver(1.45, geom, gr)



##
# ωs = Node(Float64[])
# ng1 = Node(Float64[])
# n1 = Node(Float64[])
λs = 0.7:0.05:1.1
ωs = 1 ./ λs
n1,ng1 = solve_n(ms,ωs,rwg(p))

##
λs = 1 ./ ωs
fig,ax,sc1 = scatter(λs,ng1,color=logocolors[:red])
lines!(ax,λs,ng1,color=logocolors[:red],lw=2)
lines!(ax,λs,n1,color=logocolors[:blue],lw=2)
scatter!(ax,λs,n1,color=logocolors[:blue])
fig
##
function var_ng(ωs,p)
    ng = solve_n(Zygote.dropgrad(ms),ωs,rwg(p))[2]
    mean( abs2.( ng ) ) - abs2(mean(ng))
end
var_ng(ωs,p)
@time ∂vng_RAD = Zygote.gradient(var_ng,ωs,p)

@show ∂vng_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng(ωs,x),p)[1]
@show ∂vng_err = abs.(∂vng_RAD[2] .- ∂vng_FD) ./ abs.(∂vng_FD)

# @time ∂vng_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng(ωs,x),p)
#
# Zygote.gradient((x,y)->solve_n(ms,x,rwg(y))[2],1/0.85,p)
# Zygote.gradient(ωs,p) do oms,x
# 	ngs = solve_n(Zygote.dropgrad(ms),oms,rwg(x))[2]
#     mean( abs2.( ngs ) ) - abs2(mean(ngs))
# end

## Define with constant indices

rwg2(x) = ridge_wg(x[1],x[2],x[3],0.5,2.2,1.4,Δx,Δy) # fourth param is gap between modeled object and sidewalls
n2, ng2 = solve_n(ms,ωs,rwg2(p))
##
fig,ax,sc1 = scatter(λs,ng2,color=logocolors[:red])
lines!(ax,λs,ng2,color=logocolors[:red],lw=2)
lines!(ax,λs,n2,color=logocolors[:blue],lw=2)
scatter!(ax,λs,n2,color=logocolors[:blue])
fig
##
function var_ng2(ωs,p)
    ngs = solve_n(Zygote.dropgrad(ms),ωs,rwg2(p))[2]
    mean( abs2.( ngs ) ) - abs2(mean(ngs))
end
var_ng2(ωs,p)

@show ∂vng2_RAD = Zygote.gradient(var_ng2,ωs,p)
@show ∂vng2_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng2(ωs,x),p)[1]
@show ∂vng2_err = abs.(∂vng2_RAD[2] .- ∂vng2_FD) ./ abs.(∂vng2_FD)


var_ng(ωs,p)
@show ∂vng_RAD = Zygote.gradient(var_ng,ωs,p)
@show ∂vng_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng(ωs,x),p)[1]
@show ∂vng_err = abs.(∂vng_RAD[2] .- ∂vng_FD) ./ abs.(∂vng_FD)

##

∂n_RAD = zeros(length(ωs),3)
∂n_FD = zeros(length(ωs),3)
∂n_err = zeros(length(ωs),3)

∂ng_RAD = zeros(length(ωs),3)
∂ng_FD = zeros(length(ωs),3)
∂ng_err = zeros(length(ωs),3)

for omind in 1:length(ωs)
    ∂n_RAD[omind,:] = Zygote.gradient(x->solve_n(ms,ωs[omind],rwg2(x))[1],p)[1]
    ∂ng_RAD[omind,:] = Zygote.gradient(x->solve_n(ms,ωs[omind],rwg2(x))[2],p)[1]
    ∂n_FD[omind,:] = FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,ωs[omind],rwg2(x))[1],p)[1]
    ∂ng_FD[omind,:] = FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,ωs[omind],rwg2(x))[2],p)[1]
end
∂n_err = abs.(∂n_RAD .- ∂n_FD) ./ abs.(∂n_FD)
∂ng_err = abs.(∂ng_RAD .- ∂ng_FD) ./ abs.(∂ng_FD)

##
ln = lines(collect(λs),∂n_err[:,1],color=logocolors[:green])
lines!(collect(λs),∂n_err[:,2],color=logocolors[:blue])
lines!(collect(λs),∂n_err[:,3],color=logocolors[:red])

lng = lines(collect(λs),∂ng_err[:,1],color=logocolors[:green])
lines!(collect(λs),∂ng_err[:,2],color=logocolors[:blue])
lines!(collect(λs),∂ng_err[:,3],color=logocolors[:red])

##
ei2 = εₛ⁻¹(1/1.55,rwg2(p);ms)
Zygote.gradient((x,y)->sum(sum(εₛ⁻¹(x,rwg2(y);ms))),1/1.55,p)



@time ∂sumei_FD = FiniteDifferences.grad(central_fdm(3,1),x->sum(sum(εₛ⁻¹(1/1.55,rwg2(x);ms))),p)




Zygote.gradient((x,y)->solve_n(ms,x,rwg2(y))[1],1/1.55,p)




@time ∂n2_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,1/1.55,rwg2(x))[1],p)





Zygote.gradient((x,y)->solve_n(ms,x,rwg2(y))[2],1/1.55,p)




@time ∂ng2_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,1/1.55,rwg2(x))[2],p)





omsq2,H2 = solve_ω²(ms,1.45,rwg2(p))
summag4(HH) = sum(abs2.(HH).^2)

@show ∂omsq_k_RAD = Zygote.gradient(x->solve_ω²(ms,x,rwg2(p))[1],1.45)[1]
@show ∂omsq_k_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_ω²(ms,x,rwg2(p))[1],1.45)[1]
@show ∂omsq_k_err = abs(∂omsq_k_RAD - ∂omsq_k_FD) / ∂omsq_k_FD

@show ∂sm4_k_RAD = Zygote.gradient(x->summag4(solve_ω²(ms,x,rwg2(p))[2]),1.45)[1]
@show ∂sm4_k_FD =  FiniteDifferences.grad(central_fdm(3,1),x->summag4(solve_ω²(ms,x,rwg2(p))[2]),1.45)[1]
@show ∂omsq_k_err = abs( ∂sm4_k_RAD -  ∂sm4_k_FD) /  ∂sm4_k_FD

@show ∂omsq_p_RAD =  Zygote.gradient(x->solve_ω²(ms,1.45,rwg2(x))[1],p)[1]
@show ∂omsq_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_ω²(ms,1.45,rwg2(x))[1],p)[1]
@show ∂omsq_p_err = abs.(∂omsq_p_RAD .- ∂omsq_p_FD) ./ ∂omsq_p_FD

# @show ∂omsq_p_RAD =  Zygote.gradient(x->solve_ω²(ms,1.45,rwg(x))[1],p)[1]
# @show ∂omsq_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_ω²(ms,1.45,rwg(x))[1],p)[1]
# @show ∂omsq_p_err = abs.(∂omsq_p_RAD .- ∂omsq_p_FD) ./ ∂omsq_p_FD

@show ∂sm4_p_RAD = Zygote.gradient(x->summag4(solve_ω²(ms,1.45,rwg2(x))[2]),p)[1]
@show ∂sm4_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->summag4(solve_ω²(ms,1.45,rwg2(x))[2]),p)[1]
@show ∂sm4_p_err = abs.(∂sm4_p_RAD .- ∂sm4_p_FD) ./ ∂sm4_p_FD

k2,H22 = solve_k(ms,0.7,rwg2(p))

@show ∂k_om_RAD = Zygote.gradient(x->solve_k(ms,x,rwg2(p))[1],0.7)[1]
@show ∂k_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,x,rwg2(p))[1],0.7)[1]
@show ∂k_om_err = abs(∂k_om_RAD - ∂k_om_FD) / abs(∂k_om_FD)

@show ∂sm4_om_RAD = Zygote.gradient(x->summag4(solve_k(ms,x,rwg2(p))[2]),0.7)[1]
@show ∂sm4_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->summag4(solve_k(ms,x,rwg2(p))[2]),0.7)[1]
@show ∂sm4_om_err = abs( ∂sm4_om_RAD -  ∂sm4_om_FD) /  abs(∂sm4_om_FD)

@show ∂k_p_RAD =  Zygote.gradient(x->solve_k(ms,0.7,rwg2(x))[1],p)[1]
@show ∂k_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,0.7,rwg2(x))[1],p)[1]
@show ∂k_p_err = abs.(∂k_p_RAD .- ∂k_p_FD) ./ abs.(∂k_p_FD)

@show ∂k_p_RAD =  Zygote.gradient(x->solve_k(ms,0.7,rwg(x))[1],p)[1]
@show ∂k_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,0.7,rwg(x))[1],p)[1]
@show ∂k_p_err = abs.(∂k_p_RAD .- ∂k_p_FD) ./ abs.(∂k_p_FD)

@show ∂sm4k_p_RAD = Zygote.gradient(x->summag4(solve_k(ms,0.7,rwg2(x))[2]),p)[1]
@show ∂sm4k_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->summag4(solve_k(ms,0.7,rwg2(x))[2]),p)[1]
@show ∂sm4k_p_err = abs.(∂sm4k_p_RAD .- ∂sm4k_p_FD) ./ ∂sm4k_p_FD

@show ∂sm4k_p_RAD = Zygote.gradient(x->summag4(solve_k(ms,0.7,rwg(x))[2]),p)[1]
@show ∂sm4k_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->summag4(solve_k(ms,0.7,rwg(x))[2]),p)[1]
@show ∂sm4k_p_err = abs.(∂sm4k_p_RAD .- ∂sm4k_p_FD) ./ ∂sm4k_p_FD

@show ∂sm4k_p_RAD = Zygote.gradient(x->summag4(solve_k(ms,0.7,rwg(x))[2]),p)[1]
@show ∂sm4k_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->summag4(solve_k(ms,0.7,rwg(x))[2]),p)[1]
@show ∂sm4k_p_err = abs.(∂sm4k_p_RAD .- ∂sm4k_p_FD) ./ ∂sm4k_p_FD

neff1,ng1 = solve_n(ms,0.7,rwg(p))
neff2,ng2 = solve_n(ms,0.7,rwg2(p))

@show ∂n_om_RAD = Zygote.gradient(x->solve_n(ms,x,rwg2(p))[1],0.7)[1]
@show ∂n_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,x,rwg2(p))[1],0.7)[1]
@show ∂n_om_err = abs(∂n_om_RAD - ∂n_om_FD) / abs(∂n_om_FD)

@show ∂ng_om_RAD = Zygote.gradient(x->solve_n(ms,x,rwg2(p))[2],0.7)[1]
@show ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,x,rwg2(p))[2],0.7)[1]
@show ∂ng_om_err = abs( ∂ng_om_RAD -  ∂ng_om_FD) /  abs(∂ng_om_FD)

@show ∂ng_om_RAD = Zygote.gradient(x->solve_n(ms,x,rwg(p))[2],0.7)[1]
@show ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(5,1),x->solve_n(ms,x,rwg(p))[2],0.7)[1]
@show ∂ng_om_err = abs( ∂ng_om_RAD -  ∂ng_om_FD) /  abs(∂ng_om_FD)

@show ∂n_p_RAD =  Zygote.gradient(x->solve_n(ms,0.7,rwg2(x))[1],p)[1]
@show ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,0.7,rwg2(x))[1],p)[1]
@show ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)

@show ∂n_p_RAD =  Zygote.gradient(x->solve_n(ms,0.7,rwg(x))[1],p)[1]
@show ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,0.7,rwg(x))[1],p)[1]
@show ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)

@show ∂ng_p_RAD = Zygote.gradient(x->solve_n(ms,0.7,rwg2(x))[2],p)[1]
@show ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,0.7,rwg2(x))[2],p)[1]
@show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD

@show ∂ng_p_RAD = Zygote.gradient(x->solve_n(ms,0.7,rwg(x))[2],p)[1]
@show ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,0.7,rwg(x))[2],p)[1]
@show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD

@show ∂ng_p_RAD = Zygote.gradient(x->solve_n(ms,0.8,rwg(x))[2],p)[1]
@show ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,0.8,rwg(x))[2],p)[1]
@show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD


neff1,ng1 = solve_n(ms,0.7,rwg(p))
neff2,ng2 = solve_n(ms,0.7,rwg2(p))

@show ∂n_om_RAD = Zygote.gradient(x->solve_n(ms,x,rwg2(p))[1],0.7)[1]
@show ∂n_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,x,rwg2(p))[1],0.7)[1]
@show ∂n_om_err = abs(∂n_om_RAD - ∂n_om_FD) / abs(∂n_om_FD)

@show ∂ng_om_RAD = Zygote.gradient(x->solve_n(ms,x,rwg2(p))[2],0.7)[1]
@show ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,x,rwg2(p))[2],0.7)[1]
@show ∂ng_om_err = abs( ∂ng_om_RAD -  ∂ng_om_FD) /  abs(∂ng_om_FD)

@show ∂ng_om_RAD = Zygote.gradient(x->solve_n(ms,x,rwg(p))[2],0.7)[1]
@show ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(5,1),x->solve_n(ms,x,rwg(p))[2],0.7)[1]
@show ∂ng_om_err = abs( ∂ng_om_RAD -  ∂ng_om_FD) /  abs(∂ng_om_FD)

@show ∂n_p_RAD =  Zygote.gradient(x->solve_n(ms,0.7,rwg2(x))[1],p)[1]
@show ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,0.7,rwg2(x))[1],p)[1]
@show ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)

@show ∂n_p_RAD =  Zygote.gradient(x->solve_n(ms,0.7,rwg(x))[1],p)[1]
@show ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,0.7,rwg(x))[1],p)[1]
@show ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)

@show ∂ng_p_RAD = Zygote.gradient(x->solve_n(ms,0.7,rwg2(x))[2],p)[1]
@show ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->solve_n(ms,0.7,rwg2(x))[2],p)[1]
@show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD

@show ∂ng_p_RAD = Zygote.gradient(x->solve_n(ms,0.7,rwg(x))[2],p)[1]
@show ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->solve_n(ms,0.7,rwg(x))[2],p)[1]
@show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD


f1(om,pp) = sum(sum(εₛ⁻¹(om,rwg(pp);ms)))
f1(0.7,p)
@show ∂f1_om_RAD = Zygote.gradient(x->f1(x,p),0.7)[1]
@show ∂f1_om_FD =  FiniteDifferences.grad(central_fdm(5,1),x->f1(x,p),0.7)[1]
@show ∂f1_om_err = abs( ∂f1_om_RAD -  ∂f1_om_FD) /  abs(∂f1_om_FD)

@show ∂f1_p_RAD =  Zygote.gradient(x->f1(0.7,x),p)[1]
@show ∂f1_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->f1(0.7,x),p)[1]
@show ∂f1_p_err = abs.(∂f1_p_RAD .- ∂f1_p_FD) ./ abs.(∂f1_p_FD)

using Zygote: dropgrad
function f2(om,pp)
    ε⁻¹ = εₛ⁻¹(om,rwg(pp);ms)
    k, H⃗ = solve_k(ms,om,ε⁻¹)
    (mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
    om / H_Mₖ_H(H⃗[:,1],ε⁻¹,real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗)))
end
f2(0.7,p)
@show ∂f2_om_RAD = Zygote.gradient(x->f2(x,p),0.7)[1]
@show ∂f2_om_FD =  FiniteDifferences.grad(central_fdm(5,1),x->f2(x,p),0.7)[1]
@show ∂f2_om_err = abs( ∂f2_om_RAD -  ∂f2_om_FD) /  abs(∂f2_om_FD)

@show ∂f2_p_RAD =  Zygote.gradient(x->f2(0.7,x),p)[1]
@show ∂f2_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->f2(0.7,x),p)[1]
@show ∂f2_p_err = abs.(∂f2_p_RAD .- ∂f2_p_FD) ./ abs.(∂f2_p_FD)



∂omsq_p_RAD = Zygote.gradient(x->solve_k(ms,0.7,rwg2(x))[1],p)




∂omsq_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_ω²(ms,1.45,rwg2(x))[1],p)










∂omsq_p_RAD = Zygote.gradient(x->solve_k(ms,0.65,rwg2(x))[1],p)




∂omsq_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,0.65,rwg2(x))[1],p)





∂smm_p_RAD = Zygote.gradient(x->summag4(solve_k(ms,0.65,rwg2(x))[2]),p)[1]




∂smm_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->summag4(solve_k(ms,0.65,rwg2(x))[2]),p)



##
println("### εₛ: ")
println("## primal: ")
esm = εₛ(0.8,geom,gr)
@btime εₛ(0.8,$geom,$gr) # 2.352 ms (66436 allocations: 7.75 MiB)
println("## gradients: ")
println("# Zygote: ")
@show Zygote.gradient(x->sum(sum(εₛ(x,rwg(p),gr))),0.9)[1]
@show Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)[1]
@btime Zygote.gradient(x->sum(sum(εₛ(x,rwg($p),$gr))),0.9)[1]
@btime Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),$gr))),$p)[1]
println("# ForwardDiff: ")
@show ForwardDiff.derivative(x->sum(sum(εₛ(x,rwg(p),gr))),0.9)
@show ForwardDiff.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)
@btime ForwardDiff.derivative(x->sum(sum(εₛ(x,rwg($p),$gr))),0.9)
@btime ForwardDiff.gradient(x->sum(sum(εₛ(0.8,rwg(x),$gr))),$p)
println("# ForwardDiff over Zygote (2nd order): ")
@show ForwardDiff.derivative(y->Zygote.gradient(x->sum(sum(εₛ(x,rwg(p),gr))),y)[1],0.8)
@show ForwardDiff.gradient(y->Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),y)[1],p)
@btime ForwardDiff.derivative(y->Zygote.gradient(x->sum(sum(εₛ(x,rwg($p),$gr))),y)[1],0.8)
@btime ForwardDiff.gradient(y->Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),$gr))),y)[1],$p)


println("### εₛ⁻¹: ")
println("## primal: ")
eism = εₛ⁻¹(0.8,geom,gr)
@btime εₛ⁻¹(0.8,$geom,$gr) # 2.439 ms (66439 allocations: 7.75 MiB)
println("## gradients: ")
println("# Zygote: ")
@show Zygote.gradient(x->sum(sum(εₛ⁻¹(x,rwg(p),gr))),0.9)[1]
@show Zygote.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),gr))),p)[1]
@btime Zygote.gradient(x->sum(sum(εₛ⁻¹(x,rwg($p),$gr))),0.9)[1]
@btime Zygote.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),$gr))),$p)[1]
println("# ForwardDiff: ")
@show ForwardDiff.derivative(x->sum(sum(εₛ⁻¹(x,rwg(p),gr))),0.9)
@show ForwardDiff.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),gr))),p)
@btime ForwardDiff.derivative(x->sum(sum(εₛ⁻¹(x,rwg($p),$gr))),0.9)
@btime ForwardDiff.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),$gr))),$p)
println("# ForwardDiff over Zygote (2nd order): ")
@show ForwardDiff.derivative(y->Zygote.gradient(x->sum(sum(εₛ⁻¹(x,rwg(p),gr))),y)[1],0.8)
@show ForwardDiff.gradient(y->Zygote.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),gr))),y)[1],p)
@btime ForwardDiff.derivative(y->Zygote.gradient(x->sum(sum(εₛ⁻¹(x,rwg($p),$gr))),y)[1],0.8)
@btime ForwardDiff.gradient(y->Zygote.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),$gr))),y)[1],$p)

SMatrix
using ChainRulesCore: NO_FIELDS
ChainRulesCore.rrule(T::Type{<:SMatrix}, x::AbstractMatrix) = ( T(x), dv -> (NO_FIELDS, dv) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, xs::Number...) = ( T(xs...), dv -> (NO_FIELDS, dv...) )

@Zygote.adjoint (T::Type{<:SMatrix})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
@Zygote.adjoint (T::Type{<:SMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
@Zygote.adjoint (T::Type{SMatrix{2,2,Float64,4}})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)

ChainRules.refresh_rules()
Zygote.refresh()
Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)[1]
Zygote.hessian(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)[1]

##



Zygote.gradient(x->Zygote.forwarddiff(y->sum(sum(εₛ(y...))),[0.8,rwg(x),gr]),p)
Zygote.forwarddiff(y->sum(sum(εₛ(y...))),[0.8,rwg(p),gr])
Zygote.gradient(x->sum(sum(εₛ(0.8,Zygote.forwarddiff(rwg,x),gr))),p)
ForwardDiff.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)

f1(lm,p) = εₛ(lm,rwg(p),gr)
f1(0.8,p)
Zygote.gradient(x->sum(sum(f1(0.8,x))),p)

shapes1 = rwg3(p)
geom2 = rwg2(p)
convert.(Material,getfield.(shapes1,:data))
mats0 = getfield.(shapes1,:data)
similar(mats0,Material)
Material(3.5)
Material.(getfield.(shapes1,:data))

import Base.convert
convert(::Type{Material}, x) = Material(x)
materials(rwg(p))
materials2(shapes::Vector{S}) where S<:Shape{N,N²,D,T} where {N,N²,D<:Material,T} = unique!(getfield.(shapes,:data))
materials2(shapes1)
materials(shapes1)
rwg(p)
eltype(shapes1)<:Shape{N,N²,D,T} where {N,N²,D<:Material,T}
e1 = ε_tensor(3.5)
Material(e1)
##
εs_sym = getfield.(materials(shapes2),:ε)
ε_exprs = build_function.(getfield.(materials(shapes2),:ε),λ)
εs = [ eval(εe[1]) for εe in ε_exprs ]
εs! = [ eval(εe[2]) for εe in ε_exprs ]

εs[1](0.8)

struct Geometry3{N}
	shapes::Vector{Shape{N}}
	# materials::Vector{Material}
end
Geometry3(s::Vector{S}) where S<:Shape{N} where N = Geometry3{N}(s)

shapes1 =
Geometry3(shapes1)


mats = materials(shapes2)
sinds2minds = map(s->findfirst(m->isequal(s.data,m), mats),shapes2)

csinds = corner_sinds(shapes2,ms.M̂.xyz,ms.M̂.xyzc)
sinds_pr = proc_sinds(csinds)
vxl_min = @view ms.M̂.xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
vxl_max = @view ms.M̂.xyzc[min(2,end):end,min(2,end):end,min(2,end):end]



sr1 = S_rvol(ms.M̂.corner_sinds_proc,ms.M̂.xyz,vxl_min,vxl_max,shapes)
@btime S_rvol($ms.M̂.corner_sinds_proc,$ms.M̂.xyz,$vxl_min,$vxl_max,$shapes)

sr2 = S_rvol(ms.M̂.corner_sinds_proc,ms.M̂.xyz,vxl_min,vxl_max,shapes2)
@btime S_rvol($ms.M̂.corner_sinds_proc,$ms.M̂.xyz,$vxl_min,$vxl_max,$shapes2)

corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc)

S_rvol(shapes;ms)
@btime S_rvol($shapes;ms=$ms)
@btime S_rvol(shapes2;ms)

const εᵥ = SMatrix{3,3}(1.,0.,0.,0.,1.,0.,0.,0.,1.)
fεs = map(m->fε(m)[1],mats)
λs = 0.5:0.1:1.6
ωs = 1 ./ λs
εs = [vcat([SMatrix{3,3}(fep(lm)) for fep in fεs],[εᵥ,]) for lm in λs]
minds= matinds(shapes2)

epsm = εₛ(εs[1],ms.M̂.corner_sinds_proc,minds,sr1)
@btime εₛ($εs[1],$ms.M̂.corner_sinds_proc,$minds,$sr1)

εₛ11 = [ee[1,1] for ee in epsm][:,:,1]
εₛ22 = [ee[2,2] for ee in epsm][:,:,1]
εₛ12 = [ee[1,2] for ee in epsm][:,:,1]



geom = Geometry(shapes2)

##
# check that materials/shapes lists and cross-reference index lists work by adding a few shapes
bx1 = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
				[0. , 0.1],            	# c: center
				[2.8, 0.4 ],	# r: "radii" (half span of each axis)
				SMatrix{2,2}(1.,0.,0.,1.),	    		        # axes: box axes
				MgO_LiNbO₃,					        # data: any type, data associated with box shape
			)
bx2 = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
				[-0.5 , 0.4],            	# c: center
				[0.8, 0.2 ],	# r: "radii" (half span of each axis)
				SMatrix{2,2}(1.,0.,0.,1.),	    		        # axes: box axes
				SiO₂,					        # data: any type, data associated with box shape
			)
bx3 = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
				[0.5 , 0.4],            	# c: center
				[0.2, 0.2 ],	# r: "radii" (half span of each axis)
				SMatrix{2,2}(1.,0.,0.,1.),	    		        # axes: box axes
				Si₃N₄,					        # data: any type, data associated with box shape
			)
shapes3 = vcat(shapes2, [bx1,bx2,bx3])
mats3 = materials(shapes3)
sinds2minds3 = map(s->findfirst(m->isequal(s.data,m), mats3),shapes3)


##
struct Material{T}
	ε::SMatrix{3,3,T,9}
end
n(mat::Material) = sqrt.(diag(mat.ε))
n(mat::Material,axind::Int) = sqrt(mat.ε[axind,axind])
ng(mat::Material) = ng_sym.(n.(mat))
ng(mat::Material,axind::Int) = ng_sym(n(mat,axind))
gvd(mat::Material) = gvd_sym.(n.(mat))
gvd(mat::Material,axind::Int) = gvd_sym(n(mat,axind))

using ModelingToolkit
pₑ_MgO_LiNbO₃ = (
    a₁ = 5.756,
    a₂ = 0.0983,
    a₃ = 0.202,
    a₄ = 189.32,
    a₅ = 12.52,
    a₆ = 1.32e-2,
    b₁ = 2.86e-6,
    b₂ = 4.7e-8,
    b₃ = 6.113e-8,
    b₄ = 1.516e-4,
    T₀ = 24.5,      # reference temperature in [Deg C]
)
pₒ_MgO_LiNbO₃ = (
    a₁ = 5.653,
    a₂ = 0.1185,
    a₃ = 0.2091,
    a₄ = 89.61,
    a₅ = 10.85,
    a₆ = 1.97e-2,
    b₁ = 7.941e-7,
    b₂ = 3.134e-8,
    b₃ = -4.641e-9,
    b₄ = -2.188e-6,
    T₀ = 24.5,      # reference temperature in [Deg C]
)
function n²_MgO_LiNbO₃_sym(λ, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    f = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    λ² = λ^2
    a₁ + b₁*f + (a₂ + b₂*f) / (λ² - (a₃ + b₃*f)^2) + (a₄ + b₄*f) / (λ² - a₅^2) - a₆*λ²
end
@variables λ, T
nₑ²_MgO_LiNbO₃_λT_sym = n²_MgO_LiNbO₃_sym(λ, T; pₑ_MgO_LiNbO₃...)
nₑ²_MgO_LiNbO₃_sym = substitute(nₑ²_MgO_LiNbO₃_λT_sym,[T=>pₑ_MgO_LiNbO₃.T₀])
nₒ²_MgO_LiNbO₃_λT_sym = n²_MgO_LiNbO₃_sym(λ, T; pₒ_MgO_LiNbO₃...)
nₒ²_MgO_LiNbO₃_sym = substitute(nₒ²_MgO_LiNbO₃_λT_sym,[T=>pₒ_MgO_LiNbO₃.T₀])
ε_MgO_LiNbO₃_λT_sym = Diagonal( [ nₑ²_MgO_LiNbO₃_λT_sym, nₒ²_MgO_LiNbO₃_λT_sym, nₒ²_MgO_LiNbO₃_λT_sym ] )
ε_MgO_LiNbO₃_sym = Diagonal( [ nₑ²_MgO_LiNbO₃_sym, nₒ²_MgO_LiNbO₃_sym, nₒ²_MgO_LiNbO₃_sym ] )
LN = Material(SMatrix{3,3}(ε_MgO_LiNbO₃_sym))

function materials(shapes::Vector{Shape{N,N²,D,T}}) where {N,N²,D,T}
	unique!(getfield.(shapes,:data))
end

materials2(shapes) = unique!(getfield.(shapes,:data))

struct Geometry3{N,N²,D,T}
	shapes::Vector{Shape{N,N²,D,T}}
end

wg1 = Geometry3(rwg(p))
shapes3 = vcat(shapes,shapes,shapes,shapes,shapes)
wg3 = Geometry3(shapes3)
##
xy = [ SVector(ms.M̂.x[i],ms.M̂.y[j]) for i=1:Ny,j=1:Nx ]
xyc = [SVector{2}(ms.M̂.xc[i],ms.M̂.yc[j]) for i=1:(Nx+1),j=1:(Ny+1)]

update_corner_sinds!(ms.M̂.corner_sinds,shapes,xy,xyc)
@btime update_corner_sinds!($ms.M̂.corner_sinds,$shapes,$xy,$xyc)

update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc)
@btime update_corner_sinds!($ms.M̂.corner_sinds,$shapes,$ms.M̂.xyz,$ms.M̂.xyzc)

proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc)
@btime proc_corner_sinds!($ms.M̂.corner_sinds,$ms.M̂.corner_sinds_proc)


function τ_trans(ε::AbstractMatrix{T}) where T<:Real
    return @inbounds SMatrix{3,3,T,9}(
        -1/ε[1,1],      ε[2,1]/ε[1,1],                  ε[3,1]/ε[1,1],
        ε[1,2]/ε[1,1],  ε[2,2] - ε[2,1]*ε[1,2]/ε[1,1],  ε[3,2] - ε[3,1]*ε[1,2]/ε[1,1],
        ε[1,3]/ε[1,1],  ε[2,3] - ε[2,1]*ε[1,3]/ε[1,1],  ε[3,3] - ε[3,1]*ε[1,3]/ε[1,1]
    )
end

function τ⁻¹_trans(τ::AbstractMatrix{T}) where T<:Real
    return @inbounds SMatrix{3,3,T,9}(
        -1/τ[1,1],          -τ[2,1]/τ[1,1],                 -τ[3,1]/τ[1,1],
        -τ[1,2]/τ[1,1],     τ[2,2] - τ[2,1]*τ[1,2]/τ[1,1],  τ[3,2] - τ[3,1]*τ[1,2]/τ[1,1],
        -τ[1,3]/τ[1,1],     τ[2,3] - τ[2,1]*τ[1,3]/τ[1,1],  τ[3,3]- τ[3,1]*τ[1,3]/τ[1,1]
    )
end

function avg_param(ε_fg, ε_bg, n12, rvol1)
	n = n12 / norm(n12)
	# n = normalize(n12) #n12 / norm(n12) #sqrt(sum2(abs2,n12))
    # Pick a vector that is not along n.
    h = any(iszero.(n)) ? n × normalize(iszero.(n)) :  n × SVector(1., 0. , 0.)
	v = n × h
    # Create a local Cartesian coordinate system.
    S = [n h v]  # unitary
    τ1 = τ_trans(transpose(S) * ε_fg * S)  # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans(transpose(S) * ε_bg * S)  # express param2 in S coordinates, and apply τ transform
    τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)  # volume-weighted average
    return SMatrix{3,3}(S * τ⁻¹_trans(τavg) * transpose(S))  # apply τ⁻¹ and transform back to global coordinates
end

# alleq(itr) = length(itr)==0 || all( ==(itr[1]), itr)
get_ε(shapes,ind) = ind>lastindex(shapes) ? SMatrix{3,3}(1.,0.,0.,0.,1.,0.,0.,0.,1.) : shapes[ind].data
V3(v) = isequal(length(v),3) ? v : vcat(v,zeros(3-length(v)))

function n_rvol(shape,xyz,vxl_min,vxl_max)
	r₀,n⃗ = surfpt_nearby(xyz, shape)
	rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
	return V3(n⃗),rvol
end

function _smooth(shapes,sinds_proc,xyz,vxl_min,vxl_max)
	iszero(sinds_proc[2]) && return get_ε(shapes,sinds_proc[1])
	iszero(sinds_proc[3]) && return avg_param(	shapes[sinds_proc[1]].data,
												get_ε(shapes,sinds_proc[2]),
												n_rvol(shapes[sinds_proc[1]],xyz,vxl_min,vxl_max)...
												)
	return mapreduce(i->get_ε(shapes,i),+,sinds_proc) / 8
end

function smooth(shapes,sinds_proc,xyz,xyzc)
	vxl_min = @view xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
	vxl_max = @view xyzc[min(2,end):end,min(2,end):end,min(2,end):end]
	f(sp,x,vn,vp) = let s=shapes
		_smooth(s,sp,x,vn,vp)
	end
	map(f,sinds_proc,xyz,vxl_min,vxl_max)
end

# smooth(shapes,ms::ModeSolver) = smooth(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)

function smooth(shapes;ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)
	HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()}}(
		reshape(
			reinterpret(
				reshape,
				Float64,
				smooth(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc),
				),
			(3,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz),
		)
	)
end

function smoothinv(shapes,sinds_proc,xyz,xyzc)
	vxl_min = @view xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
	vxl_max = @view xyzc[min(2,end):end,min(2,end):end,min(2,end):end]
	f(sp,x,vn,vp) = let s=shapes
		inv(_smooth(s,sp,x,vn,vp))
	end
	map(f,sinds_proc,xyz,vxl_min,vxl_max)
end

function smoothinv(shapes;ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)
	HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()}}(
		reshape(
			reinterpret(
				reshape,
				Float64,
				smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc),
				),
			(3,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz),
		)
	)
end

using StaticArrays: Dynamic
function epsi(shapes::Vector{<:Shape};ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	vxl_min = @view ms.M̂.xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
	vxl_max = @view ms.M̂.xyzc[min(2,end):end,min(2,end):end,min(2,end):end]
	f(sp,x,vn,vp) = let s=shapes
		inv(_smooth(s,sp,x,vn,vp))
	end
	eibuf = Zygote.Buffer(bounds(shapes[1])[1],3,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)
	# eibuf = Buffer(bounds(shapes[1])[1],3,3,Nx,Ny,Nz)
    for ix=1:ms.M̂.Nx,iy=1:ms.M̂.Ny,iz=1:ms.M̂.Nz
		# eps = εₛ(shapes,Zygote.dropgrad(tree),Zygote.dropgrad(g.x[i]),Zygote.dropgrad(g.y[j]),Zygote.dropgrad(g.δx),Zygote.dropgrad(g.δy))
		# eps = εₛ(shapes,x[i],y[j];tree,δx,δy)
		# epsi = inv(eps) # inv( (eps' + eps) / 2) # Hermitian(inv(eps))  # inv(Hermitian(eps)) #   # inv(eps)
        # eibuf[:,:,i,j,kk] = epsi #(epsi' + epsi) / 2
		eibuf[:,:,ix,iy,iz] = f(ms.M̂.corner_sinds_proc[ix,iy,iz],ms.M̂.xyz[ix,iy,iz],vxl_min[ix,iy,iz],vxl_max[ix,iy,iz])
    end
    # return HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},T,5,5,Array{T,5}}( real(copy(eibuf)) )
	return HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()}}( real(copy(eibuf)) )
end

function epsi2(shapes::Vector{<:Shape};ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)
end

function epsi3(shapes::Vector{<:Shape};ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	reshape(
		reinterpret(
			reshape,
			Float64,
			smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc),
			),
		(3,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz),
	)
end

##
epsi(shapes;ms)
@btime epsi($shapes;ms=$ms) #64.106 ms (613646 allocations: 27.62 MiB)

epsi2(shapes;ms)
@btime epsi2($shapes;ms=$ms) # 2.991 ms (105718 allocations: 6.86 MiB)

epsi3(shapes;ms)
@btime epsi3($shapes;ms=$ms) # 3.246 ms (105721 allocations: 6.86 MiB)

smooth(shapes,ms.M̂.corner_sinds_proc,xy,xyc)
@btime smooth($shapes,$ms.M̂.corner_sinds_proc,$xy,$xyc)

smooth(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)
@btime smooth($shapes,$ms.M̂.corner_sinds_proc,$ms.M̂.xyz,$ms.M̂.xyzc)

smooth(shapes,ms)
@btime smooth($shapes,$ms)

es = smooth(shapes,ms)

# smooth2(shapes,ms)
# @btime smooth2($shapes,$ms)

# Compare with old smoothing function
tree = tree(shapes)
es_old = [SMatrix{3,3}(εₛ(shapes,ms.M̂.x[xind],ms.M̂.y[yind];tree,δx=ms.M̂.δx,δy=ms.M̂.δy)) for xind=1:Nx,yind=1:Ny]
@assert all(es_old .≈ es[:,:,1])
@btime [SMatrix{3,3}(εₛ($shapes,$ms.M̂.x[xind],$ms.M̂.y[yind];tree,δx=ms.M̂.δx,δy=ms.M̂.δy)) for xind=1:Nx,yind=1:Ny]
# 296.386 ms (1724616 allocations: 75.20 MiB)

using ChainRules, Zygote, ForwardDiff, FiniteDifferences
f1(x) = sum(sum(smooth(rwg(x);ms)))/(128*128)
f1(p)
f2(x) = sum(sum(smoothinv(rwg(x);ms)))/(128*128)
f2(p)
Zygote.gradient(f1,p)[1]
ForwardDiff.gradient(f1,p)
FiniteDifferences.grad(central_fdm(3,1),f1,p)

println("######  btimes for f1, using regular map:")
println("f1:")
@btime f1($p)
println("FowardDiff:")
@btime ForwardDiff.gradient($f1,$p)
println("Zygote:")
@btime FiniteDifferences.grad(central_fdm(3,1),$f1,$p)
println("FiniteDifferences:")
@btime Zygote.gradient($f1,$p)[1]

println("######  btimes for f2, using pmap:")
println("f2:")
@btime f2($p)
println("FowardDiff:")
@btime ForwardDiff.gradient($f2,$p)
println("Zygote:")
@btime FiniteDifferences.grad(central_fdm(3,1),$f2,$p)
println("FiniteDifferences:")
@btime Zygote.gradient($f2,$p)[1]
# ######  btimes for f1, using regular map:
# FowardDiff:
# 8.280 ms (57116 allocations: 14.59 MiB)
# Zygote:
# 67.081 ms (1147885 allocations: 147.68 MiB)
# FiniteDifferences:
# 678.245 ms (3484021 allocations: 171.39 MiB)
# ######  btimes for f2, using pmap:
# FowardDiff:
# 8.781 ms (57116 allocations: 14.59 MiB)
# Zygote:
# 69.119 ms (1147885 allocations: 147.68 MiB)
# FiniteDifferences:
# 630.052 ms (3484021 allocations: 171.39 MiB)

##
vxl_min2 = @view xyc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
vxl_max2 = @view xyc[min(2,end):end,min(2,end):end,min(2,end):end]
fsm(sp,x,vn,vp) = let s=shapes
	_smooth(s,sp,x,vn,vp)
end
##
# corner condition where simple averaging should occur: CartesianIndex(44, 54)

I = CartesianIndex(44,54,1)
_smooth(shapes,ms.M̂.corner_sinds_proc[I],xy[I],vxl_min2[I],vxl_max2[I])
_smooth(shapes,ms.M̂.corner_sinds_proc[I],ms.M̂.xyz[I],vxl_min2[I],vxl_max2[I])

##

function avg_param2(xy,sinds)
        r₀,nout = surfpt_nearby(xy, shapes[sinds[0]])

end

shapes = rwg(p)
tree = KDTree(shapes)
n_shapes = length(shapes)

# gridpoint positions
x = ( ( Δx / Nx ) .* (0:(Nx-1))) .- Δx/2.
y = ( ( Δy / Ny ) .* (0:(Ny-1))) .- Δy/2.
xy = [ SVector(x[i],y[j]) for i=1:Ny,j=1:Nx ]
# corner positions
xc = ( ( Δx / Nx ) .* (0:Nx) ) .- ( Δx/2. * ( 1 + 1. / Nx ) )
yc = ( ( Δy / Ny ) .* (0:Ny) ) .- ( Δy/2. * ( 1 + 1. / Ny ) )
xyc = [SVector{2}(xc[i],yc[j]) for i=1:(Nx+1),j=1:(Ny+1)]
sc = Array{Int}(undef,size(xyc))
sc_ext = Array{NTuple{4,Int}}(undef,size(xy))
sc .= [(a = findfirst(isequal(findfirst(SVector(xyc[i,j]),tree)),shapes); isnothing(a) ? (n_shapes+1) : a ) for i=1:(Nx+1),j=(1:Ny+1)]
sc_ext .= [ (unq = unique!( [sc[i,j], sc[1+1,j], sc[i+1,j+1], sc[i,j+1]] ); n_unq=length(unq); n_unq==1 ? (unq[1],0,0,0) : ( n_unq==2 ?  (minimum(unq),maximum(unq),0,0)  : ( sc[i,j],  sc[i+1,j],  sc[i+1,j+1],  sc[i,j+1] ) ) )  for i=1:Nx,j=1:Ny ]

sc

128 * 128



##
# gridpoint positions
x = ( ( Δx / Nx ) .* (0:(Nx-1))) .- Δx/2.
y = ( ( Δy / Ny ) .* (0:(Ny-1))) .- Δy/2.
z = ( ( Δz / Nz ) .* (0:(Nz-1))) .- Δz/2.
xy = [ SVector(x[i],y[j]) for i=1:Ny,j=1:Nx ]
xyz = [ SVector{3}(x[i],y[j],z[k]) for i=1:Ny,j=1:Nx,k=1:Nz ]
# corner positions
xc = ( ( Δx / Nx ) .* (0:Nx) ) .- ( Δx/2. * ( 1 + 1. / Nx ) )
yc = ( ( Δy / Ny ) .* (0:Ny) ) .- ( Δy/2. * ( 1 + 1. / Ny ) )
zc = ( ( Δz / Nz ) .* (0:Nz) ) .- ( Δz/2. * ( 1 + 1. / Nz ) )
xyc = [SVector{2}(xc[i],yc[j]) for i=1:(Nx+1),j=1:(Ny+1)]
xyzc = [SVector{3}(xc[i],yc[j],zc[k]) for i=1:(Nx+1),j=1:(Ny+1),k=1:(Nz+1)]
# arrays for shape index data

corner_sinds2 = zeros(Int, Nx+1,Ny+1)
corner_sinds_proc2 = fill((0,0,0,0), Nx,Ny)
corner_sinds3 = zeros(Int, Nx+1,Ny+1,Nz+1)
corner_sinds_proc3 = fill((0,0,0,0,0,0,0,0), Nx,Ny,Nz)

# update_corner_sinds!(corner_sinds2,corner_sinds_proc2,shapes,xy,xyc)
# @btime update_corner_sinds!($corner_sinds2,$corner_sinds_proc2,$shapes,$xy,$xyc)

update_corner_sinds4!(corner_sinds2,corner_sinds_proc2,shapes,xy,xyc)
@btime update_corner_sinds4!($corner_sinds2,$corner_sinds_proc2,$shapes,$xy,$xyc)

##
function update_corner_sinds!(corner_sinds,corner_sinds_proc,shapes::AbstractVector{<:GeometryPrimitives.Shape{2}},xyz,xyzc)
	n_shapes = length(shapes)
	tree = KDTree(shapes)
	a = 0
	unq = [0,0]
	for I ∈ eachindex(xyzc)
		a = findfirst(isequal(findfirst(xyzc[I],tree)),shapes)
		corner_sinds[I] = isnothing(a) ? (n_shapes+1) : a
	end
	for I ∈ CartesianIndices(xyz)
		unq = [	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0)],
					corner_sinds[I+CartesianIndex(0,1)],
					corner_sinds[I+CartesianIndex(1,1)],
		  		]
		# unq = unique!( unq )
		unique!( unq )
		a = length(unq)
		corner_sinds_proc[I] = a==1 ? (unq[1],0,0,0) :
			( a==2 ?  (minimum(unq),maximum(unq),0,0)  :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0)],
					corner_sinds[I+CartesianIndex(0,1)],
					corner_sinds[I+CartesianIndex(1,1)],
				)
		)
	end
end

in3(x::SVector{2,<:Real}, s::GeometryPrimitives.Polygon) = all(sum(s.n .* (x' .- s.v), dims=Val(2)) .≤ 0)

function in3(x::SVector{N,<:Real}, b::GeometryPrimitives.Box{N}) where {N}
    d = b.p * (x - b.c)
    for i = 1:N
        abs(d[i]) > b.r[i] && return false  # boundary is considered inside
    end
    return true
end

function f_cinds2(shapes::Vector{S}) where {S<:Shape{N}} where N #::Int
	function f_cinds2_inner(p)
		let s=shapes # pairs(shapes)
			# x-> something(ff4(x,s),lp1) #::Int
			@inbounds for (i, a) in pairs(s)
				in3(p,a) && return i
			end
			return lastindex(s)+1
		end
	end
end

function update_corner_sinds4!(corner_sinds,corner_sinds_proc,shapes,xyz,xyzc)
	# a = 0
	unq = [0,0]
	function f_cinds(p::SVector{N,T}) where {N,T} #::Int
		let s=shapes # pairs(shapes)
			# x-> something(ff4(x,s),lp1) #::Int
			for (i, a) in pairs(s)
		        in3(p,a) && return i
				# true && return i
		    end
		    return lastindex(s)+1
		end
	end
	# corner_sinds .= f_cinds.(xyzc)
	# map!(corner_sinds,xyzc) do p
	# 	let s=shapes
	# 		for (i, a) in pairs(s)
	# 			in3(p,a) && return i
	# 		end
	# 		return lastindex(s)+1
	# 	end
	# end
	@inbounds for I in eachindex(xyzc)
		corner_sinds[I] = f_cinds(xyzc[I]) #::Int
	end
	for I ∈ CartesianIndices(xyz)
	 	unq = [		corner_sinds[I],
								corner_sinds[I+CartesianIndex(1,0)],
								corner_sinds[I+CartesianIndex(0,1)],
								corner_sinds[I+CartesianIndex(1,1)]
			  				]
		# unq = unique!( unq )
		unique!( unq )
		a = length(unq)
		println("f0")
		corner_sinds_proc[I] = isone(a) ? (unq[1],0,0,0) :
			( a===2 ?  (minimum(unq),maximum(unq),0,0)  :
				( corner_sinds[I],
							corner_sinds[I+CartesianIndex(1,0)],
							corner_sinds[I+CartesianIndex(0,1)],
							corner_sinds[I+CartesianIndex(1,1)]
						)
		)
	end
end




function update_corner_sinds!(corner_sinds,corner_sinds_proc,shapes::AbstractVector{<:GeometryPrimitives.Shape{3}},xyz,xyzc)
	n_shapes = length(shapes)
	tree = KDTree(shapes)
	a = 0
	unq = [0,0]
	for I ∈ eachindex(xyzc)
		a = findfirst(isequal(findfirst(xyzc[I],tree)),shapes)
		corner_sinds[I] = isnothing(a) ? (n_shapes+1) : a
	end
	for I ∈ eachindex(xyz)
		unq .= [	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0,0)],
					corner_sinds[I+CartesianIndex(0,1,0)],
					corner_sinds[I+CartesianIndex(1,1,0)],
					corner_sinds[I+CartesianIndex(0,0,1)],
					corner_sinds[I+CartesianIndex(1,0,1)],
					corner_sinds[I+CartesianIndex(0,1,1)],
					corner_sinds[I+CartesianIndex(1,1,1)],
		  		]
		unq = unique!( unq )
		a = length(unq)
		corner_sinds_proc[I] = a==1 ? (unq[1],0,0,0,0,0,0,0) :
			( a==2 ?  (minimum(unq),maximum(unq),0,0,0,0,0,0)  :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0,0)],
					corner_sinds[I+CartesianIndex(0,1,0)],
					corner_sinds[I+CartesianIndex(1,1,0)],
					corner_sinds[I+CartesianIndex(0,0,1)],
					corner_sinds[I+CartesianIndex(1,0,1)],
					corner_sinds[I+CartesianIndex(0,1,1)],
					corner_sinds[I+CartesianIndex(1,1,1)],
				)
		)
	end
end

function ff2(p::SVector{N}, s::Vector{S}) where {N,S<:Shape{N,N²,D,T}} where {N²,D,T<:Real}
    # for i in eachindex(s)
    #     b::Tuple{SVector{2,T}, SVector{2,T}} = bounds(s[i])
    #     # if all(b[1] .< p .< b[2]) && p ∈ s[i]  # check if p is within bounding box is faster
	# 	if in(p, s[i])  # check if p is within bounding box is faster
    #         return s[i]
    #     end
    # end
	for ss in s
        # b::Tuple{SVector{2,T}, SVector{2,T}} = bounds(ss)
        # if all(b[1] .< p .< b[2]) && p ∈ s[i]  # check if p is within bounding box is faster
		if in(p,ss)  # check if p is within bounding box is faster
            return ss
        end
    end
	# return s[1]
	return nothing
end

function ff3(p::SVector{N}, s::Vector{S}) where {N,S<:Shape{N,N²,D,T}} where {N²,D,T<:Real}
	pin = let p = p
		x->in(p,x)
	end
	findfirst(pin,s)
	# findfirst(x->in(p,x),s)
end

ff4(p,s) = let p=p
	findfirst(x->in(p,x),s)
end



ff5(p::SVector{2,T},s) where T<:Real = findfirst(x->in(p,x),s)

function ff6(p,s)
	let p=p
		y = findfirst(x->in(p,x),s)
		!isnothing(y) ? y : length(s)+1
	end
end

function ff2(p::SVector{N}, kd::KDTree{N}) where {N}
    if isempty(kd.s)
        if p[kd.ix] ≤ kd.x
            return ff2(p, kd.left)
        else
            return ff2(p, kd.right)
        end
    else
        return ff2(p, kd.s)
    end
end

function ff2(p::SVector{N}, s::Vector{S}, sbg::S) where {N,S<:Shape{N}}
    @inbounds for i in eachindex(s)
        @inbounds b::Tuple{SVector{2}, SVector{2}} = bounds(s[i])
        @inbounds if all(b[1] .< p .< b[2]) && p ∈ s[i]  # check if p is within bounding box is faster
            @inbounds return s[i]
        end
    end
    return sbg
end

function ff2(p::SVector{N}, kd::KDTree{N}, sbg::Shape{N}) where {N}
    @inbounds if isempty(kd.s)
        @inbounds if p[kd.ix] ≤ kd.x
            return ff2(p, kd.left, sbg)
        else
            return ff2(p, kd.right, sbg)
        end
    else
        return ff2(p, kd.s, sbg)
    end
end
