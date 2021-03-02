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
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
rwg(x) = ridge_wg(x[1],x[2],x[3],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy) # fourth param is gap between modeled object and sidewalls
geom = rwg(p)
ms = ModeSolver(1.45, geom, gr)

## prototype backprop through NLsolve of PDE
# from https://github.com/JuliaNLSolvers/NLsolve.jl/issues/205

using NLsolve
using Zygote
using Zygote: @adjoint
using IterativeSolvers
using LinearMaps
using SparseArrays
using LinearAlgebra
using BenchmarkTools

# nlsolve maps f to the solution x of f(x) = 0
# We have ∂x = -(df/dx)^-1 ∂f, and so the adjoint is df = -(df/dx)^-T dx
@adjoint nlsolve(f, x0; kwargs...) =
    let result = nlsolve(f, x0; kwargs...)
        result, function(vresult)
            dx = vresult[].zero
            x = result.zero
            _, back_x = Zygote.pullback(f, x)

            JT(df) = back_x(df)[1]
            # solve JT*df = -dx
            L = LinearMap(JT, length(x0))
            df = gmres(L,-dx)

            _, back_f = Zygote.pullback(f -> f(x), f)
            return (back_f(df)[1], nothing, nothing)
        end
    end

const NN = 10000
const nonlin = 0.1
const A = spdiagm(0 => fill(10.0, NN), 1 => fill(-1.0, NN-1), -1 => fill(-1.0, NN-1))
const p0 = randn(NN)
f(x, p) = A*x + nonlin*x.^2 - p
solve_x(p) = nlsolve(x -> f(x, p), zeros(NN), method=:anderson, m=10).zero
obj(p) = sum(solve_x(p))

Zygote.refresh()
g_auto, = gradient(obj, p0)
g_analytic = gmres((A + Diagonal(2*nonlin*solve_x(p0)))', ones(NN))
display(g_auto)
display(g_analytic)
@show sum(abs.(g_auto - g_analytic))

@btime gradient(obj, p0);
@btime gmres((A + Diagonal(2*nonlin*solve_x(p0)))', ones(NN));

##
f(x, p) = A*x + nonlin*x.^2 - p
g(x) = f(x,p0)
x0 = solve_x(p0)
_, back_x = Zygote.pullback(g, x0)
_, back_f = Zygote.pullback(ff -> ff(x0), g)
_, back_f2 = Zygote.pullback(pp -> f(x0,pp), p0)
dx = ones(length(x0))
JT(df) = back_x(df)[1]
# solve JT*df = -dx
L = LinearMap(JT, length(x0))
df = gmres(L,-dx)
back_f(df)[1]
back_f2(df)[1]

## Try to rewrite _solve_Δω² & solve_k for use with nlsolve instead of roots
# and check if this adjoint still works
using Zygote: dropgrad

# f(k,ωₜ,ε⁻¹) = solve_ω²(dropgrad(ms),k,ε⁻¹)[1] - ωₜ^2


function findk(ms::ModeSolver{ND},om::Real,epsi::AbstractArray{<:SMatrix{3,3},ND},k₀::Real) where ND
    f(k,ωₜ,ε⁻¹) = solve_ω²(dropgrad(ms),k,ε⁻¹)[1] - ωₜ^2
    res = nlsolve(x->f(x[1],om,epsi),[k₀,])
    k = res.zero[1]
end

function findk(ms::ModeSolver{ND},om::Real,geom::AbstractVector{S},k₀::Real) where {ND,S<:Shape}
    epsi = εₛ⁻¹(om,geom;ms)
    f(k,ωₜ,ε⁻¹) = real(solve_ω²(dropgrad(ms),real(k),real.(ε⁻¹))[1]) - real(ωₜ)^2
    res = nlsolve(x->real(f(real(x[1]),real(om),real.(epsi))),[k₀,])
    k = real(res.zero[1])
end


ei = ms.M̂.ε⁻¹

findk(ms,1/1.55,ei,1.46)
findk(ms,1/1.58,ei,1.26)

findk(ms,1/1.61,rwg(p),1.23)
findk(ms,1/1.64,rwg(p),1.20)

Zygote.gradient((om,pp)->findk(ms,om,rwg(pp),1.20), 1/1.64,p)

##
om = 1/ 1.55
k₀ = 1.2
geom= rwg(p)
epsi = εₛ⁻¹(om,geom;ms)
f1(k,ωₜ,ε⁻¹) = real(solve_ω²(dropgrad(ms),real(k),real.(ε⁻¹))[1]) - real(ωₜ)^2
f2(x) = real(f1(real(x[1]),real(om),real.(epsi)))
res = nlsolve(f2,[k₀,])

dk = SMatrix{1,1}(1.0) #SVector{1}(1.0) #1.0
k = res.zero #[1]
f2_val, back_k = Zygote.pullback(f2, k)
JT(df) = back_k(df)[1]
L = SMatrix{1,1}(ms.∂ω²∂k[1]) # LinearMap(JT, 1)
df = gmres(L,-dk) # -1/ms.∂ω²∂k[1]
_, back_f = Zygote.pullback(ff->ff(k),f2)
back_f(df)[1]

f2(k)
Zygote.gradient(ff->ff(k),f2)

f1(1.2,1/1.55,epsi)
Zygote.gradient(f1,1.2,1/1.55,epsi)
f3(k,ωₜ,p) = real(solve_ω²(dropgrad(ms),real(k),εₛ⁻¹(ωₜ,rwg(p);ms))[1]) - real(ωₜ)^2
f3(1.2,1/1.55,p)
Zygote.gradient(f3,1.2,1/1.55,p)
f3(1.26,1/1.55,p)
Zygote.gradient(f3,1.26,1/1.55,p)
f4(x) = f3(x[1],1/1.55,p)
f4([1.26,])

res2 = nlsolve(f4,[k₀,])
k2 = res2.zero
f4_val, back_k2 = Zygote.pullback(f4, k2)
JT2(df) = back_k2(df)[1]
JT2([1.0])

JT([1.0+0.1im,])

Zygote.gradient(f4,k2)


ms.∂ω²∂k[1]
ms.∂ω²∂k
domsq,domsq_domsqp = _solve_Δω²(ms,k2[1],1/1.55)

Zygote.gradient((om,pp)->f3(1.26,om,pp),1/1.55,p)
_, solve_domsq_pb = Zygote.pullback((om,pp)->real(solve_ω²(dropgrad(ms),real(k[1]),εₛ⁻¹(om,rwg(pp);ms))[1]) - real(om)^2, 1/1.55, p)
solve_domsq_pb(-1/ms.∂ω²∂k[1])




FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,1/1.55,rwg(x))[1],p)[1]




FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,x,rwg(p))[1],1/1.55)[1]


_, solve_domsq_pb = Zygote.pullback(1/1.55, p) do om, pp
    omsq,HH = solve_ω²(dropgrad(ms),real(k[1]),εₛ⁻¹(om,rwg(pp);ms))
    domsq = real(omsq) - real(om)^2
end

Zygote.gradient(ff->ff(1.26),f4)

omsqH, omsqH_pb = Zygote.pullback(k[1], epsi) do kk, einv
    omsq,HH = solve_ω²(dropgrad(ms),real(kk),einv)
end

H̄₁ = randn(ComplexF64,length(omsqH[2]))

k̄₂,eī₁ = omsqH_pb( (0.0, H̄₁) )

ω²errH, ω²errH_pb = Zygote.pullback(k[1], 1/1.55, epsi) do kk, om_in, einv
    omsq_out,HH = solve_ω²(dropgrad(ms),real(kk),einv)
    omsq_err = real(omsq_out) - real(om_in)^2
    return (omsq_err, HH)
end

k̄₂2,ω̄₁ , eī₁2 = ω²errH_pb( (0.0, H̄₁) )
k̄₂3, ω̄₁2 , eī₁3 = ω²errH_pb( (-1 / ms.∂ω²∂k[1], nothing) )

k̄₂2 ≈ k̄₂
eī₁2 ≈ eī₁

k,H = solve_k(ms,0.8,epsi)
ω²err, ω²err_pb = Zygote.pullback(0.8, epsi) do om, einv
    omsq,HH = solve_ω²(dropgrad(ms),real(k),einv)
    omsq_err = real(omsq) - real(om)^2
end

ω²err_pb(1.0)

ω̄ , eīₖ = ω²err_pb( -1 / ms.∂ω²∂k[1]  )

ω̄  ≈ ω̄₁2
eīₖ ≈ eī₁3

solve_k
