using LinearAlgebra, StaticArrays, IterativeSolvers, ChainRules, Plots
using FiniteDifferences, ForwardDiff, Zygote # ReverseDiff



#include("eigen_grad.jl")


##

a = rand(3,3)
A = Hermitian(a * a')
typeof(A) <: LinearAlgebra.RealHermSymComplexHerm


##

function f(p)
    a = [      1 + 0.1*sin(p[1])                 0.05 
               -0.05                       2 + 0.1*cos(p[1])^2        ]
    A = Hermitian(a * a')
    α, X⃗ = eigen(A)
    return α[1]*sum(abs2.(X⃗))
end

x = 0:0.05:3π
p = [ [xx 3.0 ] for xx in x ]

# @show val = f.(p)
# @show jac = [ ReverseDiff.jacobian(f, pp) for pp in p]


##
xs = 0:0.05:3π
grads = [ Zygote.gradient(x->f(x),xx)[1] for xx in xs]

plot(xs,grads)
##

v1 = [v[1] for v in val]
j11 = [j[1,1] for j in jac]

# v2 = [v[2] for v in val]
# j22 = [j[2,2] for j in jac]

l1 = plot(x,v1,label="v1")
plot!(l1,x,j11,label="j11")
# plot!(l1,x,v2,label="v2")
# plot!(l1,x,j22,label="j22")

##

function f(a)
    A = a * a'
    α, X⃗ = eigen(A)
    return α
end

inputs = rand(3,3)
@show jac = ReverseDiff.jacobian(f, inputs);
@show typeof(jac);
@show size(jac[1]);
@show size(jac[2]);

##
W = rand(2, 3); x = rand(3);
Zygote.gradient(W -> sum(W*x), W)[1]

##

function g(a)
    A = a * a'
    α, X⃗ = eigen(A)
    return sum(α)
end

x = rand(3,3)
Zygote.gradient(x -> g(x), x)[1]


## Functions

function foo(p::AbstractArray)
    A0 = Matrix(reshape(p,(3,3)))
    A = A0 * A0'
    # A = SHermitianCompact{10,Float64,sum(1:10)}(p)
    α, X⃗ = eigen(A)
    return sum(X⃗) + sum(α) + sum(p)
end

function goo(p::AbstractArray)
    A0 = Matrix(reshape(p,(3,3)))
    A = A0 * A0'
    # A = SHermitianCompact{10,Float64,sum(1:10)}(p)
    α, X⃗ = eigen(A)
    return α
end



p = randn(9)
foo(p)
goo(p)

## Finite Difference Gradients

# for p = randn(N²)
# tested with N = 3

#### Find dfoo/dx via FiniteDifferences.jl
using FiniteDifferences
@show FiniteDifferences.grad(central_fdm(3,1),foo,p)        # tuple w/ size=(N²,) Array of gradient of foo w.r.t. p⃗ components
@show FiniteDifferences.jacobian(central_fdm(3,1),goo,p)    # tuple w/ size=(N²,) Array of gradient of foo w.r.t. p⃗ components

##

#### Find dfoo/dx via ForwardDiff.jl
using ForwardDiff
@show dfoo_fmad = ForwardDiff.gradient(foo, p)
@show dgoo_fmad = ForwardDiff.jacobian(goo, p)

##

#### Find dfoo/dx via Zygote.jl
using Zygote
Zygote.gradient(foo, p)

##

foo'(randn(100))

##

foo'(randn(Float64,sum(1:10)))


##

dx = 0.03
x = collect(-π:dx:π)
f = tan
ylim = (-10,10)


p = plot(x,f.(x);ylim,label="f")
plot!(p,x,f'.(x);ylim,label="f'")


## General function of eigenproblem results -> compare with Sec. 4 of https://math.mit.edu/~stevenj/18.336/adjoint.pdf

N = 12
T = Float64 #ComplexF64

p = randn(T,sum(1:N))
A = SHermitianCompact{N,T,sum(1:N)}(p)
A = A * A'

# p = randn(T,N,N)
# A = Matrix(randn(p)
# A = A * A'

# α, X⃗ = eigen(A)


# function g(X⃗,α,p)
#     sum(X⃗) + sum(α) + sum(p)
# end

##



