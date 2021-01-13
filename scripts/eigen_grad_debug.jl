using Revise
using LinearAlgebra, StaticArrays, ArrayInterface, FFTW, LinearMaps, IterativeSolvers, ChainRules, Tullio, Plots, BenchmarkTools
using FiniteDifferences, ForwardDiff, Zygote # ReverseDiff
using OptiMode #  DataFrames, CSV,
# using ChainRulesTestUtils, Test
#include("eigen_rules.jl")

## Test AD sensitivity analysis of random matrices using rrules defined above

function proc_eigs(Xone,αone)
    sum2(x->abs2(x)^2,Xone) #* αone^4
end

function A_from_p(p)
    N = Int(sqrt(length(p)))
    A0 = Zygote.Buffer(zeros(ComplexF64,(N,N)))
    for i = 1:N
        for j=1:i
            A0[j,i] = p[(sum(1:(i-1))+1)+j-1]
            # println("ind: $((sum(1:(i-1))+1)+j -1)")
        end
    end
    for i = 2:N
        for j=1:i-1
            A0[j,i] += p[ ((sum(1:N)) + (sum(1:(i-2))+1) + j -1) ]*im
            # println("ind: $((sum(1:N)) + (sum(1:(i-1))+1) + j -1)")
        end
    end
    A = Hermitian(copy(A0))
end

function foo(p)
    # A = Hermitian(diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:N]...) + im*diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:(N-1)]...))
    A = A_from_p(p)
    F = eigen(A)
    α = F.values
    X = F.vectors
    Xone = Zygote.@showgrad(X[:,1])
    αone = Zygote.@showgrad(α[1])
    # sum(x->abs2(x)^6,Xone) * αone^2
    proc_eigs(Xone,αone)
end

function ∂foo_SJ(p,α,X,ᾱ,X̄;i=1)
    # A = Hermitian(diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:N]...) + im*diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:(N-1)]...))
    # A = A_from_p(p)
    A, Ap_pb = Zygote.pullback(A_from_p,p)
    α,X = eigen(A)
    X̄,ᾱ = Zygote.gradient(proc_eigs,X[:,1],α[1])
    P = I - X[:,i] * X[:,i]'
    b = P * X̄ #[i]
    λ₀ = IterativeSolvers.bicgstabl(A-α[i]*I,b,3)
    if isnothing(ᾱ)
        ᾱ = 0.
    end
    λ = λ₀ - ᾱ * X[:,i]
    Ā = -λ * X[:,i]'
    Ap_pb(Ā)
end

N = 30
p = randn(Float64,N^2)
# A = Hermitian(diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:N]...) + im*diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:(N-1)]...))
# A0 = zeros(ComplexF64,(N,N))
# for i = 1:N
#     for j=1:i
#         A0[j,i] = p[(sum(1:(i-1))+1)+j-1]
#         # println("ind: $((sum(1:(i-1))+1)+j -1)")
#     end
# end
# for i = 2:N
#     for j=1:i-1
#         A0[j,i] += p[ ((sum(1:N)) + (sum(1:(i-2))+1) + j -1) ]*im
#         # println("ind: $((sum(1:N)) + (sum(1:(i-1))+1) + j -1)")
#     end
# end
A = A_from_p(p)
α,X = eigen(A)
foo(p)
n_FD = 2
# plot(α,label="eigvals(A)",legend=:bottomright);scatter!(α,label=nothing)
proc_eigs(X[:,1],α[1])
foo(p)
X̄,ᾱ = Zygote.gradient(proc_eigs,X[:,1],α[1])
p̄_AD = real.(Zygote.gradient(foo,p)[1])
p̄_FD = FiniteDifferences.grad(central_fdm(n_FD, 1),foo,p)[1]
p̄_SJ = real.(∂foo_SJ(p,α,X,ᾱ,X̄)[1])
using Plots: plot, plot!, scatter, scatter!
pp = plot([-maximum(abs.(p̄_AD)),maximum(abs.(p̄_AD))],[-maximum(abs.(p̄_AD)),maximum(abs.(p̄_AD))],c=:black,label="y=x",legend=:bottomright)
scatter!(p̄_AD,p̄_FD,label="AD/FD")
scatter!(p̄_AD,p̄_SJ,label="AD/SJ")
# Āᵢⱼ_ADr = vec(real.(Ā_AD))
# Āᵢⱼ_ADi = vec(imag.(Ā_AD))
# Āᵢⱼ_FDr = vec(real.(Ā_FD))
# Āᵢⱼ_FDi = vec(imag.(Ā_FD))
# Āᵢⱼ_SJr = vec(real.(Ā_SJ))
# Āᵢⱼ_SJi = vec(imag.(Ā_SJ))
#
# using Plots: plot, plot!, scatter, scatter!
# p = plot([-1,1],[-1,1],c=:black,label="y=x",legend=:bottomright)
# scatter!(Āᵢⱼ_ADr,Āᵢⱼ_FDr,label="AD/FD_r")
# scatter!(Āᵢⱼ_ADi,Āᵢⱼ_FDi,label="AD/FD_i")
# scatter!(Āᵢⱼ_ADr,Āᵢⱼ_SJr,label="SJ/AD_r")
# scatter!(Āᵢⱼ_ADi,Āᵢⱼ_SJi,label="SJ/AD_i")

# [ ( A*X⃗[:,i] - α[i] * X⃗[:,i]) for i = 1:N]

## Now test eigen rrule fns with Helmholtz Operator matrices
"""
Default design parameters for ridge waveguide. Both MPB and OptiMode functions
should intake data in this format for convenient apples-to-apples comparison.
"""
p0 = [
    1.45,               #   propagation constant    `kz`            [μm⁻¹]
    1.7,                #   top ridge width         `w_top`         [μm]
    0.7,                #   ridge thickness         `t_core`        [μm]
    π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
    2.4,                #   core index              `n_core`        [1]
    1.4,                #   substrate index         `n_subs`        [1]
    0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
]

function kxt2c_matrix(mag,mn) #; Nx = 16, Ny = 16, Nz = 1)
    Nx, Ny, Nz = size(mag)
    kxt2c_matrix_buf = Zygote.bufferfrom(zeros(ComplexF64,(3*Nx*Ny*Nz),(2*Nx*Ny*Nz)))
    for ix=1:Nx,iy=1:Ny,iz=1:Nz,a=1:3 #,b=1:2
        q = Nx * Ny * (iz - 1) + Nx * (iy - 1) + ix
        # reference from kcross_t2c!:
        #       ds.d[1,i,j,k] = ( ds.H[1,i,j,k] * ds.mn[1,2,i,j,k] - ds.H[2,i,j,k] * ds.mn[1,1,i,j,k] ) * -ds.kpg_mag[i,j,k]
        #       ds.d[2,i,j,k] = ( ds.H[1,i,j,k] * ds.mn[2,2,i,j,k] - ds.H[2,i,j,k] * ds.mn[2,1,i,j,k] ) * -ds.kpg_mag[i,j,k]
        #       ds.d[3,i,j,k] = ( ds.H[1,i,j,k] * ds.mn[3,2,i,j,k] - ds.H[2,i,j,k] * ds.mn[3,1,i,j,k] ) * -ds.kpg_mag[i,j,k]
        # which implements d⃗ = k×ₜ₂c ⋅ H⃗
        # Here we want to explicitly define the matrix k×ₜ₂c
        # the general indexing scheme:
        # kxt2c_matrix_buf[ (3*q-2)+a-1 ,(2*q-1) + (b-1) ] <==> mn[a,b,ix,iy,iz], mag[ix,iy,iz]
        # b = 1  ( m⃗ )
        kxt2c_matrix_buf[(3*q-2)+a-1,(2*q-1)] = mn[a,2,ix,iy,iz] * mag[ix,iy,iz]
        # b = 2  ( n⃗ )
        kxt2c_matrix_buf[(3*q-2)+a-1,(2*q-1)+1] = mn[a,1,ix,iy,iz] * -mag[ix,iy,iz]
    end
    return copy(kxt2c_matrix_buf)
end

function kxt2c_matrix(p = p0;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    # kz, w, t_core, θ, n_core, n_subs, edge_gap = p
    grid = OptiMode.make_MG(Δx, Δy, Δz, Nx, Ny, Nz)
    mag,mn = calc_kpg(p[1],grid.g⃗)
    return kxt2c_matrix(mag,mn)
end

function zxt2c_matrix(mn::AbstractArray{T,5}) where T #; Nx = 16, Ny = 16, Nz = 1)
    Nx, Ny, Nz = size(mn)[3:5]
    zxt2c_matrix_buf = Zygote.bufferfrom(zeros(ComplexF64,(3*Nx*Ny*Nz),(2*Nx*Ny*Nz)))
    for ix=1:Nx,iy=1:Ny,iz=1:Nz #,b=1:2 #,a=1:2
        q = Nx * Ny * (iz - 1) + Nx * (iy - 1) + ix
        # reference from zcross_t2c!:
        #          ds.d[1,i,j,k] = -Hin[1,i,j,k] * ds.mn[2,1,i,j,k] - Hin[2,i,j,k] * ds.mn[2,2,i,j,k]
        #          ds.d[2,i,j,k] =  Hin[1,i,j,k] * ds.mn[1,1,i,j,k] + Hin[2,i,j,k] * ds.mn[1,2,i,j,k]
        #          ds.d[3,i,j,k] = 0
        # which implements d⃗ = z×ₜ₂c ⋅ H⃗
        # Here we want to explicitly define the matrix z×ₜ₂c
        # the general indexing scheme:
        # zxt2c_matrix_buf[ (3*q-2)+a-1 ,(2*q-1) + (b-1) ] <==> mn[a,b,ix,iy,iz]
        # a = 1  ( x̂ ), b = 1  ( m⃗ )
        zxt2c_matrix_buf[(3*q-2),(2*q-1)] = -mn[2,1,ix,iy,iz]
        # a = 1  ( x̂ ), b = 2  ( n⃗ )
        zxt2c_matrix_buf[(3*q-2),2*q] = -mn[2,2,ix,iy,iz]
        # a = 2  ( ŷ ), b = 1  ( m⃗ )
        zxt2c_matrix_buf[(3*q-2)+1,(2*q-1)] = mn[1,1,ix,iy,iz]
        # a = 2  ( ŷ ), b = 2  ( n⃗ )
        zxt2c_matrix_buf[(3*q-2)+1,2*q] = mn[1,2,ix,iy,iz]
    end
    return copy(zxt2c_matrix_buf)
end

function zxt2c_matrix(p = p0;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    # kz, w, t_core, θ, n_core, n_subs, edge_gap = p
    grid = OptiMode.make_MG(Δx, Δy, Δz, Nx, Ny, Nz)
    mag,mn = calc_kpg(p[1],grid.g⃗)
    return zxt2c_matrix(mn)
end

function ei_dot_rwg(p = p0;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    kz, w, t_core, θ, n_core, n_subs, edge_gap = p
    # (w,t_core,θ,edge_gap,n_core,n_subs,Δx,Δy,Δz,Nx,Ny,Nz)
    grid = OptiMode.make_MG(Δx, Δy, Δz, Nx, Ny, Nz)
    shapes = ridge_wg(w,t_core,θ,edge_gap,n_core,n_subs,Δx,Δy)
    ei_field = make_εₛ⁻¹(shapes,grid)
    # ei_matrix_buf = Zygote.bufferfrom(zeros(Float64,(3*Nx*Ny*Nz),(3*Nx*Ny*Nz)))
    ei_matrix_buf = Zygote.bufferfrom(zeros(ComplexF64,(3*Nx*Ny*Nz),(3*Nx*Ny*Nz)))
    for i=1:Nx,j=1:Ny,a=1:3,b=1:3
        q = (Ny * (j-1) + i)
        ei_matrix_buf[(3*q-2)+a-1,(3*q-2)+b-1] = ei_field[a,b,i,j,1]
    end
    # return copy(ei_matrix_buf)
    return Hermitian(copy(ei_matrix_buf))
end

function M_components(p = p0;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    kz, w, t_core, θ, n_core, n_subs, edge_gap = p
    #(kz,w,t_core,edge_gap,n_core,n_subs,Δx,Δy,Δz,NxF,NyF,NzF)
    # Nx,Ny,Nz = Zygote.ignore() do
    #     (Int(round(NxF)),Int(round(NyF)),Int(round(NzF)))
    # end
    mag, mn = calc_kpg(kz, Δx, Δy, Δz, Nx, Ny, Nz)
    kcr_t2c = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_t2c( reshape(H,(2,Nx,Ny,Nz)), mn, mag ) )::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),*(2,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕 = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end

    𝓕⁻¹ = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ifft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    kcr_c2t = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_c2t( reshape(H,(3,Nx,Ny,Nz)), mn, mag ) )::AbstractArray{ComplexF64,1},*(2,Nx,Ny,Nz),*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    eeii = ei_dot_rwg(p;Δx,Δy,Δz,Nx,Ny,Nz)
    return ( kcr_c2t, 𝓕⁻¹, eeii, 𝓕, kcr_t2c )
end

function make_M_old(p = p0;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    kz, w, t_core, θ, n_core, n_subs, edge_gap = p
    mag, mn = calc_kpg(kz, Δx, Δy, Δz, Nx, Ny, Nz)
    kcr_t2c = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_t2c( reshape(H,(2,Nx,Ny,Nz)), mn, mag ) )::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),*(2,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕 = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕⁻¹ = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ifft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    kcr_c2t = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_c2t( reshape(H,(3,Nx,Ny,Nz)), mn, mag ) )::AbstractArray{ComplexF64,1},*(2,Nx,Ny,Nz),*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    eeii = ei_dot_rwg(p;Δx,Δy,Δz,Nx,Ny,Nz)
    M = -kcr_c2t * 𝓕⁻¹ * eeii * 𝓕 * kcr_t2c
    # @assert M' ≈ M
    return Hermitian(M)
end

function make_M(p = p0;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    kz, w, t_core, θ, n_core, n_subs, edge_gap = p
    mag, mn = calc_kpg(kz, Δx, Δy, Δz, Nx, Ny, Nz)
    kcr_t2c = kxt2c_matrix(mag,mn)
    𝓕 = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕⁻¹ = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ifft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    kcr_c2t = -kcr_t2c'
    eeii = ei_dot_rwg(p;Δx,Δy,Δz,Nx,Ny,Nz)
    M = -kcr_c2t * 𝓕⁻¹ * eeii * 𝓕 * kcr_t2c
    return Hermitian(M)
end

function make_Mₖ(p = p0;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    kz, w, t_core, θ, n_core, n_subs, edge_gap = p
    mag, mn = calc_kpg(kz, Δx, Δy, Δz, Nx, Ny, Nz)
    zcr_t2c = zxt2c_matrix(mn)
    𝓕 = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕⁻¹ = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ifft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    kcr_c2t = -kxt2c_matrix(mag,mn)'
    eeii = ei_dot_rwg(p;Δx,Δy,Δz,Nx,Ny,Nz)
    -kcr_c2t * 𝓕⁻¹ * eeii * 𝓕 * zcr_t2c
end

function make_M_eidot(p,
                    eidot::Hermitian;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    kz, w, t_core, θ, n_core, n_subs, edge_gap = p
    mag, mn = calc_kpg(kz, Δx, Δy, Δz, Nx, Ny, Nz)
    kcr_t2c = kxt2c_matrix(mag,mn)
    𝓕 = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕⁻¹ = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ifft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    kcr_c2t = -transpose(kcr_t2c) #-kcr_t2c'
    M = -kcr_c2t * 𝓕⁻¹ * eidot * 𝓕 * kcr_t2c
    return Hermitian(M)
end

function make_Mₖ_eidot(p,
                    eidot::Hermitian;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    kz, w, t_core, θ, n_core, n_subs, edge_gap = p
    mag, mn = calc_kpg(kz, Δx, Δy, Δz, Nx, Ny, Nz)
    zcr_t2c = zxt2c_matrix(mn)
    𝓕 = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕⁻¹ = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ifft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    kcr_c2t = -transpose(kxt2c_matrix(mag,mn)) #-kxt2c_matrix(mag,mn)'
    -kcr_c2t * 𝓕⁻¹ * eidot * 𝓕 * zcr_t2c
end

function make_M(eidot::Hermitian,kcr_t2c)
    # kcr_t2c = kxt2c_matrix(mag,mn)
    𝓕 = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕⁻¹ = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ifft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    kcr_c2t = -kcr_t2c'
    M = -kcr_c2t * 𝓕⁻¹ * eidot * 𝓕 * kcr_t2c
    return Hermitian(M)
end

function make_M(eidot::Hermitian,mag,mn)
    kcr_t2c = kxt2c_matrix(mag,mn)
    𝓕 = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕⁻¹ = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ifft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    kcr_c2t = -kcr_t2c'
    M = -kcr_c2t * 𝓕⁻¹ * eidot * 𝓕 * kcr_t2c
    return Hermitian(M)
end

function make_Mₖ(eidot::Hermitian,mag,mn)
    zcr_t2c = zxt2c_matrix(mn)
    𝓕 = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    𝓕⁻¹ = Zygote.ignore() do
        Matrix(LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ifft(reshape(d,(3,Nx,Ny,Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,Nx,Ny,Nz),ishermitian=false,ismutating=false))
    end
    kcr_c2t = -kxt2c_matrix(mag,mn)'
    -kcr_c2t * 𝓕⁻¹ * eidot * 𝓕 * zcr_t2c
end

function proc_eigs(p,Xone,αone;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    # sum2(x->abs2(x)^2,Xone) * abs2(αone)^2
    # sqrt(real(αone)) / real(dot(Xone,make_Mₖ(p;Δx,Δy,Δz,Nx,Ny,Nz),Xone))
    # sqrt(αone) / abs(dot(Xone,make_Mₖ(p;Δx,Δy,Δz,Nx,Ny,Nz),Xone))
    sqrt(real(αone)) / real(dot(Xone,make_Mₖ(p;Δx,Δy,Δz,Nx,Ny,Nz),Xone))
end

function proc_eigs(eidot::Hermitian,mag,mn,Xone,αone)
    # sum2(x->abs2(x)^2,Xone) * abs2(αone)^2
    # sqrt(real(αone)) / real(dot(Xone,make_Mₖ(p;Δx,Δy,Δz,Nx,Ny,Nz),Xone))
    # sqrt(αone) / abs(dot(Xone,make_Mₖ(eidot,mag,mn),Xone))
    sqrt(real(αone)) / real(dot(Xone,make_Mₖ(eidot,mag,mn),Xone))
end

function proc_eigs_eidot(p,eidot::Hermitian,Xone,αone;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    # sum2(x->abs2(x)^2,Xone) * abs2(αone)^2
    sqrt(real(αone)) / real(dot(Xone,make_Mₖ_eidot(p,eidot;Δx,Δy,Δz,Nx,Ny,Nz),Xone))
end

function solve_dense(p = p0;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    # kz, w, t_core, θ, n_core, n_subs, edge_gap = p
    Eigs = eigen(make_M(p;Δx,Δy,Δz,Nx,Ny,Nz))
    Xone = Eigs.vectors[:,1]
    αone = Eigs.values[1]
    proc_eigs(p,Xone,αone;Δx,Δy,Δz,Nx,Ny,Nz)
    # proc_eigs(Xone,αone)
end

function solve_dense(eidot::Hermitian,mag,mn)
    Eigs = eigen(make_M(eidot,mag,mn))
    Xone = Eigs.vectors[:,1]
    αone = Eigs.values[1]
    proc_eigs(eidot,mag,mn,Xone,αone)
end

function solve_dense_eidot(p,
                    eidot::Hermitian;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 16,
                    Ny = 16,
                    Nz = 1)
    kz, w, t_core, θ, n_core, n_subs, edge_gap = p
    #(kz,eidot::Hermitian{ComplexF64, Matrix{ComplexF64}},Δx,Δy,Δz,Nx,Ny,Nz)
    Eigs = eigen(make_M_eidot(p,eidot;Δx,Δy,Δz,Nx,Ny,Nz))
    Xone = Eigs.vectors[:,1]
    αone = Eigs.values[1]
    proc_eigs_eidot(p,eidot,Xone,αone;Δx,Δy,Δz,Nx,Ny,Nz)
end

function ∂solve_dense_SJ(p = p0;
                    Δx  = 6.0,
                    Δy  = 4.0,
                    Δz  = 1.0,
                    Nx  = 16,
                    Ny  = 16,
                    Nz  = 1,
                    i   = 1)
    M, M_pb = Zygote.pullback(x->make_M(x;Δx,Δy,Δz,Nx,Ny,Nz),p)
    α,X = eigen(M)
    p̄2,X̄,ᾱ = Zygote.gradient(p,X[:,1],α[1]) do p,H,ω²
        proc_eigs(p,H,ω²;Δx,Δy,Δz,Nx,Ny,Nz)
    end
    P = I - X[:,i] * X[:,i]'
    b = P * X̄ #[i]
    λ₀ = IterativeSolvers.bicgstabl(M-α[i]*I,b,3)
    if isnothing(ᾱ)
        ᾱ = 0.
    end
    λ = λ₀ - ᾱ * X[:,i]
    M̄ = -λ * X[:,i]'
    p̄1 = M_pb(M̄)[1]
    if isnothing(p̄2)
        p̄2 = zeros(eltype(p),size(p))
    end
    if isnothing(p̄1)
        p̄1 = zeros(eltype(p),size(p))
    end
    p̄2 + p̄1
end
function ei_field2matrix(ei_field,Nx,Ny,Nz)
    ei_matrix_buf = Zygote.bufferfrom(zeros(ComplexF64,(3*Nx*Ny*Nz),(3*Nx*Ny*Nz)))
    for i=1:Nx,j=1:Ny,a=1:3,b=1:3
        q = (Ny * (j-1) + i)
        ei_matrix_buf[(3*q-2)+a-1,(3*q-2)+b-1] = ei_field[a,b,i,j,1]
    end
    # return copy(ei_matrix_buf)
    return Hermitian(copy(ei_matrix_buf))
end

function ei_matrix2field1(ei_matrix,Nx,Ny,Nz)
    ei_field = zeros(Float64,(3,3,Nx,Ny,Nz))
    D0 = diag(ei_matrix,0)
    D1 = diag(ei_matrix,1)
    D2 = diag(ei_matrix,2)
    for i=1:Nx,j=1:Ny,k=1:Nz #,a=1:3,b=1:3
        q = (Nz * (k-1) + Ny * (j-1) + i) # (Ny * (j-1) + i)
        ei_field[1,1,i,j,k] = real(D0[3*q-2])
        ei_field[2,2,i,j,k] = real(D0[3*q-1] )
        ei_field[3,3,i,j,k] = real(D0[3*q])
        ei_field[1,2,i,j,k] = real(D1[3*q-2])
        ei_field[2,1,i,j,k] = real(conj(D1[3*q-2]))
        ei_field[2,3,i,j,k] = real(D1[3*q-1])
        ei_field[3,2,i,j,k] = real(conj(D1[3*q-1]))
        ei_field[1,3,i,j,k] = real(D2[3*q-2])
        ei_field[3,1,i,j,k] = real(conj(D2[3*q-2]))
        # ei_matrix[(3*q-2)+a-1,(3*q-2)+b-1] = ei_field[a,b,i,j,1]
    end
    return ei_field
end

using ArrayInterface, LoopVectorization
function ei_matrix2field2(ei_matrix,Nx,Ny,Nz)
    ei_field = zeros(Float64,(3,3,Nx,Ny,Nz))
    D0 = diag(ei_matrix,0)
    D1 = diag(ei_matrix,1)
    D2 = diag(ei_matrix,2)
    @avx for i=1:Nx,j=1:Ny,k=1:Nz #,a=1:3,b=1:3
        q = (Nz * (k-1) + Ny * (j-1) + i) # (Ny * (j-1) + i)
        ei_field[1,1,i,j,k] = D0[3*q-2]
        ei_field[2,2,i,j,k] = D0[3*q-1]
        ei_field[3,3,i,j,k] = D0[3*q]
        ei_field[1,2,i,j,k] = D1[3*q-2]
        ei_field[2,1,i,j,k] = D1[3*q-2]
        ei_field[2,3,i,j,k] = D1[3*q-1]
        ei_field[3,2,i,j,k] = D1[3*q-1]
        ei_field[1,3,i,j,k] = D2[3*q-2]
        ei_field[3,1,i,j,k] = D2[3*q-2]
        # ei_matrix[(3*q-2)+a-1,(3*q-2)+b-1] = ei_field[a,b,i,j,1]
    end
    return ei_field
end

function ei_matrix2field3(ei_matrix,Nx,Ny,Nz)
    ei_field = zeros(Float64,(3,3,Nx,Ny,Nz))
    @avx for a=1:3,b=1:3,k=1:Nz,j=1:Ny,i=1:Nx
        q = (Nz * (k-1) + Ny * (j-1) + i) # (Ny * (j-1) + i)
        ei_field[a,b,i,j,k] = ei_matrix[(3*q-2)+a-1,(3*q-2)+b-1]
    end
    return ei_field
end

function ei_matrix2field4(d,λd,Nx,Ny,Nz)
    # ei_field = Hermitian(zeros(Float64,(3,3,Nx,Ny,Nz)),"U")
    ei_field = zeros(Float64,(3,3,Nx,Ny,Nz))
    @avx for k=1:Nz,j=1:Ny,i=1:Nx
        q = (Nz * (k-1) + Ny * (j-1) + i) # (Ny * (j-1) + i)
        for a=1:3 # loop over diagonals
            ei_field[a,a,i,j,k] = real( -λd[3*q-2+a-1] * conj(d[3*q-2+a-1]) )
        end
        for a2=1:2 # loop over first off diagonal
            ei_field[a2,a2+1,i,j,k] = real( -conj(λd[3*q-2+a2]) * d[3*q-2+a2-1] - λd[3*q-2+a2-1] * conj(d[3*q-2+a2]) )
            ei_field[a2+1,a2,i,j,k] = ei_field[a2,a2+1,i,j,k]  # D1[3*q-2]
        end
        # a = 1, set 1,3 and 3,1, second off-diagonal
        ei_field[1,3,i,j,k] = real( -conj(λd[3*q]) * d[3*q-2] - λd[3*q-2] * conj(d[3*q]) )
        ei_field[3,1,i,j,k] =  ei_field[1,3,i,j,k]
    end
    return ei_field
end

ei_matrix2field = ei_matrix2field4



## set discretization parameters and generate explicit dense matrices
Δx          =   6.                    # μm
Δy          =   4.                    # μm
Δz          =   1.
Nx          =   16
Ny          =   16
Nz          =   1
kz          =   p0[1] #1.45
# ω           =   1 / λ
p = p0 #[kz,w,t_core,θ,n_core,n_subs,edge_gap] #,Δx,Δy,Δz,Nx,Ny,Nz]
eid = ei_dot_rwg(p;Δx,Δy,Δz,Nx,Ny,Nz)
g = MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
ds = MaxwellData(p[1],g)
ei = make_εₛ⁻¹(ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),g)
# eii = similar(ei); [ (eii[a,b,i,j,k] = inv(ei[:,:,i,j,k])[a,b]) for a=1:3,b=1:3,i=1:Nx,j=1:Ny,k=1:Nz ] # eii = epsilon tensor field (eii for epsilon_inverse_inverse, yea it's dumb)
Mop = M̂!(ei,ds)
Mop2 = M̂(ei,ds)
Mₖop = M̂ₖ(ei,ds.mn,ds.kpg_mag,ds.𝓕,ds.𝓕⁻¹)
M = Matrix(Mop)
dMdk = Matrix(Mₖop)
mag,mn = calc_kpg(p[1],OptiMode.make_MG(Δx, Δy, Δz, Nx, Ny, Nz).g⃗)
eid = ei_dot_rwg(p0)

make_M(eid,mag,mn) ≈ M
make_Mₖ(eid,mag,mn) ≈ -dMdk
make_Mₖ_eidot(p,eid) ≈ -dMdk
make_Mₖ(p0) ≈ -dMdk

𝓕 = plan_fft(randn(ComplexF64, (3,Nx,Ny,Nz)),(2:4))
𝓕⁻¹ = plan_ifft(randn(ComplexF64, (3,Nx,Ny,Nz)),(2:4))
Mop2 = M̂(ei,mn,mag,𝓕,𝓕⁻¹)
M2 = Matrix(Mop2)
M2 ≈ M


3

##
# M̂(ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> M(H,ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)::AbstractArray{ComplexF64,1},*(2,size(ε⁻¹)[end-2:end]...),ishermitian=true,ismutating=false)
# function M(H,ε⁻¹,mn,kpg_mag,𝓕::FFTW.cFFTWPlan,𝓕⁻¹)
#     kx_c2t( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * kx_t2c(H,mn,kpg_mag), ε⁻¹), mn,kpg_mag)
# end
kxt2c_op = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_t2c( reshape(H,(2,ds.Nx,ds.Ny,ds.Nz)), ds.mn, ds.kpg_mag ) )::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),*(2,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
kxt2c = Matrix(kxt2c_op)
F_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ds.𝓕*reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)))::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
# F_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(fft(reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
F = Matrix(F_op)
einv_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec( ε⁻¹_dot( reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)), ei ) )::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
einv = Matrix(einv_op)
Finv_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(ds.𝓕⁻¹*reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)))::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
# Finv_op = LinearMap{ComplexF64}(d::AbstractArray{ComplexF64,1} -> vec(bfft(reshape(d,(3,ds.Nx,ds.Ny,ds.Nz)),(2:4)))::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
Finv = Matrix(Finv_op)
kxc2t_op = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( kx_c2t( reshape(H,(3,ds.Nx,ds.Ny,ds.Nz)), ds.mn, ds.kpg_mag ) )::AbstractArray{ComplexF64,1},*(2,ds.Nx,ds.Ny,ds.Nz),*(3,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
kxc2t = Matrix(kxc2t_op)
zxt2c_op = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> vec( zx_t2c( reshape(H,(2,ds.Nx,ds.Ny,ds.Nz)), ds.mn ) )::AbstractArray{ComplexF64,1},*(3,ds.Nx,ds.Ny,ds.Nz),*(2,ds.Nx,ds.Ny,ds.Nz),ishermitian=false,ismutating=false)
zxt2c = Matrix(zxt2c_op)

@assert -kxc2t * Finv * einv * F * kxt2c ≈ M
@assert kxc2t * Finv * einv * F * zxt2c ≈ dMdk # wrong sign?
@assert make_M(p;Δx,Δy,Δz,Nx,Ny,Nz) ≈ M
@assert make_M_eidot(p,eid;Δx,Δy,Δz,Nx,Ny,Nz) ≈ M
@assert ei_dot_rwg(p;Δx,Δy,Δz,Nx,Ny,Nz) ≈ einv
# if Finv is ifft
@assert F' ≈  Finv * ( size(F)[1]/3 )
@assert Finv' * ( size(F)[1]/3 ) ≈  F
# # if Finv is bfft
# @assert F' ≈ Finv
# @assert Finv' ≈  F
@assert kxc2t' ≈ -kxt2c
@assert kxt2c' ≈ -kxc2t

# ix = 8
# iy = 4
# q = Nx * (iy - 1) + ix
# 3q-2:3q+3 # 3q-2:3q-2+6-1
# 2q-1:2q+2 # 2q-1:2q-1+4-1
#
# real(kxt2c[3q-2:3q+3,2q-1:2q+2])
@assert kxt2c_matrix(p0) ≈ kxt2c
@assert kxt2c_matrix(mag,mn) ≈ kxt2c
@assert zxt2c_matrix(mn) ≈ zxt2c
# sum(kxt2c_matrix(p0))
# ∇sum_kxt2c1 = Zygote.gradient(x->sum(real(kxt2c_matrix(x))), p0)[1]
#
# (mag, mn), magmn_pb = Zygote.pullback(p0) do p
#     calc_kpg(p[1],OptiMode.make_MG(Δx, Δy, Δz, Nx, Ny, Nz).g⃗)
# end
#
# kxt2c, kxt2c_pb = Zygote.pullback(mag,mn) do mag,mn
#     kxt2c_matrix(mag,mn)
# end
#
# sum_kxt2c, sum_kxt2c_pb = Zygote.pullback(sum, kxt2c)
#
# # step-by-step pullback
# kxt̄2c = sum_kxt2c_pb(1)[1]
# māg,mn̄ = kxt2c_pb(kxt̄2c)
# p̄ = magmn_pb((māg,mn̄))[1]
# @assert ei_field2matrix(ei,Nx,Ny,Nz) ≈ eid
##
ei_dot_rwg(p0)
solve_dense(p0)
∂solve_dense_SJ(p0)
Zygote.gradient(solve_dense,p0)[1] ≈ ∂solve_dense_SJ(p0)

## Finite Difference End-to-end (parameters to group velocity) gradient calculation for checking AD gradients
println("####################################################################################")
println("")
println("Finite Difference End-to-end (parameters to group velocity) gradient calculation for checking AD gradients")
println("")
println("####################################################################################")
@show Δx          =   6.                    # μm
@show Δy          =   4.                    # μm
@show Δz          =   1.
@show Nx          =   16
@show Ny          =   16
@show Nz          =   1
@show p=p0
@show kz          =   p[1]
@show p̄_FD = FiniteDifferences.jacobian(central_fdm(3,1),x->solve_dense(x),p0)[1][1,:]
## End-to-end (parameters to group velocity) gradient calculation with explicit matrices
println("####################################################################################")
println("")
println("End-to-end (parameters to group velocity) gradient calculation with explicit matrices")
println("")
println("####################################################################################")
p=p0
M, M_pb = Zygote.pullback(x->make_M(x;Δx,Δy,Δz,Nx,Ny,Nz),p)
# M = make_M(p;Δx,Δy,Δz,Nx,Ny,Nz)
αX = eigen(M)
# @btime eigen(make_M($p...))
# 41.165 ms (32813 allocations: 9.78 MiB) for Nx=Ny=8, size(M)=(128,128)
# @btime eigen($M)
# 19.841 s (26 allocations: 132.17 MiB) for Nx=Ny=32, size(M)=(2048,2048)
# 864.363 ms (24 allocations: 9.04 MiB) for Nx=Ny=16, size(M)=(512,512)
# 0.021 s (24 allocations: 0.78 MiB) for Nx=Ny=8, size(M)=(128,128)
α = αX.values
X = αX.vectors
@show α[1]
proc_eigs(p,X[:,1],α[1];Δx,Δy,Δz,Nx,Ny,Nz)
solve_dense(p)
p̄2,X̄,ᾱ = Zygote.gradient(p,X[:,1],α[1]) do p,H,ω²
    proc_eigs(p,H,ω²;Δx,Δy,Δz,Nx,Ny,Nz)
end
@show p̄2
P̂ = I - X[:,1] * X[:,1]'
b = P̂ * X̄ #[1]
@show maximum(abs2.(b))
X̄ - X[:,1] * dot(X[:,1],X̄) ≈ b
λ₀ = IterativeSolvers.bicgstabl(M-α[1]*I,b,3)
@show maximum(abs2.(λ₀))
if isnothing(ᾱ)
    ᾱ = 0.
end
λ = λ₀ - ᾱ * X[:,1]
@show maximum(abs2.(λ))
M̄ = -λ * X[:,1]'
@show p̄1 = M_pb(M̄)[1]
if isnothing(p̄2)
    p̄2 = zeros(eltype(p),size(p))
end
if isnothing(p̄1)
    p̄1 = zeros(eltype(p),size(p))
end
@show p̄ = p̄2 + p̄1
@show p̄_err = abs.(p̄_FD .- p̄) ./ abs.(p̄_FD)
## (ε⁻¹ operator ,k) to group velocity gradient calculation with explicit matrices
println("####################################################################################")
println("")
println("(ε⁻¹ operator ,k) to group velocity gradient calculation with explicit matrices")
println("")
println("####################################################################################")
p = p0
eid, eid_pb = Zygote.pullback(x->ei_dot_rwg(x;Δx,Δy,Δz,Nx,Ny,Nz),p)
# eid = ei_dot_rwg(p;Δx,Δy,Δz,Nx,Ny,Nz)
M = make_M_eidot(p,eid;Δx,Δy,Δz,Nx,Ny,Nz)
αX = eigen(M)
α = αX.values
X = αX.vectors
@show ω²_eidot = α[1]
# @show ng_proc_eigs_eidot = proc_eigs_eidot(p,eid,X[:,1],α[1];Δx,Δy,Δz,Nx,Ny,Nz)
@show ng_eidot = solve_dense_eidot(p,eid;Δx,Δy,Δz,Nx,Ny,Nz)
p̄_pe,eīd_pe,X̄,ᾱ = Zygote.gradient(p,eid,X[:,1],α[1]) do p,eidot,H,ω²
    proc_eigs_eidot(p,eidot,H,ω²;Δx,Δy,Δz,Nx,Ny,Nz)
end
eīd_pe_herm = Zygote._hermitian_back(eīd_pe,eid.uplo)
@show ω̄sq_eidot = ᾱ
@show p̄_pe
P̂ = I - X[:,1] * X[:,1]'
b = P̂ * X̄ #[1]
@show maximum(abs2.(b))
X̄ - X[:,1] * dot(X[:,1],X̄) ≈ b
λ₀ = IterativeSolvers.bicgstabl(M-α[1]*I,b,3)
@show maximum(abs2.(λ₀))
if isnothing(ᾱ)
    ᾱ = 0.
end
λ = λ₀ - ᾱ * X[:,1]
@show maximum(abs2.(λ))
M̄ = -λ * X[:,1]'
kcr_c2t, 𝓕⁻¹, eeii, 𝓕, kcr_t2c = M_components(p;Δx,Δy,Δz,Nx,Ny,Nz)
# eīd1 = -𝓕 * kcr_t2c * M̄ * kcr_c2t * 𝓕⁻¹ # = (-kcr_c2t * 𝓕⁻¹)' * M̄ * (𝓕 * kcr_t2c)'
# (-kcr_c2t * 𝓕⁻¹)' * M̄ * (𝓕 * kcr_t2c)' ≈ -𝓕 * kcr_t2c * M̄ * kcr_c2t * 𝓕⁻¹
d = 𝓕 * kcr_t2c * X[:,1] ./ (Nx * Ny * Nz)
λd = 𝓕 * kcr_t2c * λ
e = eid * d
eīd_eig = -λd * d'
eīd_eig_herm = Zygote._hermitian_back(eīd_eig,eid.uplo)
λe = eid * λd
λẽ = 𝓕⁻¹ * λe
ẽ = (Nx * Ny * Nz) * 𝓕⁻¹ * e
kcr̄_t2c = -( λẽ * X[:,1]' + ẽ * λ' )
@show maximum(abs2.(d))
@show maximum(abs2.(λd))
@show maximum(abs2.(e))
@show maximum(abs2.(λe))
kcr_t2c2, kcr_t2c_pb = Zygote.pullback(kxt2c_matrix,p)
@show p̄_kcr = real(kcr_t2c_pb(kcr̄_t2c)[1])
# -𝓕 * kcr_t2c * M̄ * kcr_c2t * 𝓕⁻¹ ≈ -λd * d'
eīd = eīd_eig_herm + eīd_pe_herm
@show p̄_eid = eid_pb(eīd)[1]
if isnothing(p̄_pe)
    p̄_pe = zeros(eltype(p),size(p))
end
if isnothing(p̄_eid)
    p̄_eid = zeros(eltype(p),size(p))
end
if isnothing(p̄_kcr)
    p̄_kcr = zeros(eltype(p),size(p))
end
@show p̄ = p̄_eid + p̄_pe + p̄_kcr
@show p̄_FD
@show p̄_err = abs.(p̄_FD .- p̄) ./ abs.(p̄_FD)
# eīd_5diag = diagm([diag_idx => diag(eīd,diag_idx) for diag_idx = -2:2]...)
# eīd_3diag = diagm([diag_idx => diag(eīd,diag_idx) for diag_idx = -1:1]...)
# eīd_1diag = diagm([diag_idx => diag(eīd,diag_idx) for diag_idx = 0]...)
# @assert eid_pb(eīd)[1] ≈ eid_pb(eīd_3diag)[1]
# @show p̄1_eidot = eid_pb(eīd_3diag)[1]
# @show p̄1_eidot_5diag = eid_pb(eīd_5diag)[1]
# @show p̄1_eidot_3diag = eid_pb(eīd_3diag)[1]
# @show p̄1_eidot_1diag = eid_pb(eīd_1diag)[1]
# @show p̄1_eidot_5diag_err = abs.(p̄1_eidot .- p̄1_eidot_5diag) ./ abs.(p̄1_eidot)
# @show p̄1_eidot_3diag_err = abs.(p̄1_eidot .- p̄1_eidot_3diag) ./ abs.(p̄1_eidot)
# @show p̄1_eidot_1diag_err = abs.(p̄1_eidot .- p̄1_eidot_1diag) ./ abs.(p̄1_eidot)

# dstar = conj.(d)
# λdstar = conj.(λd)
# D0 = real( (-λd .* dstar)) #-λd .* dstar
# D1 = -λdstar[2:end] .* d[begin:end-1] + -λd[begin:end-1] .* dstar[2:end]
# D2 = -λdstar[3:end] .* d[begin:end-2] + -λd[begin:end-2] .* dstar[3:end]
# diag(eīd1_herm,0) ≈ D0
# diag(eīd1_herm,1) ≈ D1
# diag(eīd1_herm,2) ≈ D2
# @show maximum(abs2.(D0))
# @show maximum(abs2.(D1))
# @show maximum(abs2.(D2))
##
println("####################################################################################")
println("")
println("(ε⁻¹ operator ,(mn(k), mag(k)) arrays) to group velocity gradient calculation with explicit matrices")
println("")
println("####################################################################################")
p = p0
eid, eid_pb = Zygote.pullback(x->ei_dot_rwg(x;Δx,Δy,Δz,Nx,Ny,Nz),p)
(mag, mn), magmn_pb = Zygote.pullback(p0) do p
    calc_kpg(p[1],OptiMode.make_MG(Δx, Δy, Δz, Nx, Ny, Nz).g⃗)
end
kxt2c, kxt2c_pb = Zygote.pullback(mag,mn) do mag,mn
    kxt2c_matrix(mag,mn)
end
@show ng,ng_pb = Zygote.pullback(solve_dense,eid,mag,mn)
eīd,māg,mn̄ = ng_pb(1)
@show p̄_magmn = magmn_pb((real(māg),real(mn̄)))[1]
@show p̄_eid = eid_pb(eīd)[1]
@show p̄ = p̄_magmn + p̄_eid
@show p̄_FD
@show p̄_err = abs.(p̄_FD .- p̄) ./ abs.(p̄_FD)
M = make_M(eid,mag,mn)
αX = eigen(M)
α = αX.values
X = αX.vectors
@show ω² = α[1]
@show n = sqrt(ω²) / p[1]
@show ng, pe_pb = Zygote.pullback(proc_eigs,eid,mag,mn,X[:,1],α[1])
eīd_pe,māg_pe,mn̄_pe,X̄,ᾱ = pe_pb(1)
@show ω̄sq = ᾱ
@show maximum(abs2.(eīd_pe))
@show maximum(abs2.(māg_pe))
@show maximum(abs2.(mn̄_pe))
@show size(X̄)
@show maximum(abs2.(X̄))
# @show maximum(real.(X̄))
# @show maximum(imag.(X̄))
# @show minimum(real.(X̄))
# @show minimum(imag.(X̄))
# solve for adjoint field, pull back through M to get p̄ contributions
P̂ = I - X[:,1] * X[:,1]'
b = P̂ * X̄ #[1]
@show maximum(abs2.(b))
# @show maximum(real.(b))
# @show maximum(imag.(b))
# @show minimum(real.(b))
# @show minimum(imag.(b))
X̄ - X[:,1] * dot(X[:,1],X̄) ≈ b
λ₀ = IterativeSolvers.bicgstabl(M-α[1]*I,b,3)
@show maximum(abs2.(λ₀))
# @show maximum(real.(λ₀))
# @show maximum(imag.(λ₀))
# @show minimum(real.(λ₀))
# @show minimum(imag.(λ₀))
if isnothing(ᾱ)
    ᾱ = 0.
end
λ = λ₀ - ᾱ * X[:,1]
@show maximum(abs2.(λ))
# @show maximum(real.(λ))
# @show maximum(imag.(λ))
# @show minimum(real.(λ))
# @show minimum(imag.(λ))
M̄ = -λ * X[:,1]'
kcr_c2t, 𝓕⁻¹, eeii, 𝓕, kcr_t2c = M_components(p;Δx,Δy,Δz,Nx,Ny,Nz)
# eīd1 = -𝓕 * kcr_t2c * M̄ * kcr_c2t * 𝓕⁻¹ # = (-kcr_c2t * 𝓕⁻¹)' * M̄ * (𝓕 * kcr_t2c)'
# (-kcr_c2t * 𝓕⁻¹)' * M̄ * (𝓕 * kcr_t2c)' ≈ -𝓕 * kcr_t2c * M̄ * kcr_c2t * 𝓕⁻¹
d = 𝓕 * kcr_t2c * X[:,1] ./ (Nx * Ny * Nz)
λd = 𝓕 * kcr_t2c * λ
e = eid * d
λe = eid * λd
λẽ = 𝓕⁻¹ * λe
ẽ = (Nx * Ny * Nz) * 𝓕⁻¹ * e
kcr̄_t2c = -( λẽ * X[:,1]' + ẽ * λ' )
@show maximum(abs2.(d))
@show maximum(abs2.(λd))
@show maximum(abs2.(e))
@show maximum(abs2.(λe))
@show maximum(abs2.(ẽ))
@show maximum(abs2.(λẽ))
λẽ_3v = reinterpret(SVector{3,ComplexF64},λẽ)
ẽ_3v = reinterpret(SVector{3,ComplexF64},ẽ)
λ_2v = reinterpret(SVector{2,ComplexF64},λ)
H_2v = reinterpret(SVector{2,ComplexF64},X[:,1])
@show size(λẽ_3v)
@show size(ẽ_3v)
@show size(λ_2v)
@show size(H_2v)
@show maximum(norm.(λẽ_3v))
@show maximum(norm.(ẽ_3v))
@show maximum(norm.(λ_2v))
@show maximum(norm.(H_2v))
kx̄ = reshape( reinterpret(Float64, -real.( λẽ_3v .* adjoint.(conj.(H_2v)) + ẽ_3v .* adjoint.(conj.(λ_2v)) ) ), (3,2,Nx,Ny,Nz) )
@tullio māg_eigs[ix,iy,iz] := mn[a,2,ix,iy,iz] * kx̄[a,1,ix,iy,iz] - mn[a,1,ix,iy,iz] * kx̄[a,2,ix,iy,iz]
mn̄_signs = [-1 ; 1]
@tullio mn̄_eigs[a,b,ix,iy,iz] := kx̄[a,3-b,ix,iy,iz] * mag[ix,iy,iz] * mn̄_signs[b] nograd=mn̄_signs
@show maximum(abs2.(kx̄))
@show maximum(māg_eigs)
@show maximum(mn̄_eigs)
@show p̄_kcr_eigs = magmn_pb((māg_eigs,mn̄_eigs))[1]
eīd_eigs = -λd * d'
eīd_eigs_herm = Zygote._hermitian_back(eīd_eigs,eid.uplo)
eīd_pe_herm = Zygote._hermitian_back(eīd_pe,eid.uplo)
eīd_full_herm = Zygote._hermitian_back(eīd,eid.uplo)
@show p̄_eid_eigs = eid_pb(eīd_eigs_herm)[1]
@show p̄_eigs = p̄_eid_eigs + p̄_kcr_eigs



@show p̄_eid_pe = eid_pb(eīd_pe_herm)[1]
@show p̄_kcr_pe = magmn_pb((real(māg_pe),real(mn̄_pe)))[1]
@show p̄_pe = p̄_eid_pe + p̄_kcr_pe

if isnothing(p̄_pe)
    p̄_pe = zeros(eltype(p),size(p))
end
if isnothing(p̄_eigs)
    p̄_eigs = zeros(eltype(p),size(p))
end
@show p̄ = p̄_eigs + p̄_pe
@show p̄_err = abs.(p̄_FD .- p̄) ./ abs.(p̄_FD)


# māg_eigs_AD,mn̄_eigs_AD = kxt2c_pb(kcr̄_t2c)
# @assert māg_eigs ≈ real(māg_eigs_AD)
# @assert mn̄_eigs ≈ real(mn̄_eigs_AD)

# kcr_t2c2, kcr_t2c_p_pb = Zygote.pullback(kxt2c_matrix,p)
# kcr_t2c3, kcr_t2c_magmn_pb = Zygote.pullback(kxt2c_matrix,mag,mn)
# @assert kcr_t2c2 ≈ kcr_t2c
# @assert kcr_t2c3 ≈ kcr_t2c
# @assert make_M(eid,mag,mn) ≈ make_M(eid,kcr_t2c)
# M,M_pb_kcr = Zygote.pullback(make_M,eid,kcr_t2c)
# M,M_pb_magmn = Zygote.pullback(make_M,eid,mag,mn)
# eīd_eigs2,māg_eigs2,mn̄_eigs2 = M_pb(M̄)
# eīd_kcr, kcr̄_t2c1 = M_pb_kcr(M̄)
# kcr̄_t2c2 = 𝓕⁻¹ * ( (eid * 𝓕 * kcr_t2c * M̄') + (eid * 𝓕 * kcr_t2c * M̄) )
# kcr̄_t2c3 = 𝓕⁻¹ * ( (eid * 𝓕 * kcr_t2c * -X[:,1] * λ') + (eid * 𝓕 * kcr_t2c * -λ * X[:,1]' ) )
# kcr̄_t2c4 = 𝓕⁻¹ * ( ( -(Nx * Ny * Nz) * e * λ') + (-λe * X[:,1]' ) )
# kcr̄_t2c5 =  𝓕⁻¹ * -( λe * X[:,1]' + (Nx * Ny * Nz) * e * λ' )
# kcr̄_t2c2 ≈ kcr̄_t2c1
# kcr̄_t2c3 ≈ kcr̄_t2c1
# kcr̄_t2c4 ≈ kcr̄_t2c1
# kcr̄_t2c5 ≈ kcr̄_t2c1
# kcr̄_t2c2 ./ kcr̄_t2c1

# # naive ε⁻¹_bar construction
dstar = conj.(d)
λdstar = conj.(λd)
D0 = real( (-λd .* dstar)) #-λd .* dstar
D1 = -λdstar[2:end] .* d[begin:end-1] + -λd[begin:end-1] .* dstar[2:end]
D2 = -λdstar[3:end] .* d[begin:end-2] + -λd[begin:end-2] .* dstar[3:end]
diag(eīd_eigs_herm,0) ≈ D0
diag(eīd_eigs_herm,1) ≈ D1
diag(eīd_eigs_herm,2) ≈ D2
@show maximum(real.(D0))
@show maximum(real.(D1))
@show maximum(real.(D2))
@show minimum(real.(D0))
@show minimum(real.(D1))
@show minimum(real.(D2))
# # eīd = eīd_eigs_herm + eīd_pe_herm
# eīd_5diag = diagm([diag_idx => diag(eīd,diag_idx) for diag_idx = -2:2]...)
# eīd_3diag = diagm([diag_idx => diag(eīd,diag_idx) for diag_idx = -1:1]...)
# eīd_1diag = diagm([diag_idx => diag(eīd,diag_idx) for diag_idx = 0]...)
# @assert eid_pb(eīd)[1] ≈ eid_pb(eīd_5diag)[1]
#
# @show p̄_H_eigs_5diag = eid_pb(eīd_5diag)[1]
# @show p̄_H_eigs_3diag = eid_pb(eīd_3diag)[1]
# @show p̄_H_eigs_1diag = eid_pb(eīd_1diag)[1]
# @show p̄_H_eigs_5diag_err = abs.(p̄_H_eigs .- p̄_H_eigs_5diag) ./ abs.(p̄_H_eigs)
# @show p̄_H_eigs_3diag_err = abs.(p̄_H_eigs .- p̄_H_eigs_3diag) ./ abs.(p̄_H_eigs)
# @show p̄_H_eigs_1diag_err = abs.(p̄_H_eigs .- p̄_H_eigs_1diag) ./ abs.(p̄_H_eigs)

##
println("####################################################################################")
println("")
println("End-to-end (parameters to group velocity) gradient calculation with OptiMode (implicit operators)")
println("")
println("####################################################################################")
# Zygote.refresh()
function nngω_rwg_OM(p::Vector{Float64} = p0;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 128, #16,
                    Ny = 128, #16,
                    Nz = 1,
                    band_idx = 1,
                    tol = 1e-8)
                    # kz, w, t_core, θ, n_core, n_subs, edge_gap = p
                    # nng_tuple = solve_nω(kz,ridge_wg(w,t_core,θ,edge_gap,n_core,n_subs,Δx,Δy),Δx,Δy,Δz,Nx,Ny,Nz;tol)
                    nng_tuple = solve_nω(p[1],ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),Δx,Δy,Δz,Nx,Ny,Nz;tol)
                    [nng_tuple[1],nng_tuple[2]]
end
n_rwg_OM(p) = nngω_rwg_OM(p)[1]
ng_rwg_OM(p) = nngω_rwg_OM(p)[2]
# @show n_OM, n_OM_pb = Zygote.pullback(n_rwg_OM,p0)

##
@show ng_OM, ng_OM_pb = Zygote.pullback(ng_rwg_OM,p0)
@show n_OM_err = abs(n - n_OM) / n
@show ng_OM_err = abs(ng - ng_OM) / ng
@show p̄_OM = real(ng_OM_pb(1)[1])
@show p̄_OM_err = abs.(p̄_FD .- p̄_OM) ./ abs.(p̄_FD)
##
println("####################################################################################")
println("")
println("End-to-end (parameters to group velocity) gradient calculation with OptiMode (implicit operators)")
println("")
println("####################################################################################")
p = p0
Δx = 6.0
Δy = 4.0
Δz = 1.0
Nx = 16
Ny = 16
Nz = 1
band_idx = 1
tol = 1e-8
# s, s_pb = Zygote.pullback(p) do p
#     ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy)
# end

(mag, mn), magmn_pb = Zygote.pullback(p0) do p
    calc_kpg(p[1],make_MG(Δx, Δy, Δz, Nx, Ny, Nz).g⃗)
end

eid,eid_pb = Zygote.pullback(p) do p
    shapes = ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy)
    make_εₛ⁻¹(shapes,make_MG(Δx,Δy,Δz,Nx,Ny,Nz))
    # εₛ⁻¹(s,make_MG(Δx,Δy,Δz,Nx,Ny,Nz))
end

(H,ω²), eigs_pb = Zygote.pullback(p,eid) do p,eid
    solve_ω²(p[1],eid,Δx,Δy,Δz;neigs=1,eigind=1,maxiter=3000,tol)
end

ng, ng_pb = Zygote.pullback(H,ω²,eid,mag,mn) do H,ω²,eid,mag,mn
    Ha = reshape(H,(2,Nx,Ny,Nz))
    √ω² / real( dot(H, -vec( kx_c2t( ifft( ε⁻¹_dot( fft( zx_t2c(Ha,mn), (2:4) ), eid), (2:4)),mn,mag) ) ) )
end

# ng_OM, ng_pb = Zygote.pullback(p,eid) do p,eid
#     solve_nω(p[1],eid,Δx,Δy,Δz,Nx,Ny,Nz;tol)[2]
# end
# p̄_ng, eid_bar = ng_pb(1)

H̄_ng,oms̄q_ng,eīd_ng,māg_ng,mn̄_ng = ng_pb(1)
p̄_eigs, eīd_eigs = eigs_pb((H̄_ng, oms̄q_ng))
@show p̄_eigs
@show p̄_eid_eigs = eid_pb(eīd_eigs)[1]
@show p̄_eid_ng = eid_pb(eīd_ng)[1]
@show p̄_magmn_ng = magmn_pb((māg_ng,mn̄_ng))[1]

eī_eigs_OM = eīd_eigs
p̄_eid_eigs_OM = p̄_eid_eigs

@show p̄ = real(p̄_magmn_ng + p̄_eigs + p̄_eid_eigs + p̄_eid_ng)
@show p̄_err = abs.(p̄_FD .- p̄) ./ abs.(p̄_FD)

##
using ArrayInterface, LoopVectorization
function ei_f2m1(ei_field,Nx,Ny,Nz)
    ei_matrix_buf = Zygote.bufferfrom(zeros(ComplexF64,(3*Nx*Ny*Nz),(3*Nx*Ny*Nz)))
    for i=1:Nx,j=1:Ny,a=1:3,b=1:3
        q = (Ny * (j-1) + i)
        ei_matrix_buf[(3*q-2)+a-1,(3*q-2)+b-1] = ei_field[a,b,i,j,1]
    end
    # return copy(ei_matrix_buf)
    return Hermitian(copy(ei_matrix_buf))
end

function ei_f2m2(ei_field,Nx,Ny,Nz)
    ei_matrix_buf = Zygote.bufferfrom(zeros(ComplexF64,(3*Nx*Ny*Nz),(3*Nx*Ny*Nz)))
    for i=1:Nx,j=1:Ny,a=1:3,b=1:3
        q = (Ny * (j-1) + i)
        ei_matrix_buf[(3*q-2)+a-1,(3*q-2)+b-1] = ei_field[a,b,i,j,1]
    end
    # return copy(ei_matrix_buf)
    return copy(ei_matrix_buf)
end

function ei_m2f1(ei_matrix,Nx,Ny,Nz)
    ei_field = zeros(Float64,(3,3,Nx,Ny,Nz))
    D0 = diag(ei_matrix,0)
    D1 = diag(ei_matrix,1)
    D2 = diag(ei_matrix,2)
    for i=1:Nx,j=1:Ny,k=1:Nz #,a=1:3,b=1:3
        q = (Nz * (k-1) + Ny * (j-1) + i) # (Ny * (j-1) + i)
        ei_field[1,1,i,j,k] = real(D0[3*q-2])
        ei_field[2,2,i,j,k] = real(D0[3*q-1] )
        ei_field[3,3,i,j,k] = real(D0[3*q])
        ei_field[1,2,i,j,k] = real(D1[3*q-2])
        ei_field[2,1,i,j,k] = real(conj(D1[3*q-2]))
        ei_field[2,3,i,j,k] = real(D1[3*q-1])
        ei_field[3,2,i,j,k] = real(conj(D1[3*q-1]))
        ei_field[1,3,i,j,k] = real(D2[3*q-2])
        ei_field[3,1,i,j,k] = real(conj(D2[3*q-2]))
        # ei_matrix[(3*q-2)+a-1,(3*q-2)+b-1] = ei_field[a,b,i,j,1]
    end
    return ei_field
end

function ei_m2f3(ei_matrix,Nx,Ny,Nz)
    ei_field = zeros(Float64,(3,3,Nx,Ny,Nz))
    @avx for a=1:3,b=1:3,k=1:Nz,j=1:Ny,i=1:Nx
        q = (Nz * (k-1) + Ny * (j-1) + i) # (Ny * (j-1) + i)
        ei_field[a,b,i,j,k] = ei_matrix[(3*q-2)+a-1,(3*q-2)+b-1]
    end
    return ei_field
end

function ei_m2f4(d,λd,Nx,Ny,Nz)
    # ei_field = Hermitian(zeros(Float64,(3,3,Nx,Ny,Nz)),"U")
    ei_field = zeros(Float64,(3,3,Nx,Ny,Nz))
    @avx for k=1:Nz,j=1:Ny,i=1:Nx
        q = (Nz * (k-1) + Ny * (j-1) + i) # (Ny * (j-1) + i)
        for a=1:3 # loop over diagonals
            ei_field[a,a,i,j,k] = real( -λd[3*q-2+a-1] * conj(d[3*q-2+a-1]) )
        end
        for a2=1:2 # loop over first off diagonal
            ei_field[a2,a2+1,i,j,k] = real( -conj(λd[3*q-2+a2]) * d[3*q-2+a2-1] - λd[3*q-2+a2-1] * conj(d[3*q-2+a2]) )
            ei_field[a2+1,a2,i,j,k] = ei_field[a2,a2+1,i,j,k]  # D1[3*q-2]
        end
        # a = 1, set 1,3 and 3,1, second off-diagonal
        ei_field[1,3,i,j,k] = real( -conj(λd[3*q]) * d[3*q-2] - λd[3*q-2] * conj(d[3*q]) )
        ei_field[3,1,i,j,k] =  ei_field[1,3,i,j,k]
    end
    return ei_field
end

##

eid_ref, eid_ref_pb = Zygote.pullback(x->ei_dot_rwg(x;Δx,Δy,Δz,Nx,Ny,Nz),p)
eīd_eigs = -λd * d'
eīd_eigs_herm = Zygote._hermitian_back(eīd_eigs,eid_ref.uplo)
@show p̄_eid_eigs_ref = eid_ref_pb(eīd_eigs_herm)[1]

@show p̄_eid_eigs_OM
eīd_eigs_OM1 = ei_f2m1(eī_eigs_OM,Nx,Ny,Nz)
eīd_eigs_OM2 = ei_f2m2(eī_eigs_OM,Nx,Ny,Nz)
eīd_eigs_OM3 = Zygote._hermitian_back(ei_f2m1(eī_eigs_OM,Nx,Ny,Nz),eid_ref.uplo)
eīd_eigs_OM4 = Zygote._hermitian_back(ei_f2m2(eī_eigs_OM,Nx,Ny,Nz),eid_ref.uplo)
@show p̄_eid_eigs_OM1 = eid_ref_pb(eīd_eigs_OM1)[1]
@show p̄_eid_eigs_OM2 = eid_ref_pb(eīd_eigs_OM2)[1]
@show p̄_eid_eigs_OM3 = eid_ref_pb(eīd_eigs_OM3)[1]
@show p̄_eid_eigs_OM4 = eid_ref_pb(eīd_eigs_OM4)[1]

##
xlim=(320,520)
dind = 1
plt = plot(real(diag(eīd_eigs_herm,dind)),label="ref_r",lw=3,alpha=0.5,xlim=xlim)
# plot!(imag(diag(eīd_eigs_herm,dind)),label="ref_i",lw=3,alpha=0.5,) #,xlim=xlim)
plot!(real(diag(eīd_eigs_OM4,dind)),label="OM4_r",ls=:dash)
# plot!(imag(diag(eīd_eigs_OM4,dind)),label="OM4_i",ls=:dash)
plot!(real(diag(eīd_eigs_OM2,dind)),label="OM2_r",ls=:dash)
# plot!(imag(diag(eīd_eigs_OM2,dind)),label="OM2_i",ls=:dash)

##
eīd_eigs_OM = ei_field2matrix(eīd_eigs,Nx,Ny,Nz)
eīd_eigs_OM_herm = Zygote._hermitian_back(eīd_eigs_OM,eīd_eigs_OM.uplo)
@show p̄_eid_eigs2 = eid_ref_pb(eīd_eigs_OM)[1]
@show p̄_eid_eigs3 = eid_ref_pb(eīd_eigs_OM_herm)[1]
eīd_eigs_OM ≈ eīd_eigs_herm
@show diag(eīd_eigs_OM,0)./diag(eīd_eigs_herm,0)
@show diag(eīd_eigs_OM,1)./diag(eīd_eigs_herm,1)
@show diag(eīd_eigs_OM,-1)./diag(eīd_eigs_herm,-1)
eīd_eigs ≈ eī_eigs_herm

##
ng_OM_pb(1)

##
using Revise
using ChainRules, Zygote, FiniteDifferences, OptiMode
p0 = [
    1.47,               #   propagation constant    `kz`            [μm⁻¹]
    1.5,                #   top ridge width         `w_top`         [μm]
    0.7,                #   ridge thickness         `t_core`        [μm]
    π / 10.0,           #   ridge sidewall angle    `θ`             [radian]
    2.4,                #   core index              `n_core`        [1]
    1.4,                #   substrate index         `n_subs`        [1]
    0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
]

function nngω_rwg_OM(p::Vector{Float64} = p0;
                    Δx = 6.0,
                    Δy = 4.0,
                    Δz = 1.0,
                    Nx = 64,
                    Ny = 64,
                    Nz = 1,
                    band_idx = 1,
                    tol = 1e-8)
                    # kz, w, t_core, θ, n_core, n_subs, edge_gap = p
                    # nng_tuple = solve_nω(kz,ridge_wg(w,t_core,θ,edge_gap,n_core,n_subs,Δx,Δy),Δx,Δy,Δz,Nx,Ny,Nz;tol)
                    nng_tuple = solve_nω(p[1],ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),Δx,Δy,Δz,Nx,Ny,Nz;tol)
                    [nng_tuple[1],nng_tuple[2]]
end

nngω_rwg_OM(p0)




real(Zygote.gradient(x->nngω_rwg_OM(x)[1],p0)[1])








real(Zygote.gradient(x->nngω_rwg_OM(x)[2],p0)[1])








FiniteDifferences.jacobian(central_fdm(3,1),x->nngω_rwg_OM(x),p0)[1]'

## Calculate ng and gradient by hand

# params
p = p0
Δx = 6.0
Δy = 4.0
Δz = 1.0
Nx = 128
Ny = 128
Nz = 1
band_idx = 1
tol = 1e-8

# fwd pass
ε⁻¹, ε⁻¹_pb = Zygote.pullback(p) do p
     make_εₛ⁻¹(ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy), make_MG(Δx,Δy,Δz,Nx,Ny,Nz))  # MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz))
end

Hω²,Hω²_pb = Zygote.pullback(p,ε⁻¹) do p,ε⁻¹
    solve_ω²(p[1],ε⁻¹,Δx,Δy,Δz;neigs=1,eigind=1,maxiter=3000,tol)
end

H = Hω²[1][:,1]
ω² = Hω²[2]

mag_mn, mag_mn_pb = Zygote.pullback(p) do p
    # calc_kpg(p[1],Δx,Δy,Δz,Nx,Ny,Nz)
    g⃗ = Zygote.@ignore([ [gx;gy;gz] for gx in collect(fftfreq(Nx,Nx/Δx)), gy in collect(fftfreq(Ny,Ny/Δy)), gz in collect(fftfreq(Nz,Nz/Δz))])
    calc_kpg(p[1],Zygote.dropgrad(g⃗))
end

kpg_mag, mn = mag_mn

MkH, MkH_pb = Zygote.pullback(Mₖ,H,ε⁻¹,mn,kpg_mag)

ng, ng_pb = Zygote.pullback(H,MkH,ω²) do H,MkH,ω²
    -sqrt(ω²) / real(dot(H,MkH))
end

# reverse pass
H̄_ng, Mk̄H_ng, ωs̄q_ng = ng_pb(1)
H̄_MkH, eī_MkH, mn̄_MkH, maḡ_MkH = MkH_pb(Mk̄H_ng)
p̄_mnmag = mag_mn_pb((maḡ_MkH,mn̄_MkH))[1]
p̄_mnmag = mag_mn_pb((maḡ_MkH,nothing))[1]
p̄_Hω², eī_Hω² = Hω²_pb(( H̄_MkH + H̄_ng , ωs̄q_ng ))
p̄_ε⁻¹ = ε⁻¹_pb( eī_Hω² + eī_MkH )[1]
p̄ = p̄_ε⁻¹ + p̄_Hω² + real(p̄_mnmag)


function f_kpg(p)
    # g⃗ = Zygote.@ignore( [ [gx;gy;gz] for gx in collect(fftfreq(Nx,Nx/Δx)), gy in collect(fftfreq(Ny,Ny/Δy)), gz in collect(fftfreq(Nz,Nz/Δz))] )
    g⃗ = [ [gx;gy;gz] for gx in fftfreq(Nx,Nx/Δx), gy in fftfreq(Ny,Ny/Δy), gz in fftfreq(Nz,Nz/Δz)]
    kpg_mag, mn = calc_kpg(p[1],Zygote.dropgrad(g⃗))
    # sum(abs2,mn[3,:,:,:])
    sum(kpg_mag)
end
f_kpg(p0)
f_kpg([1.85,1.7,0.7,0.2243994752564138,2.4,1.4,0.5])

gradient(f_kpg,p0)
FiniteDifferences.jacobian(central_fdm(3,1),f_kpg,p0)[1][1,:]


##





nngω_rwg_OM(p0)

FiniteDifferences.jacobian(central_fdm(3,1),x->nngω_rwg_OM(x),p0)[1]'








real(Zygote.gradient(x->nngω_rwg_OM(x)[1],p0)[1])








Zygote.gradient(x->nngω_rwg_OM(x)[2],p0)[1]


Zygote.refresh()

##

## configure swept-parameter data collection
ws = collect(0.8:0.1:1.7)
ts = collect(0.5:0.1:1.3)

@show nw = length(ws)
@show nt = length(ts)
np = length(p0)

p̄_AD = zeros(Float64,(nw,nt,np))
p̄_FD = zeros(Float64,(nw,nt,np))
p̄_SJ = zeros(Float64,(nw,nt,np))

for wind in 1:nw
    for tind in 1:nt
        ww = ws[wind]
        tt = ts[tind]
        pp = copy(p0)
        pp[2] = ww
        pp[3] = tt
        p̄_AD[wind,tind,:] = Zygote.gradient(solve_dense,pp)[1]
        p̄_FD[wind,tind,:] = FiniteDifferences.grad(central_fdm(2, 1),solve_dense,pp)[1]
        p̄_SJ[wind,tind,:] = ∂solve_dense_SJ(pp)
    end
end

## collect parameter sweeps

using HDF5, Dates
function write_sweep(sw_name;
                    data_dir="/home/dodd/data/OptiMode/grad_ng_p_rwg_dense/",
                    dt_fmt=dateformat"Y-m-d--H-M-S",
                    extension=".h5",
                    kwargs...)
    timestamp = Dates.format(now(),dt_fmt)
    fname = sw_name * "_" *  timestamp * extension
    @show fpath = data_dir * fname
    h5open(fpath, "cw") do file
        for (data_name,data) in kwargs
            write(file, string(data_name), data)
        end
    end
    return fpath
end

function read_sweep(sw_name;
                    data_dir="/home/dodd/data/OptiMode/grad_ng_p_rwg_dense/",
                    dt_fmt=dateformat"Y-m-d--H-M-S",
                    extension=".h5",
                    sw_keys=["ws","ts","p0","p̄_AD","p̄_FD","p̄_SJ"]
                    )
    # choose most recently timestamped file matching sw_name tag and extension
    fname = sort(  filter(x->(prod(split(x,"_")[begin:end-1])==prod(split(sw_name,"_"))),
                        readdir(data_dir));
                by=file->DateTime(split(file[begin:end-length(extension)],"_")[end],dt_fmt)
            )[end]
    @show fpath = data_dir * fname
    ds_data = h5open(fpath, "r") do file
        @show ds_keys = keys(file)
        ds_data = Dict([k=>read(file,k) for k in sw_keys]...)
    end
    return ds_data
end

fpath_test = write_sweep("wt";ws,ts,p0,p̄_AD,p̄_FD,p̄_SJ)
ds_test = read_sweep("wt")



##  plot data from parameter sweeps
#p̄_AD
zlabels = [ "∂ng/∂k [μm]", "∂ng/∂w [μm⁻¹]", "∂ng/∂t [μm⁻¹]", "∂ng/∂θ [rad⁻¹]", "∂ng/∂ncore", "∂ng/∂nsubs]", "∂ng/∂edge_gap [μm⁻¹]"]8
#surface(ts,ws,p̄_AD[:,:,3],xlabel="t [μm]",ylabel="w [μm]",zlabel="∂ng/∂t [μm⁻¹]")
plt_p̄_AD = [ surface(p̄_AD[:,:,ind],xlabel="t [μm]",ylabel="w [μm]",zlabel=zlabels[ind]) for ind=1:np ]
plt_p̄_FD = [ surface(p̄_FD[:,:,ind],xlabel="t [μm]",ylabel="w [μm]",zlabel=zlabels[ind]) for ind=1:np ]
plt_p̄_SJ = [ surface(p̄_SJ[:,:,ind],xlabel="t [μm]",ylabel="w [μm]",zlabel=zlabels[ind]) for ind=1:np ]
plt_p̄s = [plt_p̄_AD  ,  plt_p̄_FD , plt_p̄_SJ]
plt_p̄ = [ plt_p̄s[j][ind] for j=1:3,ind=1:np ] #vcat(plt_p̄_AD,plt_p̄_SJ,plt_p̄_FD)
l = @layout [   a   b   c
                d   e   f
                g   h   i
                j   k   l
                m   n   o
                p   q   r
                s   t   u  ]

p = plot(vec(plt_p̄)..., layout = l, size=(2000,1200))

##
p̄_AD = Zygote.gradient(solve_dense,p0)[1]
p̄_FD = FiniteDifferences.grad(central_fdm(2, 1),solve_dense,p0)[1]
p̄_SJ = ∂solve_dense_SJ(p0)

using Plots: plot, plot!, scatter, scatter!
pp = plot([-maximum(abs.(p̄_AD)),maximum(abs.(p̄_AD))],[-maximum(abs.(p̄_AD)),maximum(abs.(p̄_AD))],c=:black,label="y=x",legend=:bottomright)
scatter!(p̄_AD,p̄_FD,label="AD/FD")
scatter!(p̄_AD,p̄_SJ,label="AD/SJ")

ω² = real.(α)
plot(ω²,label="ω²",legend=:topleft)
ω = sqrt.(ω²)
neff = kz ./ ω
eig_ind = 1
H = reshape(X[:,eig_ind],(size(X,1),1))
plt_neff = plot(neff,label="neff",legend=:topright)
scatter!(plt_neff,neff[1:10],label="neff",legend=:topright)
grid = OptiMode.make_MG(Δx, Δy, Δz, Nx, Ny, Nz)
shapes = ridge_wg(w,t_core,edge_gap,n_core,n_subs,Δx,Δy)
ei_field = make_εₛ⁻¹(shapes,grid)
plot_ε(ei_field,grid.x,grid.y) #;cmap=cgrad(:viridis))
plot_d⃗(H,kz,grid)


df_p_8x8 = DataFrame(   p = p,
                        p̄_AD = p̄_AD,
                        p̄_FD = p̄_FD,
                        p̄_SJ = p̄_SJ,
                    )
name="M_entry_grads_rwg_8x8"
path="/home/dodd/github/OptiMode/test/"
CSV.write(*(path,name,".csv"), df_p_8x8)


#
#
# using Zygote: @adjoint
# @adjoint (T::Type{<:SArray})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
# @adjoint (T::Type{<:SArray})(x::AbstractArray) = T(x), dv -> (nothing, dv)
# @adjoint (T::Type{<:SMatrix})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
# @adjoint (T::Type{<:SMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
# @adjoint (T::Type{<:SVector})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
# @adjoint (T::Type{<:SVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)
# Zygote.refresh()


# p̄_AD = [Zygote.gradient(solve_dense,p)[1][begin:end-4]...]
# p̄_FD = FiniteDifferences.grad(central_fdm(2, 1),solve_dense,p)[1][begin:end-4]
# p̄_SJ = [∂solve_dense_SJ(p,α,X,ᾱ,X̄)[begin:end-4]...]

##


# heatmap(real(kxt2c))
# heatmap(imag(kxt2c))
# heatmap(real(F))
# heatmap(imag(F))
# heatmap(real(einv))
# heatmap(imag(einv))
# heatmap(real(Finv))
# heatmap(imag(Finv))
# heatmap(real(kxc2t))
# heatmap(imag(kxc2t))
# heatmap(real(M))
# heatmap(imag(M))
# kxt2c_op * ds.H⃗[:,1]
# ein̄v = (-kxc2t * Finv)' * M̄ * (F * kxt2c)'
# heatmap(real(ein̄v))
# heatmap(imag(ein̄v))

##















using StaticArrays
ei = make_εₛ⁻¹( ridge_wg(p0[2],p0[3],p0[4],p0[7],p0[5],p0[6],6.0,4.0), make_MG(6.,4.,1.,64,64,1) )
eiH = HybridArray{Tuple{3,3,StaticArrays.Dynamic(),StaticArrays.Dynamic(),StaticArrays.Dynamic()}}(ei)
eis1 = [ SMatrix{3,3,Float64,9}(ei[:,:,Ixyz]) for Ixyz in CartesianIndices(size(ei)[3:5]) ]
eish1 = [ SHermitianCompact{3,Float64,6}(ei[:,:,Ixyz]) for Ixyz in CartesianIndices(size(ei)[3:5]) ]

ei[:,:,42,32,1]



eir = reshape(ei,(9,64,64,1))
eir[:,42,32,1]









eis2 = reinterpret(reshape,SMatrix{3,3,Float64,9},eir)

eis1 ≈ eis2
eis2[42,32,1]




eish1 = [ SHermitianCompact{3,Float64,6}(ei[:,:,Ixyz]) for Ixyz in CartesianIndices(size(ei)[3:5]) ]
eis1r = reinterpret(Float64,eis1)
eish1r = reinterpret(Float64,eish1)

eis1rr = reinterpret(reshape,SMatrix{3,3,Float64,9},eis1r)

SMatrix{3,3,Float64,9}(ei[:,:,42,32,1])

reinterpret(SMatrix{3,3,Float64,9},ei)



































##

@btime ei_field2matrix($ei,$Nx,$Ny,$Nz) # 534.711 μs (3 allocations: 9.00 MiB)
@btime ei_matrix2field($eid,$Nx,$Ny,$Nz) # 11.129 μs (6 allocations: 54.50 KiB)
@btime ei_matrix2field2($(real(eid)),$Nx,$Ny,$Nz) # 9.772 μs (6 allocations: 36.50 KiB)
@btime ei_matrix2field3($(real(eid)),$Nx,$Ny,$Nz) # 5.862 μs (3 allocations: 18.12 KiB)
@btime ei_matrix2field4($d,$λd,$Nx,$Ny,$Nz) # 2.702 μs (3 allocations: 18.12 KiB)
# eīd1_L, eīd1_U, eīd1_rD = LowerTriangular(eīd1), UpperTriangular(eīd1), real.(Diagonal(eīd1))
# eīd1_Herm = eīd1_U .+ eīd1_L' - eīd1_rD
  # return uplo == 'U' ? U .+ L' - rD : L .+ U' - rD
# eīd1 = transpose(-kcr_c2t * 𝓕⁻¹) * M̄ * transpose(𝓕 * kcr_t2c)
if isnothing(eīd2)
    eīd2 = zeros(eltype(eīd1),size(eīd1))
end

# eīd2_L, eīd2_U, eīd2_rD = LowerTriangular(eīd2), UpperTriangular(eīd2), real.(Diagonal(eīd2))
# eīd2_Herm = eīd2_U .+ eīd2_L' - eīd2_rD

eīd_tot1 = Zygote._hermitian_back(eīd1,eid.uplo) + Zygote._hermitian_back(eīd2,eid.uplo) #eīd1 + eīd2
eīd_tot2 = Zygote._hermitian_back(Zygote.gradient(solve_dense_eidot,p,eid::Hermitian)[2],eid.uplo)
eīd_tot3 = Zygote._hermitian_back(eīd1+eīd2,eid.uplo) # eīd1_Herm + eīd2_Herm

##
plt = plot(real(diag(eīd_tot1,-1)),
                xlim=(280,520),
                c=:black,
                label="d-1,SJ_tot",
                legend=:bottomright,
                lw=2,
                alpha=0.5,
                )
plot!(real(diag(eīd_tot1,1)),
                c=:orange,
                label="d1,SJ_tot",
                lw=2,
                alpha=0.5,
                )
plot!(real(diag(eīd_tot2,-1)),c=:red,label="d-1,AD" )
plot!(real(diag(eīd_tot2,1)),c=:blue,label="d1,AD" )
plot!(real(diag(eīd_tot3,-1)),c=:green,label="d-1,SJHerm" )
plot!(real(diag(eīd_tot3,1)),c=:magenta,label="d1,SJHerm" )
##
plt = plot(real(diag(eīd_tot1,1))+real(diag(eīd_tot1,-1)),
                xlim=(280,520),
                c=:black,
                label="d-1,SJ_tot",
                legend=:bottomright,
                lw=2,
                alpha=0.5,
                )
plot!(real(diag(eīd_tot2,1))+real(diag(eīd_tot2,-1)) )
plot!(real(diag(eīd_tot3,1))+real(diag(eīd_tot3,-1)),ls=:dash,color=:green )
##
plt = plot(real(diag(eīd_tot1,1))+real(diag(eīd_tot1,-1)),
                xlim=(280,520),
                c=:black,
                label="d-1+d1,SJ_tot",
                legend=:bottomright,
                lw=2,
                alpha=0.5,
                )
plot!(real(diag(eīd1,1))+real(diag(eīd1,-1)) )
plot!((real(diag(eīd2,1))+real(diag(eīd2,-1))) )
plot!((real(diag(eīd1,1))+real(diag(eīd1,-1))) - (real(diag(eīd2,1))+real(diag(eīd2,-1))) )
plot!(real(diag(eīd_tot2,1))+real(diag(eīd_tot2,-1)) )


##
plt = plot(real(diag(eīd1,-1)),xlim=(280,520))
plot!(real(diag(eīd1,1)) )
# plot!(real(diag(eīd2,-1)) )
# plot!(real(diag(eīd2,1)) )
plot!(real(diag(eīd_tot2,-1)) )
plot!(real(diag(eīd_tot2,1)) )
plot!(real(diag(eīd_tot2,1))+real(diag(eīd_tot2,-1)) )

##
plt = plot(real(diag(eīd_tot1,0)),
                xlim=(280,520),
                c=:black,
                label="d0,SJ_tot",
                legend=:bottomright,
                lw=3,
                alpha=0.5,
                )
plot!(real(diag(eīd1,0)),c=:red,label="d0,SJ1")
plot!(real(diag(eīd2,0)),c=:blue,label="d0,SJ2")
plot!(real(diag(eīd_tot2,0)),c=:green,label="d0,AD_tot")
##

plt = plot(imag(diag(eīd1,-1)),xlim=(280,520))
plot!(imag(diag(eīd1,1)) )
plot!(imag(diag(eīd2,-1)) )
plot!(imag(diag(eīd2,1)) )
plot!(imag(diag(eīd_tot2,-1)) )
plot!(imag(diag(eīd_tot2,1)) )


@assert solve_dense_eidot(p,eid;Δx,Δy,Δz,Nx,Ny,Nz) ≈ solve_dense(p;Δx,Δy,Δz,Nx,Ny,Nz)
eīd_tot1 ≈ eīd_tot2
real(diag(eīd_tot1,0)) ≈ real(diag(eīd_tot2,0))
real(diag(eīd_tot1,1)) ≈ real(diag(eīd_tot2,1))
real(diag(eīd_tot1,-1)) ≈ real(diag(eīd_tot2,-1))
real(diag(eīd_tot1,-1)) ≈ -real(diag(eīd_tot2,1))

eīdot1 ≈ eīdot2
real(diag(eīdot1,0)) ≈ real(diag(eīdot2,0))
real(diag(eīdot1,1)) ≈ real(diag(eīdot2,1))

real(diag(eīdot3,0)) ≈ real(diag(eīdot2,0))
real(diag(eīdot3,1)) ≈ real(diag(eīdot2,1))
real(diag(eīdot3,1)) ≈ real(diag(eīdot1,1))

real(diag(eīdot3,2)) ≈ real(diag(eīdot3,-2))
real(diag(eīdot2,2)) ≈ real(diag(eīdot2,-2))
real(diag(eīdot2,1)) ≈ real(diag(eīdot2,-1))
# heatmap(real(ein̄v))
# heatmap(imag(ein̄v))
##
function compare_eīdot(diagind;figsize=(800,800),xlims=(300,525))
        plt_comp_r = plot(
                                real(diag(eīdot2,diagind)),
                                xlim=xlims,
                                c=:red,
                                linewidth=3,
                                label="eī1rd$diagind",
                        )
        plot!(real(diag(eīdot1,-diagind)),
                                xlim=xlims,
                                c=:purple,
                                linewidth=3,
                                label="eī1rd-$diagind",
                                )
        plot!(real(diag(eīdot3,diagind)),
                                xlim=xlims,
                                c=:black,
                                linewidth=1,
                                linestyle=:dash,
                                label="eī3rd$diagind",
                                )
        plot!(real(diag(eīdot3,-diagind)),
                                xlim=xlims,
                                c=:orange,
                                linewidth=1,
                                linestyle=:dash,
                                label="eī3rd-$diagind",
                                )

        plt_comp_i = plot(imag(diag(eīdot2,diagind)),
                                xlim=xlims,
                                c=:blue,
                                linewidth=3,
                                label="eī1id$diagind",
                                )
        plot!(imag(diag(eīdot1,-diagind)),
                                xlim=xlims,
                                c=:green,
                                linewidth=3,
                                label="eī1id-$diagind",
                                )
        plot!(imag(diag(eīdot3,diagind)),
                                xlim=xlims,
                                linewidth=1,
                                c=:black,
                                linestyle=:dash,
                                label="eī3id$diagind",
                                )
        plot!(imag(diag(eīdot3,-diagind)),
                                xlim=xlims,
                                linewidth=1,
                                c=:orange,
                                linestyle=:dash,
                                label="eī3id-$diagind",
                                )
    l = @layout [   a
                    b   ]
    plot(plt_comp_r,
        plt_comp_i,
        layout=l,
        size=figsize,
        )
end

compare_eīdot(1;figsize=(800,800),xlims=(340,380))



##




##

# function A_from_p(p)
#     N = Int(sqrt(length(p)))
#     A0 = Zygote.Buffer(zeros(ComplexF64,(N,N)))
#     for i = 1:N
#         for j=1:i
#             A0[j,i] = p[(sum(1:(i-1))+1)+j-1]
#             # println("ind: $((sum(1:(i-1))+1)+j -1)")
#         end
#     end
#     for i = 2:N
#         for j=1:i-1
#             A0[j,i] += p[ ((sum(1:N)) + (sum(1:(i-2))+1) + j -1) ]*im
#             # println("ind: $((sum(1:N)) + (sum(1:(i-1))+1) + j -1)")
#         end
#     end
#     A = Hermitian(copy(A0))
# end

# function A_from_p_RD(p)
#     N = Int(sqrt(length(p)))
#     A0 = zeros(ComplexF64,(N,N)) # Zygote.Buffer(zeros(ComplexF64,(N,N)))
#     for i = 1:N
#         for j=1:i
#             A0[j,i] = p[(sum(1:(i-1))+1)+j-1]
#             # println("ind: $((sum(1:(i-1))+1)+j -1)")
#         end
#     end
#     for i = 2:N
#         for j=1:i-1
#             A0[j,i] += p[ ((sum(1:N)) + (sum(1:(i-2))+1) + j -1) ]*im
#             # println("ind: $((sum(1:N)) + (sum(1:(i-1))+1) + j -1)")
#         end
#     end
#     A = Hermitian(A0)
# end
#
# function A_from_p_real(p)
#     N = Int(sqrt(length(p)))
#     Ar = zeros(eltype(p),(N,N))
#     for i = 1:N
#         for j=1:i
#             Ar[j,i] = p[(sum(1:(i-1))+1)+j-1]
#             # println("ind: $((sum(1:(i-1))+1)+j -1)")
#         end
#     end
#     Ar
# end
#
# function A_from_p_imag(p)
#     N = Int(sqrt(length(p)))
#     Ai = zeros(eltype(p),(N,N))
#     for i = 2:N
#         for j=1:i-1
#             Ai[j,i] += p[ ((sum(1:N)) + (sum(1:(i-2))+1) + j -1) ]
#             # println("ind: $((sum(1:N)) + (sum(1:(i-1))+1) + j -1)")
#         end
#     end
#     Ai
# end
#
# function foo(p)
#     # A = Hermitian(diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:N]...) + im*diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:(N-1)]...))
#     A = Hermitian(reshape(p,(2048,2048)))#A_from_p(p)
#     F = eigen(A)
#     α = F.values
#     X = F.vectors
#     Xone = Zygote.@showgrad(X[:,1])
#     αone = Zygote.@showgrad(α[1])
#     # sum(x->abs2(x)^6,Xone) * αone^2
#     proc_eigs(Xone,αone)
# end
#
#
# function ∂foo_SJ(p,α,X,ᾱ,X̄;i=1)
#     # A = Hermitian(diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:N]...) + im*diagm([ N-nn => p[(sum(1:(nn-1))+1:sum(1:nn))] for nn=1:(N-1)]...))
#     # A = A_from_p(p)
#     # A, Ap_pb = Zygote.pullback(A_from_p,p)
#     A, Ap_pb = Zygote.pullback(x->Hermitian(reshape(x,(2048,2048))),p)
#     α,X = eigen(A)
#     X̄,ᾱ = Zygote.gradient(proc_eigs,X[:,1],α[1])
#     P = I - X[:,i] * X[:,i]'
#     b = P * X̄ #[i]
#     λ₀ = IterativeSolvers.bicgstabl(A-α[i]*I,b,3)
#     if isnothing(ᾱ)
#         ᾱ = 0.
#     end
#     λ = λ₀ - ᾱ * X[:,i]
#     Ā = -λ * X[:,i]'
#     Ap_pb(Ā)
# end



##

using ReverseDiff #: JacobianTape, JacobianConfig, jacobian, jacobian!, compile
# using LinearAlgebra: mul!

#########
# setup #
#########

# some objective functions to work with
f(a, b) = (a + b) * (a * b)'
g!(out, a, b) = mul!(out, a + b, a * b)

# pre-record JacobianTapes for `f` and `g` using inputs of shape 10x10 with Float64 elements
const f_tape = ReverseDiff.JacobianTape(f, (rand(10, 10), rand(10, 10)))
const g_tape = ReverseDiff.JacobianTape(g!, rand(10, 10), (rand(10, 10), rand(10, 10)))

# compile `f_tape` and `g_tape` into more optimized representations
const compiled_f_tape = ReverseDiff.compile(f_tape)
const compiled_g_tape = ReverseDiff.compile(g_tape)

# some inputs and work buffers to play around with
a, b = rand(10, 10), rand(10, 10)
inputs = (a, b)
output = rand(10, 10)
results = (similar(a, 100, 100), similar(b, 100, 100))
fcfg = ReverseDiff.JacobianConfig(inputs)
gcfg = ReverseDiff.JacobianConfig(output, inputs)

####################
# taking Jacobians #
####################

# with pre-recorded/compiled tapes (generated in the setup above) #
#-----------------------------------------------------------------#

# these should be the fastest methods, and non-allocating
ReverseDiff.jacobian!(results, compiled_f_tape, inputs)
ReverseDiff.jacobian!(results, compiled_g_tape, inputs)
ReverseDiff.jacobian!(results, f_tape, inputs)
ReverseDiff.jacobian!(results, g_tape, inputs)

const A_from_p_tape = ReverseDiff.JacobianTape(A_from_p, p)
const A_from_p_RD_tape = ReverseDiff.JacobianTape(A_from_p_RD, p)
const A_from_p_real_tape = ReverseDiff.JacobianTape(A_from_p_real, p)
const A_from_p_imag_tape = ReverseDiff.JacobianTape(A_from_p_imag, p)
A_from_p_real_cfg = ReverseDiff.JacobianConfig(p)

inputs = (p,)
results = (similar(p))
ReverseDiff.jacobian!(results, A_from_p_real_tape, p)
ReverseDiff.jacobian(A_from_p_RD, p)
ReverseDiff.jacobian(A_from_p_real, p)

f(rand(10, 10), rand(10, 10))

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

################################################################################
################################################################################
################################################################################
################################################################################

function ftest1(p = p0;
                Δx = 6.0,
                Δy = 4.0,
                Δz = 1.0,
                Nx = 16,
                Ny = 16,
                Nz = 1)
    kz, w, t_core, θ, n_core, n_subs, edge_gap = p
    # kz=p[1]; w=p[2]; t_core=p[3]; θ=p[4]; n_core=p[5]; n_subs=p[6]; edge_gap=p[7]
    # grid = OptiMode.make_MG(Δx, Δy, Δz, Nx, Ny, Nz)
    shapes = ridge_wg(w,t_core,θ,edge_gap,n_core,n_subs,Δx,Δy)
    # grid = Zygote.@ignore OptiMode.make_MG(6.0,4.0,1.0,16,16,1)
    # shapes = ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],6.0,4.0)
    # sum(abs.(shapes[1].v))
    # ei_field = make_εₛ⁻¹(shapes,grid)
    ei_field = make_ei1(shapes,Δx, Δy, Δz, Nx, Ny, Nz)
    sum(ei_field)
end
using ArrayInterface
ftest1(p0)
gradient(ftest1,p0)
using ReverseDiff
# ForwardDiff.gradient(ftest1,p0)
y,back =  _pullback(Context(),ftest1,p0)
@code_typed _pullback(Context(),ftest1,p0)
##
# function fp(v::SMatrix{K,2,<:Real}, data::D=nothing) where {K,D}
function fp(v::AbstractMatrix{<:Real}, data::D=nothing) where D
    K = size(v,1);
    @show typeof(v)
    # v = SMatrix{K,2}(vin)
    # v = copy(vin)
    # Sort the vertices in the counter-clockwise direction
    @show w = v .- mean(v, dims=1)  # v in center-of-mass coordinates
    ϕ = mod.(atan.(w[:,2], w[:,1]), 2π)  # SVector{K}: angle of vertices between 0 and 2π; `%` does not work for negative angle
    if !issorted(ϕ)
        # Do this only when ϕ is not sorted, because the following uses allocations.
        ind = MVector{K}(sortperm(ϕ))  # sortperm(::SVector) currently returns Vector, not MVector
        @show v = v[ind,:]  # SVector{K}: sorted v
    end

    # Calculate the increases in angle between neighboring edges.
    # ∆v = vcat(diff(v, dims=1), SMatrix{1,2}(v[1,:]-v[end,:]))  # SMatrix{K,2}: edge directions
    @show ∆v = vcat(diff(v, dims=1), transpose(v[1,:]-v[end,:]))
    ∆z = ∆v[:,1] + im * ∆v[:,2]  # SVector{K}: edge directions as complex numbers
    icurr = ntuple(identity, Val(K-1))
    inext = ntuple(x->x+1, Val(K-1))
    ∆ϕ = angle.(∆z[SVector(inext)] ./ ∆z[SVector(icurr)])  # angle returns value between -π and π

    # Check all the angle increases are positive.  If they aren't, the polygon is not convex.
    #@assert all(∆ϕ .> 0) #|| throw("v = $v should represent vertices of convex polygon.")

    n0 = [∆v[:,2] -∆v[:,1]]  # outward normal directions to edges
    # norms = SVector{K,Float64}([hypot(n0[vind,1],n0[vind,2]) for vind=1:K])
    @show n = n0 ./ hypot.(n0[:,1],n0[:,2])  # normalize
    # n0_norm = sqrt.( n0[:,1].^2 + n0[:,2].^2  )
    # nM =  Matrix(n0) ./ Vector(n0_norm)
    # n = SMatrix(nM)
    # nt = [normalize(n0[vind,:])[cind] for cind=1:2,vind=1:4]
    @show typeof(v)
    return Polygon{K,2K,D}(v,(SMatrix{4,2,Float64}(n)),data)
    # return SMatrix{K,2}(n) #Polygon{K,2K,D}(v,n,data)
end

fp(v1,6.3).n[2,2]
fp(v0,6.3)
Zygote.gradient(v -> v[1], SVector(5,5))[1]
Zygote.gradient((a,b) -> SVector(a,b)[1], 5,5)
Zygote.gradient((a,b) -> sum(SVector(a,b)), 5,5)

SMatrix{2,2,Float64,4}(1,2,3,4)
Zygote.gradient(m->m[1,1], SMatrix{2,2,Float64,4}(1,2,3,4))
gradient(v1,6.3) do a,b
    nn = fp(a,b).n[2,2] #.n
    # nn[1,1]
    #sum(abs.())
end

fp(v0,6.3)

v0 = [  0.85    -0.85   -1.     1.
        0.35    0.35    -0.35   -0.35   ]'

v1 = [  0.85     0.35
        -0.85    0.35
        -1.      -0.35
        1.       -0.35  ]
v1 ≈ v0
∆v = vcat(diff(v0, dims=1), SMatrix{1,2}(v0[1,:]-v0[end,:]))
∆v2 = vcat(diff(v0, dims=1), (v0[1,:]-v0[end,:])')
∆v2 ≈ ∆v
(v0[1,:]-v0[end,:])'
fp(v0,6.3)
fp(v0,6.3).v[1,1]

n1 = @SMatrix [ 0.      1.7
                -0.7    0.15
                0.      -2.
                0.7     0.15    ]

n1[:,1], n1[:,2]

hypot.(n1[:,1], n1[:,2])

n2M = Matrix(n1) ./ Vector(hypot.(n1[:,1],n1[:,2]))
n2SM = SMatrix{4,2,Float64,8}(n2M)
n1 ./ sqrt.( n1[:,1].^2 + n1[:,2].^2  )
@assert sqrt.( n1[:,1].^2 + n1[:,2].^2  ) ≈ hypot.(n1[:,1],n1[:,2])

abs2.(n1)

Zygote.@adjoint (T::Type{<:SMatrix})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
Zygote.@adjoint (T::Type{<:SMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)


Zygote.refresh()


##

using GeometryPrimitives
using OptiMode: make_KDTree
using Zygote: dropgrad
function make_ei1(shapes::Vector{<:GeometryPrimitives.Shape}, Δx, Δy, Δz, Nx, Ny, Nz)::Array{Float64,5}
    tree = make_KDTree(shapes)
    δx = dropgrad(Δx) / dropgrad(Nx)    # δx
    δy = dropgrad(Δy) / dropgrad(Ny)    # δy
    x = ( ( dropgrad(Δx) / dropgrad(Nx) ) .* (0:(dropgrad(Nx)-1))) .- dropgrad(Δx)/2.  # x
    y = ( ( dropgrad(Δy) / dropgrad(Ny) ) .* (0:(dropgrad(Ny)-1))) .- dropgrad(Δy)/2.  # y
    ebuf = Zygote.Buffer(Array{Float64}([1.0 2.0]),3,3,dropgrad(Nx),dropgrad(Ny),1)
    # for i=1:dropgrad(Nx),j=1:dropgrad(Ny),kk=1:dropgrad(Nz)
        # ebuf[:,:,i,j,kk] = inv(εₛ(shapes,dropgrad(tree),dropgrad(x[i]),dropgrad(y[j]),dropgrad(δx),dropgrad(δy)))
    for a=1:3,b=1:3,i=1:dropgrad(Nx),j=1:dropgrad(Ny),kk=1:dropgrad(Nz)
        ebuf[a,b,i,j,kk] = inv(εₛ(shapes,dropgrad(tree),dropgrad(x[i]),dropgrad(y[j]),dropgrad(δx),dropgrad(δy)))[a,b]
    end
    return real(copy(ebuf))
end

##
@assert typeof(ridge_wg(p0[2],p0[3],p0[4],p0[7],p0[5],p0[6],6.0,4.0))<:Vector{<:GeometryPrimitives.Shape}
ftest(p0)
Zygote.gradient(ftest,p0)

##
