using ArrayInterface, StructArrays, StaticArrays, HybridArrays, LinearAlgebra, LoopVectorization, BenchmarkTools, ChainRules, Zygote, ForwardDiff, Tullio #, CUDA

## instantiate arrays of various types to test sub-components of Maxwell Operator function
Nx = 100
Ny = 100
Nz = 1
# vanilla Arrays
H = rand(Float64,2,Nx,Ny,Nz)
d = rand(Float64,3,Nx,Ny,Nz)
ei = rand(Float64,3,3,Nx,Ny,Nz)
e = rand(Float64,3,Nx,Ny,Nz) #similar(d)
# Array{SVec} & Array{SMatrix}
Hs = [SVector{2,Float64}(H[:,i,j,k]) for i=1:Nx,j=1:Ny,k=1:Nz]
ds = [SVector{3,Float64}(d[:,i,j,k]) for i=1:Nx,j=1:Ny,k=1:Nz]
eis = [SMatrix{3,3,Float64,9}(ei[:,:,i,j,k]) for i=1:Nx,j=1:Ny,k=1:Nz]
es = similar(ds)

Hs2 = reinterpret(reshape, SVector{2,Float64}, H)
ds2 = reinterpret(reshape, SVector{3,Float64}, d)
eis2 = reinterpret(reshape, SMatrix{3,3,Float64,9}, reshape(ei,(9,Nx,Ny,Nz)))
es2 = reinterpret(reshape, SVector{3,Float64}, e)


# CuArrays
# eic = cu(ei)
# dc = cu(d)
# ec = similar(dc)
# HybridArrays
Hh = HybridArray{Tuple{StaticArrays.Dynamic(),StaticArrays.Dynamic(),StaticArrays.Dynamic(),2}}(copy(permutedims(d,(2,3,4,1))))
dh = HybridArray{Tuple{3,StaticArrays.Dynamic(),StaticArrays.Dynamic(),StaticArrays.Dynamic()}}(d)
eih = HybridArray{Tuple{3,3,StaticArrays.Dynamic(),StaticArrays.Dynamic(),StaticArrays.Dynamic()}}(ei)
eh = similar(dh)

dh2 = HybridArray{Tuple{StaticArrays.Dynamic(),StaticArrays.Dynamic(),StaticArrays.Dynamic(),3}}(copy(permutedims(d,(2,3,4,1))))
eih2 = HybridArray{Tuple{StaticArrays.Dynamic(),StaticArrays.Dynamic(),StaticArrays.Dynamic(),3,3}}(copy(permutedims(eih,(3,4,5,1,2))))
eh2 = similar(dh2)


@btime @tullio $e[a,i,j,k] =  $ei[a,b,i,j,k] * $d[b,i,j,k]; # 34.657 μs (0 allocations: 0 bytes)
@btime @tullio $eh[a,i,j,k] =  $eih[a,b,i,j,k] * $dh[b,i,j,k] ; # 192.462 μs (6 allocations: 144 bytes)
@btime @tullio $eh.data[a,i,j,k] =  $eih.data[a,b,i,j,k] * $dh.data[b,i,j,k] ; # 192.462 μs (6 allocations: 144 bytes)
@btime @tullio $eh2[i,j,k,a] = $eih2[i,j,k,a,b] * $dh2[i,j,k,b] ; # 187.520 μs (5 allocations: 128 bytes)
@btime @tullio $eh2.data[i,j,k,a] = $eih2.data[i,j,k,a,b] * $dh2.data[i,j,k,b] ; # 187.520 μs (5 allocations: 128 bytes)
@btime @tullio $es[i,j,k] = $eis[i,j,k] * $ds[i,j,k] ; # 42.807 μs (0 allocations: 0 bytes)
@btime @. $es = $eis * $ds; # 22.849 μs (0 allocations: 0 bytes)
@btime @avx @. $es = $eis * $ds; # 21.716 μs (0 allocations: 0 bytes)
@btime map!(*,$es,$eis,$ds); # 21.329 μs (0 allocations: 0 bytes)
@btime vmap!(*,$es,$eis,$ds); # 21.329 μs (0 allocations: 0 bytes)
@btime vmapt!(*,$es,$eis,$ds); # 21.327 μs (0 allocations: 0 bytes)
@btime vmapnt!(*,$es,$eis,$ds); # 21.327 μs (0 allocations: 0 bytes)
@btime vmapntt!(*,$es,$eis,$ds); # 21.318 μs (0 allocations: 0 bytes)

@btime @. $es2 = $eis2 * $ds2; # 1.420 ms (20008 allocations: 625.48 KiB)
@btime @avx @. $es2 = $eis2 * $ds2; # 1.376 ms (20008 allocations: 625.48 KiB)
@btime map!(*,$es2,$eis2,$ds2); # 21.374 μs (0 allocations: 0 bytes)
@btime vmap!(*,$es2,$eis2,$ds2); # 21.329 μs (0 allocations: 0 bytes)
@btime vmapt!(*,$es2,$eis2,$ds2); # 21.327 μs (0 allocations: 0 bytes)v
@btime vmapnt!(*,$es2,$eis2,$ds2); # 21.327 μs (0 allocations: 0 bytes)
@btime vmapntt!(*,$es2,$eis2,$ds2); # 21.318 μs (0 allocations: 0 bytes)

function many3x3inverts_avx2!(Y, X)
    @assert size(Y) === size(X)
    @assert size(Y,4) == size(Y,5) === 3
    @avx for k ∈ axes(Y,3), j ∈ axes(Y,2), i ∈ axes(Y,1), h in 0:0, l in 0:0

        X₁₁ = X[i,j,k,1+l,1+h]
        X₂₁ = X[i,j,k,2+l,1+h]
        X₃₁ = X[i,j,k,3+l,1+h]
        X₁₂ = X[i,j,k,1+l,2+h]
        X₂₂ = X[i,j,k,2+l,2+h]
        X₃₂ = X[i,j,k,3+l,2+h]
        X₁₃ = X[i,j,k,1+l,3+h]
        X₂₃ = X[i,j,k,2+l,3+h]
        X₃₃ = X[i,j,k,3+l,3+h]

        Y₁₁ = X₂₂*X₃₃ - X₂₃*X₃₂
        Y₂₁ = X₂₃*X₃₁ - X₂₁*X₃₃
        Y₃₁ = X₂₁*X₃₂ - X₂₂*X₃₁

        Y₁₂ = X₁₃*X₃₂ - X₁₂*X₃₃
        Y₂₂ = X₁₁*X₃₃ - X₁₃*X₃₁
        Y₃₂ = X₁₂*X₃₁ - X₁₁*X₃₂

        Y₁₃ = X₁₂*X₂₃ - X₁₃*X₂₂
        Y₂₃ = X₁₃*X₂₁ - X₁₁*X₂₃
        Y₃₃ = X₁₁*X₂₂ - X₁₂*X₂₁

        d = 1 / ( X₁₁*Y₁₁ + X₁₂*Y₂₁ + X₁₃*Y₃₁ )

        Y[i,j,k,1+l,1+h] = Y₁₁ * d
        Y[i,j,k,2+l,1+h] = Y₂₁ * d
        Y[i,j,k,3+l,1+h] = Y₃₁ * d
        Y[i,j,k,1+l,2+h] = Y₁₂ * d
        Y[i,j,k,2+l,2+h] = Y₂₂ * d
        Y[i,j,k,3+l,2+h] = Y₃₂ * d
        Y[i,j,k,1+l,3+h] = Y₁₃ * d
        Y[i,j,k,2+l,3+h] = Y₂₃ * d
        Y[i,j,k,3+l,3+h] = Y₃₃ * d

    end
end

eh2 = similar(eih2)
es = similar(eis)
vmap!(inv,es,eis)
ei2 = permutedims(ei,(3,4,5,1,2))
eii2 = similar(ei2)

many3x3inverts_avx2!(eh2, eih2)
many3x3inverts_avx2!(eii2, ei2)
@btime vmap!(inv,$es, $eis) # 70.073 μs (0 allocations: 0 bytes)
@btime many3x3inverts_avx2!($eh2, $eih2) # 24.790 μs (0 allocations: 0 bytes)
@btime many3x3inverts_avx2!($eii2, $ei2) # 24.663 μs (0 allocations: 0 bytes)

##
using BenchmarkTools, StaticArrays, LoopVectorization
using OptiMode: calc_kpg, MaxwellGrid
Δx = 6.0
Δy = 4.0
Δz = 1.0
Nx = 100
Ny = 100
Nz = 1
k = 1.6
Neigs = 1

T = Float64
ε⁻¹ = rand(Float64,3,3,Nx,Ny,Nz)
# H⃗ = randn(Complex{T},(2*Nx*Ny*Nz,Neigs))
# H = randn(Complex{T},(2,Nx,Ny,Nz))
# e = randn(Complex{T},(3,Nx,Ny,Nz))
# d = randn(Complex{T},(3,Nx,Ny,Nz))
H⃗ = randn(T,(2*Nx*Ny*Nz,Neigs))
H = randn(T,(2,Nx,Ny,Nz))
e = randn(T,(3,Nx,Ny,Nz))
d = randn(T,(3,Nx,Ny,Nz))
mag,mn = calc_kpg(k,Δx,Δy,Δz,Nx,Ny,Nz)
m = copy(mn[:,1,:,:,:])
n = copy(mn[:,2,:,:,:])
@btime calc_kpg($k,$Δx,$Δy,$Δz,$Nx,$Ny,$Nz) # 1.452 ms (30047 allocations: 4.89 MiB)

Hs = reinterpret(reshape, SVector{2,T}, H)
ds = reinterpret(reshape, SVector{3,T}, d)
ε⁻¹s = reinterpret(reshape, SMatrix{3,3,T,9}, reshape(ε⁻¹,(9,Nx,Ny,Nz)))
es = reinterpret(reshape, SVector{3,T}, e)
ms = reinterpret(reshape, SVector{3,T}, m)
ns = reinterpret(reshape, SVector{3,T}, n)

# H⃗2 = permutedims(H⃗,(2,3,4,1))
ε⁻¹2 = permutedims(ε⁻¹,(3,4,5,1,2))
H2 = permutedims(H,(2,3,4,1))
e2 = permutedims(e,(2,3,4,1))
d2 = permutedims(d,(2,3,4,1))
mn2 = permutedims(mn,(3,4,5,2,1))
m2 = copy(mn2[:,:,:,1,:])
n2 = copy(mn2[:,:,:,2,:])


function kx_tc1!(d,H,mn,mag)
    @fastmath @inbounds for i ∈ axes(d,2), j ∈ axes(d,3), k ∈ axes(d,4)
        @fastmath @inbounds d[1,i,j,k] = ( H[1,i,j,k] * mn[1,2,i,j,k] - H[2,i,j,k] * mn[1,1,i,j,k] ) * -mag[i,j,k]
        @fastmath @inbounds d[2,i,j,k] = ( H[1,i,j,k] * mn[2,2,i,j,k] - H[2,i,j,k] * mn[2,1,i,j,k] ) * -mag[i,j,k]
        @fastmath @inbounds d[3,i,j,k] = ( H[1,i,j,k] * mn[3,2,i,j,k] - H[2,i,j,k] * mn[3,1,i,j,k] ) * -mag[i,j,k]
    end
    return d
end

function kx_tc2!(d,H,mn,mag)
    # @assert size(Y) === size(X)
    @assert size(d,4) == 3
    @assert size(H,4) === 2
    @avx for k ∈ axes(d,3), j ∈ axes(d,2), i ∈ axes(d,1), l in 0:0
        d[i,j,k,1+l] = ( H[i,j,k,1] * mn[i,j,k,2,1+l] - H[i,j,k,2] * mn[i,j,k,1,1+l] ) * -mag[i,j,k]
        d[i,j,k,2+l] = ( H[i,j,k,1] * mn[i,j,k,2,2+l] - H[i,j,k,2] * mn[i,j,k,1,2+l] ) * -mag[i,j,k]
        d[i,j,k,3+l] = ( H[i,j,k,1] * mn[i,j,k,2,3+l] - H[i,j,k,2] * mn[i,j,k,1,3+l] ) * -mag[i,j,k]
    end
    return d
end

function kx_tc3!(d,H,mn,mag)
    # @assert size(Y) === size(X)
    @assert size(d,1) == 3
    @assert size(H,1) === 2
    @avx for k ∈ axes(d,4), j ∈ axes(d,3), i ∈ axes(d,2), l in 0:0
        d[1+l,i,j,k] = ( H[1,i,j,k] * mn[1+l,2,i,j,k] - H[2,i,j,k] * mn[1+l,1,i,j,k] ) * -mag[i,j,k]
        d[2+l,i,j,k] = ( H[1,i,j,k] * mn[2+l,2,i,j,k] - H[2,i,j,k] * mn[2+l,1,i,j,k] ) * -mag[i,j,k]
        d[3+l,i,j,k] = ( H[1,i,j,k] * mn[3+l,2,i,j,k] - H[2,i,j,k] * mn[3+l,1,i,j,k] ) * -mag[i,j,k]
    end
    return d
end

function kx_tc4!(d,H,m,n,mag)
    # @assert size(Y) === size(X)
    @assert size(d,4) == 3
    @assert size(H,4) === 2
    @avx for k ∈ axes(d,3), j ∈ axes(d,2), i ∈ axes(d,1), l in 0:0
        d[i,j,k,1+l] = ( H[i,j,k,1] * n[i,j,k,1+l] - H[i,j,k,2] * m[i,j,k,1+l] ) * -mag[i,j,k]
        d[i,j,k,2+l] = ( H[i,j,k,1] * n[i,j,k,2+l] - H[i,j,k,2] * m[i,j,k,2+l] ) * -mag[i,j,k]
        d[i,j,k,3+l] = ( H[i,j,k,1] * n[i,j,k,3+l] - H[i,j,k,2] * m[i,j,k,3+l] ) * -mag[i,j,k]
    end
    return d
end

function kx_ct1!(H,e,m,n,mag)
    # @assert size(Y) === size(X)
    @assert size(e,4) == 3
    @assert size(H,4) === 2
    @avx for k ∈ axes(H,3), j ∈ axes(H,2), i ∈ axes(H,1), l in 0:0
        H[i,j,k,1+l] =  (	e[i,j,k,1+l] * n[i,j,k,1+l] + e[i,j,k,2+l] * n[i,j,k,2+l] + e[i,j,k,3+l] * n[i,j,k,3+l]	) * -mag[i,j,k]
		H[i,j,k,2+l] =  (	e[i,j,k,1+l] * m[i,j,k,1+l] + e[i,j,k,2+l] * m[i,j,k,2+l] + e[i,j,k,3+l] * m[i,j,k,3+l]	) * mag[i,j,k]
    end
    return H
end

function eid1!(e,ei,d)
    @assert size(e,4) === 3
    @assert size(d,4) === 3
    @assert size(ei,4) === 3
    @assert size(ei,5) === 3
    @avx for k ∈ axes(e,3), j ∈ axes(e,2), i ∈ axes(e,1), l in 0:0, h in 0:0
        e[i,j,k,1+h] =  ei[i,j,k,1+h,1+l]*d[i,j,k,1+l] + ei[i,j,k,2+h,1+l]*d[i,j,k,2+l] + ei[i,j,k,3+h,1+l]*d[i,j,k,3+l]
        e[i,j,k,2+h] =  ei[i,j,k,1+h,2+l]*d[i,j,k,1+l] + ei[i,j,k,2+h,2+l]*d[i,j,k,2+l] + ei[i,j,k,3+h,2+l]*d[i,j,k,3+l]
        e[i,j,k,3+h] =  ei[i,j,k,1+h,3+l]*d[i,j,k,1+l] + ei[i,j,k,2+h,3+l]*d[i,j,k,2+l] + ei[i,j,k,3+h,3+l]*d[i,j,k,3+l]
    end
    return e
end

kx_tc1!(d,H,mn,mag)
kx_tc2!(d2,H2,mn,mag)
kx_tc3!(d,H,mn,mag)
kx_tc4!(d2,H2,m2,n2,mag)

kx_ct1!(H2,e2,m2,n2,mag)

eid1!(e2,ε⁻¹2,d2)


# kx_t2c
@btime kx_tc1!($d,$H,$mn,$mag) # 74.970 μs (0 allocations: 0 bytes) real only: 52.685 μs
@btime kx_tc2!($d2,$H2,$mn2,$mag) # 26.336 μs (0 allocations: 0 bytes) real only: 14.938 μs
@btime kx_tc3!($d,$H,$mn,$mag) # real only: 30.481 μs
@btime kx_tc4!($d2,$H2,$m2,$n2,$mag) # real only: 14.872 μs (0 allocations: 0 bytes)

# kx_c2t
@btime kx_ct1!($H2,$e2,$m2,$n2,$mag) # real only: 13.784 μs (0 allocations: 0 bytes)


# eps_inv_dot
vmap!(*,es,ε⁻¹s,ds)
@btime vmap!(*,$es,$ε⁻¹s,$ds) # real only: 22.090 μs (0 allocations: 0 bytes)
@btime eid1!($e2,$ε⁻¹2,$d2) # real only: 15.297 μs (0 allocations: 0 bytes)



# FFTs
using LinearAlgebra, FFTW

Nx,Ny,Nz = 100,100,1
FFTW.set_num_threads(4)
T = Float64
e = randn(Complex{T},(3,Nx,Ny,Nz))
d = randn(Complex{T},(3,Nx,Ny,Nz))
e2 = permutedims(e,(2,3,4,1))
d2 = permutedims(d,(2,3,4,1))
e3 = copy(e2[:,:,1,:])
d3 = copy(d2[:,:,1,:])
𝓕4! = plan_fft!(e2,(1:3),flags=FFTW.PATIENT)
𝓕⁻¹4! = plan_bfft!(d2,(1:3),flags=FFTW.PATIENT)
mul!(d2,𝓕4!,d2)
mul!(e2,𝓕⁻¹4!,e2)
@btime mul!($d2,$𝓕4!,$d2)      # 112.357 μs (0 allocations: 0 bytes)
@btime mul!($e2,$𝓕⁻¹4!,$e2)

function fftw_benchmark(Nx,Ny,Nz;T=Float64)
    e = randn(Complex{T},(Nx,Ny,Nz,3))
    d = randn(Complex{T},(Nx,Ny,Nz,3))
    for nthread in [1 4 8 12 16 24]
        println("Threads: $nthread")
        FFTW.set_num_threads(nthread)
        𝓕! = plan_fft!(e,(1:3),flags=FFTW.PATIENT)
        𝓕⁻¹! = plan_ifft!(d,(1:3),flags=FFTW.PATIENT)
        mul!(d,𝓕!,d)
        mul!(e,𝓕⁻¹!,e)
        @benchmark mul!($d,$𝓕!,$d)
        @benchmark mul!($e,$𝓕⁻¹!,$e)
    end
end

fftw_benchmark(100,100,1)


### fftw_benchmak results for Nx=Ny=100,Nz=1, T=Float64
# Threads: 1
#   110.145 μs (0 allocations: 0 bytes)
#   132.275 μs (0 allocations: 0 bytes)
#   105.332 μs (0 allocations: 0 bytes)
#   120.216 μs (0 allocations: 0 bytes)
#   120.237 μs (0 allocations: 0 bytes)
#   132.937 μs (0 allocations: 0 bytes)
#   111.051 μs (0 allocations: 0 bytes)
#   122.812 μs (0 allocations: 0 bytes)
# Threads: 2
#   122.391 μs (52 allocations: 3.00 KiB)
#   179.158 μs (52 allocations: 3.00 KiB)
#   126.739 μs (26 allocations: 1.50 KiB)
#   173.404 μs (26 allocations: 1.50 KiB)
#   93.651 μs (53 allocations: 3.03 KiB)
#   117.219 μs (52 allocations: 3.00 KiB)
#   105.269 μs (0 allocations: 0 bytes)
#   116.246 μs (26 allocations: 1.50 KiB)
# Threads: 4
#   64.103 μs (80 allocations: 5.00 KiB)
#   96.021 μs (81 allocations: 5.03 KiB)
#   47.492 μs (33 allocations: 2.00 KiB)
#   86.047 μs (33 allocations: 2.00 KiB)
#   60.454 μs (80 allocations: 5.00 KiB)
#   92.712 μs (80 allocations: 5.00 KiB)
#   45.197 μs (33 allocations: 2.00 KiB)
#   85.238 μs (33 allocations: 2.00 KiB)
# Threads: 6
#   100.160 μs (189 allocations: 11.00 KiB)
#   145.808 μs (345 allocations: 20.00 KiB)
#   52.482 μs (189 allocations: 11.00 KiB)
#   98.934 μs (345 allocations: 20.00 KiB)
#   51.032 μs (109 allocations: 7.12 KiB)
#   84.613 μs (110 allocations: 7.16 KiB)
#   47.348 μs (33 allocations: 2.00 KiB)
#   84.243 μs (33 allocations: 2.00 KiB)
# Threads: 8
#   96.731 μs (231 allocations: 14.00 KiB)
#   150.332 μs (429 allocations: 26.00 KiB)
#   60.577 μs (231 allocations: 14.00 KiB)
#   107.479 μs (429 allocations: 26.00 KiB)
#   58.915 μs (138 allocations: 9.16 KiB)
#   78.913 μs (138 allocations: 9.16 KiB)
#   51.880 μs (33 allocations: 2.00 KiB)
#   86.810 μs (132 allocations: 8.00 KiB)
# Threads: 10
#   92.914 μs (273 allocations: 17.00 KiB)
#   156.300 μs (513 allocations: 32.00 KiB)
#   69.380 μs (273 allocations: 17.00 KiB)
#   112.121 μs (513 allocations: 32.00 KiB)
#   52.422 μs (168 allocations: 11.44 KiB)
#   91.838 μs (168 allocations: 11.44 KiB)
#   58.995 μs (153 allocations: 9.50 KiB)
#   96.359 μs (153 allocations: 9.50 KiB)
# Threads: 12
#   88.757 μs (273 allocations: 17.00 KiB)
#   143.094 μs (513 allocations: 32.00 KiB)
#   66.251 μs (273 allocations: 17.00 KiB)
#   116.689 μs (513 allocations: 32.00 KiB)
#   65.331 μs (195 allocations: 13.41 KiB)
#   87.204 μs (196 allocations: 13.44 KiB)
#   56.527 μs (153 allocations: 9.50 KiB)
#   96.995 μs (153 allocations: 9.50 KiB)

## full Helmholtz Operator
using ArrayInterface, LinearAlgebra, LoopVectorization, FFTW, LinearMaps, IterativeSolvers, BenchmarkTools
using OptiMode: calc_kpg

Nx,Ny,Nz = 100,100,1
Δx,Δy,Δz = 6.0,4.0,1.0
Ninv = 1.0 / ( Nx*Ny*Nz )
k = 1.6


FFTW.set_num_threads(1)
T = Float64

Hin = randn(Complex{T},(Nx,Ny,Nz,2))
Hout = similar(Hin)
e = randn(Complex{T},(Nx,Ny,Nz,3))
d = randn(Complex{T},(Nx,Ny,Nz,3))
ε⁻¹ = randn(T,(Nx,Ny,Nz,3,3))
mag,mn = calc_kpg(k,Δx,Δy,Δz,Nx,Ny,Nz)
mn2 = permutedims(mn,(3,4,5,2,1))
m = copy(mn2[:,:,:,1,:])
n = copy(mn2[:,:,:,2,:])
𝓕! = plan_fft!(copy(e),(1:3),flags=FFTW.PATIENT)
𝓕⁻¹! = plan_bfft!(copy(d),(1:3),flags=FFTW.PATIENT)
𝓕 = plan_fft(copy(e),(1:3),flags=FFTW.PATIENT)
𝓕⁻¹ = plan_bfft(copy(d),(1:3),flags=FFTW.PATIENT)

function kx_tc!(d,H,m,n,mag)::Array{ComplexF64,4}
    # @assert size(Y) === size(X)
    # @assert size(d,4) == 3
    # @assert size(H,4) === 2
    @avx for k ∈ axes(d,3), j ∈ axes(d,2), i ∈ axes(d,1), l in 0:0
        d[i,j,k,1+l] = ( H[i,j,k,1] * n[i,j,k,1+l] - H[i,j,k,2] * m[i,j,k,1+l] ) * -mag[i,j,k]
        d[i,j,k,2+l] = ( H[i,j,k,1] * n[i,j,k,2+l] - H[i,j,k,2] * m[i,j,k,2+l] ) * -mag[i,j,k]
        d[i,j,k,3+l] = ( H[i,j,k,1] * n[i,j,k,3+l] - H[i,j,k,2] * m[i,j,k,3+l] ) * -mag[i,j,k]
    end
    return d
end

function kx_ct!(H,e,m,n,mag,Ninv)::Array{ComplexF64,4}
    # @assert size(Y) === size(X)
    # @assert size(e,4) == 3
    # @assert size(H,4) === 2
    @avx for k ∈ axes(H,3), j ∈ axes(H,2), i ∈ axes(H,1), l in 0:0
        scale = mag[i,j,k] * Ninv
        H[i,j,k,1+l] =  (	e[i,j,k,1+l] * n[i,j,k,1+l] + e[i,j,k,2+l] * n[i,j,k,2+l] + e[i,j,k,3+l] * n[i,j,k,3+l]	) * -scale  # -mag[i,j,k] * Ninv
		H[i,j,k,2+l] =  (	e[i,j,k,1+l] * m[i,j,k,1+l] + e[i,j,k,2+l] * m[i,j,k,2+l] + e[i,j,k,3+l] * m[i,j,k,3+l]	) * scale   # mag[i,j,k] * Ninv
    end
    return H
end

function eid!(e,ei,d)::Array{ComplexF64,4}
    # @assert size(e,4) === 3
    # @assert size(d,4) === 3
    # @assert size(ei,4) === 3
    # @assert size(ei,5) === 3
    @avx for k ∈ axes(e,3), j ∈ axes(e,2), i ∈ axes(e,1), l in 0:0, h in 0:0
        e[i,j,k,1+h] =  ei[i,j,k,1+h,1+l]*d[i,j,k,1+l] + ei[i,j,k,2+h,1+l]*d[i,j,k,2+l] + ei[i,j,k,3+h,1+l]*d[i,j,k,3+l]
        e[i,j,k,2+h] =  ei[i,j,k,1+h,2+l]*d[i,j,k,1+l] + ei[i,j,k,2+h,2+l]*d[i,j,k,2+l] + ei[i,j,k,3+h,2+l]*d[i,j,k,3+l]
        e[i,j,k,3+h] =  ei[i,j,k,1+h,3+l]*d[i,j,k,1+l] + ei[i,j,k,2+h,3+l]*d[i,j,k,2+l] + ei[i,j,k,3+h,3+l]*d[i,j,k,3+l]
    end
    return e
end

# function M!(Hout::AbstractArray{T,4},Hin::AbstractArray{T,4},e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv) where {T<:Real}
function M!(Hout,Hin,e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)
    kx_tc!(d,Hin,m,n,mag);
    mul!(d,𝓕!,d);
    eid!(e,ε⁻¹,d);
    mul!(d,𝓕⁻¹!,d);
    kx_ct!(Hout,e,m,n,mag,Ninv);
end

function M2!(Hout,Hin,e,ef,d,df,ε⁻¹,m,n,mag,𝓕,𝓕⁻¹,Ninv)
    kx_tc!(df,Hin,m,n,mag);
    mul!(d,𝓕,df);
    eid!(e,ε⁻¹,d);
    mul!(ef,𝓕⁻¹,e);
    kx_ct!(Hout,ef,m,n,mag,Ninv);
end

function Mnofft!(Hout,Hin,e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)
    kx_tc!(d,Hin,m,n,mag);
    eid!(e,ε⁻¹,d);
    kx_ct!(Hout,e,m,n,mag,Ninv);
end


# function M!(Hout::AbstractVector,Hin::AbstractVector,e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv) where {T<:Real}
#     s = size(mag)
#     vec(M!(reshape(Hout,(s[1],s[2],s[3],2)),reshape(Hin,(s[1],s[2],s[3],2),e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)))
# end

ef = similar(e)
df = similar(d)

Hin = randn(ComplexF64,size(Hin))
Hout = randn(ComplexF64,size(Hin))
# Mop * Hin

kx_tc!(d,Hin,m,n,mag);
mul!(d,𝓕!,d);
eid!(e,ε⁻¹,d);
mul!(e,𝓕⁻¹!,e);
kx_ct!(Hout,e,m,n,mag,Ninv);
M!(Hout,Hin,e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)
M2!(Hout,Hin,e,ef,d,df,ε⁻¹,m,n,mag,𝓕,𝓕⁻¹,Ninv)
Mnofft!(Hout,Hin,e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)

@benchmark kx_tc!($d,$Hin,$m,$n,$mag)
@benchmark mul!($d,$𝓕!,$d)
@benchmark eid!($e,$ε⁻¹,$d)
@benchmark mul!($d,$𝓕⁻¹!,$d)
@benchmark kx_ct!(Hout,e,m,n,mag,Ninv)
@benchmark M!($Hout,$Hin,$e,$d,$ε⁻¹,$m,$n,$mag,$𝓕!,$𝓕⁻¹!,$Ninv)

# FFTW threads = 4
# @btime M!($Hout,$Hin,$e,$d,$ε⁻¹,$m,$n,$mag,$𝓕!,$𝓕⁻¹!,$Ninv) # 216.260 μs (144 allocations: 8.50 KiB)
# @btime M2!($Hout,$Hin,$e,$ef,$d,$df,$ε⁻¹,$m,$n,$mag,$𝓕,$𝓕⁻¹,$Ninv) # 214.032 μs (144 allocations: 8.50 KiB)
# @btime Mnofft!($Hout,$Hin,$e,$d,$ε⁻¹,$m,$n,$mag,$𝓕!,$𝓕⁻¹!,$Ninv) # 84.440 μs (0 allocations: 0 bytes)

# FFTW threads = 1
@btime M!($Hout,$Hin,$e,$d,$ε⁻¹,$m,$n,$mag,$𝓕!,$𝓕⁻¹!,$Ninv) # 295.195 μs (0 allocations: 0 bytes)
@btime M2!($Hout,$Hin,$e,$ef,$d,$df,$ε⁻¹,$m,$n,$mag,$𝓕,$𝓕⁻¹,$Ninv) # 82.470 μs (0 allocations: 0 bytes)
@btime Mnofft!($Hout,$Hin,$e,$d,$ε⁻¹,$m,$n,$mag,$𝓕!,$𝓕⁻¹!,$Ninv) # 273.492 μs (0 allocations: 0 bytes)




##
using LinearMaps

HinV = copy(vec(Hin))
HoutV = copy(vec(Hout))
Mop = LinearMap{ComplexF64}((y,x)->vec(M!(reshape(y,(Nx,Ny,Nz,2)),reshape(x,(Nx,Ny,Nz,2)),e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)),2*length(mag),ishermitian=true,ismutating=true)
Mop_pd = LinearMap{ComplexF64}((y,x)->vec(M!(reshape(y,(Nx,Ny,Nz,2)),reshape(x,(Nx,Ny,Nz,2)),e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)),2*length(mag),ishermitian=true,ismutating=true,isposdef=true)
M̂1!(e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv) = LinearMap{ComplexF64}((y,x)->vec(M!(reshape(y,(Nx,Ny,Nz,2)),reshape(x,(Nx,Ny,Nz,2)),e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)),2*length(mag),ishermitian=true,ismutating=true)
M̂2!(e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv,Nx,Ny,Nz) = LinearMap{ComplexF64}((y,x)->vec(M!(reshape(y,(Nx,Ny,Nz,2)),reshape(x,(Nx,Ny,Nz,2)),e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)),2*length(mag),ishermitian=true,ismutating=true)
M̂3!(e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv) = LinearMap{ComplexF64}((y,x)->vec(M!(reshape(y,(size(mag)...,2)),reshape(x,(size(mag)...,2)),e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)),2*length(mag),ishermitian=true,ismutating=true)
M̂5!(e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv) = LinearMap{ComplexF64}((y::Vector{ComplexF64},x::Vector{ComplexF64})->vec(Mnofft!(reshape(y,(size(mag)...,2)),reshape(x,(size(mag)...,2)),e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv))::Vector{ComplexF64},2*length(mag),ishermitian=true,ismutating=true)
function M̂4!(e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)
    let (Nx,Ny,Nz) = size(mag), Mmap(y,x) = vec(M!(reshape(y,(Nx,Ny,Nz,2)),reshape(x,(Nx,Ny,Nz,2)),e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)), N = (2*Nx*Ny*Nz)
        Mop = LinearMap{ComplexF64}(Mmap,N,ishermitian=true,ismutating=true)
    end
end
f_Mop!(y::Vector{ComplexF64},x::Vector{ComplexF64})::Vector{ComplexF64} =  vec(Mnofft!(reshape(y,(Nx,Ny,Nz,2)),reshape(x,(Nx,Ny,Nz,2)),e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv))
N_Mop = 2*Nx*Ny*Nz
Mop5 = LinearMap{ComplexF64}(f_Mop!,f_Mop!,N_Mop,N_Mop,ishermitian=true,ismutating=true)

# function LinearAlgebra.mul!(y::AbstractVecOrMat, A::LinearMaps.FunctionMap, x::AbstractVector)
#     # LinearMaps.check_dim_mul(y, A, x)
#     return fill!(y, iszero(A.λ) ? zero(eltype(y)) : A.λ*sum(x))
# end

Mop8 = let Nx=Nx,Ny=Ny,Nz=Nz,Δx=Δx,Δy=Δy,Δz=Δz,k=k
    Ninv = 1.0 / ( Nx*Ny*Nz )
    N = 2*Nx*Ny*Nz
    shp = Nx,Ny,Nz,2
    # Hin = randn(Complex{T},(Nx,Ny,Nz,2))
    # Hout = similar(Hin)
    e = randn(Complex{T},(Nx,Ny,Nz,3))
    d = randn(Complex{T},(Nx,Ny,Nz,3))
    ε⁻¹ = randn(T,(Nx,Ny,Nz,3,3))
    mag,mn = calc_kpg(k,Δx,Δy,Δz,Nx,Ny,Nz)
    mn2 = permutedims(mn,(3,4,5,2,1))
    m = copy(mn2[:,:,:,1,:])
    n = copy(mn2[:,:,:,2,:])
    F! = plan_fft!(copy(e),(1:3),flags=FFTW.PATIENT)
    F⁻¹! = plan_bfft!(copy(d),(1:3),flags=FFTW.PATIENT)
    function M(y,x)
        kx_tc!(d,x,m,n,mag);
        mul!(d,F!,d);
        eid!(e,ε⁻¹,d);
        mul!(d,F⁻¹!,d);
        kx_ct!(y,e,m,n,mag,Ninv);
    end
    # LinearMap{ComplexF64}((2*Nx*Ny*Nz),ishermitian=true,ismutating=true) do (y,x)
    #     vec(M!(reshape(y,(Nx,Ny,Nz,2)),reshape(x,(Nx,Ny,Nz,2)),e,d,ε⁻¹,m,n,mag,F!,F⁻¹!,Ninv))
    LinearMap{ComplexF64}((y,x)->vec(M!(reshape(y,shp),reshape(x,shp),e,d,ε⁻¹,m,n,mag,F!,F⁻¹!,Ninv)),N,ishermitian=true,ismutating=true)
end

Mop1 = M̂1!(e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)
Mop2 = M̂2!(e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv,Nx,Ny,Nz)
Mop3 = M̂3!(e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)
Mop4 = M̂4!(e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)
Mop5 = M̂5!(e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)
mul!(HoutV,Mop,HinV)
mul!(HoutV,Mop1,HinV)
mul!(HoutV,Mop2,HinV)
mul!(HoutV,Mop3,HinV)
mul!(HoutV,Mop4,HinV)
mul!(HoutV,Mop_pd,HinV)
mul!(HoutV,Mop5,HinV)
mul!(HoutV,Mop8,HinV)

struct MaxwellMap{T, F} <: LinearMaps.LinearMap{Complex{T}}
    e::Array{Complex{T}, 4}
    d::Array{Complex{T}, 4}
    ε⁻¹::Array{T, 5}
    m::Array{T, 4}
    n::Array{T, 4}
    mag::Array{T, 3}
    𝓕!::FFTW.cFFTWPlan{Complex{T}}
    𝓕⁻¹!::FFTW.cFFTWPlan{Complex{T}}
    Ninv::T
    f::F
    N::Int
    _ismutating::Bool
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end

mutable struct mMaxwellMap{T, F} <: LinearMaps.LinearMap{Complex{T}}
    e::Array{Complex{T}, 4}
    d::Array{Complex{T}, 4}
    ε⁻¹::Array{T, 5}
    m::Array{T, 4}
    n::Array{T, 4}
    mag::Array{T, 3}
    𝓕!::FFTW.cFFTWPlan{Complex{T}}
    𝓕⁻¹!::FFTW.cFFTWPlan{Complex{T}}
    Ninv::T
    f::F
    N::Int
    _ismutating::Bool
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end

function mMaxwellMap{T}(f::F, N::Int;
    ismutating::Bool  = _ismutating(f),
    issymmetric::Bool = false,
    ishermitian::Bool = (T<:Real && issymmetric),
    isposdef::Bool    = false) where {T, F1, F2}
    FunctionMap{T, F1, F2}(f, fc, M, N, ismutating, issymmetric, ishermitian, isposdef)
end


# function (f::mMaxwellMap)(i, nm, TT)
#     pos_i, x_i = readvalue(f.buf, f.pos, f.len, TT; f.kw...)
#     f.pos = pos_i
#     return x_i
# end

# FFTW threads = 4
# @btime mul!($HoutV,$Mop,$HinV)  # 224.553 μs (152 allocations: 8.92 KiB)
# @btime mul!($HoutV,$Mop1,$HinV) # 217.570 μs (153 allocations: 8.94 KiB)
# @btime mul!($HoutV,$Mop2,$HinV) # 222.795 μs (150 allocations: 8.83 KiB)
# @btime mul!($HoutV,$Mop3,$HinV) # 218.957 μs (150 allocations: 8.83 KiB)
# @btime mul!($HoutV,$Mop4,$HinV) # 226.071 μs (150 allocations: 8.83 KiB)
# @btime mul!($HoutV,$Mop_pd,$HinV)  # 224.553 μs (152 allocations: 8.92 KiB)
# @btime mul!($HoutV,$Mop5,$HinV) # 217.802 μs (152 allocations: 8.92 KiB)

# FFTW threads = 1
@btime mul!($HoutV,$Mop,$HinV)  # 302.679 μs (8 allocations: 432 bytes)
@btime mul!($HoutV,$Mop1,$HinV) # 287.758 μs (9 allocations: 448 bytes)
@btime mul!($HoutV,$Mop2,$HinV) # 285.625 μs (6 allocations: 336 bytes)
@btime mul!($HoutV,$Mop3,$HinV) # 287.651 μs (6 allocations: 336 bytes)
@btime mul!($HoutV,$Mop4,$HinV) # 289.960 μs (6 allocations: 336 bytes)
@btime mul!($HoutV,$Mop_pd,$HinV) # 295.120 μs (8 allocations: 432 bytes)
@btime mul!($HoutV,$Mop5,$HinV) # 295.057 μs (8 allocations: 432 bytes)
@btime mul!($HoutV,$Mop8,$HinV)

function mymul!(y::AbstractVecOrMat, A::LinearMaps.FunctionMap, x::AbstractVector)
    # LinearMaps.ismutating(A) ? A.f(y, x) : copyto!(y, A.f(x))
    A.f(y, x)
    # return y
end

mymul!(HoutV,Mop3,HinV)
@btime mymul!($HoutV,$Mop3,$HinV)
Mop3.f(HoutV,HinV)
@btime @inbounds $Mop5.f($HoutV,$HinV)
Mop4.f
Mop6 = LinearMap{ComplexF64}((y,x)->vec(deepcopy(M!)(reshape(y,(100,100,1,2)),reshape(x,(100,100,1,2)),e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)),2*100*100*1,ishermitian=true,ismutating=true)

x = Vector{ComplexF64}(randn(ComplexF64,20000))
y = Vector{ComplexF64}(randn(ComplexF64,20000))
mymul!(y,Mop6,x)
@btime mymul!($y,$Mop6,$x)
# struct MyFnMap{T} <: LinearMaps.FunctionMap{T}
#     λ::T
#     size::Dims{2}
#     function MyFnMap(λ::T, dims::Dims{2}) where {T}
#         # all(≥(0), dims) || throw(ArgumentError("dims of MyFillMap must be non-negative"))
#         promote_type(T, typeof(λ)) == T || throw(InexactError())
#         return new{T}(λ, dims)
#     end
# end

LinearMaps._unsafe_mul!(HoutV,Mop3,HinV)
LinearMaps._unsafe_mul!(HoutV,Mop4,HinV)

@btime LinearMaps._unsafe_mul!($HoutV,$Mop3,$HinV) # 287.651 μs (6 allocations: 336 bytes)
@btime LinearMaps._unsafe_mul!($HoutV,$Mop4,$HinV)

function mul_test()
    Nx,Ny,Nz = 100,100,1
    Δx,Δy,Δz = 6.0,4.0,1.0
    Ninv = 1.0 / ( Nx*Ny*Nz )
    k = 1.6
    Hin = randn(Complex{Float64},(Nx,Ny,Nz,2))
    Hout = similar(Hin)
    e = randn(Complex{Float64},(Nx,Ny,Nz,3))
    d = randn(Complex{Float64},(Nx,Ny,Nz,3))
    ε⁻¹ = randn(T,(Nx,Ny,Nz,3,3))
    mag,mn = calc_kpg(k,Δx,Δy,Δz,Nx,Ny,Nz)
    mn2 = permutedims(mn,(3,4,5,2,1))
    m = copy(mn2[:,:,:,1,:])
    n = copy(mn2[:,:,:,2,:])
    𝓕! = plan_fft!(copy(e),(1:3)) #,flags=FFTW.PATIENT)
    𝓕⁻¹! = plan_bfft!(copy(d),(1:3)) #,flags=FFTW.PATIENT)
    Mop_test = M̂3!(e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)
    HinV = copy(vec(Hin))
    HoutV = copy(vec(Hout))
    mul!(HoutV,Mop_test,HinV)
    @benchmark mul!(HoutV,$(M̂3!(e,d,ε⁻¹,m,n,mag,𝓕!,𝓕⁻¹!,Ninv)),HinV)
end
mul_test()

##
function fnew(y,x)
    @. y = 3 + 2.0im
end

Mop4c.f = fnew

Mop4c.f(HoutV,HinV) ≈ Mop3.f(HoutV,HinV)



##
"""
    t2c: v (transverse vector) → a (cartesian vector)
"""
function t2c(v::SVector{2,ComplexF64},k::KVec)::SVector{3,ComplexF64}
    return v[1] * k.m + v[2] * k.n
end


"""
    c2t: a (cartesian vector) → v (transverse vector)
"""
function c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{2,ComplexF64}
    v0 = a ⋅ k.m
    v1 = a ⋅ k.n
    return SVector(v0,v1)
end

"""
    kcross_t2c: a (cartesian vector) = k × v (transverse vector)
"""
function kcross_t2c(v::SVector{2,ComplexF64},k::KVec)::SVector{3,ComplexF64}
    return ( v[1] * k.n - v[2] * k.m ) * k.mag
end

"""
    kcross_c2t: v (transverse vector) = k × a (cartesian vector)
"""
function kcross_c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{2,ComplexF64}
    at1 = a ⋅ k.m
    at2 = a ⋅ k.n
    v0 = -at2 * k.mag
    v1 = at1 * k.mag
    return SVector(v0,v1)
end


"""
    kcrossinv_t2c: compute a⃗ (cartestion vector) st. v⃗ (cartesian vector from two trans. vector components) ≈ k⃗ × a⃗
    This neglects the component of a⃗ parallel to k⃗ (not available by inverting this cross product)
"""
function kcrossinv_t2c(v::SVector{2,ComplexF64},k::KVec)::SVector{3,ComplexF64}
    return ( v[1] * k.n - v[2] * k.m ) * ( -1 / k.mag )
end

"""
    kcrossinv_c2t: compute  v⃗ (transverse 2-vector) st. a⃗ (cartestion 3-vector) = k⃗ × v⃗
    This cross product inversion is exact because v⃗ is transverse (perp.) to k⃗
"""
function kcrossinv_c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{2,ComplexF64}
    at1 = a ⋅ k.m
    at2 = a ⋅ k.n
    v0 = -at2 * (-1 / k.mag )
    v1 = at1 * ( -1 / k.mag )
    return SVector(v0,v1)
end

"""
    ucross_t2c: a (cartesian vector) = u × v (transverse vector)
"""
function ucross_t2c(u::SVector{3,ComplexF64},v::SVector{2,ComplexF64},k::KVec)::SVector{3,ComplexF64}
    return cross(u,t2c(v,k))
end


##

function eid1!(e::AbstractArray{T},ei::AbstractArray{T},d::AbstractArray{T},Nx,Ny,Nz)::AbstractArray{T} where T
    @fastmath @inbounds for i=1:Nx,j=1:Ny,k=1:Nz
        @fastmath @inbounds e[1,i,j,k] =  ei[1,1,i,j,k]*d[1,i,j,k] + ei[2,1,i,j,k]*d[2,i,j,k] + ei[3,1,i,j,k]*d[3,i,j,k]
        @fastmath @inbounds e[2,i,j,k] =  ei[1,2,i,j,k]*d[1,i,j,k] + ei[2,2,i,j,k]*d[2,i,j,k] + ei[3,2,i,j,k]*d[3,i,j,k]
        @fastmath @inbounds e[3,i,j,k] =  ei[1,3,i,j,k]*d[1,i,j,k] + ei[2,3,i,j,k]*d[2,i,j,k] + ei[3,3,i,j,k]*d[3,i,j,k]
    end
    return e
end

function eid2!(e::AbstractArray{T},ei::AbstractArray{T},d::AbstractArray{T},Nx,Ny,Nz)::AbstractArray{T} where T
    @fastmath @inbounds for i=1:Nx,j=1:Ny,k=1:Nz
         @simd for a = 1:3
            @fastmath @inbounds @views e[:,i,j,k] .=  ei[a,:,i,j,k]*d[a,i,j,k]
        end
    end
    return e
end

function eid3!(e::AbstractArray{SVector{3,T}},ei::AbstractArray{SMatrix{3,3,T,9}},d::AbstractArray{SVector{3,T}})::AbstractArray{SVector{3,T}} where T
    @fastmath broadcast!(*,es,eis,ds)
end

function eid4(ei::AbstractArray{T},d::AbstractArray{T})::AbstractArray{T} where T
    @tullio e[a,i,j,k] :=  ei[a,b,i,j,k] * d[b,i,j,k]
end

function eid4!(e::AbstractArray{T},ei::AbstractArray{T},d::AbstractArray{T})::AbstractArray{T} where T
    @tullio e[a,i,j,k] =  ei[a,b,i,j,k] * d[b,i,j,k] fastmath=true
end

function eid5!(es,eis, ds)
    @tullio es[i,j,k] = eis[i,j,k] * ds[i,j,k]
end

function eid5(eis, ds)
    @tullio es[i,j,k] := eis[i,j,k] * ds[i,j,k]
end



### Warmups

# Array + hand written for loop
eid1!(e,ei,d,Nx,Ny,Nz)
# Array + @simd + @views + broadcast! + for loop
eid2!(e,ei,d,Nx,Ny,Nz) ≈ eid1!(e,ei,d,Nx,Ny,Nz)
# StaticArrays+broadcast!
eid3!(es,eis,ds)
# normal Array+Tullio
eid4(ei,d)
eid4!(e,ei,d)
# CuArrays + Tullio
# eid4(eic,dc)
# eid4!(ec,eic,dc)
# Array(eid4(eic,dc))
# HybridArrays+Tullio
eid4(eih,dh)
eid4!(eh,eih,dh)
# StaticArrays+Tullio
eid5(eis,ds)
eid5!(es,eis,ds)

### Benchmarks

# Array + hand written for loop
@btime eid1!($e,$ei,$d,$Nx,$Ny,$Nz) #74.750 μs (0 allocations: 0 bytes)
# Array + @simd + @views + broadcast! + for loop
@btime eid2!($e,$ei,$d,$Nx,$Ny,$Nz) #1.308 ms (30000 allocations: 3.20 MiB)
# StaticArrays+broadcast!
@btime eid3!($es,$eis,$ds) # 22.703 μs (1 allocation: 48 bytes)
# normal Array+Tullio
@btime eid4!($e,$ei,$d) # 32.310 μs (0 allocations: 0 bytes)
@btime eid4($ei,$d) # 33.921 μs (3 allocations: 234.50 KiB)
# CuArrays + Tullio
# @btime eid4($eic,$dc) # 19.833 μs (137 allocations: 5.53 KiB), 1.069 s (2228391 allocations: 93.97 MiB) w/o KernelAbstractions
# @btime eid4!($ec,$eic,$dc) # 84.793 μs (127 allocations: 5.31 KiB)
# @btime Array(eid4($eic,$dc)) # 2.039 ms (142 allocations: 122.88 KiB)
# HybridArrays+Tullio
@btime eid4($eih,$dh) # 180.503 μs (11 allocations: 234.67 KiB)
@btime eid4!($eh,$eih,$dh) # 197.167 μs (6 allocations: 144 bytes)
@btime eid4($(eih.data),$(dh.data)) # 33.650 μs (3 allocations: 234.50 KiB)
@btime eid4!($(eh.data),$(eih.data),$(dh.data)) # 33.003 μs (0 allocations: 0 bytes)
# StaticArrays+Tullio
@btime eid5($eis,$ds) # 180.503 μs (11 allocations: 234.67 KiB)
@btime eid5!($es,$eis,$ds) # 197.167 μs (6 allocations: 144 bytes)
## grads
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
#

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

Zygote.refresh()
# using ReverseDiff: TrackedArray
# @inline Base.:+(x::TrackedArray{V,D}, y::StaticArray) where {V,D} = record_plus(x, Array(y), D)
# @inline Base.:+(x::StaticArray, y::TrackedArray{V,D}) where {V,D} = record_plus(Array(x), y, D)
# @inline Base.:-(x::TrackedArray{V,D}, y::StaticArray) where {V,D} = record_minus(x, Array(y), D)
# @inline Base.:-(x::StaticArray, y::TrackedArray{V,D}) where {V,D} = record_minus(Array(x), y, D)
# deriv!(t::StaticArray, v::AbstractArray) = deriv!(Tuple(t), Tuple(v))
# `StaticArray`s don't support mutation unless the eltype is a bits type (`isbitstype`).
# capture(t::SA) where SA <: StaticArray = istracked(t) ? SA(map(capture, t)) : copy(t)
##
function foo!(a,b,c)
    sum(sum(broadcast!(*,a,b,c)))
end

function foo(b,c)
    sum(sum(broadcast(*,b,c)))
end

function foot(ei,d)
    # @tullio esum :=  ei[a,b,i,j,k] * d[b,i,j,k]
    @tullio e :=  ei[a,b,i,j,k] * d[b,i,j,k]
end

function foots(eis::Array{SArray{Tuple{3,3},Float64,2,9},3},ds::Array{SArray{Tuple{3},Float64,1,3},3})::Float64
    # @tullio esum :=  ei[a,b,i,j,k] * d[b,i,j,k]
    @tullio es_sum :=  eis[i,j,k] * ds[i,j,k]
    sum(es_sum)
    # es = vmap(*,eis,ds)
    # sum(sum(es))
end


foo(eis,ds)
foo!(es,eis,ds)
foot(ei,d)
# foot(cu(ei),cu(dc))
foot(eih,dh)
foots(eis,ds)

@btime foo($eis,$ds) # 33.774 μs (2 allocations: 234.45 KiB)
@btime foo!($es,$eis,$ds) # 31.154 μs (0 allocations: 0 bytes)
@btime foot($ei,$d) # 125.603 μs (1 allocation: 16 bytes)
# @btime foot($eic,$dc) # 1.025 s (2198390 allocations: 91.22 MiB)
@btime foot($eih,$dh) # 80.442 μs (10 allocations: 224 bytes)
@btime foots($eis,$ds) # 32.728 μs (2 allocations: 234.45 KiB)
###### Grads!

using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile, DiffResults

##Zygote.gradient(foo,eis,ds)
Zygote.gradient(foo,eis,ds)
Zygote.gradient(foot,ei,d)
Zygote.gradient(foot,eih,dh)
Zygote.gradient(foot,eih.data,dh.data)
Zygote.gradient(foots,eis,ds)
# Zygote.gradient(foot,eic,dc)
const f_tape = GradientTape(foot, (rand(size(ei)...), rand(size(d)...)))

inputs = (ei,d)
results = (similar(ei),similar(d));
all_results = map(DiffResults.GradientResult, results);
cfg = GradientConfig(inputs)

Tracker.gradient(foot,eih,dh)


# ReverseDiff
const compiled_f_tape = compile(GradientTape(foot, (rand(size(ei)...), rand(size(d)...))))
inputs = (ei, d)
results = (similar(ei), similar(d))
all_results = map(DiffResults.GradientResult, results)
cfg = GradientConfig(inputs)
ReverseDiff.gradient!(results, compiled_f_tape, inputs)
ReverseDiff.gradient!(all_results, compiled_f_tape, inputs) # the same as the above but primal is loaded into the the provided `DiffResult` instances (see DiffResults.jl documentation).


### Benchmarks
@btime ReverseDiff.gradient!($results, $compiled_f_tape, $inputs) # 33.863 ms (11 allocations: 512 bytes)
@btime ReverseDiff.gradient!($all_results, $compiled_f_tape, $inputs) # 34.298 ms (11 allocations: 512 bytes)
@btime Zygote.gradient(foo,$eis,$ds) # 1.337 ms (50048 allocations: 6.11 MiB)
@btime Zygote.gradient(foot,$ei,$d) # 456.707 μs (83 allocations: 941.67 KiB)
# @btime Zygote.gradient(foot,$eic,$dc) # 40.263 ms (394 allocations: 16.59 KiB)
@btime Zygote.gradient(foot,$eih,$dh) # 434.074 μs (96 allocations: 941.98 KiB)
@btime Zygote.gradient(foots,$eis,$ds) # 434.074 μs (96 allocations: 941.98 KiB)

ReverseDiff.gradient(foot,eih,dh)
foot(cu(ei),cu(d))


##
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

A = rand(3,40); B = rand(40,500);
A * B ≈ mul(A, B) # true

using Tracker # or Zygote
ΔA = Tracker.gradient((A,B) -> sum(mul(A, B)), A, B)[1]
ΔA ≈ ones(3,500) * B' # true

using CUDA, KernelAbstractions # Now defined with a GPU version:
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

cu(A * B) ≈ mul(cu(A), cu(B)) # true

cu(ΔA) ≈ Tracker.gradient((A,B) -> sum(mul(A, B)), cu(A), cu(B))[1] ; # true

# Reduction over min/max:
Tracker.gradient(x -> (@tullio (max) res := x[i]^3), [1,2,3,-2,-1,3])[1]
f_cuda = (X,Y) -> sum(mul(X, Y))
@btime Tracker.gradient(f_cuda, $(cu(A)), $(cu(B))) # 435.266 μs (524 allocations: 16.63 KiB)
@btime Zygote.gradient(f_cuda, $(cu(A)), $(cu(B))) # 439.961 μs (450 allocations: 16.94 KiB)



eic

dc
