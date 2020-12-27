# using GeometryPrimitives: orthoaxes
using FFTW
using IterativeSolvers: LOBPCGIterator
export MaxwellGrid, MaxwellData, t2c, c2t, kx_c2t, kx_t2c, zx_t2c, kxinv_c2t, kxinv_t2c, ε⁻¹_dot, ε_dot_approx, M, M̂, P, P̂, Mₖ, M̂ₖ, t2c!, c2t!, kcross_c2t!, kcross_t2c!, zcross_t2c!, kcrossinv_c2t!, kcrossinv_t2c!, ε⁻¹_dot!, ε_dot_approx!, M!, M̂!, P!, P̂!, calc_kpg, H_Mₖ_H

struct MaxwellGrid{T<:Real}
    Δx::T
    Δy::T
    Δz::T
    Nx::Int
    Ny::Int
    Nz::Int
    δx::T
    δy::T
    δz::T
    x::StepRangeLen{T}
    y::StepRangeLen{T}
    z::StepRangeLen{T}
    g⃗::Array{SVector{3,T},3}
end

MaxwellGrid(Δx::T,Δy::T,Δz::T,Nx::Int,Ny::Int,Nz::Int) where {T<:Real} = MaxwellGrid{T}(
    Δx,
    Δy,
    Δz,
    Nx,
    Ny,
    Nz,
    Δx / Nx,    # δx
    Δy / Ny,    # δy
    Δz / Nz,    # δz
    ( ( Δx / Nx ) .* (0:(Nx-1))) .- Δx/2.,  # x
    ( ( Δy / Ny ) .* (0:(Ny-1))) .- Δy/2.,  # y
    ( ( Δz / Nz ) .* (0:(Nz-1))) .- Δz/2.,  # z
    # [ [gx;gy;gz] for gx in fftfreq(Nx,Nx/Δx), gy in fftfreq(Ny,Ny/Δy), gz in fftfreq(Nz,Nz/Δz)], # g⃗
	[ SVector{T,3}(gx,gy,gz) for gx in fftfreq(Nx,Nx/Δx), gy in fftfreq(Ny,Ny/Δy), gz in fftfreq(Nz,Nz/Δz)], # g⃗
)

MaxwellGrid(Δx::T,Δy::T,Nx::Int,Ny::Int) where {T<:Real} = MaxwellGrid{T}(
    Δx,
    Δy,
    1.,
    Nx,
    Ny,
    1,
    Δx / Nx,    # δx
    Δy / Ny,    # δy
    1.,    # δz
    ( ( Δx / Nx ) .* (0:(Nx-1))) .- Δx/2.,  # x
    ( ( Δy / Ny ) .* (0:(Ny-1))) .- Δy/2.,  # y
    0.0:1.0:0.0,  # z
    [ SVector{T,3}(gx,gy,gz) for gx in fftfreq(Nx,Nx/Δx), gy in fftfreq(Ny,Ny/Δy), gz in fftfreq(1,1.0)], # g⃗
)

mutable struct MaxwellData{T<:Real}
	grid::MaxwellGrid			# static grid data
	Neigs::Int					# number of eigenvalues/vectors found
	N::Int						# number of real/reciprocal space grid points (N = Nx * Ny * Nz)
	k::T
    ω²::T
    ω²ₖ::T
    ω::T
    ωₖ::T
    H⃗::Array{Complex{T},2}
    H::Array{Complex{T},4}
    e::Array{Complex{T},4}
    d::Array{Complex{T},4}
    mn::Array{T,5}
	kpg_mag::Array{T,3}
    𝓕::FFTW.cFFTWPlan
	𝓕⁻¹::AbstractFFTs.ScaledPlan
    𝓕!::FFTW.cFFTWPlan
	𝓕⁻¹!::AbstractFFTs.ScaledPlan
	solver::LOBPCGIterator
end

MaxwellData(k::T,g::MaxwellGrid{T},Neigs::Int) where {T<:Real} = MaxwellData(
	g,
	Neigs,
	N=g.Nx*g.Ny*g.Nz,
	k,
    0.0,
    0.0,
    0.0,
    0.0,
    H⃗ = randn(Complex{T},(2*g.Nx*g.Ny*g.Nz,Neigs)),
    H = randn(Complex{T},(2,g.Nx,g.Ny,g.Nz)),
    e = randn(Complex{T},(3,g.Nx,g.Ny,g.Nz)),
    d = randn(Complex{T},(3,g.Nx,g.Ny,g.Nz)),
    ( (kpg_mag, mn) = calc_kpg(k,g.Δx,g.Δy,g.Δz,g.Nx,g.Ny,g.Nz); mn),  # mn
	kpg_mag,
	plan_fft(randn(Complex{T}, (3,g.Nx,g.Ny,g.Nz))),  # planned FFT operator 𝓕
	plan_ifft(randn(Complex{T}, (3,g.Nx,g.Ny,g.Nz))),
	𝓕! = plan_fft!(randn(Complex{T}, (3,g.Nx,g.Ny,g.Nz))),
	𝓕⁻¹! = plan_ifft!(randn(Complex{T}, (3,g.Nx,g.Ny,g.Nz))),  # planned in-place FFT operator 𝓕!
	LOBPCGIterator(M̂!(ε⁻¹,H,e,d,kpg_mag,mn,𝓕!,𝓕⁻¹!), 							# Helmholtz Operator
				   false,									   					# "largest", true: find largest eigenvals, false: find smallest
				   H⃗,															 # pre-allocated eigenvectors
				   Neigs,														# number of eigenval/vec pairs to find
				   P=P̂!(ε⁻¹,H,e,d,kpg_mag,mn,𝓕!,𝓕⁻¹!),						   # (Right-)Preconditioner, approximate inverse Helmoltz Operator
				   nothing),													# Constraints
)

MaxwellData(k,g::MaxwellGrid) = MaxwellData(k,g,1)
MaxwellData(k::T,Δx::T,Δy::T,Δz::T,Nx::Int,Ny::Int,Nz::Int) where {T<:Real} = MaxwellData{T}(k,MaxwellGrid{T}(Δx,Δy,Δz,Nx,Ny,Nz))
MaxwellData(k::T,Δx::T,Δy::T,Nx::Int,Ny::Int) where {T<:Real} = MaxwellData{T}(k,MaxwellGrid{T}(Δx,Δy,Nx,Ny))


# non-Mutating Operators

function calc_kpg(kz::T,g⃗::Array{Array{T,1},3})::Tuple{Array{T,3},Array{T,5}} where {T<:Real}
	g⃗ₜ_zero_mask = Zygote.@ignore [ sum(abs2.(gg[1:2])) for gg in g⃗ ] .> 0.
	g⃗ₜ_zero_mask! = Zygote.@ignore .!(g⃗ₜ_zero_mask)
	ŷ = [0.; 1. ;0.]
	k⃗ = [0.;0.;kz]
	@tullio kpg[a,i,j,k] := k⃗[a] - g⃗[i,j,k][a] nograd=g⃗ fastmath=false
	@tullio kpg_mag[i,j,k] := sqrt <| kpg[a,i,j,k]^2 fastmath=false
	zxinds = [2; 1; 3]
	zxscales = [-1; 1. ;0.] #[[0. -1. 0.]; [-1. 0. 0.]; [0. 0. 0.]]
	@tullio kpg_nt[a,i,j,k] := zxscales[a] * kpg[zxinds[a],i,j,k] * g⃗ₜ_zero_mask[i,j,k] + ŷ[a] * g⃗ₜ_zero_mask![i,j,k]  nograd=(zxscales,zxinds,ŷ,g⃗ₜ_zero_mask,g⃗ₜ_zero_mask!) fastmath=false
	@tullio kpg_nmag[i,j,k] := sqrt <| kpg_nt[a,i,j,k]^2 fastmath=false
	@tullio kpg_n[a,i,j,k] := kpg_nt[a,i,j,k] / kpg_nmag[i,j,k] fastmath=false
	xinds1 = [2; 3; 1]
	xinds2 = [3; 1; 2]
	@tullio kpg_mt[a,i,j,k] := kpg_n[xinds1[a],i,j,k] * kpg[xinds2[a],i,j,k] - kpg[xinds1[a],i,j,k] * kpg_n[xinds2[a],i,j,k] nograd=(xinds1,xinds2) fastmath=false
	@tullio kpg_mmag[i,j,k] := sqrt <| kpg_mt[a,i,j,k]^2 fastmath=false
	@tullio kpg_m[a,i,j,k] := kpg_mt[a,i,j,k] / kpg_mmag[i,j,k] fastmath=false
	kpg_mn_basis = [[1. 0.] ; [0. 1.]]
	@tullio kpg_mn[a,b,i,j,k] := kpg_mn_basis[b,1] * kpg_m[a,i,j,k] + kpg_mn_basis[b,2] * kpg_n[a,i,j,k] nograd=kpg_mn_basis fastmath=false
	return kpg_mag, kpg_mn
end

function calc_kpg(kz::T,Δx::T,Δy::T,Δz::T,Nx::Int,Ny::Int,Nz::Int)::Tuple{Array{T,3},Array{T,5}} where {T<:Real}
	g⃗ = Zygote.@ignore [ [gx;gy;gz] for gx in fftfreq(Nx,Nx/Δx), gy in fftfreq(Ny,Ny/Δy), gz in fftfreq(Nz,Nz/Δz)]
	calc_kpg(kz,g⃗)
end
# function calc_kpg(kz::T,Δx::T,Δy::T,Δz::T,Nx::Int,Ny::Int,Nz::Int)::Tuple{Array{T,3},Array{T,5}} where T <: Real
# 	g⃗ = Zygote.@ignore [ [gx;gy;gz] for gx in fftfreq(Nx,Nx/Δx), gy in fftfreq(Ny,Ny/Δy), gz in fftfreq(Nz,Nz/Δz)]
# 	calc_kpg(kz,g⃗)
# end


# Non-Mutating Operators

"""
    t2c: v⃗ (transverse vector) → a⃗ (cartesian vector)
"""
function t2c(H,mn)
    @tullio h[a,i,j,k] := H[b,i,j,k] * mn[a,b,i,j,k] fastmath=false
end

"""
    c2t: a⃗ (cartesian vector) → v⃗ (transverse vector)
"""
function c2t(h,mn)
    @tullio H[a,i,j,k] := h[b,i,j,k] * mn[b,a,i,j,k] fastmath=false
end

"""
    kx_t2c: a⃗ (cartesian vector) = k⃗ × v⃗ (transverse vector)
"""
function kx_t2c(H,mn,kpg_mag)
	kxscales = [-1.; 1.]
	kxinds = [2; 1]
    @tullio d[a,i,j,k] := kxscales[b] * H[kxinds[b],i,j,k] * mn[a,b,i,j,k] * kpg_mag[i,j,k] nograd=(kxscales,kxinds) fastmath=false
end

"""
    kx_c2t: v⃗ (transverse vector) = k⃗ × a⃗ (cartesian vector)
"""
function kx_c2t(e⃗,mn,kpg_mag)
	kxscales = [-1.; 1.]
    kxinds = [2; 1]
    @tullio H[b,i,j,k] := kxscales[b] * e⃗[a,i,j,k] * mn[a,kxinds[b],i,j,k] * kpg_mag[i,j,k] nograd=(kxinds,kxscales) fastmath=false
end

"""
    kxinv_t2c: compute a⃗ (cartestion vector) st. v⃗ (cartesian vector from two trans. vector components) ≈ k⃗ × a⃗
    This neglects the component of a⃗ parallel to k⃗ (not available by inverting this cross product)
"""
function kxinv_t2c(H,mn,kpg_mag)
	kxinvscales = [1.; -1.]
	kxinds = [2; 1]
    @tullio e⃗[a,i,j,k] := kxscales[b] * H[kxinds[b],i,j,k] * mn[a,b,i,j,k] / kpg_mag[i,j,k] nograd=(kxscales,kxinds) fastmath=false
end

"""
    kxinv_c2t: compute  v⃗ (transverse 2-vector) st. a⃗ (cartestion 3-vector) = k⃗ × v⃗
    This cross product inversion is exact because v⃗ is transverse (perp.) to k⃗
"""
function kxinv_c2t(d⃗,mn,kpg_mag)
	kxscales = [1.; -1.]
    kxinds = [2; 1]
    @tullio H[b,i,j,k] := kxscales[b] * d⃗[a,i,j,k] * mn[a,kxinds[b],i,j,k] / kpg_mag[i,j,k] nograd=(kxinds,kxscales) fastmath=false
end

"""
    zx_t2c: a⃗ (cartesian vector) = ẑ × v⃗ (transverse vector)
"""
function zx_t2c(H,mn)
	zxinds = [2; 1; 3]
	zxscales = [-1.; 1.; 0.]
	@tullio zxH[a,i,j,k] := zxscales[a] * H[b,i,j,k] * mn[zxinds[a],b,i,j,k] nograd=(zxscales,zxinds) fastmath=false
end

"""
    ε⁻¹_dot_t: e⃗  = ε⁻¹ ⋅ d⃗ (transverse vectors)
"""
function ε⁻¹_dot_t(d⃗,ε⁻¹)
	@tullio e⃗[a,i,j,k] :=  ε⁻¹[a,b,i,j,k] * fft(d⃗,(2:4))[b,i,j,k] fastmath=false
	return ifft(e⃗,(2:4))
end

"""
    ε⁻¹_dot: e⃗  = ε⁻¹ ⋅ d⃗ (cartesian vectors)
"""
function ε⁻¹_dot(d⃗,ε⁻¹)
	@tullio e⃗[a,i,j,k] :=  ε⁻¹[a,b,i,j,k] * d⃗[b,i,j,k] fastmath=false
end

"""
    ε_dot_approx: approximate     d⃗  = ε ⋅ e⃗
                    using         d⃗  ≈  e⃗ * ( 3 / Tr(ε⁻¹) )
    (all cartesian vectors)
"""
function ε_dot_approx(e⃗,ε⁻¹)
    @tullio d⃗[b,i,j,k] := e⃗[b,i,j,k] * 3 / ε⁻¹[a,a,i,j,k] fastmath=false
end

function M(H,ε⁻¹,mn,kpg_mag)
    kx_c2t(ε⁻¹_dot_t(kx_t2c(H,mn,kpg_mag),ε⁻¹),mn,kpg_mag)
end

function M(H,ε⁻¹,mn,kpg_mag,𝓕::FFTW.cFFTWPlan,𝓕⁻¹)
    kx_c2t( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * kx_t2c(H,mn,kpg_mag), ε⁻¹), mn,kpg_mag)
end

function M(Hin::AbstractArray{ComplexF64,1},ε⁻¹,mn,kpg_mag)::Array{ComplexF64,1}
    HinA = reshape(Hin,(2,size(ε⁻¹)[end-2:end]...))
    return vec(M(HinA,ε⁻¹,mn,kpg_mag))
end

function M(Hin::AbstractArray{ComplexF64,1},ε⁻¹,mn,kpg_mag,𝓕::FFTW.cFFTWPlan,𝓕⁻¹)::Array{ComplexF64,1}
    HinA = reshape(Hin,(2,size(ε⁻¹)[end-2:end]...))
    return vec(M(HinA,ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹))
end

M̂(ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> M(H,ε⁻¹,mn,kpg_mag,𝓕,𝓕⁻¹)::AbstractArray{ComplexF64,1},*(2,size(ε⁻¹)[end-2:end]...),ishermitian=true,ismutating=false)

function P(Hin::AbstractArray{ComplexF64,4},ε⁻¹,mn,kpg_mag,𝓕::FFTW.cFFTWPlan,𝓕⁻¹)::AbstractArray{ComplexF64,4}
    kxinv_c2t( 𝓕 * ε_dot_approx( 𝓕⁻¹ * kxinv_t2c(H,mn,kpg_mag),ε⁻¹),mn,kpg_mag)
end

function P(Hin::AbstractArray{ComplexF64,1},ε⁻¹,mn,kpg_mag,𝓕)::AbstractArray{ComplexF64,1}
    HinA = reshape(Hin,(2,size(ε⁻¹)[end-2:end]...))
    return vec(P(HinA,ε⁻¹,mn,kpg_mag,𝓕))
end

P̂(ε⁻¹,mn,kpg_mag,𝓕) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> P(H,ε⁻¹,mn,kpg_mag,𝓕)::AbstractArray{ComplexF64,1},*(2,size(ε⁻¹)[end-2:end]...),ishermitian=true,ismutating=false)

function Mₖ(H,ε⁻¹,mn,kpg_mag)
    kx_c2t(ε⁻¹_dot_t(zx_t2c(H,mn),ε⁻¹),mn,kpg_mag)
end

function Mₖ(H,ε⁻¹,mn,kpg_mag,𝓕::FFTW.cFFTWPlan,𝓕⁻¹)
    kx_c2t( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * zx_t2c(H,mn), ε⁻¹), mn,kpg_mag)
end

function Mₖ(H::AbstractArray{ComplexF64,1},ε⁻¹,mn,kpg_mag)::AbstractArray{ComplexF64,1}
    Ha = reshape(H,(2,size(ε⁻¹)[end-2:end]...))
    return vec(Mₖ(Ha,ε⁻¹,mn,kpg_mag))
end

function Mₖ(H::AbstractArray{ComplexF64,1},ε⁻¹::AbstractArray{ComplexF64,5},mn,kpg_mag,𝓕::FFTW.cFFTWPlan,𝓕⁻¹)::Array{ComplexF64,1}
    Ha = reshape(H,(2,size(ε⁻¹)[end-2:end]...))
    return vec(M(Ha,ε⁻¹,mn,kpg_mag,𝓕))
end

M̂ₖ(ε⁻¹,mn,kpg_mag,𝓕) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> Mₖ(H,ε⁻¹,mn,kpg_mag,𝓕)::AbstractArray{ComplexF64,1},*(2,size(ε⁻¹)[end-2:end]...),ishermitian=true,ismutating=false)

function H_Mₖ_H(H::AbstractArray{ComplexF64,4},ε⁻¹::AbstractArray{Float64,5},kpg_mag,mn)
	kxinds = [2; 1]
	kxscales = [-1.; 1.]
	@tullio out[_] := conj.(H)[b,i,j,k] * kxscales[b] * kpg_mag[i,j,k] * ε⁻¹_dot_t(zx_t2c(H,mn),ε⁻¹)[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds) fastmath=false
	return abs(out[1])
end

# function H_Mₖ_H(H::AbstractArray{ComplexF64,1},ε⁻¹::AbstractArray{ComplexF64,5},kpg_mag,mn,𝓕::FFTW.cFFTWPlan,𝓕⁻¹)::Float64
# 	kxinds = [2; 1]
# 	kxscales = [-1.; 1.]
# 	@tullio out[_] := conj.(H)[b,i,j,k] * kxscales[b] * kpg_mag[i,j,k] * ( 𝓕⁻¹ * ε⁻¹_dot( 𝓕 * zx_t2c(H,mn), ε⁻¹) )[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds) fastmath=false
# 	return abs(out[1])
# end

# H_Mₖ_H(H::AbstractArray{ComplexF64,1},ε⁻¹::AbstractArray{ComplexF64,5},kpg_mag,mn,𝓕::FFTW.cFFTWPlan,𝓕⁻¹)  = H_Mₖ_H(reshape(H,(2,size(ε⁻¹)[end-2:end]...)),ε⁻¹,kpg_mag,mn,𝓕,𝓕⁻¹)
H_Mₖ_H(H::AbstractArray{ComplexF64,1},ε⁻¹::AbstractArray{Float64,5},kpg_mag,mn) = H_Mₖ_H(reshape(H,(2,size(ε⁻¹)[end-2:end]...)),ε⁻¹,kpg_mag,mn)
# H_Mₖ_H(H::AbstractArray{ComplexF64,2},ε⁻¹::AbstractArray{Float64,5},kpg_mag,mn,𝓕::FFTW.cFFTWPlan,𝓕⁻¹)  = H_Mₖ_H(reshape(H,(2,size(ε⁻¹)[end-2:end]...)),ε⁻¹,kpg_mag,mn,𝓕,𝓕⁻¹)
H_Mₖ_H(H::AbstractArray{ComplexF64,2},ε⁻¹::AbstractArray{Float64,5},kpg_mag,mn) = H_Mₖ_H(reshape(H,(2,size(ε⁻¹)[end-2:end]...)),ε⁻¹,kpg_mag,mn)


###### Mutating Operators #######

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

#
# function t2c!(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @fastmath @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @fastmath @inbounds ds.e[1,i,j,k] = ( Hin[1,i,j,k] * ds.mn[1,1,i,j,k] + Hin[2,i,j,k] * ds.mn[1,2,i,j,k] )
#         @fastmath @inbounds ds.e[2,i,j,k] = ( Hin[1,i,j,k] * ds.mn[2,1,i,j,k] + Hin[2,i,j,k] * ds.mn[2,2,i,j,k] )
#     	@fastmath @inbounds ds.e[3,i,j,k] = ( Hin[1,i,j,k] * ds.mn[3,1,i,j,k] + Hin[2,i,j,k] * ds.mn[3,2,i,j,k] )
# 	end
#     return ds.e
# end
#
# function c2t!(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @fastmath @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @fastmath @inbounds ds.e[1,i,j,k] =  Hin[1,i,j,k] * ds.mn[1,1,i,j,k] + Hin[2,i,j,k] * ds.mn[2,1,i,j,k] + Hin[3,i,j,k] * ds.mn[3,1,i,j,k]
#         @fastmath @inbounds ds.e[2,i,j,k] =  Hin[1,i,j,k] * ds.mn[1,2,i,j,k] + Hin[2,i,j,k] * ds.mn[2,2,i,j,k] + Hin[3,i,j,k] * ds.mn[3,2,i,j,k]
#     end
#     return ds.e
# end
#
# function zcross_t2c!(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @fastmath @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @fastmath @inbounds ds.e[1,i,j,k] = -Hin[1,i,j,k] * ds.mn[2,1,i,j,k] - Hin[2,i,j,k] * ds.mn[2,2,i,j,k]
#         @fastmath @inbounds ds.e[2,i,j,k] =  Hin[1,i,j,k] * ds.mn[1,1,i,j,k] + Hin[2,i,j,k] * ds.mn[1,2,i,j,k]
#     end
#     return ds.e
# end
#
# function kcross_t2c!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @fastmath @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @fastmath @inbounds ds.d[1,i,j,k] = ( ds.H[1,i,j,k] * ds.mn[1,2,i,j,k] - ds.H[2,i,j,k] * ds.mn[1,1,i,j,k] ) * -ds.kpg_mag[i,j,k]
#         @fastmath @inbounds ds.d[2,i,j,k] = ( ds.H[1,i,j,k] * ds.mn[2,2,i,j,k] - ds.H[2,i,j,k] * ds.mn[2,1,i,j,k] ) * -ds.kpg_mag[i,j,k]
#         @fastmath @inbounds ds.d[3,i,j,k] = ( ds.H[1,i,j,k] * ds.mn[3,2,i,j,k] - ds.H[2,i,j,k] * ds.mn[3,1,i,j,k] ) * -ds.kpg_mag[i,j,k]
#     end
#     return ds.d
# end
#
# function kcross_c2t!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @fastmath @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
# 		@fastmath @inbounds  ds.H[1,i,j,k] =  (	ds.e[1,i,j,k] * ds.mn[1,2,i,j,k] + ds.e[2,i,j,k] * ds.mn[2,2,i,j,k] + ds.e[3,i,j,k] * ds.mn[3,2,i,j,k]	) * -ds.kpg_mag[i,j,k]
# 		@fastmath @inbounds  ds.H[2,i,j,k] =  (	ds.e[1,i,j,k] * ds.mn[1,1,i,j,k] + ds.e[2,i,j,k] * ds.mn[2,1,i,j,k] + ds.e[3,i,j,k] * ds.mn[3,1,i,j,k]	) * ds.kpg_mag[i,j,k]
#     end
#     return ds.H
# end
#
# function ε⁻¹_dot!(ε⁻¹::Array{Float64,5},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @fastmath @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @fastmath @inbounds ds.e[1,i,j,k] =  ε⁻¹[1,1,i,j,k]*ds.d[1,i,j,k] + ε⁻¹[2,1,i,j,k]*ds.d[2,i,j,k] + ε⁻¹[3,1,i,j,k]*ds.d[3,i,j,k]
#         @fastmath @inbounds ds.e[2,i,j,k] =  ε⁻¹[1,2,i,j,k]*ds.d[1,i,j,k] + ε⁻¹[2,2,i,j,k]*ds.d[2,i,j,k] + ε⁻¹[3,2,i,j,k]*ds.d[3,i,j,k]
#         @fastmath @inbounds ds.e[3,i,j,k] =  ε⁻¹[1,3,i,j,k]*ds.d[1,i,j,k] + ε⁻¹[2,3,i,j,k]*ds.d[2,i,j,k] + ε⁻¹[3,3,i,j,k]*ds.d[3,i,j,k]
#         # ds.e[1,i,j,k] =  ε⁻¹[1,1,i,j,k]*ds.d[1,i,j,k] + ε⁻¹[1,2,i,j,k]*ds.d[2,i,j,k] + ε⁻¹[1,3,i,j,k]*ds.d[3,i,j,k]
#         # ds.e[2,i,j,k] =  ε⁻¹[2,1,i,j,k]*ds.d[1,i,j,k] + ε⁻¹[2,2,i,j,k]*ds.d[2,i,j,k] + ε⁻¹[2,3,i,j,k]*ds.d[3,i,j,k]
#         # ds.e[3,i,j,k] =  ε⁻¹[3,1,i,j,k]*ds.d[1,i,j,k] + ε⁻¹[3,2,i,j,k]*ds.d[2,i,j,k] + ε⁻¹[3,3,i,j,k]*ds.d[3,i,j,k]
#     end
#     return ds.e
# end
#
# function M!(ε⁻¹::Array{Float64,5},ds::MaxwellData)::Array{ComplexF64,4}
#     kcross_t2c!(ds);
#     mul!(ds.d,ds.𝓕!,ds.d);
#     ε⁻¹_dot!(ε⁻¹,ds);
# 	mul!(ds.e,ds.𝓕⁻¹!::AbstractFFTs.ScaledPlan,ds.e);
#     kcross_c2t!(ds)
# end

function M!(Hout::AbstractArray{ComplexF64,1},Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{Float64,5},ds::MaxwellData)::Array{ComplexF64,1}
    @inbounds ds.H .= reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
    M!(ε⁻¹,ds);
    @inbounds Hout .= vec(ds.H)
end

M̂!(ε⁻¹::Array{Float64,5},ds::MaxwellData) = LinearMap{ComplexF64}((2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=true) do y::AbstractVector{ComplexF64},x::AbstractVector{ComplexF64}
    M!(y,x,ε⁻¹,ds)::AbstractArray{ComplexF64,1}
    end

# function M̂!(ε⁻¹::Array{Float64,5},ds::MaxwellData)
#     function f!(y::AbstractArray{ComplexF64,1},x::AbstractArray{ComplexF64,1})::AbstractArray{ComplexF64,1}
#         M!(y,x,ε⁻¹,ds)
#     end
#     return LinearMap{ComplexF64}(f!,(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=true)
# end


### Preconditioner P̂ & Component Operators (approximate inverse operations of M̂) ###

function kcrossinv_t2c!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
    @fastmath @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @fastmath @inbounds scale::Float64 = inv(ds.kpg_mag[i,j,k])
        @fastmath @inbounds ds.e[1,i,j,k] = ( ds.H[1,i,j,k] * ds.mn[1,2,i,j,k] - ds.H[2,i,j,k] * ds.mn[1,1,i,j,k] ) * scale
        @fastmath @inbounds ds.e[2,i,j,k] = ( ds.H[1,i,j,k] * ds.mn[2,2,i,j,k] - ds.H[2,i,j,k] * ds.mn[2,1,i,j,k] ) * scale
        @fastmath @inbounds ds.e[3,i,j,k] = ( ds.H[1,i,j,k] * ds.mn[3,2,i,j,k] - ds.H[2,i,j,k] * ds.mn[3,1,i,j,k] ) * scale
    end
    return ds.e
end

function kcrossinv_c2t!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
    @fastmath @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @fastmath @inbounds scale = -inv(ds.kpg_mag[i,j,k])
        @fastmath @inbounds ds.H[1,i,j,k] =  (	ds.d[1,i,j,k] * ds.mn[1,2,i,j,k] + ds.d[2,i,j,k] * ds.mn[2,2,i,j,k] + ds.d[3,i,j,k] * ds.mn[3,2,i,j,k]	) * -scale
        @fastmath @inbounds ds.H[2,i,j,k] =  (	ds.d[1,i,j,k] * ds.mn[1,1,i,j,k] + ds.d[2,i,j,k] * ds.mn[2,1,i,j,k] + ds.d[3,i,j,k] * ds.mn[3,1,i,j,k]	) * scale
    end
    return ds.H
end

function ε_dot_approx!(ε⁻¹::AbstractArray{Float64,5},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    @fastmath @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @fastmath @inbounds ε_ave = 3. * inv( ε⁻¹[1,1,i,j,k] + ε⁻¹[2,2,i,j,k] + ε⁻¹[3,3,i,j,k] ) # tr(ε⁻¹[:,:,i,j,k])
        @fastmath @inbounds ds.d[1,i,j,k] =  ε_ave * ds.e[1,i,j,k]
        @fastmath @inbounds ds.d[2,i,j,k] =  ε_ave * ds.e[2,i,j,k]
        @fastmath @inbounds ds.d[3,i,j,k] =  ε_ave * ds.e[3,i,j,k]
    end
    return ds.d
end

function P!(ε⁻¹::AbstractArray{Float64,5},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    kcrossinv_t2c!(ds);
    # ds.𝓕⁻¹! * ds.e;
    # ldiv!(ds.e,ds.𝓕!,ds.e)
	mul!(ds.e,ds.𝓕⁻¹!,ds.e);
    ε_dot_approx!(ε⁻¹,ds);
    # ds.𝓕! * ds.d;
    mul!(ds.d,ds.𝓕!,ds.d);
    kcrossinv_c2t!(ds)
end

function P!(Hout::AbstractArray{ComplexF64,1},Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{Float64,5},ds::MaxwellData)::AbstractArray{ComplexF64,1}
    @inbounds ds.H .= reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
    P!(ε⁻¹,ds);
    @inbounds Hout .= vec(ds.H)
end

P̂!(ε⁻¹::Array{Float64,5},ds::MaxwellData) = LinearMap{ComplexF64}((2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=true) do y::AbstractVector{ComplexF64},x::AbstractVector{ComplexF64}
	P!(y,x,ε⁻¹,ds)::AbstractArray{ComplexF64,1}
    end


# function P̂!(ε⁻¹::Array{Float64,5},ds::MaxwellData)
#     function fp!(y::AbstractArray{ComplexF64,1},x::AbstractArray{ComplexF64,1})::AbstractArray{ComplexF64,1}
#         P!(y,x,ε⁻¹,ds)
#     end
#     return LinearMap{ComplexF64}(fp!,(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=true)
# end










###########################################################################################################################


# function zcross_t2c!(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         @inbounds ds.e[1,i,j,k] = -Hin[1,i,j,k] * ds.kpG[i,j,k].m[2] - Hin[2,i,j,k] * ds.kpG[i,j,k].n[2]
#         @inbounds ds.e[2,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[1]
#     end
#     return ds.e
# end
#
# function kcross_t2c!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         scale = -ds.kpg_mag #-ds.kpG[i,j,k].mag
#         ds.d[1,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[1] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[1] ) * scale
#         ds.d[2,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[2] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[2] ) * scale
#         ds.d[3,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[3] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[3] ) * scale
#     end
#     return ds.d
# end
#
# function kcross_c2t!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         scale = ds.kpG[i,j,k].mag
#         at1 = ds.e[1,i,j,k] * ds.kpG[i,j,k].m[1] + ds.e[2,i,j,k] * ds.kpG[i,j,k].m[2] + ds.e[3,i,j,k] * ds.kpG[i,j,k].m[3]
#         at2 = ds.e[1,i,j,k] * ds.kpG[i,j,k].n[1] + ds.e[2,i,j,k] * ds.kpG[i,j,k].n[2] + ds.e[3,i,j,k] * ds.kpG[i,j,k].n[3]
#         ds.H[1,i,j,k] =  -at2 * scale
#         ds.H[2,i,j,k] =  at1 * scale
#     end
#     return ds.H
# end
#
# function kcrossinv_t2c!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         scale = 1 / ds.kpG[i,j,k].mag
#         ds.e[1,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[1] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[1] ) * scale
#         ds.e[2,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[2] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[2] ) * scale
#         ds.e[3,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[3] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[3] ) * scale
#     end
#     return ds.e
# end
#
# function kcrossinv_c2t!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         scale = -1 / ds.kpG[i,j,k].mag
#         at1 = ds.d[1,i,j,k] * ds.kpG[i,j,k].m[1] + ds.d[2,i,j,k] * ds.kpG[i,j,k].m[2] + ds.d[3,i,j,k] * ds.kpG[i,j,k].m[3]
#         at2 = ds.d[1,i,j,k] * ds.kpG[i,j,k].n[1] + ds.d[2,i,j,k] * ds.kpG[i,j,k].n[2] + ds.d[3,i,j,k] * ds.kpG[i,j,k].n[3]
#         ds.H[1,i,j,k] =  -at2 * scale
#         ds.H[2,i,j,k] =  at1 * scale
#     end
#     return ds.H
# end
#
# function ε⁻¹_dot!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         ds.e[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*ds.d[1,i,j,k] + ε⁻¹[i,j,k][2,1]*ds.d[2,i,j,k] + ε⁻¹[i,j,k][3,1]*ds.d[3,i,j,k]
#         ds.e[2,i,j,k] =  ε⁻¹[i,j,k][1,2]*ds.d[1,i,j,k] + ε⁻¹[i,j,k][2,2]*ds.d[2,i,j,k] + ε⁻¹[i,j,k][3,2]*ds.d[3,i,j,k]
#         ds.e[3,i,j,k] =  ε⁻¹[i,j,k][1,3]*ds.d[1,i,j,k] + ε⁻¹[i,j,k][2,3]*ds.d[2,i,j,k] + ε⁻¹[i,j,k][3,3]*ds.d[3,i,j,k]
#         # ds.e[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][1,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][1,3]*Hin[3,i,j,k]
#         # ds.e[2,i,j,k] =  ε⁻¹[i,j,k][2,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][2,3]*Hin[3,i,j,k]
#         # ds.e[3,i,j,k] =  ε⁻¹[i,j,k][3,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][3,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,3]*Hin[3,i,j,k]
#     end
#     return ds.e
# end
#
# function ε_dot_approx!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
#     @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
#         ε_ave = 3 / tr(ε⁻¹[i,j,k])
#         ds.d[1,i,j,k] =  ε_ave * ds.e[1,i,j,k]
#         ds.d[2,i,j,k] =  ε_ave * ds.e[2,i,j,k]
#         ds.d[3,i,j,k] =  ε_ave * ds.e[3,i,j,k]
#     end
#     return ds.d
# end
#
# function M!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
#     kcross_t2c!(ds);
#     # ds.𝓕! * ds.d;
#     mul!(ds.d,ds.𝓕!,ds.d);
#     ε⁻¹_dot!(ε⁻¹,ds);
#     # ds.𝓕⁻¹! * ds.e;
#     mul!(ds.e,ds.𝓕⁻¹!,ds.e)
#     kcross_c2t!(ds)
# end
#
# function M!(Hout::AbstractArray{ComplexF64,1},Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
#     # copyto!(ds.H,reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz)))
#     @inbounds ds.H .= reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
#     M!(ε⁻¹,ds);
#     # copyto!(Hout,vec(ds.H))
#     @inbounds Hout .= vec(ds.H)
# end
#
# function M̂!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)
#     function f!(y::AbstractArray{ComplexF64,1},x::AbstractArray{ComplexF64,1})::AbstractArray{ComplexF64,1}
#         M!(y,x,ε⁻¹,ds)
#     end
#     return LinearMap{ComplexF64}(f!,(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=true)
# end
#
# function P!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
#     kcrossinv_t2c!(ds);
#     # ds.𝓕⁻¹! * ds.e;
#     mul!(ds.e,ds.𝓕⁻¹!,ds.e)
#     ε_dot_approx!(ε⁻¹,ds);
#     # ds.𝓕! * ds.d;
#     mul!(ds.d,ds.𝓕!,ds.d);
#     kcrossinv_c2t!(ds)
# end
#
# function P!(Hout::AbstractArray{ComplexF64,1},Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
#     # copyto!(ds.H,reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz)))
#     @inbounds ds.H .= reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
#     P!(ε⁻¹,ds);
#     # copyto!(Hout,vec(ds.H))
#     @inbounds Hout .= vec(ds.H)
# end
#
# function P̂!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)
#     function fp!(y::AbstractArray{ComplexF64,1},x::AbstractArray{ComplexF64,1})::AbstractArray{ComplexF64,1}
#         P!(y,x,ε⁻¹,ds)
#     end
#     return LinearMap{ComplexF64}(fp!,(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=true)
# end
