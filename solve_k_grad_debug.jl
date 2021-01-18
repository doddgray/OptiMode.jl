using Revise
include("scripts/mpb_compare.jl")

## MPB find_k data from a ridge waveguide for reference
λ = p_def[1]
Δx = 6.0
Δy = 4.0
Δz = 1.0
Nx = 128
Ny = 128
Nz = 1
ω = 1/λ
n_mpb,ng_mpb = nng_rwg_mpb(p_def)
k_mpb = n_mpb * ω
ei_mpb = ε⁻¹_rwg_mpb(p_def)
p̄_mpb_FD = ∇nng_rwg_mpb_FD(p_def)
##
H_OM, k_OM = solve_k(ω,ei_mpb,Δx,Δy,Δz)
n_OM, ng_OM = solve_n(ω,ei_mpb,Δx,Δy,Δz)

##
using LinearAlgebra, LinearMaps, IterativeSolvers
neigs=1
eigind=1
maxiter=3000
tol=1e-8
ε⁻¹ = copy(ei_mpb)
g = make_MG(Δx,Δy,Δz,size(ε⁻¹)[end-2:end]...)
ds = make_MD(k_mpb,g)
# mag,mn = calc_kpg(k_mpb,g.g⃗)
##
using  LinearAlgebra, StaticArrays, HybridArrays, ArrayInterface, LoopVectorization, LinearMaps, FFTW, AbstractFFTs, ChainRules, Zygote, Tullio, IterativeSolvers, BenchmarkTools
using OptiMode: MaxwellGrid, MaxwellData, make_εₛ⁻¹, ridge_wg, M̂!, P̂!
using StaticArrays: Dynamic, SVector

# Non-mutating Operators

"""
    kx_tc: a⃗ (cartesian vector) = k⃗ × v⃗ (transverse vector)
"""
function kx_tc(H,mn,mag)
	# kxscales = [-1.; 1.]
	# kxinds = [2; 1]
    # @tullio d[a,i,j,k] := kxscales[b] * H[kxinds[b],i,j,k] * mn[a,b,i,j,k] * kpg_mag[i,j,k] nograd=(kxscales,kxinds) fastmath=false
	@tullio d[i,j,k,a] := H[i,j,k,2] * m[i,j,k,a] * mag[i,j,k] - H[i,j,k,1] * n[i,j,k,a] * mag[i,j,k]  # nograd=(kxscales,kxinds) fastmath=false
end

"""
    kx_c2t: v⃗ (transverse vector) = k⃗ × a⃗ (cartesian vector)
"""
function kx_ct(e⃗,mn,mag)
	# mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])))
	kxscales = [-1.; 1.]
    kxinds = [2; 1]
    @tullio H[i,j,k,b] := kxscales[b] * e⃗[i,j,k,a] * mn[kxinds[b],i,j,k,a] * mag[i,j,k] nograd=(kxinds,kxscales) # fastmath=false
end

"""
    zx_t2c: a⃗ (cartesian vector) = ẑ × v⃗ (transverse vector)
"""
function zx_tc(H,mn)
	zxinds = [2; 1; 3]
	zxscales = [-1.; 1.; 0.]
	@tullio zxH[i,j,k,a] := zxscales[a] * H[i,j,k,b] * mn[b,i,j,k,zxinds[a]] nograd=(zxscales,zxinds) # fastmath=false
end

"""
    ε⁻¹_dot_t: e⃗  = ε⁻¹ ⋅ d⃗ (transverse vectors)
"""
function ε⁻¹_dot_t(d⃗,ε⁻¹)
	@tullio e⃗[i,j,k,a] :=  ε⁻¹[i,j,k,a,b] * fft(d⃗,(1:3))[i,j,k,b]  #fastmath=false
	return ifft(e⃗,(1:3))
end

"""
    ε⁻¹_dot: e⃗  = ε⁻¹ ⋅ d⃗ (cartesian vectors)
"""
function ε⁻¹_dot(d⃗,ε⁻¹)
	@tullio e⃗[i,j,k,a] :=  ε⁻¹[i,j,k,a,b] * d⃗[i,j,k,b]  #fastmath=false
end

function H_Mₖ_H(H::AbstractArray{Complex{T},4},ε⁻¹::AbstractArray{T,5},mag,m,n)::T where T<:Real
	# kxinds = [2; 1]
	# kxscales = [-1.; 1.]
	# ,i,j,k] * kxscales[b] * kpg_mag[i,j,k] * temp[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds) nograd=(kxscales,kxinds) fastmath=false
	# @tullio out := conj.(H)[b,i,j,k] * kxscales[b] * kpg_mag[i,j,k] * ε⁻¹_dot_t(zx_t2c(H,mn),ε⁻¹)[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds) nograd=(kxscales,kxinds) fastmath=false
	# return abs(out[1])
	mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3],size(m)[4])))
	real( dot(H, -kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mn), (1:3) ), ε⁻¹), (1:3)),mn,mag) ) )
end

function H_Mₖ_H(H::AbstractVector{Complex{T}},ε⁻¹::AbstractArray{T,5},mag::AbstractArray{T,3},m::AbstractArray{T,4},n::AbstractArray{T,4})::T where T<:Real
	Nx,Ny,Nz = size(mag)
	Ha = reshape(H,(Nx,Ny,Nz,2))
	H_Mₖ_H(Ha,ε⁻¹,mag,m,n)
end



# Mutating Operators

function kx_tc!(d::AbstractArray{Complex{T},4},H::AbstractArray{Complex{T},4},m::AbstractArray{T,4},n::AbstractArray{T,4},mag::AbstractArray{T,3})::AbstractArray{Complex{T},4} where T<:Real
    # @assert size(Y) === size(X)
    # @assert size(d,4) == 3
    # @assert size(H,4) === 2
    @avx for k ∈ axes(d,3), j ∈ axes(d,2), i ∈ axes(d,1), l in 0:0
	# @avx for i ∈ axes(d,1), j ∈ axes(d,2), k ∈ axes(d,3), l in 0:0
		# scale = -mag[i,j,k]
		d[i,j,k,1+l] = ( H[i,j,k,1] * n[i,j,k,1+l] - H[i,j,k,2] * m[i,j,k,1+l] ) * -mag[i,j,k]
        d[i,j,k,2+l] = ( H[i,j,k,1] * n[i,j,k,2+l] - H[i,j,k,2] * m[i,j,k,2+l] ) * -mag[i,j,k]
        d[i,j,k,3+l] = ( H[i,j,k,1] * n[i,j,k,3+l] - H[i,j,k,2] * m[i,j,k,3+l] ) * -mag[i,j,k]
    end
    return d
end

function zx_tc!(d::AbstractArray{Complex{T},4},H::AbstractArray{Complex{T},4},m::AbstractArray{T,4},n::AbstractArray{T,4})::AbstractArray{Complex{T},4} where T<:Real
    @avx for k ∈ axes(d,3), j ∈ axes(d,2), i ∈ axes(d,1), l in 0:0
		d[i,j,k,1+l] = -H[i,j,k,1] * m[i,j,k,2+l] - H[i,j,k,2] * n[i,j,k,2+l]
        d[i,j,k,2+l] =  H[i,j,k,1] * m[i,j,k,1+l] + H[i,j,k,2] * n[i,j,k,1+l]
    end
    return d
end

function kx_ct!(H::AbstractArray{Complex{T},4},e::AbstractArray{Complex{T},4},m::AbstractArray{T,4},n::AbstractArray{T,4},mag::AbstractArray{T,3},Ninv::T)::AbstractArray{Complex{T},4} where T<:Real
    @avx for k ∈ axes(H,3), j ∈ axes(H,2), i ∈ axes(H,1), l in 0:0
        scale = mag[i,j,k] * Ninv
        H[i,j,k,1+l] =  (	e[i,j,k,1+l] * n[i,j,k,1+l] + e[i,j,k,2+l] * n[i,j,k,2+l] + e[i,j,k,3+l] * n[i,j,k,3+l]	) * -scale  # -mag[i,j,k] * Ninv
		H[i,j,k,2+l] =  (	e[i,j,k,1+l] * m[i,j,k,1+l] + e[i,j,k,2+l] * m[i,j,k,2+l] + e[i,j,k,3+l] * m[i,j,k,3+l]	) * scale   # mag[i,j,k] * Ninv
    end
    return H
end

function eid!(e::AbstractArray{Complex{T},4},ε⁻¹::AbstractArray{T,5},d::AbstractArray{Complex{T},4})::AbstractArray{Complex{T},4} where T<:Real
    @avx for k ∈ axes(e,3), j ∈ axes(e,2), i ∈ axes(e,1), l in 0:0, h in 0:0
        e[i,j,k,1+h] =  ε⁻¹[i,j,k,1+h,1+l]*d[i,j,k,1+l] + ε⁻¹[i,j,k,2+h,1+l]*d[i,j,k,2+l] + ε⁻¹[i,j,k,3+h,1+l]*d[i,j,k,3+l]
        e[i,j,k,2+h] =  ε⁻¹[i,j,k,1+h,2+l]*d[i,j,k,1+l] + ε⁻¹[i,j,k,2+h,2+l]*d[i,j,k,2+l] + ε⁻¹[i,j,k,3+h,2+l]*d[i,j,k,3+l]
        e[i,j,k,3+h] =  ε⁻¹[i,j,k,1+h,3+l]*d[i,j,k,1+l] + ε⁻¹[i,j,k,2+h,3+l]*d[i,j,k,2+l] + ε⁻¹[i,j,k,3+h,3+l]*d[i,j,k,3+l]
    end
    return e
end

function kxinv_tc!(e::AbstractArray{Complex{T},4},H::AbstractArray{Complex{T},4},m::AbstractArray{T,4},n::AbstractArray{T,4},inv_mag::AbstractArray{T,3})::AbstractArray{Complex{T},4} where T<:Real
    @avx for k ∈ axes(e,3), j ∈ axes(e,2), i ∈ axes(e,1), l in 0:0
		e[i,j,k,1+l] = ( H[i,j,k,1] * n[i,j,k,1+l] - H[i,j,k,2] * m[i,j,k,1+l] ) * inv_mag[i,j,k]
        e[i,j,k,2+l] = ( H[i,j,k,1] * n[i,j,k,2+l] - H[i,j,k,2] * m[i,j,k,2+l] ) * inv_mag[i,j,k]
        e[i,j,k,3+l] = ( H[i,j,k,1] * n[i,j,k,3+l] - H[i,j,k,2] * m[i,j,k,3+l] ) * inv_mag[i,j,k]
    end
    return e
end

function kxinv_ct!(H::AbstractArray{Complex{T},4},d::AbstractArray{Complex{T},4},m::AbstractArray{T,4},n::AbstractArray{T,4},inv_mag::AbstractArray{T,3},N::T)::AbstractArray{Complex{T},4} where T<:Real
    @avx for k ∈ axes(H,3), j ∈ axes(H,2), i ∈ axes(H,1), l in 0:0
        scale = inv_mag[i,j,k] * N
        H[i,j,k,1+l] =  (	d[i,j,k,1+l] * n[i,j,k,1+l] + d[i,j,k,2+l] * n[i,j,k,2+l] + d[i,j,k,3+l] * n[i,j,k,3+l]	) * scale # inv_mag[i,j,k] * N
		H[i,j,k,2+l] =  (	d[i,j,k,1+l] * m[i,j,k,1+l] + d[i,j,k,2+l] * m[i,j,k,2+l] + d[i,j,k,3+l] * m[i,j,k,3+l]	) * -scale # inv_mag[i,j,k] * N
    end
    return H
end

function ed_approx!(d::AbstractArray{Complex{T},4},ε_ave::AbstractArray{T,3},e::AbstractArray{Complex{T},4})::AbstractArray{Complex{T},4} where T<:Real
    @avx for k ∈ axes(e,3), j ∈ axes(e,2), i ∈ axes(e,1), l in 0:0
        d[i,j,k,1+l] =  ε_ave[i,j,k]*e[i,j,k,1+l]
        d[i,j,k,2+l] =  ε_ave[i,j,k]*e[i,j,k,2+l]
        d[i,j,k,3+l] =  ε_ave[i,j,k]*e[i,j,k,3+l]
    end
    return d
end

function _P!(Hout::AbstractArray{Complex{T},4}, Hin::AbstractArray{Complex{T},4},
	e::AbstractArray{Complex{T},4}, d::AbstractArray{Complex{T},4}, ε_ave::AbstractArray{T,3},
	m::AbstractArray{T,4}, n::AbstractArray{T,4}, inv_mag::AbstractArray{T,3},
	𝓕!::FFTW.cFFTWPlan, 𝓕⁻¹!::FFTW.cFFTWPlan,
	Ninv::T)::AbstractArray{Complex{T},4} where T<:Real
	kxinv_tc!(e,Hin,m,n,inv_mag);
	mul!(e.data,𝓕⁻¹!,e.data);
    ed_approx!(d,ε_ave,e);
    mul!(d.data,𝓕!,d.data);
    kxinv_ct!(Hout,d,m,n,inv_mag,Ninv)
end

function _M!(Hout::AbstractArray{Complex{T},4}, Hin::AbstractArray{Complex{T},4},
	e::AbstractArray{Complex{T},4}, d::AbstractArray{Complex{T},4}, ε⁻¹::AbstractArray{T,5},
	m::AbstractArray{T,4}, n::AbstractArray{T,4}, mag::AbstractArray{T,3},
	𝓕!::FFTW.cFFTWPlan, 𝓕⁻¹!::FFTW.cFFTWPlan,
	Ninv::T)::AbstractArray{Complex{T},4} where T<:Real
    kx_tc!(d,Hin,m,n,mag);
    mul!(d.data,𝓕!,d.data);
    eid!(e,ε⁻¹,d);
    mul!(e.data,𝓕⁻¹!,e.data);
    kx_ct!(Hout,e,m,n,mag,Ninv)
end

function mag_m_n(k⃗::SVector{3,T},g⃗::HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T,4,4,Array{T,4}}) where T <: Real
	g⃗ₜ_zero_mask = Zygote.@ignore(  sum(abs2,g⃗[:,:,:,1:2];dims=4)[:,:,:,1] .> 0. );
	g⃗ₜ_zero_mask! = Zygote.@ignore( .!(g⃗ₜ_zero_mask) );
	local ŷ = [0.; 1. ;0.]
	local zxinds = [2; 1; 3]
	local zxscales = [-1; 1. ;0.]
	local xinds1 = [2; 3; 1]
	local xinds2 = [3; 1; 2]
	@tullio kpg[ix,iy,iz,a] := k⃗[a] - g⃗[ix,iy,iz,a] fastmath=false
	@tullio mag[ix,iy,iz] := sqrt <| kpg[ix,iy,iz,a]^2 fastmath=false
	@tullio nt[ix,iy,iz,a] := zxscales[a] * kpg[ix,iy,iz,zxinds[a]] * g⃗ₜ_zero_mask[ix,iy,iz] + ŷ[a] * g⃗ₜ_zero_mask![ix,iy,iz]  nograd=(zxscales,zxinds,ŷ,g⃗ₜ_zero_mask,g⃗ₜ_zero_mask!) fastmath=false
	@tullio nmag[ix,iy,iz] := sqrt <| nt[ix,iy,iz,a]^2 fastmath=false
	@tullio n[ix,iy,iz,a] := nt[ix,iy,iz,a] / nmag[ix,iy,iz] fastmath=false
	@tullio mt[ix,iy,iz,a] := n[ix,iy,iz,xinds1[a]] * kpg[ix,iy,iz,xinds2[a]] - kpg[ix,iy,iz,xinds1[a]] * n[ix,iy,iz,xinds2[a]] nograd=(xinds1,xinds2) fastmath=false
	@tullio mmag[ix,iy,iz] := sqrt <| mt[ix,iy,iz,a]^2 fastmath=false
	@tullio m[ix,iy,iz,a] := mt[ix,iy,iz,a] / mmag[ix,iy,iz] fastmath=false
	return mag, m, n
end

function mag_m_n(kz::T,g⃗::HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T,4,4,Array{T,4}}) where T <: Real
	mag_m_n(SVector{3,T}(0.,0.,kz),g⃗)
end

function mag_m_n(kz::T,Δx::T, Δy::T, Δz::T, Nx::Int, Ny::Int, Nz::Int) where T <: Real
	g⃗ = HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3}}(
		permutedims(
			reinterpret(reshape,Float64,
				[SVector(gx,gy,gz) for gx in fftfreq(Nx,Nx/Δx),
					gy in fftfreq(Ny,Ny/Δy),
					gz in fftfreq(Nz,Nz/Δz)],
			),
			(2,3,4,1)))
	mag_m_n(SVector{3,T}(0.,0.,kz),g⃗)
end

# mutable struct HelmholtzMap{T,Nx,Ny,Nz,N} <: LinearMap{T}
#     k⃗::Vector{T}
#     ε⁻¹::Array{T,5}
#     Δx::T
#     Δy::T
#     Δz::T
# 	g⃗::Array{T,5}
# 	mag::Array{T,3}
#     m::Array{T,3}
# 	n::Array{T,3}
#     e::Array{Complex{T},4}
#     d::Array{Complex{T},4}
#     𝓕::FFTW.cFFTWPlan
# 	𝓕⁻¹::AbstractFFTs.ScaledPlan
# 	Ninv::T
# 	shift::T
# end

mutable struct HelmholtzMap{T} <: LinearMap{T}
    k⃗::SVector{3,T}
    ε⁻¹::HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3,3},T,5,5,Array{T,5}}
	Δx::T
    Δy::T
    Δz::T
	Nx::Int
    Ny::Int
    Nz::Int
	δx::T
    δy::T
    δz::T
	N::Int
	Ninv::T
	shift::T
	g⃗::HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T,4,4,Array{T,4}}
	mag::Array{T,3} #HybridArray{Tuple{Nx,Ny,Nz},T,3,3,Array{T,3}}
    m⃗::HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T,4,4,Array{T,4}}
	n⃗::HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T,4,4,Array{T,4}}
    e::HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3},Complex{T},4,4,Array{Complex{T},4}}
    d::HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3},Complex{T},4,4,Array{Complex{T},4}}
    𝓕::FFTW.cFFTWPlan
	𝓕⁻¹::FFTW.cFFTWPlan #AbstractFFTs.ScaledPlan
	ε_ave::Array{T,3}  # for preconditioner
	inv_mag::Array{T,3} # for preconditioner
end

mutable struct HelmholtzPreconditioner{T} <: LinearMap{T}
	M̂::HelmholtzMap{T}
	# 𝓕::FFTW.cFFTWPlan
	# 𝓕⁻¹::FFTW.cFFTWPlan #AbstractFFTs.ScaledPlan
	# e::HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3},Complex{T},4,4,Array{Complex{T},4}}
    # d::HybridArray #{Tuple{Dynamic(),Dynamic(),Dynamic(),3},Complex{T},4,4,Array{Complex{T},4}}
end

# HelmholtzPreconditioner(M̂::HelmholtzMap{T}) where T = HelmholtzPreconditioner{T}(
# 	M̂,
# 	plan_fft!(randn(ComplexF64, (M̂.Nx,M̂.Ny,M̂.Nz,3)),(1:3)), # planned in-place FFT operator 𝓕!
# 	plan_bfft!(randn(ComplexF64, (M̂.Nx,M̂.Ny,M̂.Nz,3)),(1:3)),
# 	HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},Complex{T}}(randn(ComplexF64, (M̂.Nx,M̂.Ny,M̂.Nz,3))),
#     HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},Complex{T}}(randn(ComplexF64, (M̂.Nx,M̂.Ny,M̂.Nz,3))),
# )

function _g⃗(Δx,Δy,Δz,Nx,Ny,Nz)
	HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3}}(
		permutedims(
			reinterpret(reshape,Float64,
				[SVector(gx,gy,gz) for gx in fftfreq(Nx,Nx/Δx),
					gy in fftfreq(Ny,Ny/Δy),
					gz in fftfreq(Nz,Nz/Δz)],
			),
			(2,3,4,1)))
end

HelmholtzMap(k⃗::AbstractVector{T}, ε⁻¹::AbstractArray{T}, Δx::T, Δy::T, Δz::T, Nx::Int, Ny::Int, Nz::Int; shift=0. ) where {T<:Real} = HelmholtzMap{T}(
	SVector{3}(k⃗),
	ε⁻¹, #::HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3,3}}(ε⁻¹),
	Δx,
    Δy,
    Δz,
	Nx,
    Ny,
    Nz,
	Δx / Nx,    # δx
    Δy / Ny,    # δy
    Δz / Nz,    # δz
	*(Nx,Ny,Nz),
	1. / *(Nx,Ny,Nz),
	shift,
	(g⃗ = _g⃗(Δx,Δy,Δz,Nx,Ny,Nz) ; g⃗),
	( (mag, m, n) = mag_m_n(k⃗,g⃗) ; Array(mag.parent) ),
    HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(m.parent)),
	HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(n.parent)),
    HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},Complex{T}}(randn(ComplexF64, (Nx,Ny,Nz,3))),# (Array{T}(undef,(Nx,Ny,Nz,3))),
    HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},Complex{T}}(randn(ComplexF64, (Nx,Ny,Nz,3))),# (Array{T}(undef,(Nx,Ny,Nz,3))),
	plan_fft!(randn(ComplexF64, (Nx,Ny,Nz,3)),(1:3)), # planned in-place FFT operator 𝓕!
	plan_bfft!(randn(ComplexF64, (Nx,Ny,Nz,3)),(1:3)), # planned in-place iFFT operator 𝓕⁻¹!
	[ 3. * inv(ε⁻¹[ix,iy,iz,1,1]+ε⁻¹[ix,iy,iz,2,2]+ε⁻¹[ix,iy,iz,3,3]) for ix=1:Nx,iy=1:Ny,iz=1:Nz], # diagonal average ε for precond. ops
	[ inv(mm) for mm in mag ] # inverse |k⃗+g⃗| magnitudes for precond. ops
)

# function _M!(Hout::AbstractVector{Complex{T}}, Hin::AbstractVector{Complex{T}},
# 	e::AbstractArray{Complex{T},4}, d::AbstractArray{Complex{T},4}, ε⁻¹::AbstractArray{T,5},
# 	m::AbstractArray{T,4}, n::AbstractArray{T,4}, mag::AbstractArray{T,3},
# 	𝓕!::FFTW.cFFTWPlan, 𝓕⁻¹!::AbstractFFTs.ScaledPlan,
# 	Ninv::T)::AbstractVector{Complex{T}} where T<:Real
# 	@inbounds Hin_arr = reshape(Hin,(M̂.Nx,M̂.Ny,M̂.Nz,2))
# 	@inbounds Hout_arr = reshape(Hout,(M̂.Nx,M̂.Ny,M̂.Nz,2))
# 	vec( _M!(Hout_arr,Hin_arr,M̂.e,M̂.d,M̂.ε⁻¹,M̂.m⃗,M̂.n⃗,M̂.mag,M̂.𝓕,M̂.𝓕⁻¹,M̂.Ninv) )
# end

function (M̂::HelmholtzMap)(Hout::AbstractArray, Hin::AbstractArray)
	_M!(Hout,Hin,M̂.e,M̂.d,M̂.ε⁻¹,M̂.m⃗,M̂.n⃗,M̂.mag,M̂.𝓕,M̂.𝓕⁻¹,M̂.Ninv)
end

function (M̂::HelmholtzMap)(Hout::AbstractVector, Hin::AbstractVector)
	@inbounds Hin_arr = reshape(Hin,(M̂.Nx,M̂.Ny,M̂.Nz,2))
	@inbounds Hout_arr = reshape(Hout,(M̂.Nx,M̂.Ny,M̂.Nz,2))
	vec( _M!(Hout_arr,Hin_arr,M̂.e,M̂.d,M̂.ε⁻¹,M̂.m⃗,M̂.n⃗,M̂.mag,M̂.𝓕,M̂.𝓕⁻¹,M̂.Ninv) )
end

function (P̂::HelmholtzPreconditioner)(Hout::AbstractArray{T,4}, Hin::AbstractArray{T,4}) where T<:Union{Real, Complex}
	_P!(Hout,Hin,P̂.M̂.e,P̂.M̂.d,P̂.M̂.ε_ave,P̂.M̂.m⃗,P̂.M̂.n⃗,P̂.M̂.inv_mag,P̂.M̂.𝓕,P̂.M̂.𝓕⁻¹,P̂.M̂.Ninv)
end

function (P̂::HelmholtzPreconditioner)(Hout::AbstractVector{T}, Hin::AbstractVector{T}) where T<:Union{Real, Complex}
	@inbounds Hin_arr = reshape(Hin,(P̂.M̂.Nx,P̂.M̂.Ny,P̂.M̂.Nz,2))
	@inbounds Hout_arr = reshape(Hout,(P̂.M̂.Nx,P̂.M̂.Ny,P̂.M̂.Nz,2))
	vec( _P!(Hout_arr,Hin_arr,P̂.M̂.e,P̂.M̂.d,P̂.M̂.ε_ave,P̂.M̂.m⃗,P̂.M̂.n⃗,P̂.M̂.inv_mag,P̂.M̂.𝓕,P̂.M̂.𝓕⁻¹,P̂.M̂.Ninv) )
end

function update_k(M̂::HelmholtzMap{T},k⃗::SVector{3,T}) where T<:Real
	(mag, m, n) = mag_m_n(k⃗,M̂.g⃗)
	M̂.mag = Array(mag.parent)
	inv_mag = [inv(mm) for mm in M̂.mag]
    M̂.m⃗ = HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(m.parent))
	M̂.n⃗ = HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3},T}(Array(n.parent))
	M̂.k⃗ = k⃗
end

function update_k(M̂::HelmholtzMap{T},kz::T) where T<:Real
	update_k(M̂,SVector{3,T}(0.,0.,kz))
end

function Base.:(*)(M̂::HelmholtzMap, x::AbstractVector)
    #length(x) == A.N || throw(DimensionMismatch())
    y = similar(x, promote_type(eltype(M̂), eltype(x)), 2*M̂.N)
    M̂(y, x)
end

function _unsafe_mul!(y::AbstractVecOrMat, M̂::HelmholtzMap, x::AbstractVector)
    M̂(y, x)
end

function Base.:(*)(P̂::HelmholtzPreconditioner, x::AbstractVector)
    #length(x) == A.N || throw(DimensionMismatch())
    y = similar(x, promote_type(eltype(P̂.M̂), eltype(x)), 2*P̂.M̂.N)
    P̂(y, x)
end

function _unsafe_mul!(y::AbstractVecOrMat, P̂::HelmholtzPreconditioner, x::AbstractVector)
    P̂(y, x)
end

# property methods
import Base: size, eltype #, mul!
Base.size(A::HelmholtzMap) = (2*A.N, 2*A.N)
Base.size(A::HelmholtzMap,d::Int) = 2*A.N
Base.eltype(A::HelmholtzMap{T}) where {T<:Real}  = Complex{T}
LinearAlgebra.issymmetric(A::HelmholtzMap) = false # A._issymmetric
LinearAlgebra.ishermitian(A::HelmholtzMap) = true # A._ishermitian
LinearAlgebra.isposdef(A::HelmholtzMap)    = true # A._isposdef
ismutating(A::HelmholtzMap) = true # A._ismutating
#_ismutating(f) = first(methods(f)).nargs == 3
Base.size(A::HelmholtzPreconditioner) = (2*A.M̂.N, 2*A.M̂.N)
Base.size(A::HelmholtzPreconditioner,d::Int) = 2*A.M̂.N
Base.eltype(A::HelmholtzPreconditioner) = eltype(A.M̂)
LinearAlgebra.issymmetric(A::HelmholtzPreconditioner) = true # A._issymmetric
LinearAlgebra.ishermitian(A::HelmholtzPreconditioner) = true # A._ishermitian
LinearAlgebra.isposdef(A::HelmholtzPreconditioner)    = true # A._isposdef
ismutating(A::HelmholtzPreconditioner) = true # A._ismutating

import LinearAlgebra: mul!
function LinearAlgebra.mul!(y::AbstractVecOrMat, M̂::HelmholtzMap, x::AbstractVector)
    LinearMaps.check_dim_mul(y, M̂, x)
	M̂(y, x)
end

function LinearAlgebra.mul!(y::AbstractVecOrMat, P̂::HelmholtzPreconditioner, x::AbstractVector)
    LinearMaps.check_dim_mul(y, P̂, x)
	P̂(y, x)
end

LinearAlgebra.transpose(P̂::HelmholtzPreconditioner) = P̂
LinearAlgebra.ldiv!(c,P̂::HelmholtzPreconditioner,b) = mul!(c,P̂,b) # P̂(c, b) #


##
eltype(M̂)
eltype(P̂)
size(M̂)
M̂(Hout,Hin)
M̂(vec(Hout),vec(Hin))
*(M̂,vec(Hin))




##
p = [
    1.45,               #   propagation constant    `kz`            [μm⁻¹]
    1.7,                #   top ridge width         `w_top`         [μm]
    0.7,                #   ridge thickness         `t_core`        [μm]
    π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
    2.4,                #   core index              `n_core`        [1]
    1.4,                #   substrate index         `n_subs`        [1]
    0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
]
Δx = 6.0
Δy = 4.0
Δz = 1.0
Nx = 128
Ny = 128
Nz = 1
kz = p[1]

# ε⁻¹ = permutedims(ei_mpb,(3,4,5,1,2))
g = MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
ds = MaxwellData(kz,g)
ei = make_εₛ⁻¹(ridge_wg(p[2],p[3],p[4],p[7],p[5],p[6],Δx,Δy),g)
ε⁻¹ = HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3,3},Float64,5,5,Array{Float64,5}}(permutedims(ei,(3,4,5,1,2)))
k⃗ = SVector(0.,0.,kz)
N = *(Nx,Ny,Nz)
Ninv = 1. / N
g⃗ = HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3}}(
	permutedims(
		reinterpret(reshape,Float64,
			[SVector(gx,gy,gz) for gx in fftfreq(Nx,Nx/Δx),
				gy in fftfreq(Ny,Ny/Δy),
				gz in fftfreq(Nz,Nz/Δz)],
		),
		(2,3,4,1)))
mag,m,n = mag_m_n(kz,Δx, Δy, Δz, Nx, Ny, Nz)
n⃗ = HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3}}(n.parent)
m⃗ = HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3}}(m.parent)
Hin = HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),2}}(randn(ComplexF64,(Nx,Ny,Nz,2)))
d = HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3}}(randn(ComplexF64,(Nx,Ny,Nz,3)))
e = HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),3}}(randn(ComplexF64,(Nx,Ny,Nz,3)))
Hout = HybridArray{Tuple{Dynamic(),Dynamic(),Dynamic(),2}}(randn(ComplexF64,(Nx,Ny,Nz,2)))
# 𝓕 = plan_fft!(randn(ComplexF64, (Nx,Ny,Nz,3)),(1:3))
# 𝓕 = plan_fft!(randn(ComplexF64, (Nx,Ny,Nz,3)),(1:3))
# 𝓕⁻¹ = plan_ifft!(randn(ComplexF64, (Nx,Ny,Nz,3)),(1:3))
M̂ = HelmholtzMap(k⃗, ε⁻¹, Δx, Δy, Δz, Nx, Ny, Nz)
P̂ = HelmholtzPreconditioner(M̂)


solver = LOBPCGIterator(M̂,false,randn(eltype(M̂),(size(M̂)[1],1)),P̂,nothing)

update_k(M̂,1.49)
M̂.k⃗[3]


res = IterativeSolvers.lobpcg!(solver;log=false,maxiter=3000,not_zeros=false,tol=1e-8)
H =  res.X[:,1]
ω² =  real(res.λ[1])
ω = sqrt(ω²)
neff = p[1] / ω
ng = ω / H_Mₖ_H(H,M̂.ε⁻¹,M̂.mag,M̂.m⃗,M̂.n⃗)



##
kx_tc!(d,Hin,m,n,mag)
kx_tc!(d,Hin,m⃗,n⃗,mag)
mul!(d.data,𝓕,d.data)
eid!(e,ε⁻¹,d)
mul!(e.data,𝓕⁻¹,e.data)
kx_ct!(Hout,e,m⃗,n⃗,mag,Ninv)
_M!(Hout,Hin,e, d, ε⁻¹, m, n, mag, 𝓕, 𝓕⁻¹, Ninv)

@btime kx_tc!($d,$Hin,$m⃗,$n⃗,$mag)
@btime mul!($d.data,$𝓕,$d.data)
@btime eid!($e,$ε⁻¹,$d)
@btime mul!($e.data,$𝓕⁻¹,$e.data)
@btime kx_ct!($Hout,$e,$m⃗,$n⃗,$mag,$Ninv)
@btime _M!($Hout,$Hin,$e, $d, $ε⁻¹, $m, $n, $mag, $𝓕, $𝓕⁻¹, $Ninv)
##

*(M̂,vec(Hin))

##


solver.A

res = lobpcg!(solver;log=false,maxiter=2000,not_zeros=false,tol=1e-8)
real(res.λ[1])








update_k(solver.A,1.55)

res = lobpcg!(solver;log=false,maxiter=2000,not_zeros=false,tol=1e-8)
real(res.λ[1])


for kk in 1.4:0.01:1.5
    @show kk
    @show nn = kk / sqrt(real(lobpcg!(solver; log=false, maxiter=1000, not_zeros=false,tol).λ[1]))
end
res.λ[eigind])

@show solver.iteration




X = solver.XBlocks.block
solver.XBlocks



##


# res = IterativeSolvers.lobpcg(M̂!(ε⁻¹,ds),false,ds.H⃗;P=P̂!(ε⁻¹,ds),maxiter,tol)
# H =  res.X[:,eigind]                       # eigenmode wavefn. magnetic fields in transverse pol. basis
# ω² =  real(res.λ[eigind])                     # eigenmode temporal freq.,  neff = kz / ω, kz = k[3]
# ω = sqrt(ω²)
#
# res2 = IterativeSolvers.lobpcg(M̂!(ei,ds),false,ds.H⃗;P=P̂!(ei,ds),maxiter=3000,tol=1e-8)
# H2=  res2.X[:,1]
# ω²2 =  real(res2.λ[1])
# ω2 = sqrt(ω²2)
# neff2 = p[1] / ω2
#
# function b2f(v::AbstractVector;Nx=128,Ny=128,Nz=1)
# 	vec( permutedims( reshape(v,(Nx,Ny,Nz,2)), (4,1,2,3) ) )
# end
#
# function f2b(v::AbstractVector;Nx=128,Ny=128,Nz=1)
# 	vec( permutedims( reshape(v,(2,Nx,Ny,Nz)), (2,3,4,1) ) )
# end
#
# H2 ./ b2f(M̂*f2b(H2))
#
# Mop = M̂!(ei,ds)
# Pop = P̂!(ei,ds)
#
# (Mop * H2) ./ (b2f(M̂*f2b(H2)))
#
# PH2_1 = ldiv!(similar(H2),Pop,copy(H2))
# PH2_2 = b2f(ldiv!(similar(H2),HelmholtzPreconditioner(M̂),f2b(copy(H2))))
# PH2_3 = (b2f(P̂(similar(H2),f2b(copy(H2)))))
# PH2_1 ./ PH2_2
#
# (H2) ./ (b2f(f2b(H2)))
#
# P̂*f2b(H2)
