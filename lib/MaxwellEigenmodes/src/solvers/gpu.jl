####################################################################################################
# GPU-capable (device- and precision-generic) eigensolver backend
#
# `GPUSolver{T}` runs the plane-wave Helmholtz eigensolves on flat arrays using only
# backend-agnostic operations: broadcasts over array views, AbstractFFTs plans created
# from the device arrays themselves (FFTW on `Array`, CUFFT on `CuArray`), and
# KrylovKit's array-generic `eigsolve`/`linsolve`. The same code therefore executes on
# the CPU (`device=:cpu`, any precision — also the reference/debug path) and on CUDA
# GPUs (`device=:cuda`, requires `using CUDA`; see `ext/MaxwellEigenmodesCUDAExt.jl`).
#
# The adjoint (gradient back-propagation) for `solve_k` with this backend is implemented
# in the same device-generic style in `rrule(::typeof(solve_k), …, ::GPUSolver)` below,
# so the eigensolve adjoint also runs on the GPU.
####################################################################################################

export GPUSolver

"""
    GPUSolver(T=Float32; device=:cuda, krylovdim=50, maxiter=200, logger=NullLogger())

Device- and precision-generic eigensolver backend for `solve_ω²`/`solve_k`.

- `T ∈ (Float32, Float64)` selects the floating-point precision of the eigensolve.
- `device=:cuda` runs on an NVIDIA GPU via CUDA.jl (load `CUDA` to activate the package
  extension; requires `CUDA.functional()`); `device=:cpu` runs the identical generic
  code path on the CPU, which is useful as a reference and for testing.

Inputs (`ω`, `ε⁻¹`) and outputs (`kmags`, `evecs`) remain host `Float64` data for
interoperability with the rest of the pipeline; conversion and device transfer happen
internally. With `T=Float32` the achievable `eig_tol`/`k_tol` are limited to a few
hundred times `eps(Float32)`; tolerances are clamped accordingly.

Gradients: `rrule(solve_k, ω, ε⁻¹, grid, ::GPUSolver)` implements the adjoint method
with the same device-generic operations, so back-propagation through GPU-accelerated
mode solves also runs on the GPU.
"""
struct GPUSolver{T<:Union{Float32,Float64},L<:AbstractLogger} <: AbstractEigensolver{L}
    logger::L
    device::Symbol
    krylovdim::Int
    maxiter::Int
end

function GPUSolver(::Type{T}=Float32; device::Symbol=:cuda, krylovdim::Int=50,
    maxiter::Int=200, logger::L=NullLogger()) where {T,L}
    return GPUSolver{T,L}(logger, device, krylovdim, maxiter)
end

solver_precision(::GPUSolver{T}) where {T} = T

# Device transfer hook. The CUDA package extension adds a method for Val{:cuda}.
_device_array(::Val{:cpu}, A::AbstractArray) = A
_device_array(::Val{D}, A::AbstractArray) where {D} =
    error("GPUSolver device `:$D` is not available. For `:cuda`, load CUDA.jl (`using CUDA`) on a machine with a functional GPU.")
_device_array(solver::GPUSolver, A::AbstractArray) = _device_array(Val(solver.device), A)

# clamp solver tolerances to what the working precision can support
_clamped_tol(tol, ::Type{T}) where {T} = max(T(tol), 50 * eps(T))

####################################################################################################
# Generic (broadcast-only) Maxwell operator kernels on flat arrays
#
# Array layouts (column-major, leading small axes):
#   H, e, d : (2 or 3, Ns...) Complex{T};  mn : (3, 2, Ns...) T;  mag : (Ns...) T
#   ε⁻¹     : (3, 3, Ns...) T
####################################################################################################

@inline _sl(A::AbstractArray, i::Int) = view(A, i, ntuple(_ -> Colon(), ndims(A) - 1)...)
@inline _sl2(A::AbstractArray, i::Int, j::Int) = view(A, i, j, ntuple(_ -> Colon(), ndims(A) - 2)...)

"d (3-cpt) = [(k+g)×]ₜc H (2-cpt): d_a = -( H₁ mn_{a2} - H₂ mn_{a1} ) mag"
function _kx_tc_g!(d, H, mn, mag)
    H1, H2 = _sl(H, 1), _sl(H, 2)
    @inbounds for a in 1:3
        _sl(d, a) .= (H2 .* _sl2(mn, a, 1) .- H1 .* _sl2(mn, a, 2)) .* mag
    end
    return d
end

"H (2-cpt) = [(k+g)×]cₜ e (3-cpt), scaled by `scale` (= mag*Ninv for the Helmholtz apply)"
function _kx_ct_g!(H, e, mn, mag, Ninv)
    e1, e2, e3 = _sl(e, 1), _sl(e, 2), _sl(e, 3)
    @inbounds begin
        _sl(H, 1) .= (e1 .* _sl2(mn, 1, 2) .+ e2 .* _sl2(mn, 2, 2) .+ e3 .* _sl2(mn, 3, 2)) .* mag .* (-Ninv)
        _sl(H, 2) .= (e1 .* _sl2(mn, 1, 1) .+ e2 .* _sl2(mn, 2, 1) .+ e3 .* _sl2(mn, 3, 1)) .* mag .* Ninv
    end
    return H
end

"d (3-cpt) = [ẑ×]ₜc H (2-cpt): d₁ = -(H₁ mn_{21} + H₂ mn_{22}); d₂ = H₁ mn_{11} + H₂ mn_{12}; d₃ = 0"
function _zx_tc_g!(d, H, mn)
    H1, H2 = _sl(H, 1), _sl(H, 2)
    @inbounds begin
        _sl(d, 1) .= .-(H1 .* _sl2(mn, 2, 1) .+ H2 .* _sl2(mn, 2, 2))
        _sl(d, 2) .= H1 .* _sl2(mn, 1, 1) .+ H2 .* _sl2(mn, 1, 2)
        _sl(d, 3) .= 0
    end
    return d
end

"e (3-cpt) = ε⁻¹ d (3-cpt), pointwise 3×3 mat-vec"
function _eid_g!(e, ε⁻¹, d)
    d1, d2, d3 = _sl(d, 1), _sl(d, 2), _sl(d, 3)
    @inbounds for i in 1:3
        _sl(e, i) .= _sl2(ε⁻¹, i, 1) .* d1 .+ _sl2(ε⁻¹, i, 2) .* d2 .+ _sl2(ε⁻¹, i, 3) .* d3
    end
    return e
end

"out (3-cpt) = a × b for flat (3, Ns...) arrays"
function _cross_g!(out, a, b)
    a1, a2, a3 = _sl(a, 1), _sl(a, 2), _sl(a, 3)
    b1, b2, b3 = _sl(b, 1), _sl(b, 2), _sl(b, 3)
    @inbounds begin
        _sl(out, 1) .= a2 .* b3 .- a3 .* b2
        _sl(out, 2) .= a3 .* b1 .- a1 .* b3
        _sl(out, 3) .= a1 .* b2 .- a2 .* b1
    end
    return out
end

####################################################################################################
# Workspace: device-resident operator data + scratch buffers + FFT plans
####################################################################################################

struct GPUHelmholtz{T,ND}
    ε⁻¹::AbstractArray   # (3,3,Ns...) T, device
    mn::AbstractArray    # (3,2,Ns...) T, device
    mag::AbstractArray   # (Ns...)     T, device
    d::AbstractArray     # (3,Ns...)   Complex{T}, device scratch
    e::AbstractArray     # (3,Ns...)   Complex{T}, device scratch
    F!::Any              # in-place forward FFT plan over spatial dims
    B!::Any              # in-place backward (unnormalized) FFT plan
    Ninv::T
    Ns::NTuple{ND,Int}
end

"""
Build the device-resident Helmholtz workspace at wavevector magnitude `kmag` for
precision `T`. `ε⁻¹d` must already be a device array of element type `T`; the k-space
polarization basis fields (`mag`, `mn`) are computed on the host in Float64 and
transferred at precision `T`.
"""
function _gpu_helmholtz(kmag::Real, ε⁻¹d::AbstractArray, grid::Grid{ND}, solver::GPUSolver{T}) where {ND,T}
    mag64, mn64 = mag_mn(Float64(kmag), grid)
    magd = _device_array(solver, T.(mag64))
    mnd = _device_array(solver, T.(mn64))
    Ns = size(grid)
    d = similar(ε⁻¹d, Complex{T}, (3, Ns...))
    e = similar(d)
    F! = plan_fft!(d, 2:(ND+1))
    B! = plan_bfft!(d, 2:(ND+1))
    return GPUHelmholtz{T,ND}(ε⁻¹d, mnd, magd, d, e, F!, B!, T(inv(prod(Ns))), Ns)
end

"apply the Helmholtz operator: Hout = M̂ Hin (flat-vector in/out, device arrays)"
function _gpu_M!(Houtv::AbstractVector, Hinv::AbstractVector, W::GPUHelmholtz{T}) where {T}
    Hin = reshape(Hinv, (2, W.Ns...))
    Hout = reshape(Houtv, (2, W.Ns...))
    _kx_tc_g!(W.d, Hin, W.mn, W.mag)
    mul!(W.d, W.F!, W.d)
    _eid_g!(W.e, W.ε⁻¹, W.d)
    mul!(W.e, W.B!, W.e)
    _kx_ct_g!(Hout, W.e, W.mn, W.mag, W.Ninv)
    return Houtv
end

_gpu_M(Hinv::AbstractVector, W::GPUHelmholtz) = _gpu_M!(similar(Hinv), Hinv, W)

"⟨H|∂M̂/∂k|H⟩ (= ∂ω²∂k for a normalized eigenvector), device-generic HMₖH"
function _gpu_HMkH(Hv::AbstractVector, W::GPUHelmholtz{T}) where {T}
    H = reshape(Hv, (2, W.Ns...))
    Hout = reshape(similar(Hv), (2, W.Ns...))
    _zx_tc_g!(W.d, H, W.mn)
    mul!(W.d, W.F!, W.d)
    _eid_g!(W.e, W.ε⁻¹, W.d)
    mul!(W.e, W.B!, W.e)
    _kx_ct_g!(Hout, W.e, W.mn, W.mag, W.Ninv)
    return -real(dot(Hv, vec(Hout)))
end

####################################################################################################
# solve_ω² / solve_k
####################################################################################################

function _gpu_eigsolve(W::GPUHelmholtz{T}, x₀::AbstractVector, nev::Int, tol::Real, solver::GPUSolver{T}) where {T}
    evals, evecs, _ = KrylovKit.eigsolve(x -> _gpu_M(x, W), x₀, nev, :SR;
        tol=_clamped_tol(tol, T), maxiter=solver.maxiter, krylovdim=solver.krylovdim, ishermitian=true)
    if length(evals) < nev
        # Lanczos can terminate early on an invariant subspace (e.g. when the starting
        # vector is an exact eigenvector); restart from a random vector.
        evals, evecs, _ = KrylovKit.eigsolve(x -> _gpu_M(x, W), _gpu_rand_x0(W, solver), nev, :SR;
            tol=_clamped_tol(tol, T), maxiter=solver.maxiter, krylovdim=solver.krylovdim, ishermitian=true)
    end
    return real.(evals[1:nev]), evecs[1:nev]
end

_gpu_rand_x0(W::GPUHelmholtz{T}, solver::GPUSolver{T}) where {T} =
    _device_array(solver, randn(Complex{T}, 2 * prod(W.Ns)))

function solve_ω²(ms::ModeSolver{ND,Tms}, solver::GPUSolver{T}; nev=1, maxiter=200, tol=1e-8,
    log=false, f_filter=nothing) where {ND,Tms<:Real,T}
    ε⁻¹d = _device_array(solver, T.(ms.M̂.ε⁻¹))
    W = _gpu_helmholtz(ms.M̂.k⃗[3], ε⁻¹d, ms.grid, solver)
    evals, evecs = _gpu_eigsolve(W, _gpu_rand_x0(W, solver), nev, tol, solver)
    evals_out = Float64.(evals)
    evecs_out = [ComplexF64.(Array(ev)) for ev in evecs]
    ms.H⃗[:, 1:nev] .= hcat(evecs_out...)
    ms.ω²[1:nev] .= complex.(evals_out)
    return evals_out, evecs_out
end

function solve_k(ω::T0, ε⁻¹::AbstractArray{T0}, grid::Grid{ND,T0}, solver::GPUSolver{T}; nev=1,
    max_eigsolves=60, maxiter=200, k_tol=1e-8, eig_tol=1e-8, log=false, kguess=nothing,
    Hguess=nothing, f_filter=nothing, overwrite=false) where {ND,T0<:Real,T}
    ε⁻¹d = _device_array(solver, T.(ε⁻¹))
    ωT = T(ω)
    k_tolT = _clamped_tol(k_tol, T)
    kmags = Vector{Float64}(undef, nev)
    evecs = Vector{Vector{ComplexF64}}(undef, nev)
    k = T(isnothing(kguess) ? k_guess(ω, ε⁻¹) : kguess)
    for eigind in 1:nev
        x₀ = nothing
        local ev
        # Newton iteration on Δω²(k) = ω²(k) - ω², using ∂ω²/∂k = 2⟨H|∂M̂/∂k|H⟩
        for _ in 1:max_eigsolves
            W = _gpu_helmholtz(k, ε⁻¹d, grid, solver)
            x₀ === nothing && (x₀ = _gpu_rand_x0(W, solver))
            evals, evs = _gpu_eigsolve(W, x₀, eigind, eig_tol, solver)
            ev = evs[eigind]
            x₀ = ev .+ T(1e-3) .* _gpu_rand_x0(W, solver)
            Δω² = evals[eigind] - ωT^2
            ∂ω²∂k = 2 * _gpu_HMkH(ev, W)
            Δk = Δω² / ∂ω²∂k
            k -= Δk
            abs(Δk) < k_tolT && break
        end
        kmags[eigind] = Float64(k)
        evecs[eigind] = ComplexF64.(Array(ev))
    end
    return kmags, evecs
end

####################################################################################################
# Adjoint: device-generic rrule for solve_k with the GPU backend.
#
# Mirrors the adjoint-method pullback of the CPU rrule in `grads/solve.jl`, with all
# heavy operations (the (M̂-ω²)λ=b adjoint linear solve, FFTs, and the ε̄⁻¹ / k̄
# accumulations) expressed as device-generic broadcasts so they run on the GPU.
####################################################################################################

"ε̄⁻¹[i,j] = -herm(λd ⊗ d†): device-generic version of `ε⁻¹_bar`, accumulated into eibar"
function _gpu_eibar_accum!(eibar, λd, d)
    @inbounds for i in 1:3
        _sl2(eibar, i, i) .+= .-real.(_sl(λd, i) .* conj.(_sl(d, i)))
        for j in (i+1):3
            Bij = .-real.(_sl(λd, i) .* conj.(_sl(d, j))) .- real.(_sl(λd, j) .* conj.(_sl(d, i)))
            _sl2(eibar, i, j) .+= Bij
            _sl2(eibar, j, i) .+= Bij
        end
    end
    return eibar
end

"""
k̄ contribution from the (mag, m, n) basis-field sensitivities (∇ₖmag_m_n, dk̂ = ẑ),
device-generic on flat (3, Ns...) arrays.
"""
function _gpu_k̄_magmn(māg, kx̄_m, kx̄_n, mag, mn)
    m = view(mn, :, 1, ntuple(_ -> Colon(), ndims(mn) - 2)...)
    n = view(mn, :, 2, ntuple(_ -> Colon(), ndims(mn) - 2)...)
    kpg_over_mag = similar(m, eltype(mag), size(m))
    _cross_g!(kpg_over_mag, m, n)
    kpg_over_mag ./= reshape(mag, (1, size(mag)...))
    # k̄ from māg: Σ māg ⋅ (kp̂g ⋅ ẑ) ⋅ mag
    k̄ = sum(māg .* _sl(kpg_over_mag, 3) .* mag)
    # q = kp̂g_over_mag × ẑ = (kpg_y, -kpg_x, 0)/|k+g|
    q = similar(kpg_over_mag)
    _sl(q, 1) .= _sl(kpg_over_mag, 2)
    _sl(q, 2) .= .-_sl(kpg_over_mag, 1)
    _sl(q, 3) .= 0
    mxq = similar(q)
    # k̄ -= Σ m̄ ⋅ (m × q);  m̄ = kx̄_m .* mag
    _cross_g!(mxq, m, q)
    k̄ -= sum((kx̄_m .* reshape(mag, (1, size(mag)...))) .* mxq)
    # k̄ -= Σ n̄ ⋅ (n × q);  n̄ = kx̄_n .* mag
    _cross_g!(mxq, n, q)
    k̄ -= sum((kx̄_n .* reshape(mag, (1, size(mag)...))) .* mxq)
    return k̄
end

function rrule(::typeof(solve_k), ω::T0, ε⁻¹::AbstractArray{T0}, grid::Grid{ND,T0},
    solver::GPUSolver{T}; nev=1, max_eigsolves=60, maxiter=200, k_tol=1e-8, eig_tol=1e-8,
    log=false, kguess=nothing, Hguess=nothing, f_filter=nothing, overwrite=false) where {ND,T0<:Real,T}

    kmags, evecs = solve_k(ω, ε⁻¹, grid, solver; nev, max_eigsolves, maxiter, k_tol, eig_tol,
        log, kguess, Hguess, f_filter, overwrite)
    ε⁻¹d = _device_array(solver, T.(ε⁻¹))
    Ns = size(grid)

    function gpu_solve_k_pullback(ΔΩ)
        k̄mags, ēvecs = ΔΩ
        ω_bar = zero(Float64)
        eibar_d = fill!(similar(ε⁻¹d), zero(T))
        ωT = T(ω)
        for (eigind, k̄, ēv, kmag, ev) in zip(1:nev, k̄mags, ēvecs, kmags, evecs)
            W = _gpu_helmholtz(kmag, ε⁻¹d, grid, solver)
            evd = _device_array(solver, Complex{T}.(ev))
            k̄ᵢ = k̄ isa AbstractZero ? zero(T) : T(k̄)
            ∂ω²∂k = 2 * _gpu_HMkH(evd, W)
            k̄ₕ = zero(T)
            if !(ēv isa AbstractZero)
                ēvd = _device_array(solver, Complex{T}.(ēv))
                # adjoint linear solve: (M̂ - ω²) λ = ēv - ⟨ev,ēv⟩ ev, with λ ⟂ ev
                b = ēvd .- (dot(evd, ēvd) .* evd)
                λ⃗, _ = KrylovKit.linsolve(x -> _gpu_M(x, W) .- (ωT^2) .* x, b;
                    tol=_clamped_tol(eig_tol, T), maxiter=1000, krylovdim=solver.krylovdim, ishermitian=true)
                λ⃗ = λ⃗ .- (dot(evd, λ⃗) .* evd)
                evg = reshape(evd, (2, Ns...))
                λ = reshape(λ⃗, (2, Ns...))
                # d = 𝓕 kx_tc(ev) * Ninv ;  λd = 𝓕 kx_tc(λ)
                d = similar(W.d)
                _kx_tc_g!(d, evg, W.mn, W.mag)
                mul!(d, W.F!, d)
                d .*= W.Ninv
                λd = similar(W.d)
                _kx_tc_g!(λd, λ, W.mn, W.mag)
                mul!(λd, W.F!, λd)
                _gpu_eibar_accum!(eibar_d, λd, d)
                # k̄ via the (mag, m, n) sensitivities
                λd .*= W.Ninv
                λẽ = similar(W.d)
                _eid_g!(λẽ, W.ε⁻¹, λd)
                mul!(λẽ, W.B!, λẽ)
                ẽ = similar(W.d)
                _eid_g!(ẽ, W.ε⁻¹, d)
                mul!(ẽ, W.B!, ẽ)
                ev1 = _sl(evg, 1); ev2 = _sl(evg, 2)
                λ1 = _sl(λ, 1); λ2 = _sl(λ, 2)
                kx̄_m = real.(λẽ .* conj.(reshape(ev2, (1, Ns...))) .+ ẽ .* conj.(reshape(λ2, (1, Ns...))))
                kx̄_n = .-real.(λẽ .* conj.(reshape(ev1, (1, Ns...))) .+ ẽ .* conj.(reshape(λ1, (1, Ns...))))
                m = view(W.mn, :, 1, ntuple(_ -> Colon(), ND)...)
                n = view(W.mn, :, 2, ntuple(_ -> Colon(), ND)...)
                māg = dropdims(sum(n .* kx̄_n .+ m .* kx̄_m; dims=1); dims=1)
                k̄ₕ = -_gpu_k̄_magmn(māg, kx̄_m, kx̄_n, W.mag, W.mn)
            end
            # combine k̄ and k̄ₕ: ω̄ and the eigenvalue-path ε̄⁻¹ term
            scale = (k̄ᵢ + k̄ₕ) / ∂ω²∂k
            λ⃗2 = scale .* _device_array(solver, Complex{T}.(ev))
            evg = reshape(_device_array(solver, Complex{T}.(ev)), (2, Ns...))
            d = similar(W.d)
            _kx_tc_g!(d, evg, W.mn, W.mag)
            mul!(d, W.F!, d)
            d .*= W.Ninv
            λd = similar(W.d)
            _kx_tc_g!(λd, reshape(λ⃗2, (2, Ns...)), W.mn, W.mag)
            mul!(λd, W.F!, λd)
            _gpu_eibar_accum!(eibar_d, λd, d)
            ω_bar += Float64(2 * ωT * scale)
        end
        ei_bar = Float64.(Array(eibar_d))
        return (NoTangent(), ω_bar, ei_bar, NoTangent(), NoTangent())
    end
    return ((kmags, evecs), gpu_solve_k_pullback)
end
