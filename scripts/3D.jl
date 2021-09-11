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

Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 0.5, 128, 128, 32;
grid = Grid(Δx,Δy,Δz,Nx,Ny,Nz)
LNx = rotate(MgO_LiNbO₃,Matrix(MRP(RotY(π/2))),name=:LiNbO₃_X);
LNxN = NumMat(LNx;expr_module=@__MODULE__())
SiO₂N = NumMat(SiO₂;expr_module=@__MODULE__())
Si₃N₄N = NumMat(Si₃N₄;expr_module=@__MODULE__())
AlOxN = NumMat(αAl₂O₃;expr_module=@__MODULE__())
##
# rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNxN,SiO₂N,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).
rwg3D(x) = ridge_wg_partial_etch3D(x[1],x[2],x[3],x[4],0.5,Si₃N₄N,SiO₂N,Δx,Δy,Δz)

p = [
       0.8,                #   top ridge width         `w_top`         [μm]
       0.4,                #   ridge thickness         `t_core`        [μm]
       0.75, #0.5,                #   ridge thickness         `t_core`        [μm]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
               ];
ω = inv(1.55)

rwg1 = rwg3D(p)
ε⁻¹ = smooth(ω,p,:fεs,true,rwg3D,grid)
kguess = 0.25 # k_guess(ω,ε⁻¹)
k⃗ = SVector(0.0,0.0,kguess)
M̂ = HelmholtzMap(k⃗, ε⁻¹, grid)
ms = ModeSolver(kguess, ε⁻¹, grid; nev=4, maxiter=300, tol=1e-7)
k1,Hv1 = solve_k(ω,p,rwg3D,grid;nev=1,eigind=1);
##
using KrylovKit
using IterativeSolvers
using DFTK: LOBPCG
evals,evecs,convinfo = eigsolve(x->ms.M̂*x,randn(ComplexF64,size(ms.H⃗,1)),size(ms.H⃗,2),:SR;maxiter=100,tol=1e-6,krylovdim=6,verbosity=2)
resIS = lobpcg(ms.eigs_itr; true,not_zeros=false,maxiter=100,tol=1e-6)
resDF = LOBPCG(ms.M̂,randn(ComplexF64,size(ms.H⃗)),I,ms.P̂,1e-6,300; display_progress=true)
##
function mag_mn1!(mag,mn::AbstractArray{T1,NDp2},k⃗::SVector{3,T2},g⃗) where {T1<:Real,T2<:Real,NDp2}
	local ẑ = SVector(0.,0.,1.)
	local ŷ = SVector(0.,1.,0.)
	# mv = view(mn,1:3,1,eachindex(g⃗)...)
	# nv = view(mn,1:3,2,eachindex(g⃗)...)
	# mvs = reinterpret(reshape,SVector{3,T1},mv)
	# nvs = reinterpret(reshape,SVector{3,T1},nv)
	kpg = zero(k⃗)
	@fastmath @inbounds for i ∈ eachindex(g⃗)
		@inbounds kpg = k⃗ - g⃗[i]
		@inbounds mag[i] = norm(kpg)
		@inbounds mn[1:3,2,i] .=  ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : SVector(-1.,0.,0.) #[-1.,0.,0.] #ŷ
		@inbounds mn[1:3,1,i] .=  normalize( cross( mn[1:3,2,i], kpg )  )
	end
	# return mag,m,n
	return mag, mn
end

function mag_mn2!(mag,mn::AbstractArray{T1,NDp2},k⃗::SVector{3,T2},g⃗) where {T1<:Real,T2<:Real,NDp2}
	local ẑ = SVector(0.,0.,1.)
	local ŷ = SVector(0.,1.,0.)
	mv = view(mn,1:3,1,eachindex(g⃗)...)
	nv = view(mn,1:3,2,eachindex(g⃗)...)
	mvs = reinterpret(reshape,SVector{3,T1},mv)
	nvs = reinterpret(reshape,SVector{3,T1},nv)
	kpg = zero(k⃗)
	@fastmath @inbounds for i ∈ eachindex(g⃗)
		@inbounds kpg = k⃗ - g⃗[i]
		@inbounds mag[i] = norm(kpg)
		@inbounds nvs[i] =  ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : SVector(-1.,0.,0.) #[-1.,0.,0.] #ŷ
		@inbounds mvs[i] =  normalize( cross( nvs[i], kpg )  )
	end
	# return mag,m,n
	return mag, mn
end


##
gr = grid
g⃗s = g⃗(gr)
mag, m⃗, n⃗ = mag_m_n(k⃗,g⃗s)
mag, mn = mag_mn(k⃗,g⃗s)
mag, mn = mag_mn(k⃗,grid)

@btime mag_m_n($k⃗,$g⃗s)
@btime mag_mn($k⃗,$g⃗s)
@btime mag_mn($k⃗,$grid)

##

g⃗s = g⃗(gr)
mag, m⃗, n⃗ = mag_m_n(k⃗,g⃗s)
d0 = randn(Complex{Float64}, (3,size(gr)...))
d1 = randn(Complex{Float64}, (3,128,128))
fftax = _fftaxes(gr)
Fp1 = plan_fft!(d0,fftax,flags=FFTW.PATIENT)
Fp2 = plan_bfft!(d0,fftax,flags=FFTW.PATIENT)
Fp3 = plan_fft(d0,fftax,flags=FFTW.PATIENT)
Fp4 = plan_bfft(d0,fftax,flags=FFTW.PATIENT)

fftax1 = (2:3)
Fp11 = plan_fft!(d1,fftax1,flags=FFTW.PATIENT)
Fp21 = plan_bfft!(d1,fftax1,flags=FFTW.PATIENT)
Fp31 = plan_fft(d1,fftax1,flags=FFTW.PATIENT)
Fp41 = plan_bfft(d1,fftax1,flags=FFTW.PATIENT)

# return HelmholtzMap{3,T}(
HelmholtzMap(
                SVector{3,Float64}(k⃗),
                gr.Nx,
                gr.Ny,
                gr.Nz,
                N(gr),
                1. / N(gr),
                g⃗s,
                mag,
                m⃗,
                n⃗,
                HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,m⃗)),
                HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,n⃗)),
            HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},Complex{Float64}}(d0),# (Array{Float64}(undef,(Nx,Ny,Nz,3))),
            HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},Complex{Float64}}(d0),# (Array{Float64}(undef,(Nx,Ny,Nz,3))),
                plan_fft!(d0,fftax,flags=FFTW.PATIENT), # planned in-place FFT operator 𝓕!
                plan_bfft!(d0,fftax,flags=FFTW.PATIENT), # planned in-place iFFT operator 𝓕⁻¹!
                plan_fft(d0,fftax,flags=FFTW.PATIENT), # planned in-place FFT operator 𝓕!
                plan_bfft(d0,fftax,flags=FFTW.PATIENT), # planned in-place iFFT operator 𝓕⁻¹!
                ε⁻¹,
                [ 3. * inv(sum(diag(ε⁻¹[:,:,I]))) for I in eachindex(gr)], #[ 3. * inv(sum(diag(einv))) for einv in ε⁻¹],
                [ inv(mm) for mm in mag ], # inverse |k⃗+g⃗| magnitudes for precond. ops
                0.0,
        )

##
# g⃗s = g⃗(grid)
# mag, m⃗, n⃗ = mag_m_n(k⃗,g⃗s)
# d0 = randn(Complex{Float64}, (3,size(grid)...))
# fftax = _fftaxes(grid)
# [ 3. * inv(sum(diag(ε⁻¹[:,:,ix,iy,iz]))) for ix=1:grid.Nx,iy=1:grid.Ny,iz=1:grid.Nz]
# epsave = [ 3. * inv(sum(diag(ε⁻¹[:,:,I]))) for I in CartesianIndices(size(grid))]
# ε⁻¹[:,:,gridinds[64,64,64]]


##



##


##
