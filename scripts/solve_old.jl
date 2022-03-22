
export solve_ω, _solve_Δω², solve_k, solve_n, solve, ng, k_guess, solve_nω, solve_ω², replan_ffts!, filter_eigs
function replan_ffts!(ms::ModeSolver{3,T}) where T<:Real
	ms.M̂.𝓕! = plan_fft!(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
	ms.M̂.𝓕⁻¹! = plan_bfft!(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
	ms.M̂.𝓕 = plan_fft(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
	ms.M̂.𝓕⁻¹ = plan_bfft(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
end

function replan_ffts!(ms::ModeSolver{2,T}) where T<:Real
	ms.M̂.𝓕! = plan_fft!(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny)),(2:3),flags=FFTW.PATIENT);
	ms.M̂.𝓕⁻¹! = plan_bfft!(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny)),(2:3),flags=FFTW.PATIENT);
	ms.M̂.𝓕 = plan_fft(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny)),(2:3),flags=FFTW.PATIENT);
	ms.M̂.𝓕⁻¹ = plan_bfft(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny)),(2:3),flags=FFTW.PATIENT);
end


"""
################################################################################
#																			   #
#	Routines to shield expensive initialization calculations from memory-	   #
#						intensive reverse-mode AD				  			   #
#																			   #
################################################################################
"""

import Base: show

function show(res::IterativeSolvers.LOBPCGResults;ind=1,color=:blue)
	itr_nums = map(x->getfield(x,:iteration)[ind],res.trace)
	residuals = map(x->getfield(x,:residual_norms)[ind],res.trace)
	log10res = log10.(residuals)
	log10res_min, log10res_max = extrema(log10res)
	resplt = UnicodePlots.lineplot(itr_nums,
		log10res;
		color,
		ylim=[floor(log10res_min),ceil(log10res_max)],
		xlim=[extrema(itr_nums)...],
		xlabel="iteration #",
		ylabel="log10(residuals)",
		title="LOBPCG Convergence (eig #: $ind)",
		)
	annotate!(resplt,:r,1,"converged: $(res.converged)")
	annotate!(resplt,:r,2,"iterations: $(res.iterations)   ($(res.maxiter) max)")
	annotate!(resplt,:r,3,"eigenvalue: $(res.λ[ind])")
	annotate!(resplt,:r,4,"residual_norm: $(res.residual_norms[ind])   (tol: $(res.tolerance))")
end


"""
################################################################################
#																			   #
#						solve_ω² methods: (ε⁻¹, k) --> (H, ω²)				   #
#																			   #
################################################################################
"""

# add try/catch with
# res = DFTK.LOBPCG(ms.M̂,rand(ComplexF64,size(ms.M̂)[1],1),I,ms.P̂,1e-8,3000)

# struct Eigensolver end
# struct IS_LOBPCG <: Eigensolver end
# struct DFTK_LOBPCG <: Eigensolver end



"""
f_filter takes in a ModeSolver and an eigenvalue/vector pair `αX` and outputs boolean,
ex. f_filter = (ms,αX)->sum(abs2,𝓟x(ms.grid)*αX[2])>0.9
where the modesolver `ms` is passed for access to any auxilary information
"""
function filter_eigs(ms::ModeSolver{ND,T},f_filter::Function)::Tuple{Vector{T},Matrix{Complex{T}}} where {ND,T<:Real}
	ω²H_filt = filter(ω²H->f_filter(ms,ω²H), [(real(ms.ω²[i]),ms.H⃗[:,i]) for i=1:length(ms.ω²)] )
	return copy(getindex.(ω²H_filt,1)), copy(hcat(getindex.(ω²H_filt,2)...)) # ω²_filt, H_filt
	# return getindex.(ω²H_filt,1), hcat(getindex.(ω²H_filt,2)...) # ω²_filt, H_filt
end

# function _solve_ω²(ms::ModeSolver{ND,T},::;nev=1,eigind=1,maxiter=100,tol=1.6e-8

function solve_ω²(ms::ModeSolver{ND,T};nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing)::Tuple{Vector{T},Matrix{Complex{T}}} where {ND,T<:Real}

		# evals,evecs,convinfo = eigsolve(x->ms.M̂*x,ms.H⃗,size(ms.H⃗,2),:SR;maxiter,tol,krylovdim=50)
		evals,evecs,convinfo = eigsolve(x->ms.M̂*x,ms.H⃗,size(ms.H⃗,2),:SR;maxiter,tol,krylovdim=50) #,verbosity=2)
		copyto!(ms.H⃗,hcat(evecs...)[1:size(ms.H⃗,2)])
		copyto!(ms.ω²,evals[1:size(ms.H⃗,2)])

		# res = lobpcg!(ms.eigs_itr; log,not_zeros=false,maxiter,tol)

		# res = LOBPCG(ms.M̂,ms.H⃗,I,ms.P̂,tol,maxiter)
		# copyto!(ms.H⃗,res.X)
		# copyto!(ms.ω²,res.λ)


	if isnothing(f_filter)
		return   (copy(real(ms.ω²)), copy(ms.H⃗))
	else
		return filter_eigs(ms, f_filter)
	end
end

function solve_ω²(ms::ModeSolver{ND,T},k::TK,ε⁻¹::AbstractArray{SMatrix{3,3,T,9},ND};nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
		# nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where T<:Real
	@ignore(update_k!(ms,k))
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms; nev, eigind, maxiter, tol, log, f_filter)
end

function solve_ω²(ms::ModeSolver{ND,T},k::TK;nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
		# nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where T<:Real
	@ignore(update_k!(ms,k))
	solve_ω²(ms; nev, eigind, maxiter, tol, log, f_filter)
end


"""
################################################################################
#																			   #
#						solve_k methods: (ε⁻¹, ω) --> (H, k)				   #
#																			   #
################################################################################
"""

"""
modified solve_ω version for Newton solver, which wants (x -> f(x), f(x)/f'(x)) as input to solve f(x) = 0
"""
function _solve_Δω²(ms::ModeSolver{ND,T},k::TK,ωₜ::T;nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,TK}
	# println("k: $(k)")
	ω²,H⃗ = solve_ω²(ms,k; nev, eigind, maxiter, tol, log, f_filter)
	Δω² = ω²[eigind] - ωₜ^2
	# ms.∂ω²∂k[eigind] = 2 * HMₖH(H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n) # = 2ω*ωₖ; ωₖ = ∂ω/∂kz = group velocity = c / ng; c = 1 here
	∂ω²∂k = 2 * HMₖH(H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn)
	ms.∂ω²∂k[eigind] = ∂ω²∂k
	# println("Δω²: $(Δω²)")
	# println("∂ω²∂k: $(∂ω²∂k)")
    return Δω² , Δω² / ∂ω²∂k #Δω² / copy(ms.∂ω²∂k[eigind])
end

function solve_k(ms::ModeSolver{ND,T},ω::T;nev=1,eigind=1,maxiter=100,tol=1e-8,atol=tol,maxevals=60,log=false,f_filter=nothing)::Tuple{T,Vector{Complex{T}}} where {ND,T<:Real} #
	if iszero(ms.M̂.k⃗[3])
		ms.M̂.k⃗ = SVector(0., 0., ω*ñₘₐₓ(ms.M̂.ε⁻¹))
	end
    kz = Roots.find_zero(x -> _solve_Δω²(ms,x,ω;nev,eigind,maxiter,tol,f_filter), ms.M̂.k⃗[3], Roots.Newton(); atol,maxevals,) #  verbose=true,)
	if isnothing(f_filter)
		Hv = ms.H⃗[:,eigind]
	else
		Hv = filter_eigs(ms, f_filter)[2][:,eigind]
	end
    return ( copy(kz), copy(Hv) ) # maybe copy(ds.H⃗) instead?
end

function solve_k(ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{T};nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing)::Tuple{T,Vector{Complex{T}}} where {ND,T<:Real} #
	Zygote.@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_k(ms, ω; nev, eigind, maxiter, tol, log, f_filter)
end

function solve_k(ω::T,p::AbstractVector,geom_fn::F,grid::Grid{ND};kguess=nothing,Hguess=nothing,nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing)::Tuple{T,Vector{Complex{T}}} where {ND,T<:Real,F<:Function}
	ε⁻¹ = smooth(ω,p,:fεs,true,geom_fn,grid)
	ms = ignore() do
		kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
		ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, tol)
		if !isnothing(Hguess)
			ms.H⃗ = reshape(Hguess,size(ms.H⃗))
		end
		return ms
	end
	solve_k(ms, ω; nev, eigind, maxiter, tol, log, f_filter)
end

function solve_k(ω::AbstractVector{T},p::AbstractVector,geom_fn::F,grid::Grid{ND};kguess=nothing,Hguess=nothing,nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function}
	ε⁻¹ = smooth(ω,p,:fεs,true,geom_fn,grid)
	# ms = @ignore(ModeSolver(k_guess(first(ω),first(ε⁻¹)), first(ε⁻¹), grid; nev, maxiter, tol))
	ms = ignore() do
		kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
		ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, tol)
		if !isnothing(Hguess)
			ms.H⃗ = Hguess
		end
		return ms
	end
	nω = length(ω)
	k = Buffer(ω,nω)
	Hv = Buffer([1.0 + 3.0im, 2.1+4.0im],(size(ms.M̂)[1],nω))
	for ωind=1:nω
		@ignore(update_ε⁻¹(ms,ε⁻¹[ωind]))
		kHv = solve_k(ms,ω[ωind]; nev, eigind, maxiter, tol, log, f_filter)
		k[ωind] = kHv[1]
		Hv[:,ωind] = kHv[2]
	end
	return copy(k), copy(Hv)
end

"""
################################################################################
#																			   #
#						`solve` methods: (ε⁻¹, ω) --> (E, n, ng, gvd)		   #
#																			   #
################################################################################
"""

function ∇ₖmag_mn2(māg,mn̄s,mag,mns)
	m = view(mns,1,:,:,:)
	n = view(mns,2,:,:,:)
	@tullio kp̂g_over_mag[i,ix,iy] := m[mod(i-2),ix,iy] * n[mod(i-1),ix,iy] / mag[ix,iy] - m[mod(i-1),ix,iy] * n[mod(i-2),ix,iy] / mag[ix,iy] (i in 1:3)
	kp̂g_over_mag_x_dk̂ = _cross(kp̂g_over_mag,dk̂)
	@tullio k̄_mag := māg[ix,iy] * mag[ix,iy] * kp̂g_over_mag[j,ix,iy] * dk̂[j]
	@tullio k̄_mn := -conj(mn̄s)[imn,i,ix,iy] * mns[imn,mod(i-2),ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-1),ix,iy] + conj(mn̄s)[imn,i,ix,iy] * mns[imn,mod(i-1),ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-2),ix,iy] (i in 1:3)
	k̄_magmn = k̄_mag + k̄_mn
	return k̄_magmn
end

function solve(ω::T,p::AbstractVector,geom_fn::F,grid::Grid{ND};kguess=nothing,Hguess=nothing,dk̂=SVector(0.0,0.0,1.0),nev=1,eigind=1,maxiter=500,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function} # output type ::Tuple{T,T,T,Vector{Complex{T}}}
	ε,ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fεs,:fnn̂gs,:fnn̂gs),[false,true,false,true],geom_fn,grid));
	# ε = smooth(ω,p,:fεs,false,geom_fn,grid)
	# ε⁻¹ = smooth(ω,p,:fεs,true,geom_fn,grid)
	# nng = smooth(ω,p,:fnn̂gs,false,geom_fn,grid)
	# nng⁻¹ = smooth(ω,p,:fnn̂gs,true,geom_fn,grid)

	ngvd,nng2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,false],geom_fn,grid,volfrac_smoothing));
	ms = ignore() do
		kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
		ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, tol)
		if !isnothing(Hguess)
			ms.H⃗ = Hguess
		end
		return ms
	end
	k, Hv = solve_k(ms,ω,ε⁻¹;nev,eigind,maxiter,tol,log, f_filter) #ω²_tol)
	neff::T = k/ω

	# calculate effective group index `ng`
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	Ninv 		= 		1. / N(grid)
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	m = flat(m⃗)
	n = flat(n⃗)
	mns = copy(vcat(reshape(m,1,3,Ns...),reshape(n,1,3,Ns...)))
    Hₜ = reshape(Hv,(2,Ns...))
	D = 1im * fft( kx_tc( Hₜ,mns,mag), _fftaxes(grid) )
	E = ε⁻¹_dot( D, ε⁻¹)
	# E = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
	# H = inv(ω) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
	H = fft( tc(Hₜ,mns), (2:3) ) * (-1im * ω)
	P = 2*real(_sum_cross_z(conj(E),H))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
	# W = dot(E,_dot((ε+nng),E))             # energy density per unit length
	W = real(dot(E,_dot(nng,E))) + (N(grid)* (ω^2))     # energy density per unit length
	ng = real( W / P )

	# calculate GVD = ∂(ng) / ∂ω = (∂²k)/(∂ω²)
	W̄ = inv(P)
	om̄₁₁ = 2*ω * N(grid) * W̄
	nnḡ = _outer(E,E) * W̄
	# H̄ = (-2*ng*W̄) * _cross(repeat([0.,0.,1.],outer=(1,Ns...)), E)
	# Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,repeat([0.,0.,1.],outer=(1,Ns...))) )
	H̄ = (-2*ng*W̄) * _cross(dk̂, E)
	Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,dk̂) )
	om̄₁₂ = dot(H,H̄) / ω
	om̄₁ = om̄₁₁ + om̄₁₂
	# eī₁ = _outer(Ē,D) ####################################
	𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),(2:3))
	𝓕⁻¹_H̄ = bfft( H̄ ,(2:3))
	H̄ₜ = 1im*( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mns,mag) + ω*ct(𝓕⁻¹_H̄,mns) )
	local one_mone = [1.0im, -1.0im]
	@tullio 𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ[i,j,ix,iy] := one_mone[i] * reverse(Hₜ;dims=1)[i,ix,iy] * conj(𝓕⁻¹_ε⁻¹_Ē)[j,ix,iy] nograd=one_mone
	∂ω²∂k_nd = 2 * HMₖH(Hv,ε⁻¹,mag,m,n)

	##### grad solve k
	# solve_adj!(λ⃗,M̂,H̄,ω^2,H⃗,eigind)
	M̂2 = HelmholtzMap(k,ε⁻¹,grid)
	λ⃗	= eig_adjt(
		M̂2,								 # Â
		ω^2, 							# α
		Hv, 					 		 # x⃗
		0.0, 							# ᾱ
		vec(H̄ₜ);								 # x̄
		# λ⃗₀,
		P̂	= HelmholtzPreconditioner(M̂2),
	)
	### k̄ₕ, eīₕ = ∇M̂(k,ε⁻¹,λ⃗,H⃗,grid)
	λ = reshape(λ⃗,(2,Ns...))
	λd 	= 	fft(kx_tc( λ , mns, mag ),_fftaxes(grid))
	# eīₕ	 = 	 ε⁻¹_bar(vec(D * (Ninv * -1.0im)), vec(λd), Ns...) ##########################
	λẽ  =   bfft(ε⁻¹_dot(λd , ε⁻¹),_fftaxes(grid))
	ẽ 	 =   bfft(E * -1.0im,_fftaxes(grid))
	@tullio mn̄s_kx0[i,j,ix,iy] := -1.0im * one_mone[i] * reverse(conj(Hₜ);dims=1)[i,ix,iy] * (Ninv*λẽ)[j,ix,iy] + -1.0im * one_mone[i] * reverse(conj(λ);dims=1)[i,ix,iy] * (Ninv*ẽ)[j,ix,iy]  nograd=one_mone
	# @tullio mn̄s_kx0[i,j,ix,iy] := -1.0im * one_mone[i] * reverse(conj(Hₜ);dims=1)[i,ix,iy] * λẽ[j,ix,iy] + -1.0im * one_mone[i] * reverse(conj(λ);dims=1)[i,ix,iy] * ẽ[j,ix,iy]  nograd=one_mone
	# @tullio mn̄s_kx0[i,j,ix,iy] := -1.0im * one_mone[i] * reverse(conj(Hₜ);dims=1)[i,ix,iy] * λẽ[j,ix,iy] + -1.0im * one_mone[i] * reverse(conj(λ);dims=1)[i,ix,iy] * ẽ[j,ix,iy]  nograd=one_mone
	@tullio mn̄s[i,j,ix,iy] := mag[ix,iy] * (mn̄s_kx0-conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ))[i,j,ix,iy]  + 1im*ω*conj(Hₜ)[i,ix,iy]*𝓕⁻¹_H̄[j,ix,iy]
	@tullio māg[ix,iy] := mns[a,b,ix,iy] * (mn̄s_kx0-conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ))[a,b,ix,iy]
	# k̄ = ∇ₖmag_mn(māg,mn̄s,mag,mns)
	@tullio kp̂g_over_mag[i,ix,iy] := m[mod(i-2),ix,iy] * n[mod(i-1),ix,iy] / mag[ix,iy] - m[mod(i-1),ix,iy] * n[mod(i-2),ix,iy] / mag[ix,iy] (i in 1:3)
	kp̂g_over_mag_x_dk̂ = _cross(kp̂g_over_mag,dk̂)
	@tullio k̄_mag := māg[ix,iy] * mag[ix,iy] * kp̂g_over_mag[j,ix,iy] * dk̂[j]
	@tullio k̄_mn := -conj(mn̄s)[imn,i,ix,iy] * mns[imn,mod(i-2),ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-1),ix,iy] + conj(mn̄s)[imn,i,ix,iy] * mns[imn,mod(i-1),ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-2),ix,iy] (i in 1:3)
	k̄ = k̄_mag + k̄_mn
	### \ k̄ₕ, eīₕ = ∇M̂(k,ε⁻¹,λ⃗,H⃗,grid)

	# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
	λₖ  = ( k̄ / ∂ω²∂k_nd ) * Hₜ #reshape(λ⃗ₖ, (2,Ns...))
	λdₖ	=	fft(kx_tc( λₖ , mns, mag ),_fftaxes(grid))
	# eīₖ = ε⁻¹_bar(vec(D* (Ninv * -1.0im)), vec(λdₖ), Ns...) ####################################
	om̄₂  =  2ω * k̄ / ∂ω²∂k_nd
	##### \grad solve k
	om̄₃ = dot(herm(nnḡ), ngvd)
	om̄₄ = dot( herm(_outer(Ē+(λd+λdₖ)*(Ninv * -1.0im),D) ), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
	# @show om̄₄_old = dot( ( eīₖ + eīₕ + eī₁ ), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
	gvd = real( om̄₁ + om̄₂ + om̄₃ + om̄₄ )

	return ( neff, ng, gvd, E )
end


function solve_old(ω::T,p::AbstractVector,geom_fn::F,grid::Grid{ND};kguess=nothing,Hguess=nothing,dk̂=SVector(0.0,0.0,1.0),nev=1,eigind=1,maxiter=500,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function} # output type ::Tuple{T,T,T,Vector{Complex{T}}}
	# ε⁻¹,nng⁻¹ = smooth(ω,p,(:fεs,:fnn̂gs),[true,true],geom_fn,grid)
	# ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fnn̂gs,:fnn̂gs),[true,false,true],geom_fn,grid));
	ε,ε⁻¹,nng,nng⁻¹ = deepcopy(smooth(ω,p,(:fεs,:fεs,:fnn̂gs,:fnn̂gs),[false,true,false,true],geom_fn,grid));
	# ngvd = copy(smooth((ω,),p,:fnĝvds,false,SMatrix{3,3,T,9}(0.,0.,0.,0.,0.,0.,0.,0.,0.),geom_fn,grid,volfrac_smoothing));
	ngvd,nng2 = deepcopy(smooth(ω,p,(:fnĝvds,:fnn̂gs),[false,false],geom_fn,grid,volfrac_smoothing));

	# ε⁻¹,nng⁻¹,ngvd = smooth(ω,p,(:fεs,:fnn̂gs,:fnĝvds),[true,true,false],geom_fn,grid)
	# ms = @ignore(ModeSolver(k_guess(ω,ε⁻¹), ε⁻¹, grid; nev, maxiter, tol))
	ms = ignore() do
		kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
		ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, tol)
		if !isnothing(Hguess)
			ms.H⃗ = Hguess
		end
		return ms
	end
	# update_ε⁻¹(ms,ε⁻¹)
	# k, Hv = solve_k(ω,p,geom_fn,grid; nev, eigind, maxiter, tol, log, f_filter)
	k, Hv = solve_k(ms,ω,ε⁻¹;nev,eigind,maxiter,tol,log, f_filter) #ω²_tol)
	# (mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
	# ng::T = ω / HMₖH(Hv,real(nng⁻¹),real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗))) #  material disp. included
	neff::T = k/ω

	# calculate effective group index `ng`
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	mag,m⃗,n⃗ = mag_m_n(k,grid)
	mns = vcat(reshape(flat(m⃗),1,3,Ns...),reshape(flat(n⃗),1,3,Ns...))
    Hₜ = reshape(Hv,(2,Ns...))
	D = 1im * fft( kx_tc( Hₜ,mns,mag), (2:1+ND) )
	E = ε⁻¹_dot( D, ε⁻¹)
	# E = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
	# H = inv(ω) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
    H = fft( tc(Hₜ,mns), (2:3) ) * (-1im * ω)
	P = 2*real(_sum_cross_z(conj(E),H))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
	# W = dot(E,_dot((ε+nng),E))             # energy density per unit length
    W = real(dot(E,_dot(nng,E))) + (N(grid)* (ω^2))     # energy density per unit length
	ng = real( W / P )

	# calculate GVD = ∂(ng) / ∂ω = (∂²k)/(∂ω²)
	W̄ = inv(P)
	om̄₁₁ = 2*ω * N(grid) * W̄
	nnḡ = _outer(E,E) * W̄
	# H̄ = (-2*ng*W̄) * _cross(repeat([0.,0.,1.],outer=(1,Ns...)), E)
	# Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,repeat([0.,0.,1.],outer=(1,Ns...))) )
	H̄ = (-2*ng*W̄) * _cross(dk̂, E)
	Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,dk̂) )
	om̄₁₂ = dot(H,H̄) / ω
	om̄₁ = om̄₁₁ + om̄₁₂
	eī₁ = _outer(Ē,D)
	𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),(2:3))
	𝓕⁻¹_H̄ = bfft( H̄ ,(2:3))
	H̄ₜ = 1im*( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mns,mag) + ω*ct(𝓕⁻¹_H̄,mns) )
	# 𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ = 1im *_outer(_dot(repeat([0.0+0.0im 1.0+0.0im ;-1.0+0.0im 0.0+0.0im ],outer=(1,1,Ns...)), Hₜ), 𝓕⁻¹_ε⁻¹_Ē )
	local one_mone = [1.0im, -1.0im]
	@tullio 𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ[i,j,ix,iy] := one_mone[i] * reverse(Hₜ;dims=1)[i,ix,iy] * conj(𝓕⁻¹_ε⁻¹_Ē)[j,ix,iy] nograd=one_mone
	@tullio māg[ix,iy] := mns[a,b,ix,iy] * -conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ)[a,b,ix,iy]
	mn̄s = -conj( 1im*ω*_outer(Hₜ,𝓕⁻¹_H̄) + _mult(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ,mag))
	# m̄ = reinterpret(reshape,SVector{3,T},view(mn̄s,1,:,:,:))
	# n̄ = reinterpret(reshape,SVector{3,T},view(mn̄s,2,:,:,:))
	m̄ = reinterpret(reshape,SVector{3,eltype(mn̄s)},view(mn̄s,1,:,:,:))
	n̄ = reinterpret(reshape,SVector{3,eltype(mn̄s)},view(mn̄s,2,:,:,:))
	k̄ = ∇ₖmag_m_n(māg,m̄,n̄,mag,m⃗,n⃗;dk̂)
	∂ω²∂k_nd = 2 * HMₖH(Hv,ε⁻¹,mag,flat(m⃗),flat(n⃗))
	( _, _, om̄₂, eī₂ ) = ∇solve_k(
		(k̄,vec(H̄ₜ)),
		(k,Hv),
		∂ω²∂k_nd,
		ω,
		ε⁻¹,
		grid;
		eigind,
	)
	om̄₃ = dot(herm(nnḡ), ngvd)
	om̄₄ = dot(herm(eī₁+eī₂), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
	gvd = real( om̄₁ + om̄₂ + om̄₃ + om̄₄ )

	return ( neff, ng, gvd, E )
end

function solve(ω::AbstractVector{T},p::AbstractVector,geom_fn::F,grid::Grid{ND};nev=1,eigind=1,maxiter=500,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function} # output type ::Tuple{T,T,T,Vector{Complex{T}}}
	nω = length(ω)
	neff = Buffer(ω,nω)
	ng = Buffer(ω,nω)
	gvd = Buffer(ω,nω)
	E = Buffer([1.0 + 3.0im, 2.1+4.0im],(3,size(grid)...,nω))
	ε⁻¹_nng⁻¹_ngvd = smooth(ω,p,(:fεs,:fnn̂gs,:fnĝvds),[true,true,false],geom_fn,grid)
	# create ModeSolver and solve for first frequency
	om = first(ω)
	ms = @ignore(ModeSolver(k_guess(om,ε⁻¹_nng⁻¹_ngvd[1,1]), ε⁻¹_nng⁻¹_ngvd[1,1], grid; nev, maxiter, tol))
	k,Hv = solve_k(ms,om; nev, eigind, maxiter, tol, log, f_filter)
	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
	neff[1] = k/om
	ng[1] = om / HMₖH(Hv,ε⁻¹_nng⁻¹_ngvd[2,1],real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗))) #  material disp. included
	gvd[1] = 0.0
	E[:,:,:,1] = E⃗(k,reshape(Hv,(2,size(grid)...)),om,ε⁻¹_nng⁻¹_ngvd[1,1],ε⁻¹_nng⁻¹_ngvd[2,1],grid; normalized=true, nnginv=true)
	# solve at all remaining frequencies by updating the ModeSolver
	for ωind=2:nω
		om = ω[ωind]
		@ignore(update_ε⁻¹(ms,ε⁻¹_nng⁻¹_ngvd[1,ωind]))
		k,Hv = solve_k(ms,om; nev, eigind, maxiter, tol, log, f_filter)
		(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
		neff[ωind] = k/om
		ng[ωind] = om / HMₖH(Hv,ε⁻¹_nng⁻¹_ngvd[2,ωind],real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗))) #  material disp. included
		gvd[ωind] = 0.0
		E[:,:,:,ωind] = E⃗(k,reshape(Hv,(2,size(grid)...)),om,ε⁻¹_nng⁻¹_ngvd[1,ωind],ε⁻¹_nng⁻¹_ngvd[2,ωind],grid; normalized=true, nnginv=true)
	end
	return copy(neff), copy(ng), copy(gvd), copy(E)
end

# function solve_n(ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{<:SMatrix{3,3},ND},nnginv::AbstractArray{<:SMatrix{3,3},ND};nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,ω²_tol=tol,f_filter=nothing) where {ND,T<:Real}
# 	@ignore(update_ε⁻¹(ms,ε⁻¹))
# 	k, H⃗ = solve_k(ms,ω,ε⁻¹;nev,eigind,maxiter,tol,log, f_filter) #ω²_tol)
# 	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
# 	ng = ω / HMₖH(H⃗,nnginv,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗))) # new, material disp. included
# 	gvd = ∂²ω²∂k²(ω,geom,k,Hv,grid)
# 	neff, ng, gvd = neff_ng_gvd(ω,geom,k,Hv,ms.grid; eigind)
# 	E = E⃗(k,reshape(H⃗,(2,size(ms.grid)...)),ω,geom,ms.grid; svecs=false, normalized=true)
# 	return ( k/ω, ng, E )
# end


# function solve(ms::ModeSolver{ND,T},ω::T,geom::Geometry;nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
# 	ε⁻¹ = εₛ⁻¹(ω,geom;ms) # make_εₛ⁻¹(ω,shapes,dropgrad(ms))
# 	nnginv = nngₛ⁻¹(ω,geom;ms)
# 	# solve_n(ms, ω,ε⁻¹,nnginv; nev, eigind, maxiter, tol, log, f_filter)
# 	# update_ε⁻¹(ms,ε⁻¹)
# 	k, Hv = solve_k(ms,ω,ε⁻¹;nev,eigind,maxiter,tol,log, f_filter) #ω²_tol)
# 	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
# 	# ng::T = ω / HMₖH(Hv,nnginv,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗))) # new, material disp. included
# 	# neff::T = k/ω
# 	# gvd = 0.0
# 	neff, ng, gvd = neff_ng_gvd(ω,geom,k,Hv,ms.grid; eigind)
# 	E = E⃗(k,reshape(Hv,(2,size(ms.grid)...)),ω,geom,ms.grid; normalized=true)
# 	return ( neff, ng, gvd, E )
# end

function solve_n(ω::T,geom::Geometry,gr::Grid{ND};nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
	ms = @ignore(ModeSolver(kguess(ω,geom), geom, gr;nev))
	solve_n(dropgrad(ms), ω, geom; nev, eigind, maxiter, tol, log, f_filter)
end

# prev method could also be:
# function solve_n(ω::Real,geom::Vector{<:Shape},gr::Grid{ND,T};nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,ω²_tol=tol, k₀=kguess(ω,geom)) where {ND,T<:Real}
# 	ms::ModeSolver{ND,T} = @ignore( ModeSolver(k₀, geom, gr) );
# 	solve_n(ms,ω,geom;nev,eigind,maxiter,tol,log)
# end

# function solve_n(ms::ModeSolver{ND,T},ω::Vector{T};nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
# 	nω = length(ω)
# 	n = Buffer(ω,nω)
# 	ng = Buffer(ω,nω)
# 	@inbounds for ωind=1:nω
# 		@inbounds nng = solve_n(ms,ω[ωind]; nev, eigind, maxiter, tol, log)
# 		@inbounds n[ωind] = nng[1]
# 		@inbounds ng[ωind] = nng[2]
# 	end
# 	return ( copy(n), copy(ng) )
# end
#
# function solve_n(ms::ModeSolver{ND,T},ω::Vector{T},ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
# 	@ignore(update_ε⁻¹(ms,ε⁻¹))
# 	nω = length(ω)
# 	n = Buffer(ω,nω)
# 	ng = Buffer(ω,nω)
# 	@inbounds for ωind=1:nω
# 		@inbounds nng = solve_n(ms,ω[ωind],ε⁻¹; nev, eigind, maxiter, tol, log)
# 		@inbounds n[ωind] = nng[1]
# 		@inbounds ng[ωind] .= nng[2]
# 	end
# 	return ( copy(n), copy(ng) )
# end




function _solve_n_serial(ms::ModeSolver{ND,T},ωs::Vector{T},geom::Geometry;nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,ω²_tol=tol,wp=nothing,f_filter=nothing) where {ND,T<:Real}

	nω = length(ωs)
	ns = Buffer(ωs,nω)
	ngs = Buffer(ωs,nω)
	# @inbounds for ωind=1:nω
	# 	@inbounds nng = solve_n(ms, ωs[ωind], geom; nev, eigind, maxiter, tol, log)
	# 	# @inbounds nng = solve_n(ms, ωs[ωind], εₛ⁻¹(ωs[ωind],geom;ms=dropgrad(ms)); nev, eigind, maxiter, tol, log)
	# 	@inbounds ns[ωind] = nng[1]
	# 	@inbounds ngs[ωind] = nng[2]
	# end
	Srvol = S_rvol(geom;ms=dropgrad(ms))
	# ms_copies = @ignore( [ deepcopy(ms) for om in 1:length(ωs) ] )

	nω = length(ωs)
	n_buff = Buffer(ωs,nω)
	ng_buff = Buffer(ωs,nω)
	gvd_buff = Buffer(ωs,nω)
	E_buff = Buffer(ms.H⃗,(3,size(ms.grid)...,nω))
	for ωind=1:nω
		ωinv = inv(ωs[ωind])
		es = vcat(map(f->SMatrix{3,3}(f( ωinv )),geom.fεs),[εᵥ,])
		eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
		ε⁻¹_ω = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)
		# ei_new = εₛ⁻¹(es,eis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
		k_ω, Hv_ω = solve_k(ms, ωs[ωind], ε⁻¹_ω; nev, eigind, maxiter, tol, log, f_filter) #ω²_tol)
		(mag,m⃗,n⃗) = mag_m_n(k_ω,dropgrad(ms.M̂.g⃗))
		### dispersive ng calculation
		# nngs_ω = vcat( nn̂g.(materials(geom), ωinv) ,[εᵥ,]) # = √.(ε̂) .* nĝ (elementwise product of index and group index tensors) for each material, vacuum permittivity tensor appended
		# nngis_ω = inv.(nngs_ω)
		# nngs_ω = vcat(map(f->SMatrix{3,3}(f( ωinv )),geom.fnn̂gs),[εᵥ,])
		# nngis_ω = inv.(nngs_ω)	# corresponding list of inverse dielectric tensors for each material
		# # nngi_ω = εₛ⁻¹(nngs,nngis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
		# nngi_ω = εₛ⁻¹(nngs_ω,nngis_ω,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)
		#
		# ng_ω = ωs[ωind] / HMₖH(Hv_ω[:,eigind],nngi_ω,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗))) # new spatially smoothed ε⁻¹ tensor array
		# gvd_ω = ∂²ω²∂k²(ωs[ωind],geom,k_ω,Hv_ω,ms.grid; eigind)
		neff_ω, ng_ω, gvd_ω = neff_ng_gvd(ωs[ωind],geom,k_ω,Hv_ω,ms.grid; eigind)


		E_buff[1:3,axes(ms.grid)...,ωind] = E⃗(k_ω,reshape(Hv_ω,(2,size(ms.grid)...)),ωs[ωind],geom,ms.grid; normalized=true)
		n_buff[ωind] =  neff_ω #k_ω/ωs[ωind]
		ng_buff[ωind] = ng_ω
		gvd_buff[ωind] = gvd_ω
	end
	return ( copy(n_buff), copy(ng_buff), copy(gvd_buff), copy(E_buff) )
end

function _solve_n_parallel(ms::ModeSolver{ND,T},ωs::Vector{T},geom::Vector{<:Shape};nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,ω²_tol=tol,wp=nothing) where {ND,T<:Real}
	# pids = length(workers())<n_procs ? addprocs(n_procs) : workers()
	ω0 = ωs[Int(ceil(nω/2))]
	nng0 = solve_n(ms,ω0,shapes)
	ms_copies = [ deepcopy(ms) for om in 1:nω ]
	shapess = [deepcopy(shapes) for om in 1:nω ]
	nng = pmap(wp,ms_copies,ωs,shapess) do m,om,s
		@ignore( replan_ffts!(m) );
		solve_n(dropgrad(m),om,s)
	end
	n = [res[1] for res in nng]; ng = [res[2] for res in nng]
	return n, ng
end


function solve_n(ms::ModeSolver{ND,T},ωs::Vector{T},geom::Geometry;nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,ω²_tol=tol,wp=nothing,f_filter=nothing) where {ND,T<:Real}
	_solve_n_serial(ms, ωs, geom; nev, eigind, maxiter, tol, log,f_filter=dropgrad(f_filter))
end

function solve_n(ωs::Vector{T},geom::Geometry,gr::Grid;nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,ω²_tol=tol,wp=nothing,f_filter=nothing) where {ND,T<:Real}
	ms = @ignore(ModeSolver(kguess(first(ωs),geom), geom, gr))
	_solve_n_serial(ms,ωs, geom; nev, eigind, maxiter, tol, log,f_filter=dropgrad(f_filter))
end



	# ε⁻¹ = make_εₛ⁻¹(shapes,dropgrad(ms))
	# @ignore(update_ε⁻¹(ms,ε⁻¹))

	# m = @ignore( ModeSolver(1.45, shapes, 6., 4., 1., 128, 128, 1) )
	# ms_copies = @ignore( [ deepcopy(ms) for om in 1:length(ωs) ] )
	# nng = map((x,y)->solve_n(y,x,geom), ωs, ms_copies)
	# n = [res[1] for res in nng]
	# ng = [res[2] for res in nng]

	# Srvol = S_rvol(geom;ms=dropgrad(ms))
	# ms_copies = @ignore( [ deepcopy(ms) for om in 1:length(ωs) ] )

	# nω = length(ωs)
	# n_buff = Buffer(ωs,nω)
	# ng_buff = Buffer(ωs,nω)
	# for ωind=1:nω
	# 	# calculate ε⁻¹ for current ω
	# 	# m = ms_copies[ωind]
	# 	# @ignore( replan_ffts!(m) );
	# 	# es = vcat(εs(ms.geom,( 1. / ωs[ωind] )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
	# 	# eis = inv.(es)
	# 	# ε⁻¹_ω = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)
	# 	ε⁻¹_ω = εₛ⁻¹(ωs[ωind],geom;ms=dropgrad(ms))
	# 	# solve for n, ng with new ε⁻¹
	# 	nng = solve_n(ms,ωs[ωind],ε⁻¹_ω; nev, eigind, maxiter, tol, log)
	# 	# nng = solve_n(ms_copies[ωind],ω[ωind],ε⁻¹; nev, eigind, maxiter, tol, log)
	# 	n_buff[ωind] = nng[1]
	# 	ng_buff[ωind] = nng[2]
	# end
	# return ( copy(n_buff), copy(ng_buff) )

	# if isnothing(wp)			# solve_n for all ωs on current process
	# 	n_buff = Buffer(ωs,nω)
	# 	ng_buff = Buffer(ωs,nω)
	# 	for ωind=1:nω
	# 		nng = solve_n(ms,ωs[ωind],shapes; nev, eigind, maxiter, tol, log)
	# 		# nng = solve_n(ms_copies[ωind],ω[ωind],ε⁻¹; nev, eigind, maxiter, tol, log)
	# 		n_buff[ωind] = nng[1]
	# 		ng_buff[ωind] = nng[2]
	# 	end
	# 	# return ( copy(n), copy(ng) )
	# 	n = copy(n_buff); ng = copy(ng_buff);
	# else						# distribute solve_n's across worker pool wp
	# 	# pids = length(workers())<n_procs ? addprocs(n_procs) : workers()
	# 	ω0 = ωs[Int(ceil(nω/2))]
	# 	nng0 = solve_n(ms,ω0,shapes)
	# 	ms_copies = [ deepcopy(ms) for om in 1:nω ]
	# 	shapess = [deepcopy(shapes) for om in 1:nω ]
	# 	nng = pmap(wp,ms_copies,ωs,shapess) do m,om,s
	# 		@ignore( replan_ffts!(m) );
	# 		solve_n(dropgrad(m),om,s)
	# 	end
	# 	# nng = pmap(ωs) do om
	# 	# 	m = @ignore( ModeSolver(1.45, shapes, 6., 4., 1., 128, 128, 1) )
	# 	# 	solve_n(
	# 	# 		m,
	# 	# 		om,
	# 	# 		shapes,
	# 	# 		)
	# 	# end
	# 	n = [res[1] for res in nng]; ng = [res[2] for res in nng]
	# end
# 	return n, ng
# end
# 	n = Buffer(ω,nω)
# 	ng = Buffer(ω,nω)
# 	for ωind=1:nω
# 		# nng = solve_n(ms,ω[ωind],ε⁻¹; nev, eigind, maxiter, tol, log)
# 		nng = solve_n(ms_copies[ωind],ω[ωind],ε⁻¹; nev, eigind, maxiter, tol, log)
# 		n[ωind] = nng[1]
# 		ng[ωind] = nng[2]
# 	end
# 	return ( copy(n), copy(ng) )
# end

# nng = vmap(ms_copies,ω) do m,om
# 	solve_n(m,om,ε⁻¹)
# end
# n = [ res[1] for res in nng ]
# ng = [ res[2] for res in nng ]
# return n, ng
# end


# function solve(ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{<:SMatrix{3,3},ND},nng⁻¹::AbstractArray{<:SMatrix{3,3},ND};nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
# 	@ignore(update_ε⁻¹(ms,ε⁻¹))
# 	k, H⃗ = solve_k(ms,ω,ε⁻¹;nev,eigind,maxiter,tol,log) #ω²_tol)
# 	mag,m⃗,n⃗ = mag_m_n(k,grid)
#
# 	∂ω²∂k_nondisp 	= 2 * HMₖH(H⃗[:,eigind],ε⁻¹  ,real(mag),real(flat(m⃗)),real(flat(n⃗)))
# 	∂ω²∂k_disp 		= 2 * HMₖH(H⃗[:,eigind],nng⁻¹,real(mag),real(flat(m⃗)),real(flat(n⃗)))
# 	ng = 2ω / ∂ω²∂k_disp # new, material disp. included
# 	# calculate second order dispersion
# 	k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind)
# 	( _, _, om̄₁, eī₁ ) = ∇solve_k(	  (k̄,H̄),
# 									 	(k,H⃗[:,eigind]),
# 									  	∂ω²∂k_nondisp,
# 									   	ω,
# 									    ε⁻¹,
# 										grid; eigind)
#
#
# 	ng = 2ω / ∂ω²∂k_disp # new, material disp. included
#
# 	end
#     return ( k/ω, ng )
# end


"""
################################################################################
#																			   #
#					solve_nω methods: (ε⁻¹, k) --> (n, ng)					   #
#						(mostly for debugging gradients)					   #
#																			   #
################################################################################
"""

function solve_nω(ms::ModeSolver{T},k,ε⁻¹::AbstractArray{T,5};nev=1,eigind=1,maxiter=100,tol=1e-8,log=false) where T<:Real
    ω², H⃗ = solve_ω²(ms,k,ε⁻¹;nev,eigind,maxiter,tol,log)
	ω = sqrt(ω²)
	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
	ng = ω / HMₖH(H⃗,ε⁻¹,real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗)))
	return ( k/ω, ng )
end

# function solve_nω(ms::ModeSolver{T},k,shapes::Vector{<:Shape};nev=1,eigind=1,maxiter=100,tol=1e-8,log=false) where T<:Real
# 	g::MaxwellGrid = make_MG(ms.M̂.Δx,ms.M̂.Δy,ms.M̂.Δz,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)
# 	# ε⁻¹ = HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},Float64,5,5,Array{Float64,5}}( make_εₛ⁻¹(shapes,g) )
# 	ε⁻¹ = make_εₛ⁻¹(shapes,dropgrad(ms))
# 	ω², H⃗ = solve_ω²(ms,k,ε⁻¹;nev,eigind,maxiter,tol,log)
# 	ω = sqrt(ω²)
# 	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
# 	ng = ω / HMₖH(H⃗,ε⁻¹,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
# 	return ( k/ω, ng )
# end

# function solve_nω(kz::T,shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;neigs=1,eigind=1,maxiter=100,tol=1e-8) where T<:Real
# 	# g::MaxwellGrid = make_MG(Zygote.dropgrad(Δx),Zygote.dropgrad(Δy),Zygote.dropgrad(Δz),Zygote.dropgrad(Nx),Zygote.dropgrad(Ny),Zygote.dropgrad(Nz)) #Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
# 	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
# 	# ds::MaxwellData = make_MD(kz,g) # MaxwellData(kz,g)
# 	# kpg_mag,kpg_mn = calc_kpg(kz,Zygote.dropgrad(Δx),Zygote.dropgrad(Δy),Zygote.dropgrad(Δz),Zygote.dropgrad(Nx),Zygote.dropgrad(Ny),Zygote.dropgrad(Nz))
# 	# mag,mn = calc_kpg(kz,Δx,Δy,Δz,Nx,Ny,Nz)
# 	mag,mn = calc_kpg(kz,g.g⃗)
# 	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
# 	H,ω² = solve_ω²(kz,ε⁻¹,Δx,Δy,Δz;neigs,eigind,maxiter,tol)
# 	# println("ω² = $ω²")
# 	ω = sqrt(ω²)
# 	Ha = reshape(H,(2,Nx,Ny,Nz))
# 	# ng = -ω / real( dot(Ha, kx_c2t( ifft( ε⁻¹_dot( fft( zx_t2c(Ha,ds.mn), (2:4) ), ε⁻¹), (2:4)),ds.mn,ds.mag) ) )
# 	ng = ω / HMₖH(Ha,ε⁻¹,mag,mn[:,1,:,:,:],mn[:,2,:,:,:])
# 	# ng = ω / real( dot(H, -vec( kx_c2t( ifft( ε⁻¹_dot( fft( zx_t2c(Ha,mn), (2:4) ), ε⁻¹), (2:4)),mn,mag) ) ) )
# 	# ng = -ω / real( dot(Ha, kx_c2t( ifft( ε⁻¹_dot( fft( zx_t2c(Ha,Zygote.@showgrad(mn)), (2:4) ), ε⁻¹), (2:4)), Zygote.@showgrad(mn),Zygote.@showgrad(mag)) ) )
# 	( kz/ω, ng )
# end

"""
################################################################################
#																			   #
#							   Plotting methods					   			   #
#																			   #
################################################################################
"""
# function uplot(ms::ModeSolver;xlim=[0.5,1.8])
# 	ls_mats = uplot(ms.materials;xlim)
# end

# uplot(ch::IterativeSolvers.ConvergenceHistory; kwargs...) = lineplot(log10.(ch.data[:resnorm]); name="log10(resnorm)", kwargs...)


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


using  IterativeSolvers, Roots # , KrylovKit
export solve_ω, _solve_Δω², solve_k, solve_n, ng, k_guess, solve_nω, solve_ω², make_εₛ⁻¹, make_MG, make_MD


"""
################################################################################
#																			   #
#	Routines to shield expensive initialization calculations from memory-	   #
#						intensive reverse-mode AD				  			   #
#																			   #
################################################################################
"""

k_guess(ω,ε⁻¹::Array{Float64,5}) = ( kg = Zygote.@ignore ( first(ω) * sqrt(1/minimum([minimum(ε⁻¹[a,a,:,:,:]) for a=1:3])) ); kg  )
k_guess(ω,shapes::Vector{<:Shape}) = ( kg = Zygote.@ignore ( first(ω) * √εₘₐₓ(shapes) ); kg  )
make_MG(Δx,Δy,Δz,Nx,Ny,Nz) = (g = Zygote.@ignore (MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)); g)::MaxwellGrid
make_MD(k,g::MaxwellGrid) = (ds = Zygote.@ignore (MaxwellData(k,g)); ds)::MaxwellData
make_KDTree(shapes::Vector{<:Shape}) = (tree = Zygote.@ignore (KDTree(shapes)); tree)::KDTree

function make_εₛ⁻¹(shapes::Vector{<:Shape},g::MaxwellGrid)::Array{Float64,5}
    tree = make_KDTree(shapes)
    eibuf = Zygote.Buffer(Array{Float64}(undef),3,3,g.Nx,g.Ny,1)
    for i=1:g.Nx,j=1:g.Ny,kk=1:g.Nz
		# eps = εₛ(shapes,Zygote.dropgrad(tree),Zygote.dropgrad(g.x[i]),Zygote.dropgrad(g.y[j]),Zygote.dropgrad(g.δx),Zygote.dropgrad(g.δy))
		eps = εₛ(shapes,tree,g.x[i],g.y[j],g.δx,g.δy)
		epsi = inv(eps) # inv( (eps' + eps) / 2) # Hermitian(inv(eps))  # inv(Hermitian(eps)) #   # inv(eps)
        eibuf[:,:,i,j,kk] = epsi #(epsi' + epsi) / 2
    end
    return real(copy(eibuf))
end


"""
################################################################################
#																			   #
#						solve_ω² methods: (ε⁻¹, k) --> (H, ω²)				   #
#																			   #
################################################################################
"""

function solve_ω²(kz::T,ε⁻¹::Array{Float64,5},ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8) where T<:Real
	# Δk = k - ds.k
	ds.k = kz
	ds.kpg_mag, ds.mn = calc_kpg(kz,ds.Δx,ds.Δy,ds.Δz,ds.Nx,ds.Ny,ds.Nz)
    # res = IterativeSolvers.lobpcg(M̂(ε⁻¹,ds),false,neigs;P=P̂(ε⁻¹,ds),maxiter,tol)
    res = IterativeSolvers.lobpcg(M̂!(ε⁻¹,ds),false,ds.H⃗;P=P̂!(ε⁻¹,ds),maxiter,tol)
    H =  res.X #[:,eigind]                       # eigenmode wavefn. magnetic fields in transverse pol. basis
    ds.ω² =  real(res.λ[eigind])                     # eigenmode temporal freq.,  neff = kz / ω, kz = k[3]
	ds.H⃗ .= H
	ds.ω = ( ds.ω² > 0. ? sqrt(ds.ω²) : 0. )
    # ds.ω²ₖ = 2 * H_Mₖ_H(Ha,ε⁻¹,kpg_mn,kpg_mag,ds.𝓕,ds.𝓕⁻¹) # = 2ω*ωₖ; ωₖ = ∂ω/∂kz = group velocity = c / ng; c = 1 here
    return (H, ds.ω^2) #(H, ds.ω²) #, ωₖ
end
# @btime: solve_ω²(1.5,$ε⁻¹_mpb;ds=$ds)
# 536.372 ms (17591 allocations: 125.75 MiB)

# function solve_ω²(k::Array{<:Real},ε⁻¹::Array{Float64,5},ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
# 	outs = [solve_ω²(kk,ε⁻¹,ds;neigs,eigind,maxiter,tol) for kk in k]
#     ( [o[1] for o in outs], [o[2] for o in outs] )
# end

function solve_ω²(kz,ε⁻¹::Array{Float64,5},g::MaxwellGrid;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    solve_ω²(kz,ε⁻¹,make_MD(first(kz),g);neigs,eigind,maxiter,tol)
end
# @btime:
# 498.442 ms (13823 allocations: 100.19 MiB)

function solve_ω²(kz::T,ε⁻¹::Array{T,5},Δx::T,Δy::T,Δz::T;neigs=1,eigind=1,maxiter=3000,tol=1e-8) where T<:Real
    solve_ω²(kz,ε⁻¹,make_MG(Δx,Δy,Δz,size(ε⁻¹)[end-2:end]...);neigs,eigind,maxiter,tol)
end

# function solve_k(ω::Union{Number,Vector{<:Number}},shapes::Vector{<:Shape},g::MaxwellGrid;kguess=k_guess(ω,shapes),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
#     solve_k(ω,shapes,make_MD(kguess,g)::MaxwellData;neigs,eigind,maxiter,tol)
# end

# function solve_k(ω::Union{Number,Vector{<:Number}},shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;kguess=k_guess(ω,shapes),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
# 	g = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
#     solve_k(ω,shapes,g;kguess,neigs,eigind,maxiter,tol)
# end

function solve_ω²(kz,shapes::Vector{<:GeometryPrimitives.Shape},g::MaxwellGrid;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
	ds::MaxwellData = make_MD(kz,g)
	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
	solve_ω²(kz,ε⁻¹,ds;neigs,eigind,maxiter,tol)
end

function solve_ω²(kz,shapes::Vector{<:GeometryPrimitives.Shape},Δx,Δy,Δz,Nx,Ny,Nz;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
	ds::MaxwellData = make_MD(kz,g)
	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
	solve_ω²(kz,ε⁻¹,ds;neigs,eigind,maxiter,tol)
end

# function solve_k(ω::Number,shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;kguess=k_guess(ω,shapes),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
# 	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
# 	ds::MaxwellData = make_MD(kguess,g)
# 	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
#     # solve_k(ω,ε⁻¹,ds;neigs,eigind,maxiter,tol)
# 	kz = Roots.find_zero(k -> _solve_Δω²(k,ω,ε⁻¹,ds;neigs,eigind,maxiter,tol), ds.k, Roots.Newton())
# 	return ( copy(ds.H⃗), kz )
# end


# function solve_ω²(k::Vector{<:Number},shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
# 	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
# 	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
#     outs = [solve_ω²(kk,ε⁻¹,make_MD(kk,g)::MaxwellData;neigs,eigind,maxiter,tol) for kk in k]
# 	return ( [o[1] for o in outs], [o[2] for o in outs] ) #( copy(ds.H⃗), kz )
# end

"""
################################################################################
#																			   #
#						solve_ω methods: (ε⁻¹, k) --> (H, ω)				   #
#																			   #
################################################################################
"""

function solve_ω(k::T,ε⁻¹::Array{Float64,5},ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8) where T<:Real
	# Δk = k - ds.k
	ds.k = k
	ds.kpg_mag, ds.mn = calc_kpg(k,ds.Δx,ds.Δy,ds.Δz,ds.Nx,ds.Ny,ds.Nz)
    # res = IterativeSolvers.lobpcg(M̂(ε⁻¹,ds),false,neigs;P=P̂(ε⁻¹,ds),maxiter,tol)
    res = IterativeSolvers.lobpcg(M̂!(ε⁻¹,ds),false,ds.H⃗;P=P̂!(ε⁻¹,ds),maxiter,tol)
    H =  res.X #[:,eigind]                       # eigenmode wavefn. magnetic fields in transverse pol. basis
    ω =  √(real(res.λ[eigind]))                     # eigenmode temporal freq.,  neff = kz / ω, kz = k[3]
	ds.H⃗ .= H
    ds.ω² = ω^2; ds.ω = ω;
    # ds.ω²ₖ = 2 * H_Mₖ_H(Ha,ε⁻¹,kpg_mn,kpg_mag,ds.𝓕,ds.𝓕⁻¹) # = 2ω*ωₖ; ωₖ = ∂ω/∂kz = group velocity = c / ng; c = 1 here
    return H, ω #, ωₖ
end
# @btime: solve_ω(1.5,$ε⁻¹_mpb;ds=$ds)
# 536.372 ms (17591 allocations: 125.75 MiB)

function solve_ω(k::Array{<:Real},ε⁻¹::Array{Float64,5},ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
	outs = [solve_ω(kk,ε⁻¹,ds;neigs,eigind,maxiter,tol) for kk in k]
    ( [o[1] for o in outs], [o[2] for o in outs] )
end

function solve_ω(k,ε⁻¹::Array{Float64,5},g::MaxwellGrid;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    solve_ω(k,ε⁻¹,MaxwellData(first(k),g);neigs,eigind,maxiter,tol)
end
# @btime:
# 498.442 ms (13823 allocations: 100.19 MiB)

function solve_ω(k,ε⁻¹::AbstractArray,Δx,Δy,Δz;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    solve_ω(k,ε⁻¹,MaxwellGrid(Δx,Δy,Δz,size(ε⁻¹)[end-2:end]...);neigs,eigind,maxiter,tol)
end

# function solve_ω(k,shapes::Vector{<:Shape},Δx,Δy,Δz;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
#     solve_ω(k,ε⁻¹,MaxwellGrid(Δx,Δy,Δz,size(ε⁻¹)[end-2:end]...);neigs,eigind,maxiter,tol)
# end
#
# function solve_ω(k::Union{Number,Vector{<:Number}},shapes::Vector{<:Shape},ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
#     solve_k(ω,make_εₛ⁻¹(shapes,ds.grid)::Array{Float64,5},ds;neigs,eigind,maxiter,tol)
# end

"""
################################################################################
#																			   #
#						solve_k methods: (ε⁻¹, ω) --> (H, k)				   #
#																			   #
################################################################################
"""


"""
modified solve_ω version for Newton solver, which wants (x -> f(x), f(x)/f'(x)) as input to solve f(x) = 0
"""
function _solve_Δω²(k,ωₜ,ε⁻¹::Array{Float64,5},ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    ds.k = k
	ds.kpg_mag, ds.mn = calc_kpg(k,ds.Δx,ds.Δy,ds.Δz,ds.Nx,ds.Ny,ds.Nz)
    res = IterativeSolvers.lobpcg(M̂!(ε⁻¹,ds),false,ds.H⃗;P=P̂!(ε⁻¹,ds),maxiter,tol)
    ds.H⃗ .=  res.X #[:,eigind]                      # eigenmode wavefn. magnetic fields in transverse pol. basis
    ds.ω² =  (real(res.λ[eigind]))                # eigenmode temporal freq.,  neff = kz / ωₖ, kz = k[3]
    Δω² = ds.ω² - ωₜ^2
    # ω²ₖ =   2 * real( ( H[:,eigind]' * M̂ₖ(ε⁻¹,ds) * H[:,eigind] )[1])  # = 2ω*ωₖ; ωₖ = ∂ω/∂kz = group velocity = c / ng; c = 1 here
	# Ha = reshape(H,(2,size(ε⁻¹)[end-2:end]...))
	ds.ω²ₖ = 2 * H_Mₖ_H(ds.H⃗,ε⁻¹,ds.kpg_mag,ds.mn) #,ds.𝓕,ds.𝓕⁻¹) # = 2ω*ωₖ; ωₖ = ∂ω/∂kz = group velocity = c / ng; c = 1 here
    return Δω² , Δω² / ds.ω²ₖ
end

# function solve_k(ω,ε⁻¹;Δx=6.0,Δy=4.0,Δz=1.0,k_guess=ω*sqrt(1/minimum([minimum(ε⁻¹[a,a,:,:,:]) for a=1:3])),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
function solve_k(ω::Number,ε⁻¹::Array{Float64,5},ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    kz = Roots.find_zero(k -> _solve_Δω²(k,ω,ε⁻¹,ds;neigs,eigind,maxiter,tol), ds.k, Roots.Newton())
    return ( copy(ds.H⃗), kz ) # maybe copy(ds.H⃗) instead?
end

function solve_k(ω::Vector{<:Number},ε⁻¹::Array{Float64,5},ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    outs = [solve_k(om,ε⁻¹,ds;neigs,eigind,maxiter,tol) for om in ω]
    ( [o[1] for o in outs], [o[2] for o in outs] )
end

function solve_k(ω::Number,ε⁻¹::Array{Float64,5},g::MaxwellGrid;kguess=k_guess(ω,ε⁻¹),neigs=1,eigind=1,maxiter=1000,tol=1e-6)
	solve_k(ω,ε⁻¹,make_MD(kguess,g);neigs,eigind,maxiter,tol)
end

function solve_k(ω::Number,ε⁻¹::Array{Float64,5},Δx,Δy,Δz;kguess=k_guess(ω,ε⁻¹),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
	g = make_MG(Δx,Δy,Δz,size(ε⁻¹)[end-2:end]...) #MaxwellGrid(Δx,Δy,Δz,size(ε⁻¹)[end-2:end]...)
	ds = make_MD(kguess,g) 			# MaxwellData(kguess,g)
    solve_k(ω,ε⁻¹,ds;neigs,eigind,maxiter,tol)
end

function solve_k(ω::Union{Number,Vector{<:Number}},shapes::Vector{<:Shape},ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    solve_k(ω,make_εₛ⁻¹(shapes,ds.grid)::Array{Float64,5},ds;neigs,eigind,maxiter,tol)
end

function solve_k(ω::Union{Number,Vector{<:Number}},shapes::Vector{<:Shape},g::MaxwellGrid;kguess=k_guess(ω,shapes),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    solve_k(ω,shapes,make_MD(kguess,g)::MaxwellData;neigs,eigind,maxiter,tol)
end

# function solve_k(ω::Union{Number,Vector{<:Number}},shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;kguess=k_guess(ω,shapes),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
# 	g = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
#     solve_k(ω,shapes,g;kguess,neigs,eigind,maxiter,tol)
# end

function solve_k(ω::Number,shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;kguess=k_guess(ω,shapes),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
	ds::MaxwellData = make_MD(kguess,g)
	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
    # solve_k(ω,ε⁻¹,ds;neigs,eigind,maxiter,tol)
	kz = Roots.find_zero(k -> _solve_Δω²(k,ω,ε⁻¹,ds;neigs,eigind,maxiter,tol), ds.k, Roots.Newton())
	return ( copy(ds.H⃗), kz )
end

function solve_k(ω::Vector{<:Number},shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;kguess=k_guess(ω,shapes),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
	ds::MaxwellData = make_MD(kguess,g)
	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
    outs = [solve_k(om,ε⁻¹,ds;neigs,eigind,maxiter,tol) for om in ω]
	return ( [o[1] for o in outs], [o[2] for o in outs] ) #( copy(ds.H⃗), kz )
end


"""
################################################################################
#																			   #
#						solve_n methods: (ε⁻¹, ω) --> (n, ng)				   #
#																			   #
################################################################################
"""

function solve_n(ω::Number,ε⁻¹::AbstractArray,ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
	k = Roots.find_zero(k -> _solve_Δω²(k,ω,ε⁻¹,ds;neigs,eigind,maxiter,tol), ds.k, Roots.Newton())
	( k / ω , 2ω / ds.ω²ₖ ) # = ( n , ng )
end

function solve_n(ω::Array{<:Real,1},ε⁻¹::AbstractArray,ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    outs = [solve_n(om,ε⁻¹,ds;neigs,eigind,maxiter,tol) for om in ω]
    ( [o[1] for o in outs], [o[2] for o in outs] )
end

function solve_n(ω,ε⁻¹::Array{Float64,5},g::MaxwellGrid;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
	k_guess = first(ω) * sqrt(1/minimum([minimum(ε⁻¹[a,a,:,:,:]) for a=1:3]))
	solve_n(ω,ε⁻¹,MaxwellData(k_guess,g);neigs,eigind,maxiter,tol)
end
# @btime:
# 498.442 ms (13823 allocations: 100.19 MiB)

# function solve_n(ω,ε⁻¹::AbstractArray,Δx,Δy,Δz;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
#     solve_n(ω,ε⁻¹,MaxwellGrid(Δx,Δy,Δz,size(ε⁻¹)[end-2:end]...);neigs,eigind,maxiter,tol)
# end

function solve_n(ω::Array{<:Real},ε⁻¹::Array{<:Real,5},Δx::T,Δy::T,Δz::T;eigind=1,maxiter=3000,tol=1e-8) where T<:Real
	H,k = solve_k(ω, ε⁻¹,Δx,Δy,Δz;eigind,maxiter,tol)
	( k ./ ω, [ ω[i] / H_Mₖ_H(H[i],ε⁻¹,calc_kpg(k[i],Δx,Δy,Δz,size(ε⁻¹)[end-2:end]...)...) for i=1:length(ω) ] ) # = (n, ng)
end

function solve_n(ω,ε⁻¹,Δx,Δy,Δz;eigind=1,maxiter=3000,tol=1e-8)
	Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)
	H,kz = solve_k(ω, ε⁻¹,Δx,Δy,Δz)
	mag, mn = calc_kpg(kz,Δx,Δy,Δz,Nx,Ny,Nz)
	ng = ω / H_Mₖ_H(H,ε⁻¹,mag,mn)
	( kz/ω, ng )
end

# function solve_n(ω::Array{<:Real},shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;kguess=k_guess(ω,shapes),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
# 	H,k = solve_k(ω,shapes,Δx,Δy,Δz,Nx,Ny,Nz;kguess,neigs,eigind,maxiter,tol)
# 	( k ./ ω, [ ω[i] / H_Mₖ_H(H[i],ε⁻¹,calc_kpg(k[i],Δx,Δy,Δz,Nx,Ny,Nz)...) for i=1:length(ω) ] ) # = (n, ng)
# end

# function solve_n(ω::Number,shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;kguess=k_guess(ω,shapes),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
# 	H,k = solve_k(ω,shapes,Δx,Δy,Δz,Nx,Ny,Nz;kguess,neigs,eigind,maxiter,tol)
# 	ng = ω / H_Mₖ_H(H,ε⁻¹,calc_kpg(k,Δx,Δy,Δz,Nx,Ny,Nz)...)
# 	( k/ω, ng )
# end

function solve_n(ω::Number,shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;kguess=k_guess(ω,shapes),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
	H,kz = solve_k(ω,ε⁻¹,Δx,Δy,Δz;kguess,neigs,eigind,maxiter,tol)
	kpg_mag,kpg_mn = calc_kpg(kz,Zygote.dropgrad(Δx),Zygote.dropgrad(Δy),Zygote.dropgrad(Δz),Zygote.dropgrad(Nx),Zygote.dropgrad(Ny),Zygote.dropgrad(Nz))
	ng = ω / H_Mₖ_H(H,ε⁻¹,kpg_mag,kpg_mn)
	( kz/ω, ng )
end

function solve_n(ω::Array{<:Real},shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;kguess=k_guess(ω,shapes),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
	H,k = solve_k(ω,shapes,Δx,Δy,Δz,Nx,Ny,Nz;kguess,neigs,eigind,maxiter,tol)
	( k ./ ω, [ ω[i] / H_Mₖ_H(H[i],ε⁻¹,calc_kpg(k[i],Δx,Δy,Δz,Nx,Ny,Nz)...) for i=1:length(ω) ] ) # = (n, ng)
end

"""
################################################################################
#																			   #
#					solve_nω methods: (ε⁻¹, k) --> (n, ng)					   #
#						(mostly for debugging gradients)					   #
#																			   #
################################################################################
"""

function solve_nω(kz::T,shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;neigs=1,eigind=1,maxiter=3000,tol=1e-8) where T<:Real
	# g::MaxwellGrid = make_MG(Zygote.dropgrad(Δx),Zygote.dropgrad(Δy),Zygote.dropgrad(Δz),Zygote.dropgrad(Nx),Zygote.dropgrad(Ny),Zygote.dropgrad(Nz)) #Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
	# ds::MaxwellData = make_MD(kz,g) # MaxwellData(kz,g)
	# kpg_mag,kpg_mn = calc_kpg(kz,Zygote.dropgrad(Δx),Zygote.dropgrad(Δy),Zygote.dropgrad(Δz),Zygote.dropgrad(Nx),Zygote.dropgrad(Ny),Zygote.dropgrad(Nz))
	# mag,mn = calc_kpg(kz,Δx,Δy,Δz,Nx,Ny,Nz)
	mag,mn = calc_kpg(kz,g.g⃗)
	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
	H,ω² = solve_ω²(kz,ε⁻¹,Δx,Δy,Δz;neigs,eigind,maxiter,tol)
	# println("ω² = $ω²")
	ω = sqrt(ω²)
	Ha = reshape(H,(2,Nx,Ny,Nz))
	# ng = -ω / real( dot(Ha, kx_c2t( ifft( ε⁻¹_dot( fft( zx_t2c(Ha,ds.mn), (2:4) ), ε⁻¹), (2:4)),ds.mn,ds.mag) ) )
	ng = ω / H_Mₖ_H(Ha,ε⁻¹,mag,mn)
	# ng = ω / real( dot(H, -vec( kx_c2t( ifft( ε⁻¹_dot( fft( zx_t2c(Ha,mn), (2:4) ), ε⁻¹), (2:4)),mn,mag) ) ) )
	# ng = -ω / real( dot(Ha, kx_c2t( ifft( ε⁻¹_dot( fft( zx_t2c(Ha,Zygote.@showgrad(mn)), (2:4) ), ε⁻¹), (2:4)), Zygote.@showgrad(mn),Zygote.@showgrad(mag)) ) )
	( kz/ω, ng )
end

function solve_nω(kz::T,ε⁻¹::Array{T,5},Δx,Δy,Δz,Nx,Ny,Nz;neigs=1,eigind=1,maxiter=3000,tol=1e-8) where T<:Real
	# g::MaxwellGrid = make_MG(Zygote.dropgrad(Δx),Zygote.dropgrad(Δy),Zygote.dropgrad(Δz),Zygote.dropgrad(Nx),Zygote.dropgrad(Ny),Zygote.dropgrad(Nz)) #Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
	# ds::MaxwellData = make_MD(kz,g) # MaxwellData(kz,g)
	# kpg_mag,kpg_mn = calc_kpg(kz,Zygote.dropgrad(Δx),Zygote.dropgrad(Δy),Zygote.dropgrad(Δz),Zygote.dropgrad(Nx),Zygote.dropgrad(Ny),Zygote.dropgrad(Nz))
	# mag,mn = calc_kpg(kz,Δx,Δy,Δz,Nx,Ny,Nz)
	mag,mn = calc_kpg(kz,g.g⃗)
	# ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
	H,ω² = solve_ω²(kz,ε⁻¹,Δx,Δy,Δz;neigs,eigind,maxiter,tol)
	# println("ω² = $ω²")
	@show ω = sqrt(ω²)
	Ha = reshape(H,(2,Nx,Ny,Nz))
	# ng = -ω / real( dot(Ha, kx_c2t( ifft( ε⁻¹_dot( fft( zx_t2c(Ha,ds.mn), (2:4) ), ε⁻¹), (2:4)),ds.mn,ds.mag) ) )
	# ng = ω / H_Mₖ_H(Ha,ε⁻¹,mag,mn)
	ng = ω / real( dot(H, -vec( kx_c2t( ifft( ε⁻¹_dot( fft( zx_t2c(Ha,mn), (2:4) ), ε⁻¹), (2:4)),mn,mag) ) ) )
	# ng = -ω / real( dot(Ha, kx_c2t( ifft( ε⁻¹_dot( fft( zx_t2c(Ha,Zygote.@showgrad(mn)), (2:4) ), ε⁻¹), (2:4)), Zygote.@showgrad(mn),Zygote.@showgrad(mag)) ) )
	( kz/ω, ng )
end


function solve_nω(kz::Array{<:Real},shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
	Hω = [solve_ω(kz[i],ε⁻¹,Δx,Δy,Δz;neigs,eigind,maxiter,tol) for i=1:length(kz)]
	ω² = [res[2] for res in Hω]
	ω = sqrt.(ω²)
	H = [res[1] for res in Hω]
	( kz ./ ω, [ ω[i] / H_Mₖ_H(H[i],ε⁻¹,calc_kpg(kz[i],Δx,Δy,Δz,Nx,Ny,Nz)...) for i=1:length(kz) ] ) # = (n, ng)
end

# using Zygote: @showgrad, dropgrad

# MkHa = Mₖ(Ha,ε⁻¹,kpg_mn,kpg_mag) #,ds.𝓕,ds.𝓕⁻¹)
# kxinds = [2; 1]
# kxscales = [-1.; 1.]
# @show size(H)
# temp = abs2.(H) #ε⁻¹_dot(zx_t2c(Ha,kpg_mn),ε⁻¹)
# Hastar = conj.(Ha)
# @tullio HMkH := Hastar[b,i,j,k] * kxscales[b] * kpg_mag[i,j,k] * temp[a,i,j,k] * kpg_mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds) nograd=(kxscales,kxinds,Hastar) fastmath=false verbose=2
# ng = ω / abs(HMkH)
# ng = sum(abs2,temp)
# ng = ω / real(H_Mₖ_H(H,ε⁻¹,kpg_mag,kpg_mn))
# ng = ω / H_Mₖ_H(H,ε⁻¹,Zygote.dropgrad(kpg_mag),Zygote.dropgrad(kpg_mn))
