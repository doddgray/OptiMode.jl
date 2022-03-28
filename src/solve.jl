export solve_ω², _solve_Δω², solve_k, solve_k_single, filter_eigs
export AbstractEigensolver

"""
################################################################################
#																			   #
#						solve_ω² methods: (ε⁻¹, k) --> (H, ω²)				   #
#																			   #
################################################################################
"""
abstract type AbstractEigensolver end

# abstract type AbstractLinearSolver end

"""
	solve_ω²(ms::ModeSolver, solver::AbstractEigensolver; kwargs...)

	Find a few extremal eigenvalue/eigenvector pairs of the `HelmholtzOperator` map 
	in the modesolver object `ms`. The eigenvalues physically correspond to ω², the
	square of the temporal frequencies of electromagnetic resonances (modes) of the
	dielectric structure being modeled with Bloch wavevector k⃗, a 3-vector of spatial
	frequencies.
"""
function solve_ω²(ms::ModeSolver{ND,T}, solver::TS; kwargs...)::Tuple{Vector{T},Vector{Vector{Complex{T}}}} where {ND,T<:Real,TS<:AbstractEigensolver} end

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

# # function _solve_ω²(ms::ModeSolver{ND,T},::;nev=1,eigind=1,maxiter=100,tol=1.6e-8

# function solve_ω²(ms::ModeSolver{ND,T},solver::AbstractEigensolver;nev=1,maxiter=200,k_tol=1e-8,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real}
# 	evals,evecs = _solve_ω²(ms,solver;nev,eigind,maxiter,tol,log,f_filter)
# 	# @assert isequal(size(ms.H⃗,2),nev) # check that the modesolver struct is consistent with the number of eigenvalue/vector pairs `nev`
# 	# evals_res = evals[1:nev]
# 	# evecs_res = vec.(evecs[1:nev])
# 	# copyto!(ms.H⃗,hcat(evecs_res...)) 
# 	# copyto!(ms.ω²,evals_res)
	
# 	# res = lobpcg!(ms.eigs_itr; log,not_zeros=false,maxiter,tol)

# 	# res = LOBPCG(ms.M̂,ms.H⃗,I,ms.P̂,tol,maxiter)
# 	# copyto!(ms.H⃗,res.X)
# 	# copyto!(ms.ω²,res.λ)


# 	# if isnothing(f_filter)
# 	# 	return   (copy(real(ms.ω²)), copy(ms.H⃗))
# 	# else
# 	# 	return filter_eigs(ms, f_filter)
# 	# end
# 	return evals, evecs
# end

function solve_ω²(ms::ModeSolver{ND,T},k::TK,solver::AbstractEigensolver;nev=1,maxiter=100,tol=1e-8,
	log=false,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
	# @ignore(update_k!(ms,k))
	update_k!(ms,k)
	solve_ω²(ms,solver; nev, maxiter, tol, log, f_filter)
end

function solve_ω²(ms::ModeSolver{ND,T},k::TK,ε⁻¹::AbstractArray{T},solver::AbstractEigensolver;nev=1,
	maxiter=100,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
	@ignore(update_k!(ms,k))
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms,solver; nev, maxiter, tol, log, f_filter)
end

function solve_ω²(k::TK,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::AbstractEigensolver;nev=1,maxiter=100,
	tol=1e-8,log=false,evecs_guess=nothing,f_filter=nothing) where {ND,T<:Real,TK<:Union{T,SVector{3,T}}}
	ms = ignore() do
		ms = ModeSolver(k, ε⁻¹, grid; nev, maxiter, tol)
		if !isnothing(Hguess)
			ms.H⃗ = reshape(Hguess,(2*length(grid),2))
		end
		return ms
	end
	solve_ω²(ms,solver; nev, maxiter, tol, log, f_filter)
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
function _solve_Δω²(ms::ModeSolver{ND,T},k::TK,ωₜ::T,evec_out::Vector{Complex{T}},solver::AbstractEigensolver;nev=1,
	eigind=1,maxiter=100,eig_tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,TK}
	# println("k: $(k)")
	evals,evecs = solve_ω²(ms,k,solver; nev, maxiter, tol=eig_tol, log, f_filter)
	evec_out[:] = copy(evecs[eigind]) #copyto!(evec_out,evecs[eigind])
	Δω² = evals[eigind] - ωₜ^2
	# ∂ω²∂k = 2 * HMₖH(evecs[eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn) # = 2ω*(∂ω/∂|k|); ∂ω/∂|k| = group velocity = c / ng; c = 1 here
	∂ω²∂k = 2 * HMₖH(evec_out,ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn) # = 2ω*(∂ω/∂|k|); ∂ω/∂|k| = group velocity = c / ng; c = 1 here
	ms.∂ω²∂k[eigind] = ∂ω²∂k
	ms.ω²[eigind] = evals[eigind]
	# println("Δω²: $(Δω²)")
	# println("∂ω²∂k: $(∂ω²∂k)")
    return Δω² , ( Δω² / ∂ω²∂k )
end

# ::Tuple{T,Vector{Complex{T}}}
function solve_k_single(ms::ModeSolver{ND,T},ω::T,solver::AbstractEigensolver;nev=1,eigind=1,
	maxiter=100,max_eigsolves=60,k_tol=1e-10,eig_tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real} #
    evec_out = Vector{Complex{T}}(undef,size(ms.H⃗,1))
	kmag = Roots.find_zero(
		x -> _solve_Δω²(ms,x,ω,evec_out,solver;nev,eigind,maxiter,eig_tol,f_filter),	# f(x), it will find zeros of this function
		ms.M̂.k⃗[3],				  # initial guess, previous |k|(ω) solution
		Roots.Newton(); 			# iterative zero-finding algorithm
		atol=k_tol,					# absolute |k| convergeance tolerance 
		maxevals=max_eigsolves,		# max Newton iterations before it gives up
		#verbose=true,
	)
	return kmag, evec_out #copy(ms.H⃗[:,eigind])
end

# ::Tuple{T,Vector{Complex{T}}}
function solve_k(ms::ModeSolver{ND,T},ω::T,solver::AbstractEigensolver;nev=1,maxiter=100,k_tol=1e-8,eig_tol=1e-8,
	max_eigsolves=60,log=false,f_filter=nothing) where {ND,T<:Real} #
	kmags = Vector{T}(undef,nev)
	evecs = Matrix{Complex{T}}(undef,(size(ms.H⃗,1),nev))
	for (idx,eigind) in enumerate(1:nev)
		# idx>1 && copyto!(ms.H⃗,repeat(evecs[:,idx-1],1,size(ms.H⃗,2)))
		kmag, evec = solve_k_single(ms,ω,solver;nev,eigind,maxiter,max_eigsolves,k_tol,eig_tol,log)
		kmags[idx] = kmag
		evecs[:,idx] =  canonicalize_phase(evec,kmag,ms.M̂.ε⁻¹,ms.grid)
	end
	return kmags, collect(copy.(eachcol(evecs))) #evecs #[copy(ev) for ev in eachcol(evecs)] #collect(eachcol(evecs))
end

function solve_k(ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{T},solver::AbstractEigensolver;nev=1,
	max_eigsolves=60, maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real} 
	Zygote.@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_k(ms, ω, solver; nev, maxiter, max_eigsolves, k_tol, eig_tol, log, f_filter)
end

function solve_k(ω::T,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::AbstractEigensolver;nev=1,
	max_eigsolves=60,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,kguess=nothing,Hguess=nothing,
	f_filter=nothing) where {ND,T<:Real} 
	# ms = ignore() do
	# 	kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
	# 	ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, eig_tol)
	# 	if !isnothing(Hguess)
	# 		ms.H⃗ = reshape(Hguess,size(ms.H⃗))
	# 	end
	# 	return ms
	# end
	ms = ModeSolver(k_guess(ω,ε⁻¹), ε⁻¹, grid; nev, maxiter, tol=eig_tol)
	solve_k(ms, ω, solver; nev, maxiter, max_eigsolves, k_tol, eig_tol, log, f_filter,)
end

# function solve_k(ω::T,p::AbstractVector,geom_fn::F,grid::Grid{ND},solver::AbstractEigensolver;kguess=nothing,Hguess=nothing,nev=1,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function}
# 	ε⁻¹ = smooth(ω,p,:fεs,true,geom_fn,grid)
# 	ms = ignore() do
# 		kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
# 		ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, tol)
# 		if !isnothing(Hguess)
# 			ms.H⃗ = reshape(Hguess,size(ms.H⃗))
# 		end
# 		return ms
# 	end
# 	solve_k(ms, ω, solver; nev, maxiter, tol, log, f_filter)
# end

# function solve_k(ω::AbstractVector{T},p::AbstractVector,geom_fn::F,grid::Grid{ND},solver::AbstractEigensolver;kguess=nothing,Hguess=nothing,nev=1,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function}
# 	ε⁻¹ = smooth(ω,p,:fεs,true,geom_fn,grid)
# 	# ms = @ignore(ModeSolver(k_guess(first(ω),first(ε⁻¹)), first(ε⁻¹), grid; nev, maxiter, tol))
# 	ms = ignore() do
# 		kguess = isnothing(kguess) ? k_guess(ω,ε⁻¹) : kguess
# 		ms = ModeSolver(kguess, ε⁻¹, grid; nev, maxiter, tol)
# 		if !isnothing(Hguess)
# 			ms.H⃗ = Hguess
# 		end
# 		return ms
# 	end
# 	nω = length(ω)
# 	k = Buffer(ω,nω)
# 	Hv = Buffer([1.0 + 3.0im, 2.1+4.0im],(size(ms.M̂)[1],nω))
# 	for ωind=1:nω
# 		@ignore(update_ε⁻¹(ms,ε⁻¹[ωind]))
# 		kHv = solve_k(ms,ω[ωind],solver; nev, maxiter, tol, log, f_filter)
# 		k[ωind] = kHv[1]
# 		Hv[:,ωind] = kHv[2]
# 	end
# 	return copy(k), copy(Hv)
# end






# ω::T,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::AbstractEigensolver;nev=1,
# 	max_eigsolves=60,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,kguess=nothing,Hguess=nothing,
# 	f_filter=nothing) where {ND,T<:Real} 

# function solve_disp(ω::T,p::AbstractVector,geom_fn::F,grid::Grid{ND};kguess=nothing,Hguess=nothing,dk̂=SVector(0.0,0.0,1.0),nev=1,eigind=1,maxiter=500,tol=1e-8,log=false,f_filter=nothing) where {ND,T<:Real,F<:Function} # output type ::Tuple{T,T,T,Vector{Complex{T}}}

# function solve_disp(ω::Real,ε::AbstractArray{<:Real},∂ε_∂ω::AbstractArray{<:Real},∂²ε_∂ω²::AbstractArray{<:Real},
# 	grid::Grid{ND,T},solver::AbstractEigensolver;nev=1,max_eigsolves=60,maxiter=500,k_tol=1e-8,eig_tol=1e-8,
# 	log=false,kguess=nothing,Hguess=nothing,dk̂=SVector(0.0,0.0,1.0),log=false,f_filter=nothing) where {ND,T<:Real}
# 	ε⁻¹             =   sliceinv_3x3(ε);
# 	kmags,evecs   	=	solve_k(ω,ε⁻¹,grid,solver;nev,eig_tol,k_tol)
# 	ngs 			=	map((k,ev)->group_index(k,ev,ω,ε⁻¹,∂ε_∂ω,grid),kmags,evecs) 
# 	neffs = inv(ω) * kmags

# 	# # calculate effective group index `ng`
# 	# Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# 	# Ninv 		= 		1. / N(grid)
# 	# mag,m⃗,n⃗ = mag_m_n(k,grid)
# 	# m = flat(m⃗)
# 	# n = flat(n⃗)
# 	# mns = copy(vcat(reshape(m,1,3,Ns...),reshape(n,1,3,Ns...)))
#     # Hₜ = reshape(Hv,(2,Ns...))
# 	# D = 1im * fft( kx_tc( Hₜ,mns,mag), _fftaxes(grid) )
# 	# E = ε⁻¹_dot( D, ε⁻¹)
# 	# # E = 1im * ε⁻¹_dot( fft( kx_tc( Hₜ,mns,mag), (2:1+ND) ), ε⁻¹)
# 	# # H = inv(ω) * fft(tc(kx_ct( ifft( E, (2:1+ND) ), mns,mag), mns),(2:1+ND) )
# 	# H = fft( tc(Hₜ,mns), (2:3) ) * (-1im * ω)
# 	# P = 2*real(_sum_cross_z(conj(E),H))    # integrated Poyting flux parallel to ẑ: P = ∫dA S⃗⋅ẑ
# 	# # W = dot(E,_dot((ε+nng),E))             # energy density per unit length
# 	# W = real(dot(E,_dot(nng,E))) + (N(grid)* (ω^2))     # energy density per unit length
# 	# ng = real( W / P )

# 	# calculate GVD = ∂(ng) / ∂ω = (∂²k)/(∂ω²)
# 	W̄ = inv(P)
# 	om̄₁₁ = 2*ω * N(grid) * W̄
# 	nnḡ = _outer(E,E) * W̄
# 	# H̄ = (-2*ng*W̄) * _cross(repeat([0.,0.,1.],outer=(1,Ns...)), E)
# 	# Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,repeat([0.,0.,1.],outer=(1,Ns...))) )
# 	H̄ = (-2*ng*W̄) * _cross(dk̂, E)
# 	Ē = 2W̄*( _dot(nng,E) - ng * _cross(H,dk̂) )
# 	om̄₁₂ = dot(H,H̄) / ω
# 	om̄₁ = om̄₁₁ + om̄₁₂
# 	# eī₁ = _outer(Ē,D) ####################################
# 	𝓕⁻¹_ε⁻¹_Ē = bfft(ε⁻¹_dot( Ē, ε⁻¹),(2:3))
# 	𝓕⁻¹_H̄ = bfft( H̄ ,(2:3))
# 	H̄ₜ = 1im*( kx_ct(𝓕⁻¹_ε⁻¹_Ē,mns,mag) + ω*ct(𝓕⁻¹_H̄,mns) )
# 	local one_mone = [1.0im, -1.0im]
# 	@tullio 𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ[i,j,ix,iy] := one_mone[i] * reverse(Hₜ;dims=1)[i,ix,iy] * conj(𝓕⁻¹_ε⁻¹_Ē)[j,ix,iy] nograd=one_mone
# 	∂ω²∂k_nd = 2 * HMₖH(Hv,ε⁻¹,mag,m,n)

# 	##### grad solve k
# 	# solve_adj!(λ⃗,M̂,H̄,ω^2,H⃗,eigind)
# 	M̂2 = HelmholtzMap(k,ε⁻¹,grid)
# 	λ⃗	= eig_adjt(
# 		M̂2,								 # Â
# 		ω^2, 							# α
# 		Hv, 					 		 # x⃗
# 		0.0, 							# ᾱ
# 		vec(H̄ₜ);								 # x̄
# 		# λ⃗₀,
# 		P̂	= HelmholtzPreconditioner(M̂2),
# 	)
# 	### k̄ₕ, eīₕ = ∇M̂(k,ε⁻¹,λ⃗,H⃗,grid)
# 	λ = reshape(λ⃗,(2,Ns...))
# 	λd 	= 	fft(kx_tc( λ , mns, mag ),_fftaxes(grid))
# 	# eīₕ	 = 	 ε⁻¹_bar(vec(D * (Ninv * -1.0im)), vec(λd), Ns...) ##########################
# 	λẽ  =   bfft(ε⁻¹_dot(λd , ε⁻¹),_fftaxes(grid))
# 	ẽ 	 =   bfft(E * -1.0im,_fftaxes(grid))
# 	@tullio mn̄s_kx0[i,j,ix,iy] := -1.0im * one_mone[i] * reverse(conj(Hₜ);dims=1)[i,ix,iy] * (Ninv*λẽ)[j,ix,iy] + -1.0im * one_mone[i] * reverse(conj(λ);dims=1)[i,ix,iy] * (Ninv*ẽ)[j,ix,iy]  nograd=one_mone
# 	# @tullio mn̄s_kx0[i,j,ix,iy] := -1.0im * one_mone[i] * reverse(conj(Hₜ);dims=1)[i,ix,iy] * λẽ[j,ix,iy] + -1.0im * one_mone[i] * reverse(conj(λ);dims=1)[i,ix,iy] * ẽ[j,ix,iy]  nograd=one_mone
# 	# @tullio mn̄s_kx0[i,j,ix,iy] := -1.0im * one_mone[i] * reverse(conj(Hₜ);dims=1)[i,ix,iy] * λẽ[j,ix,iy] + -1.0im * one_mone[i] * reverse(conj(λ);dims=1)[i,ix,iy] * ẽ[j,ix,iy]  nograd=one_mone
# 	@tullio mn̄s[i,j,ix,iy] := mag[ix,iy] * (mn̄s_kx0-conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ))[i,j,ix,iy]  + 1im*ω*conj(Hₜ)[i,ix,iy]*𝓕⁻¹_H̄[j,ix,iy]
# 	@tullio māg[ix,iy] := mns[a,b,ix,iy] * (mn̄s_kx0-conj(𝓕⁻¹_ε⁻¹_Ē_xHₜᵀ))[a,b,ix,iy]
# 	# k̄ = ∇ₖmag_mn(māg,mn̄s,mag,mns)
# 	@tullio kp̂g_over_mag[i,ix,iy] := m[mod(i-2),ix,iy] * n[mod(i-1),ix,iy] / mag[ix,iy] - m[mod(i-1),ix,iy] * n[mod(i-2),ix,iy] / mag[ix,iy] (i in 1:3)
# 	kp̂g_over_mag_x_dk̂ = _cross(kp̂g_over_mag,dk̂)
# 	@tullio k̄_mag := māg[ix,iy] * mag[ix,iy] * kp̂g_over_mag[j,ix,iy] * dk̂[j]
# 	@tullio k̄_mn := -conj(mn̄s)[imn,i,ix,iy] * mns[imn,mod(i-2),ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-1),ix,iy] + conj(mn̄s)[imn,i,ix,iy] * mns[imn,mod(i-1),ix,iy] * kp̂g_over_mag_x_dk̂[mod(i-2),ix,iy] (i in 1:3)
# 	k̄ = k̄_mag + k̄_mn
# 	### \ k̄ₕ, eīₕ = ∇M̂(k,ε⁻¹,λ⃗,H⃗,grid)

# 	# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
# 	λₖ  = ( k̄ / ∂ω²∂k_nd ) * Hₜ #reshape(λ⃗ₖ, (2,Ns...))
# 	λdₖ	=	fft(kx_tc( λₖ , mns, mag ),_fftaxes(grid))
# 	# eīₖ = ε⁻¹_bar(vec(D* (Ninv * -1.0im)), vec(λdₖ), Ns...) ####################################
# 	om̄₂  =  2ω * k̄ / ∂ω²∂k_nd
# 	##### \grad solve k
# 	om̄₃ = dot(herm(nnḡ), ngvd)
# 	om̄₄ = dot( herm(_outer(Ē+(λd+λdₖ)*(Ninv * -1.0im),D) ), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
# 	# @show om̄₄_old = dot( ( eīₖ + eīₕ + eī₁ ), ∂ε⁻¹_∂ω(ε,ε⁻¹,nng,ω))
# 	gvd = real( om̄₁ + om̄₂ + om̄₃ + om̄₄ )

# 	return ( neff, ng, gvd, E )
# end










