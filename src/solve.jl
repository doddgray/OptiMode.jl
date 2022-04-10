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
	f_filter=nothing,overwrite=false) where {ND,T<:Real} 
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


function rrule(::typeof(solve_k), ω::T,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::TS;nev=1,
	max_eigsolves=60,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,kguess=nothing,Hguess=nothing,
	f_filter=nothing,overwrite=false) where {ND,T<:Real,TS<:AbstractEigensolver} 
	kmags,evecs = solve_k(ω, ε⁻¹, grid, solver; nev, maxiter, max_eigsolves, k_tol, eig_tol, log, kguess, Hguess,
	f_filter, overwrite)
	# g⃗ = copy(ms.M̂.g⃗)
	# (mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(k) do x
	# 	mag_m_n(x,dropgrad(g⃗))
	# end
	gridsize = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	# ε⁻¹_copy = copy(ε⁻¹)
	function solve_k_pullback(ΔΩ)
		ei_bar = zero(ε⁻¹)
		ω_bar = zero(ω)
		k̄mags, ēvecs = ΔΩ
		for (eigind, k̄, ēv, k, ev) in zip(1:nev, k̄mags, ēvecs, kmags, evecs)
			ms = ModeSolver(k, ε⁻¹, grid; nev, maxiter)
			ms.H⃗[:,eigind] = copy(ev)
			# replan_ffts!(ms)	# added  to check if this enables pmaps to work without crashing
			λ⃗ = randn(eltype(ev),size(ev)) # similar(ev)
			λd =  similar(ms.M̂.d)
			λẽ = similar(ms.M̂.d)
			∂ω²∂k = 2 * HMₖH(ev,ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.mn)
			# println("\tsolve_k_pullback:")
			# println("k̄ (bar): $k̄")
			# println("\tsolve_k pullback for eigind=$eigind:")
			# println("\t\tω² (target): $(ω^2)")
			# # println("\t\t∂ω²∂k (recorded): $(domsq_dk_solns[eigind])")
			# println("\t\t∂ω²∂k (recalc'd): $(∂ω²∂k)")
			# (mag,m⃗,n⃗), mag_m_n_pb = Zygote.pullback(kk->mag_m_n(kk,g⃗(ms.grid)),k)
			ev_grid = reshape(ev,(2,gridsize...))
			if isa(k̄,AbstractZero)
				k̄ = 0.0
			end
			if !isa(ēv,AbstractZero)
				# solve_adj!(ms,ēv,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = ēv - dot(ev,ēv)*ev
				λ⃗ = eig_adjt(ms.M̂, ω^2, ev, 0.0, ēv; λ⃗₀=randn(eltype(ev),size(ev)), P̂=ms.P̂)
				# @show val_magmax(λ⃗)
				# @show dot(ev,λ⃗)
				λ⃗ 	-= 	 dot(ev,λ⃗) * ev
				λ	=	reshape(λ⃗,(2,gridsize...))
				d = _H2d!(ms.M̂.d, ev_grid * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( ev_grid , mn2, mag )  * ms.M̂.Ninv
				λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
				ei_bar += ε⁻¹_bar(vec(ms.M̂.d), vec(λd), gridsize...) # eīₕ  # prev: ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), gridsize...)
				# @show val_magmax(ei_bar)
				### back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
				λd *=  ms.M̂.Ninv
				λẽ_sv = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,ms ) )
				ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
				# @show val_magmax(λẽ)
				# @show val_magmax(reinterpret(reshape,Complex{T},ẽ))
				kx̄_m⃗ = real.( λẽ_sv .* conj.(view( ev_grid,2,axes(grid)...)) .+ ẽ .* conj.(view(λ,2,axes(grid)...)) )
				kx̄_n⃗ =  -real.( λẽ_sv .* conj.(view( ev_grid,1,axes(grid)...)) .+ ẽ .* conj.(view(λ,1,axes(grid)...)) )
				m⃗ = reinterpret(reshape, SVector{3,Float64},ms.M̂.mn[:,1,axes(grid)...])
				n⃗ = reinterpret(reshape, SVector{3,Float64},ms.M̂.mn[:,2,axes(grid)...])
				māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
				# @show k̄ₕ_old = -mag_m_n_pb(( māg, kx̄_m⃗.*ms.M̂.mag, kx̄_n⃗.*ms.M̂.mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
				k̄ₕ = -∇ₖmag_m_n(
					māg,
					kx̄_m⃗.*ms.M̂.mag, # m̄,
					kx̄_n⃗.*ms.M̂.mag, # n̄,
					ms.M̂.mag,
					m⃗,
					n⃗;
					dk̂=SVector(0.,0.,1.), # dk⃗ direction
				)
				# @show k̄ₕ
			else
				# eīₕ = zero(ε⁻¹)#fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
				k̄ₕ = 0.0
			end
			# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω_bar and eīₖ
			λ⃗ = ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * ev
			d = _H2d!(ms.M̂.d, ev_grid * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( ev_grid , mn2, mag )  * ms.M̂.Ninv
			λd = _H2d!(λd,reshape(λ⃗,(2,gridsize...)),ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )			
			ei_bar += ε⁻¹_bar(vec(ms.M̂.d), vec(λd), gridsize...) # eīₖ # epsinv_bar = eīₖ + eīₕ
			ω_bar +=  ( 2*ω * (k̄ + k̄ₕ ) / ∂ω²∂k )  
			# @show ω_bar
		end
		return (NoTangent(), ω_bar , ei_bar,ZeroTangent(),NoTangent())
	end
	return ((kmags, evecs), solve_k_pullback)
end












