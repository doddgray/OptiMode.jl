export solve_ω², _solve_Δω², solve_k, filter_eigs

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


# # function find_k(ω::Real,ε::AbstractArray,grid::Grid{ND};num_bands=2,band_min=1,band_max=num_bands,filename_prefix="f01",data_path=pwd(),kwargs...) where ND
# function rrule(::typeof(find_k),ω::Real,ε⁻¹,grid::Grid{ND};nev=1,eigind=1,maxiter=300,tol=1e-8,log=false,f_filter=nothing) where {ND}
# 		# ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{T};nev=1,eigind=1,maxiter=300,tol=1e-8,log=false,f_filter=nothing
# 	kmags, evecs = find_k(ω,ε,grid);
	
# 	function find_k_pullback(ΔΩ)
# 		k̄, ēvecs = ΔΩ
# 		ε⁻¹ = sliceinv_3x3(ε)
# 		ms = ModeSolver(kmags[1], ε⁻¹, grid; nev, maxiter, tol)
# 		# println("\tsolve_k_pullback:")
# 		# println("k̄ (bar): $k̄")
# 		update_k!(ms,k)
# 		update_ε⁻¹(ms,ε⁻¹) #ε⁻¹)
# 		ms.ω²[eigind] = omsq_soln # ω^2
# 		ms.∂ω²∂k[eigind] = ∂ω²∂k
# 		copyto!(ms.H⃗, Hv)
# 		replan_ffts!(ms)	# added  to check if this enables pmaps to work without crashing
# 		λ⃗ = similar(Hv)
# 		λd =  similar(ms.M̂.d)
# 		λẽ = similar(ms.M̂.d)
# 		# ε⁻¹_bar = similar(ε⁻¹)
# 		# ∂ω²∂k = ms.∂ω²∂k[eigind] # copy(ms.∂ω²∂k[eigind])
# 		# Ns = size(ms.grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
# 		# Nranges = eachindex(ms.grid)

# 		H = reshape(Hv,(2,Ns...))
# 		# if typeof(k̄)==ZeroTangent()
# 		if isa(k̄,AbstractZero)
# 			k̄ = 0.
# 		end
# 		# if typeof(ēvecs) != ZeroTangent()
# 		if !isa(ēvecs,AbstractZero)
# 			# solve_adj!(ms,ēvecs,eigind) 												# overwrite ms.λ⃗ with soln to (M̂ + ω²I) λ⃗ = ēvecs - dot(Hv,ēvecs)*Hv
# 			solve_adj!(λ⃗,ms.M̂,ēvecs,omsq_soln,Hv,eigind;log=false)
# 			# solve_adj!(ms,ēvecs,ω^2,Hv,eigind)
# 			λ⃗ -= dot(Hv,λ⃗) * Hv
# 			λ = reshape(λ⃗,(2,Ns...))
# 			d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
# 			λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
# 			# eīₕ = ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), Ns...)
# 			eīₕ = ε⁻¹_bar(vec(ms.M̂.d), vec(λd), Ns...)
# 			# eīₕ = copy(ε⁻¹_bar)
# 			# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
# 			λd *=  ms.M̂.Ninv
# 			λẽ_sv = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,ms ) )
# 			ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(ms.M̂.e,ms.M̂.d,ms) )
# 			kx̄_m⃗ = real.( λẽ_sv .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
# 			kx̄_n⃗ =  -real.( λẽ_sv .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
# 			māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
# 			k̄ₕ = -mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1] # m̄ = kx̄_m⃗ .* mag, n̄ = kx̄_n⃗ .* mag, #NB: not sure why this is needs to be negated, inputs match original version
# 		else
# 			eīₕ = zero(ε⁻¹)#fill(SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.),size(ε⁻¹))
# 			k̄ₕ = 0.0
# 		end
# 		# combine k̄ₕ with k̄, scale by ( 2ω / ∂ω²∂k ) and calculate ω̄ and eīₖ
# 		copyto!(λ⃗, ( (k̄ + k̄ₕ ) / ∂ω²∂k ) * Hv )
# 		λ = reshape(λ⃗,(2,Ns...))
# 		d = _H2d!(ms.M̂.d, H * ms.M̂.Ninv, ms) # =  ms.M̂.𝓕 * kx_tc( H , mn2, mag )  * ms.M̂.Ninv
# 		λd = _H2d!(λd,λ,ms) # ms.M̂.𝓕 * kx_tc( reshape(λ⃗,(2,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)) , mn2, mag )
# 		# ε⁻¹_bar!(ε⁻¹_bar, vec(ms.M̂.d), vec(λd), Ns...)
# 		# eīₖ = copy(ε⁻¹_bar)
# 		eīₖ = ε⁻¹_bar(vec(ms.M̂.d), vec(λd), Ns...)
# 		# ε⁻¹_bar = eīₖ + eīₕ
# 		eibar = eīₖ + eīₕ
# 		ω̄  =  ( 2ω * (k̄ + k̄ₕ ) / ∂ω²∂k )  #2ω * k̄ₖ / ms.∂ω²∂k[eigind]
# 		# if !(typeof(k)<:SVector)
# 		# 	k̄_kx = k̄_kx[3]
# 		# end
# 		# ms.ω̄  = 2ω * ( k̄_kx  / ms.∂ω²∂k[eigind] ) # = 2ω * ω²̄
# 		return (NoTangent(), ZeroTangent(), ω̄  , eibar)
# 	end

# 	return ((k, Hv), solve_k_pullback)
# end