using  IterativeSolvers, Roots # , KrylovKit
export solve_ω, _solve_Δω², solve_k, solve_n, ng, k_guess, solve_nω, solve_ω², replan_ffts!
using Zygote: Buffer

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


function make_εₛ⁻¹(shapes::Vector{<:Shape},M̂::HelmholtzMap{T}) where T<:Real
	make_εₛ⁻¹(shapes;Δx=M̂.Δx,Δy=M̂.Δy,Δz=M̂.Δz,Nx=M̂.Nx,Ny=M̂.Ny,Nz=M̂.Nz,
		 	δx=M̂.δx,δy=M̂.δy,δz=M̂.δz,x=M̂.x,y=M̂.y,z=M̂.z)
end

function make_εₛ⁻¹(shapes::Vector{<:Shape},ms::ModeSolver{T}) where T<:Real
	make_εₛ⁻¹(shapes;Δx=ms.M̂.Δx,Δy=ms.M̂.Δy,Δz=ms.M̂.Δz,Nx=ms.M̂.Nx,Ny=ms.M̂.Ny,
		 	Nz=ms.M̂.Nz,δx=ms.M̂.δx,δy=ms.M̂.δy,δz=ms.M̂.δz,x=ms.M̂.x,y=ms.M̂.y,z=ms.M̂.z)
end


function make_εₛ⁻¹(ω,shapes::Vector{<:Shape},ms::ModeSolver{T}) where T<:Real
    eibuf = Buffer(bounds(shapes[1])[1],3,3,Nx,Ny,Nz)
	eps_shapes = vcat( [ s.data(1/ω) for s in shapes ], [Diagonal([1,1,1]),] )
	# eibuf = Buffer(bounds(shapes[1])[1],3,3,Nx,Ny,Nz)
    for I ∈ eachindex(ms.M̂.xyz)
		sinds = ms.M̂.corner_sinds_proc[I]
		# eps = εₛ(shapes,Zygote.dropgrad(tree),Zygote.dropgrad(g.x[i]),Zygote.dropgrad(g.y[j]),Zygote.dropgrad(g.δx),Zygote.dropgrad(g.δy))
		# eps = εₛ(shapes,x[i],y[j];tree,δx,δy)
		if sinds[2]==0
			eibuf[:,:,I[1],I[2],I[3]] = inv(eps_shapes[sinds[1]])
		elseif sinds[3]==0
			r₀,nout = surfpt_nearby(ms.M̂.xyz(I), shapes[sinds[1]])
			rvol = volfrac((ms.M̂.xyzc[I], ms.M̂.xyzc[I+CartesianIndex(1,1,1)]),nout,r₀)
			eibuf[:,:,I[1],I[2],I[3]] = inv(avg_param(
					eps_shapes[sinds[1]],
					eps_shapes[sinds[2]],
					[nout[1];nout[2];0],
                    rvol,
				)
			)
		else
			eibuf[:,:,I[1],I[2],I[3]] = 8. * inv( sum( getindex( eps_shapes,[sinds...] ) ) )
		end
		# epsi = inv(eps) # inv( (eps' + eps) / 2) # Hermitian(inv(eps))  # inv(Hermitian(eps)) #   # inv(eps)
        # eibuf[:,:,I...] = epsi #(epsi' + epsi) / 2
    end
    # return HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},T,5,5,Array{T,5}}( real(copy(eibuf)) )
	return HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()}}( real(copy(eibuf)) )
end

# function make_εₛ⁻¹(shapes::Vector{<:Shape},g::MaxwellGrid)::Array{T,5} where T<:Real
#     tree = make_KDTree(shapes)
#     eibuf = Zygote.Buffer(Array{T}(undef),3,3,g.Nx,g.Ny,1)
#     for i=1:g.Nx,j=1:g.Ny,kk=1:g.Nz
# 		# eps = εₛ(shapes,Zygote.dropgrad(tree),Zygote.dropgrad(g.x[i]),Zygote.dropgrad(g.y[j]),Zygote.dropgrad(g.δx),Zygote.dropgrad(g.δy))
# 		eps = εₛ(shapes,tree,g.x[i],g.y[j],g.δx,g.δy)
# 		epsi = inv(eps) # inv( (eps' + eps) / 2) # Hermitian(inv(eps))  # inv(Hermitian(eps)) #   # inv(eps)
#         eibuf[:,:,i,j,kk] = epsi #(epsi' + epsi) / 2
#     end
#     return real(copy(eibuf))
# end

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

function solve_ω²(ms::ModeSolver{ND,T};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
		# ; kwargs...) where T<:Real
		# ;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
		res = lobpcg!(ms.eigs_itr; log,not_zeros=false,maxiter,tol=1.6e-8)
		# res = lobpcg!(ms.eigs_itr; log=true,not_zeros=false,maxiter,tol=1e-12)
		# res = lobpcg(ms.M̂, false, 1; P=ms.P̂, log=true,maxiter,tol=1e-12)
		# M̂ = HelmholtzMap(ms.M̂.k⃗, ms.M̂.ε⁻¹, ms.grid)
		# P̂ = HelmholtzPreconditioner(M̂)
		# eigs_itr = LOBPCGIterator(M̂,false,randn(eltype(M̂),(size(M̂)[1],1)),P̂,nothing)
		# res = lobpcg!(eigs_itr; log=true,not_zeros=false,maxiter,tol=1e-7)
		# # show(res)
		# println("\t\t\t\tEigs (LOBPCG) @ k = $( ms.M̂.k⃗[3] ) :")
		# println("\t\t\t\t\tconverged: $(res.converged)")
		# println("\t\t\t\t\titerations: $(res.iterations)   ($(res.maxiter) max)")
		# println("\t\t\t\t\teigenvalue: $(res.λ[eigind])")
		# println("\t\t\t\t\tresidual_norm: $(res.residual_norms[eigind])   (tol: $(res.tolerance))")
		# return (real(ms.ω²[eigind]), ms.H⃗[:,eigind])
		# return (copy(real(ms.ω²[eigind])), copy(ms.H⃗[:,eigind]))
		return (copy(real(ms.ω²)), copy(ms.H⃗))
		# copyto!(ms.ω²,res.λ)
		# copyto!(ms.H⃗,res.X)
		# return (copy(real(res.λ[eigind])), copy(res.X[:,eigind]))
end

# function _solve_ω²(k,ε⁻¹,gr;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false)
# 		M̂ = HelmholtzMap(k, ε⁻¹, gr)
# 		P̂ = HelmholtzPreconditioner(M̂)
# 		eigs_itr = LOBPCGIterator(M̂,false,randn(eltype(M̂),(size(M̂)[1],1)),P̂,nothing)
# 		res = lobpcg!(eigs_itr; log=true,not_zeros=false,maxiter,tol=1e-12)
# 		# show(res)
# 		println("\t\t\t\tEigs (LOBPCG) @ k = $k :")
# 		println("\t\t\t\t\tconverged: $(res.converged)")
# 		println("\t\t\t\t\titerations: $(res.iterations)   ($(res.maxiter) max)")
# 		println("\t\t\t\t\teigenvalue: $(res.λ[eigind])")
# 		println("\t\t\t\t\tresidual_norm: $(res.residual_norms[eigind])   (tol: $(res.tolerance))")
# 		# return (real(ms.ω²[eigind]), ms.H⃗[:,eigind])
# 		# return (copy(real(ms.ω²[eigind])), copy(ms.H⃗[:,eigind]))
# 		return (copy(real(res.λ[eigind])), copy(res.X[:,eigind]))
# end



function solve_ω²(ms::ModeSolver{ND,T},k::Union{T,SVector{3,T}},ε⁻¹::AbstractArray{SMatrix{3,3,T,9},ND};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
		# nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
	@ignore(update_k!(ms,k))
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms; nev, eigind, maxiter, tol, log)
end

function solve_ω²(ms::ModeSolver{ND,T},k::Union{T,SVector{3,T}},shapes::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
		# nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
	ε⁻¹ = εₛ⁻¹(shapes;ms=dropgrad(ms))
	solve_ω²(ms,k,ε⁻¹; nev, eigind, maxiter, tol, log)
end

function solve_ω²(k::Union{T,SVector{3,T}},shapes::Vector{<:Shape},gr::Grid{ND};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
	ms = @ignore(ModeSolver(k, shapes, gr)) # ; nev, eigind, maxiter, tol, log))
	ε⁻¹ = εₛ⁻¹(shapes;ms=dropgrad(ms))
	solve_ω²(ms,k,ε⁻¹; nev, eigind, maxiter, tol, log)
end

function solve_ω²(ms::ModeSolver{ND,T},k::Union{T,SVector{3,T}};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
	# @ignore(update_k(ms,k)
	update_k!(ms,k)
	solve_ω²(ms; nev, eigind, maxiter, tol, log)
end

function solve_ω²(ms::ModeSolver{ND,T},k::Vector{T}; nev=1,eigind=1,
		maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
	ω² = Buffer(k,length(k))
	H = Buffer(ms.H⃗,length(k),size(ms.M̂)[1])
	@inbounds for kind=1:length(k)
		@inbounds ω²H = solve_ω²(ms,k[kind]; nev, eigind, maxiter, tol, log)
		# @show size(ω²H[1])
		# @show size(ω²H[2])
		@inbounds ω²[kind] = ω²H[1]
		@inbounds H[kind,:] .= ω²H[2]
	end
	return ( copy(ω²), copy(H) )
	# [ ( @ignore(update_k!(ms,kk)); solve_ω²(ms; kwargs...) ) for kk in k ]
end

function solve_ω²(ms::ModeSolver{ND,T},k::Vector{T},ε⁻¹::AbstractArray{SMatrix{3,3,T,9},ND}; nev=1,eigind=1,
		maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
	ε⁻¹ = εₛ⁻¹(shapes;ms=dropgrad(ms))
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms,k; nev, eigind, maxiter, tol, log)
end

function solve_ω²(k::Vector{T},ε⁻¹::AbstractArray{SMatrix{3,3,T,9},ND},gr::Grid{ND}; nev=1,eigind=1,
		maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
	ms = @ignore(ModeSolver(first(k), shapes, gr))  #; nev, eigind, maxiter, tol, log))
	ε⁻¹ = εₛ⁻¹(shapes;ms=dropgrad(ms))
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms,k; nev, eigind, maxiter, tol, log)
end

function solve_ω²(ms::ModeSolver{ND,T},shapes::Vector{<:Shape};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
	ε⁻¹ = εₛ⁻¹(shapes;ms=dropgrad(ms))
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms; nev, eigind, maxiter, tol, log)
end

function solve_ω²(ms::ModeSolver{ND,T},ε⁻¹::AbstractArray{SMatrix{3,3,T,9},ND};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real}
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms; nev, eigind, maxiter, tol, log)
end

# function solve_ω²(k::Union{T,SVector{3,T}},ε⁻¹::AbstractArray{T,5}; Δx::T,Δy::T,Δz::T,
# 		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
# 	ms = @ignore(ModeSolver(k, ε⁻¹; kwargs...))
# 	solve_ω²(ms;kwargs...)
# end



# function solve_ω²(k::Vector{T},ε⁻¹::AbstractArray{T,5}; kwargs...) where T<:Real
# 	ms = @ignore(ModeSolver(k, ε⁻¹; kwargs...))
# 	[ ( @ignore(update_k(ms,kk)); solve_ω²(ms; kwargs...) ) for kk in k ]
# end

# # Legacy code to be removed soon
# function solve_ω²(kz::T,ε⁻¹::Array{Float64,5},ds::MaxwellData;neigs=1,eigind=1,maxiter=3000,tol=1e-8) where T<:Real
# 	# Δk = k - ds.k
# 	ds.k = kz
# 	ds.kpg_mag, ds.mn = calc_kpg(kz,ds.Δx,ds.Δy,ds.Δz,ds.Nx,ds.Ny,ds.Nz)
#     # res = IterativeSolvers.lobpcg(M̂(ε⁻¹,ds),false,neigs;P=P̂(ε⁻¹,ds),maxiter,tol)
#     res = IterativeSolvers.lobpcg(M̂!(ε⁻¹,ds),false,ds.H⃗;P=P̂!(ε⁻¹,ds),maxiter,tol)
#     H =  res.X #[:,eigind]                       # eigenmode wavefn. magnetic fields in transverse pol. basis
#     ds.ω² =  real(res.λ[eigind])                     # eigenmode temporal freq.,  neff = kz / ω, kz = k[3]
# 	ds.H⃗ .= H
# 	ds.ω = ( ds.ω² > 0. ? sqrt(ds.ω²) : 0. )
#     # ds.ω²ₖ = 2 * H_Mₖ_H(Ha,ε⁻¹,kpg_mn,kpg_mag,ds.𝓕,ds.𝓕⁻¹) # = 2ω*ωₖ; ωₖ = ∂ω/∂kz = group velocity = c / ng; c = 1 here
#     return (H, ds.ω^2) #(H, ds.ω²) #, ωₖ
# end

# function solve_ω²(kz,ε⁻¹::Array{Float64,5},g::MaxwellGrid;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
#     solve_ω²(kz,ε⁻¹,make_MD(first(kz),g);neigs,eigind,maxiter,tol)
# end

# function solve_ω²(kz::T,ε⁻¹::Array{T,5},Δx::T,Δy::T,Δz::T;neigs=1,eigind=1,maxiter=3000,tol=1e-8) where T<:Real
#     solve_ω²(kz,ε⁻¹,make_MG(Δx,Δy,Δz,size(ε⁻¹)[end-2:end]...);neigs,eigind,maxiter,tol)
# end

# function solve_ω²(kz,shapes::Vector{<:GeometryPrimitives.Shape},g::MaxwellGrid;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
# 	ds::MaxwellData = make_MD(kz,g)
# 	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
# 	solve_ω²(kz,ε⁻¹,ds;neigs,eigind,maxiter,tol)
# end

# function solve_ω²(kz,shapes::Vector{<:GeometryPrimitives.Shape},Δx,Δy,Δz,Nx,Ny,Nz;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
# 	g::MaxwellGrid = make_MG(Δx,Δy,Δz,Nx,Ny,Nz)  	# MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)
# 	ds::MaxwellData = make_MD(kz,g)
# 	ε⁻¹::Array{Float64,5} = make_εₛ⁻¹(shapes,g)
# 	solve_ω²(kz,ε⁻¹,ds;neigs,eigind,maxiter,tol)
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

function _solve_Δω²(ms::ModeSolver{ND,T},k::Union{T,SVector{3,T}},ωₜ::T;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real} #,ω²_tol=1e-6)
	ω²,H⃗ = solve_ω²(ms,k; nev, eigind, maxiter, tol=1e-12, log)
	Δω² = copy(ω²[eigind]) - ωₜ^2
	ms.∂ω²∂k[eigind] = 2 * H_Mₖ_H(H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n) # = 2ω*ωₖ; ωₖ = ∂ω/∂kz = group velocity = c / ng; c = 1 here
    return Δω² , Δω² / copy(ms.∂ω²∂k[eigind])
end

# function _solve_Δω²(ms::ModeSolver{ND,T},k::Union{T,SVector{3,T}},ωₜ::T,ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where {ND,T<:Real} #,ω²_tol=1e-6)
# 	# ε⁻¹ = εₛ⁻¹(ω,geom;ms=dropgrad(ms))
# 	# ω²,H⃗ = solve_ω²(ms,k,ε⁻¹; nev, eigind, maxiter, tol, log)
# 	# Δω² = ω²[eigind] - ωₜ^2
# 	f(k,ωₜ,ε⁻¹) = solve_ω²(dropgrad(ms),k,ε⁻¹; nev, eigind, maxiter, tol, log)[1] - ωₜ^2
#
# 	ms.∂ω²∂k[eigind] = 2 * H_Mₖ_H(ms.H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n) # = 2ω*ωₖ; ωₖ = ∂ω/∂kz = group velocity = c / ng; c = 1 here
#     return Δω² , Δω² / ms.∂ω²∂k[eigind]
# end

# 	∂ω²∂k, ∂ω²∂k_pb = Zygote.pullback(ms.H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n) do H,ei,mag,m,n
# 		2 * H_Mₖ_H(H,ei,mag,m,n)
# 	end
# 	ms.∂ω²∂k[eigind] = ∂ω²∂k # = 2ω*ωₖ; ωₖ = ∂ω/∂kz = group velocity = c / ng; c = 1 here
#     return Δω² , Δω² / ∂ω²∂k
# end

# function solve_k(ω,ε⁻¹;Δx=6.0,Δy=4.0,Δz=1.0,k_guess=ω*sqrt(1/minimum([minimum(ε⁻¹[a,a,:,:,:]) for a=1:3])),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
function solve_k(ms::ModeSolver{ND,T},ω::T;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real} #
	if iszero(ms.M̂.k⃗[3])
		# println("iszero(ms.M̂.k⃗[3]) cond. in solve_k: ms.M̂.k⃗ = $(ms.M̂.k⃗)")
		ms.M̂.k⃗ = SVector(0., 0., ω*ñₘₐₓ(ms.M̂.ε⁻¹))
	end
    kz = Roots.find_zero(x -> _solve_Δω²(ms,x,ω;nev,eigind,maxiter,tol), ms.M̂.k⃗[3], Roots.Newton()) #; verbose=true) #;rtol=ω²_tol)
	# println("\tkz = $kz")
    return ( kz, copy(ms.H⃗) ) # maybe copy(ds.H⃗) instead?
end

function solve_k(ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_k(ms, ω; nev, eigind, maxiter, tol, log)
end

function solve_k(ms::ModeSolver{ND,T},ω::T,geom::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	ε⁻¹ = εₛ⁻¹(ω,geom;ms=dropgrad(ms)) # make_εₛ⁻¹(shapes,dropgrad(ms))
	solve_k(ms, ω, ε⁻¹; nev, eigind, maxiter, tol, log)
end

function solve_k(ω::T,geom::Vector{<:Shape},gr::Grid{ND};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	ms = @ignore(ModeSolver(kguess(ω,geom), geom, gr))
	ε⁻¹ = εₛ⁻¹(ω,geom;ms=dropgrad(ms))
	solve_k(ms, ω, ε⁻¹; nev, eigind, maxiter, tol, log)
end

function solve_k(ms::ModeSolver{ND,T},ω::Vector{T};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	nω = length(ω)
	k = Buffer(ω,nω)
	H = Buffer(ms.H⃗,nω,size(ms.M̂)[1],nev)
	@inbounds for ωind=1:nω
		@inbounds kH = solve_k(ms,ω[ωind]; nev, eigind, maxiter, tol, log)
		@inbounds k[ωind] = kH[1]
		@inbounds H[ωind,:,:] .= kH[2]
	end
	return ( copy(k), copy(H) )
end

function solve_k(ms::ModeSolver{ND,T},ω::Vector{T},ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	nω = length(ω)
	k = Buffer(ω,nω)
	H = Buffer(ms.H⃗,nω,size(ms.M̂)[1],nev)
	@inbounds for ωind=1:nω
		@inbounds kH = solve_k(ms,ω[ωind]; nev, eigind, maxiter, tol, log)
		@inbounds k[ωind] = kH[1]
		@inbounds H[ωind,:,:] .= kH[2]
	end
	return ( copy(k), copy(H) )
end

function solve_k(ms::ModeSolver{ND,T},ω::Vector{T},shapes::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	nω = length(ω)
	k = Buffer(ω,nω)
	H = Buffer(ms.H⃗,nω,size(ms.M̂)[1],nev)
	@inbounds for ωind=1:nω
		@inbounds kH = solve_k(ms,ω[ωind],εₛ⁻¹(ω[ωind],shapes;ms); nev, eigind, maxiter, tol, log)
		@inbounds k[ωind] = kH[1]
		@inbounds H[ωind,:,:] .= kH[2]
	end
	return ( copy(k), copy(H) )
end

function solve_k(ω::Vector{T},shapes::Vector{<:Shape},gr::Grid{ND};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	# ms = @ignore(ModeSolver(k, shapes, gr))
	ms = @ignore(ModeSolver(kguess(ω,geom), geom, gr))
	solve_k(ms, ω, shapes; nev, eigind, maxiter, tol, log)
end

"""
################################################################################
#																			   #
#						solve_n methods: (ε⁻¹, ω) --> (n, ng)				   #
#																			   #
################################################################################
"""

# function solve_n(ms::ModeSolver{ND,T},ω::T;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real} #
#     k, H⃗ = solve_k(ms,ω;nev,eigind,maxiter,tol,log) #ω²_tol)
# 	# ng = ω / H_Mₖ_H(H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n)
# 	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
# 	# original, non-dispersive ng calculation ("geometric dispersion" only, material dispersion ignored)
# 	ng = ω / H_Mₖ_H(H⃗[:,eigind],ms.M̂.ε⁻¹,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
# 	# dispersive ng calculation
# 	nnginv = inv.(nngₛ(ω,rwg_pe(p_jank);ms))
# 	ng = ω / H_Mₖ_H(ms.H⃗[:,eigind],nnginv,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
#     return ( k/ω, ng )
# end

function solve_n(ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	k, H⃗ = solve_k(ms,ω,ε⁻¹;nev,eigind,maxiter,tol,log) #ω²_tol)
	# ng = ω / H_Mₖ_H(H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n)
	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
	ng = ω / H_Mₖ_H(H⃗[:,eigind],ε⁻¹,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
    return ( k/ω, ng )
end

function solve_n(ms::ModeSolver{ND,T},ω::T,ε⁻¹::AbstractArray{<:SMatrix{3,3},ND},nnginv::AbstractArray{<:SMatrix{3,3},ND};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol,ng_nodisp=true) where {ND,T<:Real}
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	k, H⃗ = solve_k(ms,ω,ε⁻¹;nev,eigind,maxiter,tol,log) #ω²_tol)
	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
	if ng_nodisp
		HMₖH = H_Mₖ_H(H⃗[:,eigind],ε⁻¹,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))	# old, no material disp, TODO remove this after using for comparison
		println("old HMₖH: $(HMₖH)")
		ng = ω / HMₖH
		println("old ng: $ng")
		# ng = ω / H_Mₖ_H(H⃗[:,eigind],ε⁻¹,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))	# old, no material disp, TODO remove this after using for comparison
	else
		HMₖH = H_Mₖ_H(H⃗[:,eigind],nnginv,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
		println("new HMₖH: $(HMₖH)")
		ng = ω / HMₖH
		println("new ng: $ng")
		# ng = ω / H_Mₖ_H(H⃗[:,eigind],nnginv,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗))) # new, material disp. included
	end
    return ( k/ω, ng )
end


function solve_n(ms::ModeSolver{ND,T},ω::T,geom::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	ε⁻¹ = εₛ⁻¹(ω,geom;ms) # make_εₛ⁻¹(ω,shapes,dropgrad(ms))
	nnginv = nngₛ⁻¹(ω,geom;ms)
	# solve_n(ms, ω,ε⁻¹; nev, eigind, maxiter, tol, log)

	solve_n(ms, ω,ε⁻¹,nnginv; nev, eigind, maxiter, tol, log)

	# @ignore(update_ε⁻¹(ms,ε⁻¹))
	# k, H⃗ = solve_k(ms,ω,ε⁻¹;nev,eigind,maxiter,tol,log) #ω²_tol)
	# (mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
	# ### original, non-dispersive ng calculation ("geometric dispersion" only, material dispersion ignored)
	# ng = ω / H_Mₖ_H(H⃗[:,eigind],ms.M̂.ε⁻¹,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
	# ### dispersive ng calculation
	# # nnginv = real(inv.(nngₛ(ω,geom;ms)))
	# # ng = ω / H_Mₖ_H(ms.H⃗[:,eigind],nnginv,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
	# return ( k/ω, ng )
end

function solve_n(ω::T,geom::Vector{<:Shape},gr::Grid{ND};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	ms = @ignore(ModeSolver(kguess(ω,geom), geom, gr))
	solve_n(dropgrad(ms), ω, geom; nev, eigind, maxiter, tol, log)
end

# prev method could also be:
# function solve_n(ω::Real,geom::Vector{<:Shape},gr::Grid{ND,T};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol, k₀=kguess(ω,geom)) where {ND,T<:Real}
# 	ms::ModeSolver{ND,T} = @ignore( ModeSolver(k₀, geom, gr) );
# 	solve_n(ms,ω,geom;nev,eigind,maxiter,tol,log)
# end

function solve_n(ms::ModeSolver{ND,T},ω::Vector{T};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	nω = length(ω)
	n = Buffer(ω,nω)
	ng = Buffer(ω,nω)
	@inbounds for ωind=1:nω
		@inbounds nng = solve_n(ms,ω[ωind]; nev, eigind, maxiter, tol, log)
		@inbounds n[ωind] = nng[1]
		@inbounds ng[ωind] = nng[2]
	end
	return ( copy(n), copy(ng) )
end

function solve_n(ms::ModeSolver{ND,T},ω::Vector{T},ε⁻¹::AbstractArray{<:SMatrix{3,3},ND};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where {ND,T<:Real}
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	nω = length(ω)
	n = Buffer(ω,nω)
	ng = Buffer(ω,nω)
	@inbounds for ωind=1:nω
		@inbounds nng = solve_n(ms,ω[ωind],ε⁻¹; nev, eigind, maxiter, tol, log)
		@inbounds n[ωind] = nng[1]
		@inbounds ng[ωind] .= nng[2]
	end
	return ( copy(n), copy(ng) )
end

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

using Distributed

function _solve_n_serial(ms::ModeSolver{ND,T},ωs::Vector{T},geom::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol,wp=nothing,ng_nodisp=false) where {ND,T<:Real}

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
	ms_copies = @ignore( [ deepcopy(ms) for om in 1:length(ωs) ] )

	nω = length(ωs)
	n_buff = Buffer(ωs,nω)
	ng_buff = Buffer(ωs,nω)
	for ωind=1:nω
		ωinv = inv(ωs[ωind])
		es = vcat(εs(geom,ωinv),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
		eis = inv.(es)
		ε⁻¹_ω = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)
		# ε⁻¹_ω = εₛ⁻¹(ωs[ωind],geom;ms=dropgrad(ms))
		# @ignore(update_ε⁻¹(ms,ε⁻¹_ω))
		k, H⃗ = solve_k(ms, ωs[ωind], ε⁻¹_ω; nev, eigind, maxiter, tol, log) #ω²_tol)
		(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
		if ng_nodisp
		### original, non-dispersive ng calculation ("geometric dispersion" only, material dispersion ignored)
			ng_ω =  ωs[ωind] / H_Mₖ_H(H⃗[:,eigind],ε⁻¹_ω,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
		else
			### dispersive ng calculation
			nngs_ω = vcat( nn̂g.(materials(geom), ωinv) ,[εᵥ,]) # = √.(ε̂) .* nĝ (elementwise product of index and group index tensors) for each material, vacuum permittivity tensor appended
			nngis_ω = inv.(nngs_ω)
			nngi_ω = εₛ⁻¹(nngs_ω,nngis_ω,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)
			# nnginv_ω_sm = real(inv.(εₛ(nngs_ω,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)))  # new spatially smoothed ε⁻¹ tensor array
			ng_ω = ωs[ωind] / H_Mₖ_H(H⃗[:,eigind],nngi_ω,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗))) # new spatially smoothed ε⁻¹ tensor array
		end
		ns[ωind] = k/ωs[ωind]
		ngs[ωind] = ng_ω
	end
	return ( copy(ns), copy(ngs) )
end

# function _solve_n_parallel(ms::ModeSolver{ND,T},ωs::Vector{T},geom::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol,wp=nothing) where {ND,T<:Real}
# 	solve_n_single(ms, ωs, geom; nev, eigind, maxiter, tol, log)
# end

function solve_n(ms::ModeSolver{ND,T},ωs::Vector{T},geom::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol,wp=nothing,ng_nodisp=false) where {ND,T<:Real}
	_solve_n_serial(ms, ωs, geom; nev, eigind, maxiter, tol, log, ng_nodisp)
end

function solve_n(ωs::Vector{T},geom::Vector{<:Shape},gr::Grid;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol,wp=nothing,ng_nodisp=false) where {ND,T<:Real}
	ms = @ignore(ModeSolver(kguess(first(ωs),geom), geom, gr))
	_solve_n_serial(ms,ωs, geom; nev, eigind, maxiter, tol, log, ng_nodisp)
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

"""
################################################################################
#																			   #
#					solve_nω methods: (ε⁻¹, k) --> (n, ng)					   #
#						(mostly for debugging gradients)					   #
#																			   #
################################################################################
"""

function solve_nω(ms::ModeSolver{T},k,ε⁻¹::AbstractArray{T,5};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
    ω², H⃗ = solve_ω²(ms,k,ε⁻¹;nev,eigind,maxiter,tol,log)
	ω = sqrt(ω²)
	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
	ng = ω / H_Mₖ_H(H⃗,ε⁻¹,real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗)))
	return ( k/ω, ng )
end

# function solve_nω(ms::ModeSolver{T},k,shapes::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
# 	g::MaxwellGrid = make_MG(ms.M̂.Δx,ms.M̂.Δy,ms.M̂.Δz,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)
# 	# ε⁻¹ = HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},Float64,5,5,Array{Float64,5}}( make_εₛ⁻¹(shapes,g) )
# 	ε⁻¹ = make_εₛ⁻¹(shapes,dropgrad(ms))
# 	ω², H⃗ = solve_ω²(ms,k,ε⁻¹;nev,eigind,maxiter,tol,log)
# 	ω = sqrt(ω²)
# 	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
# 	ng = ω / H_Mₖ_H(H⃗,ε⁻¹,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
# 	return ( k/ω, ng )
# end

# function solve_nω(kz::T,shapes::Vector{<:Shape},Δx,Δy,Δz,Nx,Ny,Nz;neigs=1,eigind=1,maxiter=3000,tol=1e-8) where T<:Real
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
# 	ng = ω / H_Mₖ_H(Ha,ε⁻¹,mag,mn[:,1,:,:,:],mn[:,2,:,:,:])
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
function uplot(ms::ModeSolver;xlim=[0.5,1.8])
	ls_mats = uplot(ms.materials;xlim)
end
