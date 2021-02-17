using  IterativeSolvers, Roots # , KrylovKit
export solve_ω, _solve_Δω², solve_k, solve_n, ng, k_guess, solve_nω, solve_ω², make_MG, make_MD, replan_ffts!
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
make_MG(Δx,Δy,Δz,Nx,Ny,Nz) = (g = Zygote.@ignore (MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz)); g)::MaxwellGrid
make_MD(k,g::MaxwellGrid) = (ds = Zygote.@ignore (MaxwellData(k,g)); ds)::MaxwellData

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

function make_εₛ⁻¹(shapes::Vector{<:Shape},g::MaxwellGrid)::Array{T,5} where T<:Real
    tree = make_KDTree(shapes)
    eibuf = Zygote.Buffer(Array{T}(undef),3,3,g.Nx,g.Ny,1)
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

function solve_ω²(ms::ModeSolver;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
		# ; kwargs...) where T<:Real
		# ;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
		res = lobpcg!(ms.eigs_itr; log,not_zeros=false,maxiter,tol)
		return (real(ms.ω²[eigind]), ms.H⃗[:,eigind])
end

function solve_ω²(ms::ModeSolver{T},k::Union{T,SVector{3,T}},ε⁻¹::AbstractArray{T,5};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
		# nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
	@ignore(update_k!(ms,k))
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms; nev, eigind, maxiter, tol, log)
end

function solve_ω²(ms::ModeSolver{T},k::Union{T,SVector{3,T}},shapes::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
		# nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
	ε⁻¹ = make_εₛ⁻¹(shapes,dropgrad(ms))
	@ignore(update_k!(ms,k))
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms; nev, eigind, maxiter, tol, log)
end

function solve_ω²(ms::ModeSolver{T},k::Union{T,SVector{3,T}};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
	# @ignore(update_k(ms,k)
	update_k!(ms,k)
	solve_ω²(ms; nev, eigind, maxiter, tol, log)
end

function solve_ω²(ms::ModeSolver{T},k::Vector{T}; nev=1,eigind=1,
		maxiter=3000,tol=1e-8,log=false) where T<:Real
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

function solve_ω²(ms::ModeSolver{T},shapes::Vector{<:Shape};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
	ε⁻¹ = make_εₛ⁻¹(shapes,dropgrad(ms))
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms; nev, eigind, maxiter, tol, log)
end

function solve_ω²(ms::ModeSolver{T},ε⁻¹::AbstractArray{T,5};
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_ω²(ms; nev, eigind, maxiter, tol, log)
end

function solve_ω²(k::Union{T,SVector{3,T}},ε⁻¹::AbstractArray{T,5}; Δx::T,Δy::T,Δz::T,
		nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
	ms = @ignore(ModeSolver(k, ε⁻¹; kwargs...))
	solve_ω²(ms;kwargs...)
end

function solve_ω²(k::Vector{T},ε⁻¹::AbstractArray{T,5}; kwargs...) where T<:Real
	ms = @ignore(ModeSolver(k, ε⁻¹; kwargs...))
	[ ( @ignore(update_k(ms,kk)); solve_ω²(ms; kwargs...) ) for kk in k ]
end

# Legacy code to be removed soon
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

function solve_ω²(kz,ε⁻¹::Array{Float64,5},g::MaxwellGrid;neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    solve_ω²(kz,ε⁻¹,make_MD(first(kz),g);neigs,eigind,maxiter,tol)
end

function solve_ω²(kz::T,ε⁻¹::Array{T,5},Δx::T,Δy::T,Δz::T;neigs=1,eigind=1,maxiter=3000,tol=1e-8) where T<:Real
    solve_ω²(kz,ε⁻¹,make_MG(Δx,Δy,Δz,size(ε⁻¹)[end-2:end]...);neigs,eigind,maxiter,tol)
end

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

"""
################################################################################
#																			   #
#						solve_ω methods: (ε⁻¹, k) --> (H, ω)				   #
#																			   #
################################################################################
"""


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

function _solve_Δω²(ms::ModeSolver{T},k::Union{T,SVector{3,T}},ωₜ::T;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real #,ω²_tol=1e-6)
	ω²,H⃗ = solve_ω²(ms,k; nev, eigind, maxiter, tol, log)
	Δω² = ω²[eigind] - ωₜ^2
	ms.∂ω²∂k[eigind] = 2 * H_Mₖ_H(ms.H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n) # = 2ω*ωₖ; ωₖ = ∂ω/∂kz = group velocity = c / ng; c = 1 here
    return Δω² , Δω² / ms.∂ω²∂k[eigind]
end
# 	∂ω²∂k, ∂ω²∂k_pb = Zygote.pullback(ms.H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n) do H,ei,mag,m,n
# 		2 * H_Mₖ_H(H,ei,mag,m,n)
# 	end
# 	ms.∂ω²∂k[eigind] = ∂ω²∂k # = 2ω*ωₖ; ωₖ = ∂ω/∂kz = group velocity = c / ng; c = 1 here
#     return Δω² , Δω² / ∂ω²∂k
# end

# function solve_k(ω,ε⁻¹;Δx=6.0,Δy=4.0,Δz=1.0,k_guess=ω*sqrt(1/minimum([minimum(ε⁻¹[a,a,:,:,:]) for a=1:3])),neigs=1,eigind=1,maxiter=3000,tol=1e-8)
function solve_k(ms::ModeSolver{T},ω::T;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real #
	if ms.M̂.k⃗[3]==0.
		ms.M̂.k⃗ = SVector(0., 0., ω*sqrt(1/minimum([minimum(ε⁻¹[a,a,:,:,:]) for a=1:3])))
	end
    kz = Roots.find_zero(x -> _solve_Δω²(ms,x,ω;nev,eigind,maxiter,tol), ms.M̂.k⃗[3], Roots.Newton()) #;rtol=ω²_tol)
    return ( kz, ms.H⃗ ) # maybe copy(ds.H⃗) instead?
end

function solve_k(ms::ModeSolver{T},ω::T,ε⁻¹::AbstractArray{T,5};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_k(ms, ω; nev, eigind, maxiter, tol, log)
end

function solve_k(ms::ModeSolver{T},ω::T,shapes::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real
	ε⁻¹ = make_εₛ⁻¹(shapes,dropgrad(ms))
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	solve_k(ms, ω; nev, eigind, maxiter, tol, log)
end

function solve_k(ms::ModeSolver{T},ω::Vector{T};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real
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

function solve_k(ms::ModeSolver{T},ω::Vector{T},ε⁻¹::AbstractArray{T,5};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real
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

function solve_k(ms::ModeSolver{T},ω::Vector{T},shapes::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real
	ε⁻¹ = make_εₛ⁻¹(shapes,dropgrad(ms))
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



"""
################################################################################
#																			   #
#						solve_n methods: (ε⁻¹, ω) --> (n, ng)				   #
#																			   #
################################################################################
"""

function solve_n(ms::ModeSolver{T},ω::T;nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real #
    k, H⃗ = solve_k(ms,ω;nev,eigind,maxiter,tol,log) #ω²_tol)
	# ng = ω / H_Mₖ_H(H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n)
	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
	ng = ω / H_Mₖ_H(H⃗[:,eigind],ms.M̂.ε⁻¹,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
    return ( k/ω, ng )
end

function solve_n(ms::ModeSolver{T},ω::T,ε⁻¹::AbstractArray{T,5};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real
	@ignore(update_ε⁻¹(ms,ε⁻¹))
	k, H⃗ = solve_k(ms,ω,ε⁻¹;nev,eigind,maxiter,tol,log) #ω²_tol)
	# ng = ω / H_Mₖ_H(H⃗[:,eigind],ms.M̂.ε⁻¹,ms.M̂.mag,ms.M̂.m,ms.M̂.n)
	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
	ng = ω / H_Mₖ_H(H⃗[:,eigind],ε⁻¹,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
    return ( k/ω, ng )
end

function solve_n(ms::ModeSolver{T},ω::T,shapes::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real
	ε⁻¹ = make_εₛ⁻¹(ω,shapes,dropgrad(ms))
	solve_n(ms, ω,ε⁻¹; nev, eigind, maxiter, tol, log)
end

function solve_n(ms::ModeSolver{T},ω::Vector{T};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real
	nω = length(ω)
	n = Buffer(ω,nω)
	ng = Buffer(ω,nω)
	@inbounds for ωind=1:nω
		@inbounds nng = solve_n(ms,ω[ωind]; nev, eigind, maxiter, tol, log)
		@inbounds n[ωind] = nng[1]
		@inbounds ng[ωind] .= nng[2]
	end
	return ( copy(n), copy(ng) )
end

function solve_n(ms::ModeSolver{T},ω::Vector{T},ε⁻¹::AbstractArray{T,5};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real
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

function replan_ffts!(ms::ModeSolver{T}) where T<:Real
	ms.M̂.𝓕! = plan_fft!(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
	ms.M̂.𝓕⁻¹! = plan_bfft!(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
	ms.M̂.𝓕 = plan_fft(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
	ms.M̂.𝓕⁻¹ = plan_bfft(randn(Complex{T}, (3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)),(2:4),flags=FFTW.PATIENT);
end

using Distributed

function solve_n(ω::T,shapes::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol) where T<:Real
	ms = @ignore( ModeSolver(1.45, shapes, 6., 4., 1., 128, 128, 1) );
	solve_n(ms,ω,shapes;nev,eigind,maxiter,tol,log)
end

function solve_n(ms::ModeSolver{T},ωs::Vector{T},shapes::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false,ω²_tol=tol,wp=nothing) where T<:Real
	# ε⁻¹ = make_εₛ⁻¹(shapes,dropgrad(ms))
	# @ignore(update_ε⁻¹(ms,ε⁻¹))
	# ms_copies = [ deepcopy(ms) for om in 1:length(ωs) ]
	# m = @ignore( ModeSolver(1.45, shapes, 6., 4., 1., 128, 128, 1) )

	# nng = pmap(x->solve_n(x,shapes), ωs)
	# n = [res[1] for res in nng]
	# ng = [res[2] for res in nng]
	update_corner_sinds!(ms,shapes)
	nω = length(ωs)
	n_buff = Buffer(ωs,nω)
	ng_buff = Buffer(ωs,nω)
	for ωind=1:nω
		nng = solve_n(ms,ωs[ωind],shapes; nev, eigind, maxiter, tol, log)
		# nng = solve_n(ms_copies[ωind],ω[ωind],ε⁻¹; nev, eigind, maxiter, tol, log)
		n_buff[ωind] = nng[1]
		ng_buff[ωind] = nng[2]
	end
	return ( copy(n_buff), copy(ng_buff) )

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
	# return n, ng
end
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

function solve_nω(ms::ModeSolver{T},k,shapes::Vector{<:Shape};nev=1,eigind=1,maxiter=3000,tol=1e-8,log=false) where T<:Real
	g::MaxwellGrid = make_MG(ms.M̂.Δx,ms.M̂.Δy,ms.M̂.Δz,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)
	# ε⁻¹ = HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},Float64,5,5,Array{Float64,5}}( make_εₛ⁻¹(shapes,g) )
	ε⁻¹ = make_εₛ⁻¹(shapes,dropgrad(ms))
	ω², H⃗ = solve_ω²(ms,k,ε⁻¹;nev,eigind,maxiter,tol,log)
	ω = sqrt(ω²)
	(mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
	ng = ω / H_Mₖ_H(H⃗,ε⁻¹,real(mag),real(reinterpret(reshape,T,m⃗)),real(reinterpret(reshape,T,n⃗)))
	return ( k/ω, ng )
end

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
	ng = ω / H_Mₖ_H(Ha,ε⁻¹,mag,mn[:,1,:,:,:],mn[:,2,:,:,:])
	# ng = ω / real( dot(H, -vec( kx_c2t( ifft( ε⁻¹_dot( fft( zx_t2c(Ha,mn), (2:4) ), ε⁻¹), (2:4)),mn,mag) ) ) )
	# ng = -ω / real( dot(Ha, kx_c2t( ifft( ε⁻¹_dot( fft( zx_t2c(Ha,Zygote.@showgrad(mn)), (2:4) ), ε⁻¹), (2:4)), Zygote.@showgrad(mn),Zygote.@showgrad(mag)) ) )
	( kz/ω, ng )
end
