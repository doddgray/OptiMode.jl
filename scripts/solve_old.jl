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
