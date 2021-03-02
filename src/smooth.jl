export εₛ, εₛ⁻¹,  corner_sinds, corner_sinds!, proc_sinds, proc_sinds!, avg_param, S_rvol, matinds, _εₛ⁻¹_init, _εₛ_init
export make_εₛ⁻¹, make_εₛ⁻¹_fwd, make_KDTree # legacy junk to remove or update
export εₘₐₓ, ñₘₐₓ, nₘₐₓ # utility functions for automatic good guesses, move to geometry or solve?

function τ_trans(ε::AbstractMatrix{T}) where T<:Real
    return @inbounds SMatrix{3,3,T,9}(
        -1/ε[1,1],      ε[2,1]/ε[1,1],                  ε[3,1]/ε[1,1],
        ε[1,2]/ε[1,1],  ε[2,2] - ε[2,1]*ε[1,2]/ε[1,1],  ε[3,2] - ε[3,1]*ε[1,2]/ε[1,1],
        ε[1,3]/ε[1,1],  ε[2,3] - ε[2,1]*ε[1,3]/ε[1,1],  ε[3,3] - ε[3,1]*ε[1,3]/ε[1,1]
    )
end

function τ⁻¹_trans(τ::AbstractMatrix{T}) where T<:Real
    return @inbounds SMatrix{3,3,T,9}(
        -1/τ[1,1],          -τ[2,1]/τ[1,1],                 -τ[3,1]/τ[1,1],
        -τ[1,2]/τ[1,1],     τ[2,2] - τ[2,1]*τ[1,2]/τ[1,1],  τ[3,2] - τ[3,1]*τ[1,2]/τ[1,1],
        -τ[1,3]/τ[1,1],     τ[2,3] - τ[2,1]*τ[1,3]/τ[1,1],  τ[3,3]- τ[3,1]*τ[1,3]/τ[1,1]
    )
end

function avg_param(ε_fg, ε_bg, S, rvol1)
    τ1 = τ_trans(transpose(S) * ε_fg * S)  # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans(transpose(S) * ε_bg * S)  # express param2 in S coordinates, and apply τ transform
    τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)   # volume-weighted average
    return SMatrix{3,3}(S * τ⁻¹_trans(τavg) * transpose(S))  # apply τ⁻¹ and transform back to global coordinates
end


function corner_sinds(shapes::Vector{S},xyz,xyzc::Array{T}) where {S<:GeometryPrimitives.Shape{2},T<:SVector{N}} where N
	ps = pairs(shapes)
	lsp1 = length(shapes) + 1
	map(xyzc) do p
		let ps=ps, lsp1=lsp1
			for (i, a) in ps #pairs(s)
				in(p::T,a::S)::Bool && return i
			end
			return lsp1
		end
	end
end

function corner_sinds!(corner_sinds,shapes::Vector{S},xyz,xyzc::Array{T}) where {S<:GeometryPrimitives.Shape{2},T<:SVector{N}} where N
	ps = pairs(shapes)
	lsp1 = length(shapes) + 1
	map!(corner_sinds,xyzc) do p
		let ps=ps, lsp1=lsp1
			for (i, a) in ps #pairs(s)
				in(p::T,a::S)::Bool && return i
			end
			return lsp1
		end
	end
end

function proc_sinds(corner_sinds::AbstractArray{Int,2})
	unq = [0,0]
	sinds_proc = fill((0,0,0,0),size(corner_sinds).-1) #zeros(eltype(first(corner_sinds)),size(corner_sinds).-1)
	@inbounds for I ∈ CartesianIndices(sinds_proc)
	 	unq = [		corner_sinds[I],
								corner_sinds[I+CartesianIndex(1,0)],
								corner_sinds[I+CartesianIndex(0,1)],
								corner_sinds[I+CartesianIndex(1,1)]
			  				]
		unique!( unq )
		sinds_proc[I] = isone(lastindex(unq)) ? (unq[1],0,0,0) :
			( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0) ) :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0)],
					corner_sinds[I+CartesianIndex(0,1)],
					corner_sinds[I+CartesianIndex(1,1)]
				)
		)
	end
	return sinds_proc
end

function proc_sinds!(sinds_proc::AbstractArray{T,2},corner_sinds::AbstractArray{Int,2}) where T
	unq = [0,0]
	@inbounds for I ∈ CartesianIndices(sinds_proc)
	 	unq = [		corner_sinds[I],
								corner_sinds[I+CartesianIndex(1,0)],
								corner_sinds[I+CartesianIndex(0,1)],
								corner_sinds[I+CartesianIndex(1,1)]
			  				]
		unique!( unq )
		sinds_proc[I] = isone(lastindex(unq)) ? (unq[1],0,0,0) :
			( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0) ) :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0)],
					corner_sinds[I+CartesianIndex(0,1)],
					corner_sinds[I+CartesianIndex(1,1)]
				)
		)
	end
end


function proc_sinds(corner_sinds::AbstractArray{Int,3})
	unq = [0,0]
	sinds_proc = fill((0,0,0,0,0,0,0,0),size(corner_sinds).-1) #zeros(eltype(first(corner_sinds)),size(corner_sinds).-1)
	@inbounds for I ∈ CartesianIndices(sinds_proc)
	 	unq = [		corner_sinds[I],
								corner_sinds[I+CartesianIndex(1,0,0)],
								corner_sinds[I+CartesianIndex(0,1,0)],
								corner_sinds[I+CartesianIndex(1,1,0)]
			  				]
		unique!( unq )
		sinds_proc[I] = isone(lastindex(unq)) ? (unq[1],0,0,0,0,0,0,0) :
			( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0,0,0,0,0) ) :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0,0)],
					corner_sinds[I+CartesianIndex(0,1,0)],
					corner_sinds[I+CartesianIndex(1,1,0)],
					corner_sinds[I+CartesianIndex(0,0,1)],
					corner_sinds[I+CartesianIndex(1,0,1)],
					corner_sinds[I+CartesianIndex(0,1,1)],
					corner_sinds[I+CartesianIndex(1,1,1)]
				)
		)
	end
	return sinds_proc
end

function proc_sinds!(sinds_proc::AbstractArray{T,3},corner_sinds::AbstractArray{Int,3}) where T
	unq = [0,0]
	@inbounds for I ∈ CartesianIndices(sinds_proc)
	 	unq = [		corner_sinds[I],
								corner_sinds[I+CartesianIndex(1,0,0)],
								corner_sinds[I+CartesianIndex(0,1,0)],
								corner_sinds[I+CartesianIndex(1,1,0)]
			  				]
		unique!( unq )
		sinds_proc[I] = isone(lastindex(unq)) ? (unq[1],0,0,0,0,0,0,0) :
			( lastindex(unq)===2 ?  ( xtrm=extrema(unq); (xtrm[1],xtrm[2],0,0,0,0,0,0) ) :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0,0)],
					corner_sinds[I+CartesianIndex(0,1,0)],
					corner_sinds[I+CartesianIndex(1,1,0)],
					corner_sinds[I+CartesianIndex(0,0,1)],
					corner_sinds[I+CartesianIndex(1,0,1)],
					corner_sinds[I+CartesianIndex(0,1,1)],
					corner_sinds[I+CartesianIndex(1,1,1)]
				)
		)
	end
end

# matinds(geom::Geometry) = vcat(map(s->findfirst(m->isequal(s.data,m), materials(geom.shapes)),geom.shapes),length(geom.shapes)+1)
matinds(geom::Geometry) = vcat(map(s->findfirst(m->isequal(ε(Material(s.data)),ε(m)), materials(geom)),geom.shapes),length(geom.shapes)+1)
matinds(geom::Vector{<:Shape}) = vcat(map(s->findfirst(m->isequal(ε(Material(s.data)),ε(m)), materials(geom)),geom),length(geom)+1)
matinds(shapes,mats) = vcat(map(s->findfirst(m->isequal(ε(Material(s.data)),ε(m)),mats),shapes),length(shapes)+1)

_get_ε(shapes,ind) = ind>lastindex(shapes) ? SMatrix{3,3}(1.,0.,0.,0.,1.,0.,0.,0.,1.) : shapes[ind].data
_get_ε(εs,ind,matinds) = ind>lastindex(shapes) ? SMatrix{3,3}(1.,0.,0.,0.,1.,0.,0.,0.,1.) : εs[matinds[ind]]
_V3(v) = isequal(length(v),3) ? v : vcat(v,zeros(3-length(v)))

function n_rvol(shape,xyz,vxl_min,vxl_max)
	r₀,n⃗ = surfpt_nearby(xyz, shape)
	rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
	return _V3(n⃗),rvol
end

function normcart(n0::AbstractVector{T}) where T<:Real #::SMatrix{3,3,T,9} where T<:Real
	# Create `S`, a local Cartesian coordinate system from a surface-normal
	# 3-vector n0 (pointing outward) from shape
	n = n0 / norm(n0)
	# Pick `h` to be a vector that is not along n.
	h = any(iszero.(n)) ? n × normalize(iszero.(n)) :  n × SVector(1., 0. , 0.)
	v = n × h
	S = SMatrix{3,3,T,9}([n h v])  # unitary
end

function _S_rvol(sinds_proc,xyz,vxl_min,vxl_max,shapes)
	if iszero(sinds_proc[2])
		return (SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.), 0.)
	else
		r₀,n⃗ = surfpt_nearby(xyz, shapes[sinds_proc[1]])
		rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
		return normcart(_V3(n⃗)), rvol
	end
end

function S_rvol(sinds_proc,xyz,vxl_min,vxl_max,shapes)
	f(sp,x,vn,vp) = let s=shapes
		_S_rvol(sp,x,vn,vp,s)
	end
	map(f,sinds_proc,xyz,vxl_min,vxl_max)
end

function vxl_min(x⃗c::AbstractArray{T,2}) where T
	@view x⃗c[1:max((end-1),1),1:max((end-1),1)]
end

function vxl_min(x⃗c::AbstractArray{T,3}) where T
	@view x⃗c[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
end

function vxl_max(x⃗c::AbstractArray{T,2}) where T
	@view x⃗c[min(2,end):end,min(2,end):end]
end

function vxl_max(x⃗c::AbstractArray{T,3}) where T
	@view x⃗c[min(2,end):end,min(2,end):end,min(2,end):end]
end

function S_rvol(geom;ms::ModeSolver)
	Zygote.@ignore( ms.geom = geom )	# update ms.geom
	xyz = Zygote.@ignore(x⃗(ms.grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
	xyzc = Zygote.@ignore(x⃗c(ms.grid))
	Zygote.@ignore(corner_sinds!(ms.corner_sinds,geom,xyz,xyzc))
	Zygote.@ignore(proc_sinds!(ms.sinds_proc,ms.corner_sinds))
	f(sp,x,vn,vp) = let g=geom
		_S_rvol(sp,x,vn,vp,g)
	end
	map(f,ms.sinds_proc,xyz,vxl_min(xyzc),vxl_max(xyzc))
end

function _εₛ(εs,sinds_proc,matinds,Srvol)
	iszero(sinds_proc[2]) && return εs[matinds[sinds_proc[1]]]
	iszero(sinds_proc[3]) && return avg_param(	εs[matinds[sinds_proc[1]]],
												εs[matinds[sinds_proc[2]]],
												Srvol[1],
												Srvol[2]
												)
	return mapreduce(i->εs[matinds[sinds_proc[i]]],+,sinds_proc) / 8
end

function εₛ(εs,sinds_proc,matinds,Srvol)
	f(sp,srv) = let es=εs, mi=matinds
		_εₛ(es,sp,mi,srv)
	end
	map(f,sinds_proc,Srvol)
end

function εₛ(lm::Real,geom::Vector{S},gr::Grid) where S<:GeometryPrimitives.Shape
	xyz = Zygote.@ignore(x⃗(gr))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
	xyzc = Zygote.@ignore(x⃗c(gr))
	sinds = Zygote.@ignore(corner_sinds(geom,xyz,xyzc))  	# shape indices at pixel/voxel corners,
	sinds_proc = Zygote.@ignore(proc_sinds(sinds))  		# processed corner shape index lists for each pixel/voxel, should efficiently indicate whether averaging is needed and which ε⁻¹ to use otherwise
	mats = Zygote.@ignore(materials(geom))
	minds = Zygote.@ignore(matinds(geom,mats))
	vxl_min = Zygote.@ignore( @view xyzc[1:max((end-1),1),1:max((end-1),1)] )
	vxl_max = Zygote.@ignore( @view xyzc[min(2,end):end,min(2,end):end] )
	Srvol = S_rvol(sinds_proc,xyz,vxl_min,vxl_max,geom)
	εs = vcat([mm.fε.(lm) for mm in mats],[εᵥ,])
	εₛ(εs,sinds_proc,minds,Srvol)
end

function _εₛ_init(lm::Real,geom::Vector{S},gr::Grid) where S<:GeometryPrimitives.Shape
	xyz = Zygote.@ignore(x⃗(gr))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
	xyzc = Zygote.@ignore(x⃗c(gr))
	sinds = Zygote.@ignore(corner_sinds(geom,xyz,xyzc))  	# shape indices at pixel/voxel corners,
	sinds_proc = Zygote.@ignore(proc_sinds(sinds))  		# processed corner shape index lists for each pixel/voxel, should efficiently indicate whether averaging is needed and which ε⁻¹ to use otherwise
	mats = Zygote.@ignore(materials(geom))
	minds = Zygote.@ignore(matinds(geom,mats))
	vxl_min = Zygote.@ignore( @view xyzc[1:max((end-1),1),1:max((end-1),1)] )
	vxl_max = Zygote.@ignore( @view xyzc[min(2,end):end,min(2,end):end] )
	Srvol = S_rvol(sinds_proc,xyz,vxl_min,vxl_max,geom)
	εs = vcat([mm.fε.(lm) for mm in mats],[εᵥ,])
	εsm = εₛ(εs,sinds_proc,minds,Srvol)
	return (sinds,sinds_proc,Srvol,mats,minds,εsm)
end

function _εₛ⁻¹(εs,ε⁻¹s,sinds_proc,matinds,Srvol)
	iszero(sinds_proc[2]) && return ε⁻¹s[matinds[sinds_proc[1]]]
	iszero(sinds_proc[3]) && return inv(avg_param(	εs[matinds[sinds_proc[1]]],
												εs[matinds[sinds_proc[2]]],
												Srvol[1],
												Srvol[2]
												))
	return inv(mapreduce(i->εs[matinds[sinds_proc[i]]],+,sinds_proc)) * 8
end

function εₛ⁻¹(εs,ε⁻¹s,sinds_proc,matinds,Srvol)
	f(sp,srv) = let es=εs, eis=ε⁻¹s, mi=matinds
		_εₛ⁻¹(es,eis,sp,mi,srv)
		# ei_nonHerm = _εₛ(es,sp,mi,srv)
		# inv( (ei_nonHerm' + ei_nonHerm) / 2 )
	end
	map(f,sinds_proc,Srvol)
end

function εₛ⁻¹(lm::Real,geom::Vector{S},gr::Grid) where S<:GeometryPrimitives.Shape
	xyz = Zygote.@ignore(x⃗(gr))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
	xyzc = Zygote.@ignore(x⃗c(gr))
	sinds = Zygote.@ignore(corner_sinds(geom,xyz,xyzc))  	# shape indices at pixel/voxel corners,
	sinds_proc = Zygote.@ignore(proc_sinds(sinds))  		# processed corner shape index lists for each pixel/voxel, should efficiently indicate whether averaging is needed and which ε⁻¹ to use otherwise
	mats = Zygote.@ignore(materials(geom))
	minds = Zygote.@ignore(matinds(geom,mats))
	vxl_min = Zygote.@ignore( @view xyzc[1:max((end-1),1),1:max((end-1),1)] )
	vxl_max = Zygote.@ignore( @view xyzc[min(2,end):end,min(2,end):end] )
	Srvol = S_rvol(sinds_proc,xyz,vxl_min,vxl_max,geom)
	εs = vcat([mm.fε.(lm) for mm in mats],[εᵥ,])
	ε⁻¹s = inv.(εs)
	εₛ⁻¹(εs,ε⁻¹s,sinds_proc,minds,Srvol)
end

function εₛ⁻¹(ω,Srvol::AbstractArray{Tuple{SMatrix{3,3,T,9},T}};ms::ModeSolver) where T
	es = vcat(εs(ms.geom,( 1. / ω )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
	ei_new = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
end

function εₛ⁻¹(ω,geom::AbstractVector{<:Shape};ms::ModeSolver)
	Srvol = S_rvol(geom;ms)
	es = vcat(εs(geom,( 1. / ω )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
	ei_new = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
end

function εₛ⁻¹(geom::AbstractVector{<:Shape};ms::ModeSolver)
	om_prev = Zygote.@ignore(sqrt(real(ms.ω²[1])))
	Srvol = S_rvol(geom;ms)
	es = vcat(εs(geom,( 1. / om_prev )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
	ei_new = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
end


function _εₛ⁻¹_init(lm::Real,geom::Vector{S},gr::Grid) where S<:GeometryPrimitives.Shape
	xyz = Zygote.@ignore(x⃗(gr))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
	xyzc = Zygote.@ignore(x⃗c(gr))
	sinds = Zygote.@ignore(corner_sinds(geom,xyz,xyzc))  	# shape indices at pixel/voxel corners,
	sinds_proc = Zygote.@ignore(proc_sinds(sinds))  		# processed corner shape index lists for each pixel/voxel, should efficiently indicate whether averaging is needed and which ε⁻¹ to use otherwise
	mats = Zygote.@ignore(materials(geom))
	minds = Zygote.@ignore(matinds(geom,mats))
	vxl_min = Zygote.@ignore( @view xyzc[1:max((end-1),1),1:max((end-1),1)] )
	vxl_max = Zygote.@ignore( @view xyzc[min(2,end):end,min(2,end):end] )
	Srvol = S_rvol(sinds_proc,xyz,vxl_min,vxl_max,geom)
	εs = vcat([mm.fε.(lm) for mm in mats],[εᵥ,])
	ε⁻¹s = inv.(εs)
	εism = εₛ⁻¹(εs,ε⁻¹s,sinds_proc,minds,Srvol)
	return (sinds,sinds_proc,Srvol,mats,minds,εism)
end

##### Utilities for good guesses, should move elsewhere (solve?)
function εₘₐₓ(shapes::AbstractVector{<:GeometryPrimitives.Shape})
    maximum(vec([shapes[i].data[j,j] for j=1:3,i=1:size(shapes)[1]]))
end
ñₘₐₓ(ε⁻¹)::Float64 = √(maximum(3 ./ tr.(ε⁻¹)))
nₘₐₓ(ε)::Float64 = √(maximum(reinterpret(Float64,ε)))


##### Legacy code, please remove soon

function εₛ(shapes::AbstractVector{<:GeometryPrimitives.Shape{2}},x::Real,y::Real;tree::KDTree,δx::Real,δy::Real,npix_sm::Int=1)
    s1 = @ignore(findfirst(SVector(x+δx/2.,y+δy/2.),tree))
    s2 = @ignore(findfirst(SVector(x+δx/2.,y-δy/2.),tree))
    s3 = @ignore(findfirst(SVector(x-δx/2.,y-δy/2.),tree))
    s4 = @ignore(findfirst(SVector(x-δx/2.,y+δy/2.),tree))

    ε1 = isnothing(s1) ? εᵥ : s1.data
    ε2 = isnothing(s2) ? εᵥ : s2.data
    ε3 = isnothing(s3) ? εᵥ : s3.data
    ε4 = isnothing(s4) ? εᵥ : s4.data

    if (ε1==ε2==ε3==ε4)
        return ε1
    else
        sinds = @ignore ( [ isnothing(ss) ? length(shapes)+1 : findfirst(isequal(ss),shapes) for ss in [s1,s2,s3,s4]] )
        n_unique = @ignore( length(unique(sinds)) )
        if n_unique==2
            s_fg = @ignore(shapes[minimum(dropgrad(sinds))])
            r₀,nout = surfpt_nearby([x; y], s_fg)
            rvol = volfrac((SVector{2}(x-δx/2.,y-δy/2.), SVector{2}(x+δx/2.,y+δy/2.)),nout,r₀)
            sind_bg = @ignore(maximum(dropgrad(sinds))) #max(sinds...)
            ε_bg = sind_bg > length(shapes) ? εᵥ : shapes[sind_bg].data
            return avg_param(
                    s_fg.data,
                    ε_bg,
                    [nout[1];nout[2];0],
                    rvol,)
        else
            return +(ε1,ε2,ε3,ε4)/4.
        end
    end
end

make_KDTree(shapes::AbstractVector{<:Shape}) = (tree = @ignore (KDTree(shapes)); tree)::KDTree

function make_εₛ⁻¹(shapes::Vector{<:Shape{N}};Δx::Real,Δy::Real,Δz::Real,Nx::Int,Ny::Int,Nz::Int,
	 	δx=Δx/Nx, δy=Δy/Ny, δz=Δz/Nz, x=( ( δx .* (0:(Nx-1))) .- Δx/2. ),
		y=( ( δy .* (0:(Ny-1))) .- Δy/2. ), z=( ( δz .* (0:(Nz-1))) .- Δz/2. ) ) where N
    tree = make_KDTree(shapes)
    eibuf = Buffer(bounds(shapes[1])[1],3,3,Nx,Ny,Nz)
	# eibuf = Buffer(bounds(shapes[1])[1],3,3,Nx,Ny,Nz)
    for i=1:Nx,j=1:Ny,kk=1:Nz
		# eps = εₛ(shapes,Zygote.dropgrad(tree),Zygote.dropgrad(g.x[i]),Zygote.dropgrad(g.y[j]),Zygote.dropgrad(g.δx),Zygote.dropgrad(g.δy))
		eps = εₛ(shapes,x[i],y[j];tree,δx,δy)
		epsi = inv(eps) # inv( (eps' + eps) / 2) # Hermitian(inv(eps))  # inv(Hermitian(eps)) #   # inv(eps)
        eibuf[:,:,i,j,kk] = epsi #(epsi' + epsi) / 2
    end
    # return HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},T,5,5,Array{T,5}}( real(copy(eibuf)) )
	return HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()}}( real(copy(eibuf)) )
end

function make_εₛ⁻¹_fwd(shapes::Vector{<:Shape{N}};Δx::Real,Δy::Real,Δz::Real,Nx::Int,Ny::Int,Nz::Int,
	 	δx=Δx/Nx, δy=Δy/Ny, δz=Δz/Nz, x=( ( δx .* (0:(Nx-1))) .- Δx/2. ),
		y=( ( δy .* (0:(Ny-1))) .- Δy/2. ), z=( ( δz .* (0:(Nz-1))) .- Δz/2. ) ) where N
    Zygote.forwarddiff(shapes) do shapes
		make_εₛ⁻¹(shapes;Δx,Δy,Δz,Nx,Ny,Nz,δx,δy,δz,x,y,z)
	end
end
