# function vec3D(v::AbstractVector{T}) where T
# 	vout = (length(v)==3 ? v : SVector{3,T}(v[1],v[2],0.))
# 	return vout
# end

# function normcart(n0::AbstractVector{T}) where T #::SMatrix{3,3,T,9} where T<:Real
# 	# Create `S`, a local Cartesian coordinate system from a surface-normal
# 	# 3-vector n0 (pointing outward) from shape
# 	# n = n0 / norm(n0)
# 	n = vec3D(n0) / norm(n0)
# 	# Pick `h` to be a vector that is not along n.
# 	h = any(iszero.(n)) ? n × normalize(iszero.(n)) :  n × SVector(1., 0. , 0.)
# 	v = n × h
# 	S = SMatrix{3,3,T,9}([n h v])  # unitary
# end

# function normcart(n0::SVector{2,T}) where T #::SMatrix{3,3,T,9} where T<:Real
# 	# Create `S`, a local Cartesian coordinate system from a surface-normal
# 	# 3-vector n0 (pointing outward) from shape
# 	n = SVector{3,T}(n0[1],n0[2],0.) / norm(n0)
# 	# Pick `h` to be a vector that is not along n.
# 	h = any(iszero.(n)) ? n × normalize(iszero.(n)) :  n × SVector(1., 0. , 0.)
# 	v = n × h
# 	S = SMatrix{3,3,T,9}([n h v])  # unitary
# end



##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################





# function smooth(sinds::AbstractArray{NTuple{NI, TI}},shapes,minds,mat_vals,xx,vxl_min,vxl_max) where {NI,TI<:Int}
# function smooth(sinds::NTuple{NI, TI},shapes,minds,mat_vals,xx,vxl_min,vxl_max) where {NI,TI<:Int}
# 	@inbounds if iszero(sinds[2])
# 		return mat_vals[minds[first(sinds)]]
# 	elseif iszero(sinds[3])
# 		r₀,n⃗ = surfpt_nearby(xx, shapes[first(sinds)])
# 		rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
# 		return @inbounds avg_param(
# 			mat_vals[minds[sinds[1]]],
# 			mat_vals[minds[sinds[2]]],
# 			normcart(n⃗),
# 			rvol,
# 		)
# 	else
# 		return @inbounds mapreduce(i->mat_vals[minds[i]],+,sinds) / NI  # naive averaging to be used
# 	end
# end

# # extra `matvals_inv` input for cases where smoothed, inverted, matrix-valued quantity is to be calculated
# function smooth(sinds::NTuple{NI, TI},shapes,minds,mat_vals,mat_vals_inv,calcinv,xx,vxl_min,vxl_max) where {NI,TI<:Int}
# 	@inbounds if iszero(sinds[2])
# 		return first(calcinv) ? mat_vals_inv[minds[first(sinds)]] : mat_vals[minds[first(sinds)]]
# 	elseif iszero(sinds[3])
# 		r₀,n⃗ = surfpt_nearby(xx, shapes[first(sinds)])
# 		rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
# 		avg = @inbounds avg_param(
# 			mat_vals[minds[sinds[1]]],
# 			mat_vals[minds[sinds[2]]],
# 			normcart(n⃗),
# 			rvol)
# 		return first(calcinv) ? inv(avg) : avg
# 	else
# 		avg = @inbounds (mapreduce(i->mat_vals[minds[i]],+,sinds) / NI)  # naive averaging to be used
# 		return first(calcinv) ? inv(avg) : avg
# 	end
# end

# # function smooth(sinds::AbstractArray{NTuple{NI, TI}},shapes,minds,mat_vals::Matrix,xx,vxl_min,vxl_max) where {NI,TI<:Int}
# function smooth(sinds::NTuple{NI, TI},shapes,minds,mat_vals::Matrix,xx,vxl_min,vxl_max) where {NI,TI<:Int}
# 	@inbounds if iszero(sinds[2])
# 		return mat_vals[minds[first(sinds)],:]
# 	elseif iszero(sinds[3])
# 		r₀,n⃗ = surfpt_nearby(xx, shapes[first(sinds)])
# 		rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
# 		return @inbounds [ avg_param(
# 			mat_vals[minds[sinds[1]],i],
# 			mat_vals[minds[sinds[2]],i],
# 			normcart(n⃗),
# 			rvol,
# 		) for i=1:size(mat_vals,2)]
# 	else
# 		return @inbounds [ (mapreduce(i->mat_vals[minds[i],j],+,sinds) / NI) for j=1:size(mat_vals,2)]  # use naive averaging when >2 materials found at pixel corners
# 	end
# end

# # function smooth(sinds::AbstractArray{NTuple{NI, TI}},shapes,minds,mat_vals::Matrix,xx,vxl_min,vxl_max) where {NI,TI<:Int}
# function smooth(sinds::NTuple{NI, TI},shapes,minds,mat_vals::Matrix,mat_vals_inv::Matrix,calcinv,xx,vxl_min,vxl_max) where {NI,TI<:Int}
# 	@inbounds if iszero(sinds[2])
# 		return @inbounds [ (calcinv[fn_idx] ? mat_vals_inv[minds[first(sinds)],fn_idx] : mat_vals[minds[first(sinds)],fn_idx]) for fn_idx=1:size(mat_vals,2)]
# 	elseif iszero(sinds[3])
# 		r₀,n⃗ = surfpt_nearby(xx, shapes[first(sinds)])
# 		rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
# 		return @inbounds [ ( calcinv[fn_idx] ? inv(avg_param(mat_vals[minds[sinds[1]],fn_idx],mat_vals[minds[sinds[2]],fn_idx],normcart(n⃗),rvol)) : avg_param(mat_vals[minds[sinds[1]],fn_idx],mat_vals[minds[sinds[2]],fn_idx],normcart(n⃗),rvol) ) for fn_idx=1:size(mat_vals,2) ]
# 	else
# 		avg = @inbounds [ (mapreduce(i->mat_vals[minds[i],j],+,sinds) / NI) for j=1:size(mat_vals,2)]  # use naive averaging when >2 materials found at pixel corners
# 		return @inbounds [ ( calcinv[fn_idx] ? inv(avg[fn_idx]) : avg[fn_idx] ) for fn_idx=1:size(mat_vals,2) ]
# 	end
# end

# function smooth(ω::T1,p::AbstractVector{T2},fnames::NTuple{N,Symbol},f_geom::F,grid::Grid{ND}) where {ND,N,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	n_fns=length(fnames)
# 	om_p = vcat(ω,p)
# 	xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
# 	vxlmin,vxlmax = vxl_minmax(xyzc)

# 	arr_flatB = Zygote.Buffer(om_p,9,size(grid)...,n_fns)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		shapes = getfield(geom,:shapes)
# 		om_inv = inv(first(om_p))
# 		mat_vals = mapreduce(ss->[ map(f->(mat=SMatrix{3,3}(f(om_inv)); 0.5*(mat+mat')),getfield(geom,ss))... ], hcat, fnames)
# 		sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
# 		smoothed_vals_nested = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# 			Tuple(smooth(sinds,shapes,geom.material_inds,mat_vals,xx,vn,vp))
# 		end
# 		smoothed_vals = hcat( [map(x->getindex(x,i),smoothed_vals_nested) for i=1:n_fns]...)
# 		smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
# 		return smoothed_vals_rr  # new spatially smoothed ε tensor array
# 	end
# 	copyto!(arr_flatB,copy(arr_flat))
# 	arr_flat_r = copy(arr_flatB)
# 	Nx = size(grid,1)
# 	Ny = size(grid,2)
# 	fn_arrs = [hybridize(view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,axes(grid)...,n),grid) for n=1:n_fns]
# 	return fn_arrs
# end

# function smooth(ω::AbstractVector{T1},p::AbstractVector{T2},fnames::NTuple{NF,Symbol},f_geom::F,grid::Grid{ND}) where {ND,NF,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	n_ω = length(ω)
# 	n_fns=length(fnames)

# 	xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
# 	vxlmin,vxlmax = vxl_minmax(xyzc)

# 	arr_omcat = mapreduce(hcat,ω) do om
# 		om_p = vcat(om,p)
# 		arr_flatB = Zygote.Buffer(om_p,9,size(grid)...,n_fns)
# 		arr_flat = Zygote.forwarddiff(om_p) do om_p
# 			geom = f_geom(om_p[2:n_p+1])
# 			shapes = getfield(geom,:shapes)
# 			om_inv = inv(first(om_p))
# 			mat_vals = mapreduce(ss->[ map(f->(mat=SMatrix{3,3}(f(om_inv)); 0.5*(mat+mat')),getfield(geom,ss))... ], hcat, fnames)
# 			sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
# 			smoothed_vals_nested = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# 				Tuple(smooth(sinds,shapes,geom.material_inds,mat_vals,xx,vn,vp))
# 			end
# 			smoothed_vals = hcat( [map(x->getindex(x,i),smoothed_vals_nested) for i=1:n_fns]...)
# 			smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
# 			return smoothed_vals_rr  # new spatially smoothed ε tensor array
# 		end
# 		copyto!(arr_flatB,copy(arr_flat))
# 		arr_flat_r = copy(arr_flatB)
# 		Nx = size(grid,1)
# 		Ny = size(grid,2)
# 		fn_arrs = [hybridize(view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,axes(grid)...,n),grid) for n=1:n_fns]
# 		return fn_arrs
# 	end
# 	return arr_omcat
# end

# function smooth(ω::T1,p::AbstractVector{T2},fnames::NTuple{NF,Symbol},invert_fn::Vector{Bool},f_geom::F,grid::Grid{ND}) where {ND,NF,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	n_fns=length(fnames)
# 	om_p = vcat(ω,p)
# 	xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
# 	vxlmin,vxlmax = vxl_minmax(xyzc)

# 	arr_flatB = Zygote.Buffer(om_p,9,size(grid)...,n_fns)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		shapes = getfield(geom,:shapes)
# 		om_inv = inv(first(om_p))
# 		mat_vals = mapreduce(ss->[ map(f->(mat=SMatrix{3,3}(f(om_inv)); 0.5*(mat+mat')),getfield(geom,ss))... ], hcat, fnames)
# 		mat_vals_inv = inv.(mat_vals)
# 		# calcinv = repeat([invert_fn...]',size(mat_vals,1))
# 		sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
# 		smoothed_vals_nested = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# 			Tuple(smooth(sinds,shapes,geom.material_inds,mat_vals,mat_vals_inv,invert_fn,xx,vn,vp))
# 		end
# 		smoothed_vals = hcat( [map(x->getindex(x,i),smoothed_vals_nested) for i=1:n_fns]...)
# 		smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
# 		return smoothed_vals_rr  # new spatially smoothed ε tensor array
# 	end
# 	copyto!(arr_flatB,copy(arr_flat))
# 	arr_flat_r = real(copy(arr_flatB)) # copy(arr_flatB)
# 	Nx = size(grid,1)
# 	Ny = size(grid,2)
# 	fn_arrs = [hybridize(view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,axes(grid)...,n),grid) for n=1:n_fns]
# 	return fn_arrs
# end

# function smooth(ω::AbstractVector{T1},p::AbstractVector{T2},fnames::NTuple{NF,Symbol},invert_fn::Vector{Bool},f_geom::F,grid::Grid{ND}) where {ND,NF,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	n_ω = length(ω)
# 	n_fns=length(fnames)

# 	xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
# 	vxlmin,vxlmax = vxl_minmax(xyzc)


# 	arr_omcat = mapreduce(hcat,ω) do om
# 		om_p = vcat(om,p)
# 		arr_flatB = Zygote.Buffer(om_p,9,size(grid)...,n_fns)
# 		arr_flat = Zygote.forwarddiff(om_p) do om_p
# 			geom = f_geom(om_p[2:n_p+1])
# 			shapes = getfield(geom,:shapes)
# 			om_inv = inv(first(om_p))
# 			mat_vals = mapreduce(ss->[ map(f->(mat=SMatrix{3,3}(f(om_inv)); 0.5*(mat+mat')),getfield(geom,ss))... ], hcat, fnames)
# 			mat_vals_inv = inv.(mat_vals)
# 			sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
# 			smoothed_vals_nested = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# 				Tuple(smooth(sinds,shapes,geom.material_inds,mat_vals,mat_vals_inv,invert_fn,xx,vn,vp))
# 			end
# 			smoothed_vals = hcat( [map(x->getindex(x,i),smoothed_vals_nested) for i=1:n_fns]...)
# 			smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
# 			return smoothed_vals_rr  # new spatially smoothed ε tensor array
# 		end
# 		copyto!(arr_flatB,copy(arr_flat))
# 		arr_flat_r = copy(arr_flatB)
# 		Nx = size(grid,1)
# 		Ny = size(grid,2)
# 		fn_arrs = [hybridize(view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,axes(grid)...,n),grid) for n=1:n_fns]
# 		return fn_arrs
# 	end
# 	return arr_omcat
# end

# function smooth(ω::T1,p::AbstractVector{T2},fname::Symbol,invert_fn::Bool,f_geom::F,grid::Grid{ND}) where {ND,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	om_p = vcat(ω,p)
# 	xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
# 	vxlmin,vxlmax = vxl_minmax(xyzc)

# 	arr_flatB = Zygote.Buffer(om_p,9,size(grid)...)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		shapes = getfield(geom,:shapes)
# 		om_inv = inv(first(om_p))
# 		mat_vals = map(f->(mat=SMatrix{3,3}(f(om_inv)); 0.5*(mat+mat')),getfield(geom,fname))
# 		mat_vals_inv = inv.(mat_vals)
# 		# sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
# 		sinds = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
# 		smoothed_vals = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# 			smooth(sinds,shapes,geom.material_inds,mat_vals,mat_vals_inv,invert_fn,xx,vn,vp)
# 		end
# 		smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
# 		return smoothed_vals_rr  # new spatially smoothed ε tensor array
# 	end
# 	copyto!(arr_flatB,copy(arr_flat))
# 	arr_flat_r = copy(arr_flatB)
# 	return hybridize(reshape(arr_flat_r,(3,3,size(grid)...)),grid)
# end



# # testing:
# function kottke_smoothing(sinds::NTuple{NI, TI},shapes,minds,mat_vals,mat_vals_inv,calcinv,xx,vxl_min,vxl_max) where {NI,TI<:Int}
# 	@inbounds if iszero(sinds[2])
# 		return @inbounds [ (calcinv[fn_idx] ? mat_vals_inv[minds[first(sinds)],fn_idx] : mat_vals[minds[first(sinds)],fn_idx]) for fn_idx=1:size(mat_vals,2)]
# 	elseif iszero(sinds[3])
# 		r₀,n⃗ = surfpt_nearby(xx, shapes[first(sinds)])
# 		rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
# 		return @inbounds [ ( calcinv[fn_idx] ? inv(avg_param(mat_vals[minds[sinds[1]],fn_idx],mat_vals[minds[sinds[2]],fn_idx],normcart(n⃗),rvol)) : avg_param(mat_vals[minds[sinds[1]],fn_idx],mat_vals[minds[sinds[2]],fn_idx],normcart(n⃗),rvol) ) for fn_idx=1:size(mat_vals,2) ]
# 	else
# 		avg = @inbounds [ (mapreduce(i->mat_vals[minds[i],j],+,sinds) / NI) for j=1:size(mat_vals,2)]  # use naive averaging when >2 materials found at pixel corners
# 		return @inbounds [ ( calcinv[fn_idx] ? inv(avg[fn_idx]) : avg[fn_idx] ) for fn_idx=1:size(mat_vals,2) ]
# 	end
# end

# function volfrac_smoothing(sinds::NTuple{NI, TI},shapes,minds,mat_vals,mat_vals_inv,calcinv,xx,vxl_min,vxl_max) where {NI,TI<:Int}
# 	@inbounds if iszero(sinds[2])
# 		return @inbounds [ (calcinv[fn_idx] ? mat_vals_inv[minds[first(sinds)],fn_idx] : mat_vals[minds[first(sinds)],fn_idx]) for fn_idx=1:size(mat_vals,2)]
# 	elseif iszero(sinds[3])
# 		r₀,n⃗ = surfpt_nearby(xx, shapes[first(sinds)])
# 		rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
# 		return @inbounds [ ( calcinv[fn_idx] ? inv( rvol*mat_vals[minds[sinds[1]],fn_idx] + (1-rvol)*mat_vals[minds[sinds[2]],fn_idx] ) : ( rvol*mat_vals[minds[sinds[1]],fn_idx] + (1-rvol)*mat_vals[minds[sinds[2]],fn_idx] ) ) for fn_idx=1:size(mat_vals,2) ]
# 	else
# 		avg = @inbounds [ (mapreduce(i->mat_vals[minds[i],j],+,sinds) / NI) for j=1:size(mat_vals,2)]  # use naive averaging when >2 materials found at pixel corners
# 		return @inbounds [ ( calcinv[fn_idx] ? inv(avg[fn_idx]) : avg[fn_idx] ) for fn_idx=1:size(mat_vals,2) ]
# 	end
# end

# function smooth(ω::AbstractVector{T1},p::AbstractVector{T2},fname::Symbol,invert_fn::Bool,f_geom::F,grid::Grid{ND}) where {ND,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	om_p = vcat(ω,p)
# 	xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
# 	vxlmin,vxlmax = vxl_minmax(xyzc)

# 	arr_omcat = map(ω) do om #mapreduce(hcat,ω) do om
# 		om_p = vcat(om,p)
# 		arr_flatB = Zygote.Buffer(om_p,9,size(grid)...)
# 		arr_flat = Zygote.forwarddiff(om_p) do om_p
# 			om_p = vcat(om,p)
# 			geom = f_geom(om_p[2:n_p+1])
# 			shapes = getfield(geom,:shapes)
# 			om_inv = inv(first(om_p))
# 			mat_vals = map(f->(mat=SMatrix{3,3}(f(om_inv)); 0.5*(mat+mat')),getfield(geom,fname))
# 			mat_vals_inv = inv.(mat_vals)
# 			sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
# 			smoothed_vals = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# 				# kottke_smoothing(sinds::NTuple{NI, TI},shapes,minds,mat_vals,mat_vals_inv,calcinv,xx,vxl_min,vxl_max)
# 				kottke_smoothing(sinds,shapes,geom.material_inds,mat_vals,mat_vals_inv,invert_fn,xx,vn,vp)
# 				# smooth(sinds,shapes,geom.material_inds,mat_vals,mat_vals_inv,invert_fn,xx,vn,vp)
# 			end
# 			smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
# 			return smoothed_vals_rr  # new spatially smoothed ε tensor array
# 		end
# 		copyto!(arr_flatB,copy(arr_flat))
# 		arr_flat_r = copy(arr_flatB)
# 		return hybridize(reshape(arr_flat_r,(3,3,size(grid)...)),grid)
# 	end
# 	return arr_omcat
# end

# function smooth(ω::T1,p::AbstractVector{T2},fnames::NTuple{NF,Symbol},invert_fn::Vector{Bool},f_geom::F,grid::Grid{ND},smoothing_fn::TF) where {NF,T1<:Real,T2<:Real,F<:Function,TF<:Function,ND}
# 	n_p = length(p)
# 	n_fns=length(fnames)
# 	om_p = vcat(ω,p)
# 	xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
# 	vxlmin,vxlmax = vxl_minmax(xyzc)

# 	arr_flatB = Zygote.Buffer(om_p,9,size(grid)...,n_fns)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		shapes = getfield(geom,:shapes)
# 		om_inv = inv(first(om_p))
# 		mat_vals = mapreduce(ss->[ map(f->(mat=SMatrix{3,3}(f(om_inv)); 0.5*(mat+mat')),getfield(geom,ss))... ], hcat, fnames)
# 		mat_vals_inv = inv.(mat_vals)
# 		# calcinv = repeat([invert_fn...]',size(mat_vals,1))
# 		sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
# 		smoothed_vals_nested = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# 			Tuple(smoothing_fn(sinds,shapes,geom.material_inds,mat_vals,mat_vals_inv,invert_fn,xx,vn,vp))
# 		end
# 		smoothed_vals = hcat( [map(x->getindex(x,i),smoothed_vals_nested) for i=1:n_fns]...)
# 		smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
# 		return smoothed_vals_rr  # new spatially smoothed ε tensor array
# 	end
# 	copyto!(arr_flatB,copy(arr_flat))
# 	arr_flat_r = copy(arr_flatB)
# 	Nx = size(grid,1)
# 	Ny = size(grid,2)
# 	fn_arrs = [hybridize(view(reshape(arr_flat_r,3,3,size(grid)...,n_fns),1:3,1:3,axes(grid)...,n),grid) for n=1:n_fns]
# 	return fn_arrs
# end

# function smooth(ω::NTuple{N1,T1},p::AbstractVector{T2},fname::Symbol,invert_fn::Bool,default_val::TA,f_geom::F,grid::Grid,smoothing_fn::TF) where {N1,T1<:Real,T2<:Real,TA<:AbstractArray,F,TF}
# 	n_p = length(p)
# 	# om_p = vcat(ω...,p)
# 	xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
# 	vxlmin,vxlmax = vxl_minmax(xyzc)

# 	# om_p = vcat(SVector{N1,T1}(ω...),p)
# 	# om_p = vcat(Vector(ω...),p)
# 	om_p = vcat(ω...,p)
# 	arr_flatB = Zygote.Buffer([1.0, 2.0],(length(default_val),size(grid)...))
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		# om_p = vcat(om,p)
# 		geom = f_geom(om_p[N1+1:N1+n_p])
# 		shapes = getfield(geom,:shapes)
# 		om_inv = inv.(om_p[1:N1])
# 		mat_vals = map(f->convert(TA,f(om_inv...)),getfield(geom,fname))
# 		# println("default_val 1: $default_val")
# 		# mat_vals = map(f->promote(default_val,f(om_inv...))[2],getfield(geom,fname))
# 		# println("default_val 2: $default_val")
# 		# mat_vals = map(f->f(om_inv...),getfield(geom,fname))
# 		sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
# 		if invert_fn
# 			mat_vals_inv = inv.(mat_vals)
# 			smoothed_vals = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# 				smoothing_fn(sinds,shapes,geom.material_inds,mat_vals,mat_vals_inv,invert_fn,xx,vn,vp)[1]
# 			end
# 		else
# 			smoothed_vals = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# 				smoothing_fn(sinds,shapes,geom.material_inds,mat_vals,mat_vals,invert_fn,xx,vn,vp)[1]
# 			end
# 		end
# 		smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
# 		# smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
# 		return smoothed_vals_rr  # new spatially smoothed ε tensor array
# 	end
# 	copyto!(arr_flatB,copy(arr_flat))
# 	arr_flat_r = copy(arr_flatB)
# 	return HybridArray{Tuple{size(default_val)...,Dynamic(),Dynamic()}}(reshape(arr_flat_r,(size(default_val)...,size(grid)...)))
# end


# function smooth(ω::AbstractVector{NTuple{N1,T1}},p::AbstractVector{T2},fname::Symbol,invert_fn::Bool,default_val::TA,f_geom::F,grid::Grid,smoothing_fn::TF) where {N1,T1<:Real,T2<:Real,TA<:AbstractArray,F,TF}
# 	n_p = length(p)
# 	# xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyz = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	# xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
# 	xyzc = Zygote.@ignore(x⃗c(grid))
# 	vxlmin,vxlmax = vxl_minmax(xyzc)

# 	arr_omcat = map(ω) do om #mapreduce(hcat,ω) do om
# 		om_p = vcat(SVector{N1}(om...),p)
# 		arr_flatB = Zygote.Buffer(om_p,(length(default_val),size(grid)...))
# 		arr_flat = Zygote.forwarddiff(om_p) do om_p
# 			geom = f_geom(om_p[N1+1:N1+n_p])
# 			shapes = getfield(geom,:shapes)
# 			om_inv = inv.(om_p[1:N1])
# 			mat_vals = map(f->convert(TA,f(om_inv...)),getfield(geom,fname))
# 			sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
# 			if invert_fn
# 				mat_vals_inv = inv.(mat_vals)
# 				smoothed_vals = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# 					smoothing_fn(sinds,shapes,geom.material_inds,mat_vals,mat_vals_inv,invert_fn,xx,vn,vp)[1]
# 				end
# 			else
# 				smoothed_vals = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# 					smoothing_fn(sinds,shapes,geom.material_inds,mat_vals,mat_vals,invert_fn,xx,vn,vp)[1]
# 				end
# 			end

# 			smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
# 			return smoothed_vals_rr  # new spatially smoothed ε tensor array
# 		end
# 		copyto!(arr_flatB,copy(arr_flat))
# 		arr_flat_r = copy(arr_flatB)
# 		return HybridArray{Tuple{size(default_val)...,Dynamic(),Dynamic()}}(reshape(arr_flat_r,(size(default_val)...,size(grid)...)))
# 	end
# 	return arr_omcat
# end

# # function smooth(ω::AbstractVector{T1},p::AbstractVector{T2},fname::Symbol,invert_fn::Bool,default_val::TA,f_geom::F,grid::Grid{ND},smoothing_fn::TF) where {T1<:Real,T2<:Real,TA<:AbstractArray,F,TF,ND}
# # 	n_p = length(p)
# # 	xyz::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# # 	xyzc::Array{SVector{3, Float64},ND} = Zygote.@ignore(x⃗c(grid))
# # 	vxlmin,vxlmax = vxl_minmax(xyzc)

# # 	arr_omcat = map(ω) do om #mapreduce(hcat,ω) do om
# # 		om_p = vcat([om,],p)
# # 		arr_flatB = Zygote.Buffer(om_p,(length(default_val),size(grid)...))
# # 		arr_flat = Zygote.forwarddiff(om_p) do om_p
# # 			geom = f_geom(om_p[2:1+n_p])
# # 			shapes = getfield(geom,:shapes)
# # 			om_inv = inv(om_p[1])
# # 			mat_vals = map(f->convert(TA,f(om_inv)),getfield(geom,fname))
# # 			sinds::Matrix{NTuple{4, Int64}} = Zygote.@ignore(proc_sinds(corner_sinds(shapes,xyzc)))
# # 			if invert_fn
# # 				mat_vals_inv = inv.(mat_vals)
# # 				smoothed_vals = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# # 					smoothing_fn(sinds,shapes,geom.material_inds,mat_vals,mat_vals_inv,invert_fn,xx,vn,vp)[1]
# # 				end
# # 			else
# # 				smoothed_vals = map(sinds,xyz,vxlmin,vxlmax) do sinds,xx,vn,vp
# # 					smoothing_fn(sinds,shapes,geom.material_inds,mat_vals,mat_vals,invert_fn,xx,vn,vp)[1]
# # 				end
# # 			end

# # 			# smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
# # 			# return smoothed_vals_rr  # new spatially smoothed ε tensor array
# # 			return smoothed_vals
# # 		end
# # 		copyto!(arr_flatB,copy(arr_flat))
# # 		arr_flat_r = copy(arr_flatB)
# # 		# return HybridArray{Tuple{size(default_val)...,Dynamic(),Dynamic()}}(reshape(arr_flat_r,(size(default_val)...,size(grid)...)))
# # 	end
# # 	return arr_omcat
# # end

# # smoothed_vals_rr = copy(reinterpret(eltype(first(smoothed_vals)),smoothed_vals))
# # 			return smoothed_vals_rr  # new spatially smoothed ε tensor array
# # 		end
# # 		copyto!(arr_flatB,copy(arr_flat))
# # 		arr_flat_r = copy(arr_flatB)
# # 		return hybridize(reshape(arr_flat_r,(3,3,size(grid)...)),grid)


##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################





##### Legacy code, please remove soon

# _V3(v) = isequal(length(v),3) ? v : vcat(v,zeros(3-length(v)))
#
# function n_rvol(shape,xyz,vxl_min,vxl_max)
# 	r₀,n⃗ = surfpt_nearby(xyz, shape)
# 	rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
# 	return _V3(n⃗),rvol
# end
#
# function normcart(n0::AbstractVector{T}) where T<:Real #::SMatrix{3,3,T,9} where T<:Real
# 	# Create `S`, a local Cartesian coordinate system from a surface-normal
# 	# 3-vector n0 (pointing outward) from shape
# 	n = n0 / norm(n0)
# 	# Pick `h` to be a vector that is not along n.
# 	h = any(iszero.(n)) ? n × normalize(iszero.(n)) :  n × SVector(1., 0. , 0.)
# 	v = n × h
# 	S = SMatrix{3,3,T,9}([n h v])  # unitary
# end
#
# function _S_rvol(sinds_proc,xyz,vxl_min,vxl_max,shapes)
# 	if iszero(sinds_proc[2])
# 		return (SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.), 0.)
# 	elseif iszero(sinds_proc[3])
# 		r₀,n⃗ = surfpt_nearby(_V3(xyz), shapes[sinds_proc[1]])
# 		rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
# 		return normcart(_V3(n⃗)), rvol
# 	else
# 		return (SMatrix{3,3}(0.,0.,0.,0.,0.,0.,0.,0.,0.), 0.)  # naive averaging to be used
# 	end
# end
#
# function S_rvol(sinds_proc,xyz,vxl_min,vxl_max,shapes)
# 	f(sp,x,vn,vp) = let s=shapes
# 		_S_rvol(sp,x,vn,vp,s)
# 	end
# 	map(f,sinds_proc,xyz,vxl_min,vxl_max)
# end
#
# function vxl_min(x⃗c::AbstractArray{T,2}) where T
# 	@view x⃗c[1:max((end-1),1),1:max((end-1),1)]
# end
#
# function vxl_min(x⃗c::AbstractArray{T,3}) where T
# 	@view x⃗c[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
# end
#
# function vxl_max(x⃗c::AbstractArray{T,2}) where T
# 	@view x⃗c[min(2,end):end,min(2,end):end]
# end
#
# function vxl_max(x⃗c::AbstractArray{T,3}) where T
# 	@view x⃗c[min(2,end):end,min(2,end):end,min(2,end):end]
# end
#
# function S_rvol(geom;ms::ModeSolver)
# 	Zygote.@ignore( ms.geom = geom )	# update ms.geom
# 	xyz = Zygote.@ignore(x⃗(ms.grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc = Zygote.@ignore(x⃗c(ms.grid))
# 	Zygote.@ignore(corner_sinds!(ms.corner_sinds,geom.shapes,xyz,xyzc))
# 	Zygote.@ignore(proc_sinds!(ms.sinds_proc,ms.corner_sinds))
# 	f(sp,x,vn,vp) = let s=geom.shapes
# 		_S_rvol(sp,x,vn,vp,s)
# 	end
# 	map(f,ms.sinds_proc,xyz,vxl_min(xyzc),vxl_max(xyzc))
# end
#
# function S_rvol(geom::Geometry,grid::Grid)
# 	xyz = Zygote.@ignore(x⃗(grid))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc = Zygote.@ignore(x⃗c(grid))
# 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	f(sp,x,vn,vp) = let s=geom.shapes
# 		_S_rvol(sp,x,vn,vp,s)
# 	end
# 	map(f,ps,xyz,vxl_min(xyzc),vxl_max(xyzc))
# end
#
# function _εₛ(es,sinds_proc::NTuple{N},minds,Srvol) where N
# 	iszero(sinds_proc[2]) && return es[minds[sinds_proc[1]]]
# 	iszero(sinds_proc[3]) && return avg_param(	es[minds[sinds_proc[1]]],
# 												es[minds[sinds_proc[2]]],
# 												Srvol[1],
# 												Srvol[2]
# 												)
# 	return mapreduce(i->es[minds[sinds_proc[i]]],+,sinds_proc) / N
# end
#
# function εₛ(εs,sinds_proc,matinds,Srvol)
# 	f(sp,srv) = let es=εs, mi=matinds
# 		e_nonHerm = _εₛ(es,sp,mi,srv)
# 		(e_nonHerm' + e_nonHerm) / 2
# 	end
# 	map(f,sinds_proc,Srvol)
# end
#
# function εₛ(ω,geom::AbstractVector{<:Shape};ms::ModeSolver)
# 	Srvol = S_rvol(geom;ms)
# 	es = vcat(εs(geom,( 1. / ω )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	ei_new = εₛ(es,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function nngₛ(ω,geom::AbstractVector{<:Shape};ms::ModeSolver)
# 	Srvol = S_rvol(geom;ms)
# 	# nngs0 = nn̂g.(materials(geom),( 1. / ω )) # = √.(ε̂) .* nĝ (elementwise product of index and group index tensors)
# 	# nngs = vcat( nngs0 ,[εᵥ,]) # ( √dielectric tensor * ng tensor ) for each material, vacuum permittivity tensor appended
# 	nngs = vcat(nn̂gs(geom,( 1. / ω )),[εᵥ,])
# 	εₛ(nngs,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function εₛ(ω,geom::AbstractVector{<:Shape},grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom,grid))
# 	minds = Zygote.@ignore(matinds(geom))
# 	es = vcat(εs(geom,( 1. / ω )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	ei_new = εₛ(es,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε tensor array
# end
#
# function nngₛ(ω,geom::AbstractVector{<:Shape},grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom,grid))
# 	minds = Zygote.@ignore(matinds(geom))
# 	# nngs0 = nn̂g.(materials(geom),( 1. / ω )) # = √.(ε̂) .* nĝ (elementwise product of index and group index tensors)
# 	# nngs = vcat( nngs0 ,[εᵥ,]) # ( √dielectric tensor * ng tensor ) for each material, vacuum permittivity tensor appended
# 	nngs = vcat(nn̂gs(geom,( 1. / ω )),[εᵥ,])
# 	εₛ(nngs,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε tensor array
# end
#
# function ngvdₛ(ω,geom::AbstractVector{<:Shape},grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom,grid))
# 	minds = Zygote.@ignore(matinds(geom))
# 	# ngvds0 = nĝvd.(materials(geom),( 1. / ω )) # = √.(ε̂) .* nĝ (elementwise product of index and group index tensors)
# 	# ngvds = vcat( ngvds0 ,[εᵥ,]) # ( √dielectric tensor * ng tensor ) for each material, vacuum permittivity tensor appended
# 	ngvds = vcat(nĝvds(geom,( 1. / ω )),[εᵥ,])
# 	εₛ(ngvds,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε tensor array
# end
#
# function εₛ_nngₛ_ngvdₛ(ω,geom::AbstractVector{<:Shape},grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom,grid))
# 	minds = Zygote.@ignore(matinds(geom))
# 	es = vcat(εs(geom,( 1. / ω )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	# nngs = vcat( nn̂g.(materials(geom),( 1. / ω )) ,[εᵥ,]) # ( √dielectric tensor * ng tensor ) for each material, vacuum permittivity tensor appended
# 	# ngvds = vcat( nĝvd.(materials(geom),( 1. / ω )) ,[εᵥ,])
# 	nngs = vcat(nn̂gs(geom,( 1. / ω )),[εᵥ,])
# 	ngvds = vcat(nĝvds(geom,( 1. / ω )),[εᵥ,])
# 	return εₛ(es,dropgrad(ps),dropgrad(minds),Srvol), εₛ(nngs,dropgrad(ps),dropgrad(minds),Srvol), εₛ(ngvds,dropgrad(ps),dropgrad(minds),Srvol)
# end
#
# function _εₛ_init(lm::Real,geom::Vector{S},gr::Grid) where S<:GeometryPrimitives.Shape
# 	xyz = Zygote.@ignore(x⃗(gr))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc = Zygote.@ignore(x⃗c(gr))
# 	sinds = Zygote.@ignore(corner_sinds(geom,xyz,xyzc))  	# shape indices at pixel/voxel corners,
# 	sinds_proc = Zygote.@ignore(proc_sinds(sinds))  		# processed corner shape index lists for each pixel/voxel, should efficiently indicate whether averaging is needed and which ε⁻¹ to use otherwise
# 	mats = Zygote.@ignore(materials(geom))
# 	minds = Zygote.@ignore(matinds(geom,mats))
# 	vxl_min = Zygote.@ignore( @view xyzc[1:max((end-1),1),1:max((end-1),1)] )
# 	vxl_max = Zygote.@ignore( @view xyzc[min(2,end):end,min(2,end):end] )
# 	Srvol = S_rvol(sinds_proc,xyz,vxl_min,vxl_max,geom)
# 	# εs = vcat([mm.fε.(lm) for mm in mats],[εᵥ,])
# 	es = vcat(εs(geom, lm),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	εsm = εₛ(es,sinds_proc,minds,Srvol)
# 	return (sinds,sinds_proc,Srvol,mats,minds,εsm)
# end
#
# function _εₛ⁻¹(εs,ε⁻¹s,sinds_proc,matinds,Srvol)
# 	iszero(sinds_proc[2]) && return ε⁻¹s[matinds[sinds_proc[1]]]
# 	iszero(sinds_proc[3]) && return inv(avg_param(	εs[matinds[sinds_proc[1]]],
# 												εs[matinds[sinds_proc[2]]],
# 												Srvol[1],
# 												Srvol[2]
# 												))
# 	return inv(mapreduce(i->εs[matinds[sinds_proc[i]]],+,sinds_proc)) * 8
# end
#
# function εₛ⁻¹(εs,ε⁻¹s,sinds_proc,matinds,Srvol)
# 	f(sp,srv) = let es=εs, eis=ε⁻¹s, mi=matinds
# 		# _εₛ⁻¹(es,eis,sp,mi,srv)
# 		ei_nonHerm = _εₛ(es,sp,mi,srv)
# 		inv( (ei_nonHerm' + ei_nonHerm) / 2 )
# 	end
# 	map(f,sinds_proc,Srvol)
# end
#
#
# function εₛ⁻¹(ω,Srvol::AbstractArray{Tuple{SMatrix{3,3,T,9},T}};ms::ModeSolver) where T
# 	es = vcat(εs(ms.geom,( 1. / ω )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material # previously I ran `inv( (eps' + eps) / 2)` at each point
# 	ei_new = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function εₛ⁻¹(ω,geom::AbstractVector{<:Shape};ms::ModeSolver)
# 	Srvol = S_rvol(geom;ms)
# 	es = vcat(εs(geom,( 1. / ω )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# 	ei_new = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function εₛ⁻¹(ω,geom::AbstractVector{<:Shape},grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom,grid))
# 	minds = Zygote.@ignore(matinds(geom))
# 	es = vcat(εs(geom,( 1. / ω )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# 	ei_new = εₛ⁻¹(es,eis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# # function εₛ⁻¹(ω,geom::Geometry;ms::ModeSolver)
# # 	Srvol = S_rvol(geom;ms)
# # 	es = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fεs),[εᵥ,])
# # 	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# # 	ei_new = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# # end
# #
# # function εₛ⁻¹(ω,geom::Geometry,grid::Grid)
# # 	Srvol = S_rvol(geom,grid)
# # 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# # 	minds = geom.material_inds
# # 	es = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fεs),[εᵥ,])
# # 	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# # 	ei_new = εₛ⁻¹(es,eis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# # end
#
# function nngₛ⁻¹(ω,geom::AbstractVector{<:Shape};ms::ModeSolver)
# 	Srvol = S_rvol(geom;ms)
# 	# nngs0 = nn̂g.(materials(geom),( 1. / ω )) # = √.(ε̂) .* nĝ (elementwise product of index and group index tensors)
# 	# nngs = vcat( nngs0 ,[εᵥ,]) # ( √dielectric tensor * ng tensor ) for each material, vacuum permittivity tensor appended
# 	nngs = vcat(nn̂gs(geom,( 1. / ω )),[εᵥ,])
# 	nngis = inv.(nngs)
# 	εₛ⁻¹(nngs,nngis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed nng⁻¹ tensor array
# end
#
#
# function nngₛ⁻¹(ω,geom::AbstractVector{<:Shape},grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom,grid))
# 	minds = Zygote.@ignore(matinds(geom))
# 	# nngs0 = nn̂g.(materials(geom),( 1. / ω )) # = √.(ε̂) .* nĝ (elementwise product of index and group index tensors)
# 	# nngs = vcat( nngs0 ,[εᵥ,]) # ( √dielectric tensor * ng tensor ) for each material, vacuum permittivity tensor appended
# 	nngs = vcat(nn̂gs(geom,( 1. / ω )),[εᵥ,])
# 	nngis = inv.(nngs)	# corresponding list of inverse dielectric tensors for each material
# 	εₛ⁻¹(nngs,nngis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function ngvdₛ⁻¹(ω,geom::AbstractVector{<:Shape},grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom,grid))
# 	minds = Zygote.@ignore(matinds(geom))
# 	# ngvds0 = nĝvd.(materials(geom),( 1. / ω )) # = √.(ε̂) .* nĝ (elementwise product of index and group index tensors)
# 	# ngvds = vcat( ngvds0 ,[εᵥ,]) # ( √dielectric tensor * ng tensor ) for each material, vacuum permittivity tensor appended
# 	ngvds = vcat(nĝvds(geom,( 1. / ω )),[εᵥ,])
# 	ngvdis = inv.(ngvds)	# corresponding list of inverse dielectric tensors for each material
# 	εₛ⁻¹(ngvds,ngvdis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function εₛ⁻¹_nngₛ⁻¹_ngvdₛ⁻¹(ω,geom::AbstractVector{<:Shape},grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom,grid))
# 	minds = Zygote.@ignore(matinds(geom))
# 	es = vcat(εs(geom,( 1. / ω )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	# nngs = vcat( nn̂g.(materials(geom),( 1. / ω )) ,[εᵥ,]) # ( √dielectric tensor * ng tensor ) for each material, vacuum permittivity tensor appended
# 	# ngvds = vcat( nĝvd.(materials(geom),( 1. / ω )) ,[εᵥ,])
# 	nngs = vcat(nn̂gs(geom,( 1. / ω )),[εᵥ,])
# 	ngvds = vcat(nĝvds(geom,( 1. / ω )),[εᵥ,])
# 	return εₛ⁻¹(es,inv.(es),dropgrad(ps),dropgrad(minds),Srvol), εₛ⁻¹(nngs,inv.(nngs),dropgrad(ps),dropgrad(minds),Srvol), εₛ⁻¹(ngvds,inv.(ngvds),dropgrad(ps),dropgrad(minds),Srvol)
# end
#
# function εₛ⁻¹(geom::AbstractVector{<:Shape};ms::ModeSolver)
# 	om_prev = Zygote.@ignore(sqrt(real(ms.ω²[1])))
# 	Srvol = S_rvol(geom;ms)
# 	es = vcat(εs(geom,( 1. / om_prev )),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# 	ei_new = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
#
# function _εₛ⁻¹_init(lm::Real,geom::Vector{S},gr::Grid) where S<:GeometryPrimitives.Shape
# 	xyz = Zygote.@ignore(x⃗(gr))			# (Nx × Ny × Nz) 3-Array of (x,y,z) vectors at pixel/voxel centers
# 	xyzc = Zygote.@ignore(x⃗c(gr))
# 	sinds = Zygote.@ignore(corner_sinds(geom,xyzc))  	# shape indices at pixel/voxel corners,
# 	sinds_proc = Zygote.@ignore(proc_sinds(sinds))  		# processed corner shape index lists for each pixel/voxel, should efficiently indicate whether averaging is needed and which ε⁻¹ to use otherwise
# 	mats = materials(geom) # vcat(εs(geom,( 1. / ω )),[εᵥ,]) # Zygote.@ignore(vcat((εs(geom,lm),[εᵥ,])))  #materials(geom))
# 	minds = Zygote.@ignore(matinds(geom))
# 	vxl_min = Zygote.@ignore( @view xyzc[1:max((end-1),1),1:max((end-1),1)] )
# 	vxl_max = Zygote.@ignore( @view xyzc[min(2,end):end,min(2,end):end] )
# 	Srvol = S_rvol(sinds_proc,xyz,vxl_min,vxl_max,geom)
# 	# es =  vcat([mats[minds[i]].fε.(lm) for i=1:length(geom)],[εᵥ,]) # vcat([mm.fε.(lm) for mm in mats],[εᵥ,]) #vcat([mm.fε.(lm) for mm in mats],[εᵥ,])
# 	es = vcat(εs(geom, lm),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	eis = inv.(es)
# 	εism = εₛ⁻¹(es,eis,sinds_proc,minds,Srvol)
# 	return (sinds,sinds_proc,Srvol,mats,minds,εism)
# end
#
# # smoothed parameter functions using Geometry struct
#
# function εₛ(ω,geom::Geometry,grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	minds = geom.material_inds
# 	es = map(f->SMatrix{3,3}(f( 1. / ω )),geom.fεs)		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	ei_new = εₛ(es,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε tensor array
# end
#
# function nngₛ(ω,geom::Geometry,grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	minds = geom.material_inds
# 	nngs = map(f->SMatrix{3,3}(f( 1. / ω )),geom.fnn̂gs)
# 	εₛ(nngs,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε tensor array
# end
#
# function ngvdₛ(ω,geom::Geometry,grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	minds = geom.material_inds
# 	ngvds = map(f->SMatrix{3,3}(f( 1. / ω )),geom.fnĝvds)
# 	εₛ(ngvds,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε tensor array
# end
#
# function χ⁽²⁾ₛ(ω,geom::Geometry,grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	minds = geom.material_inds
# 	χ⁽²⁾s = map(f->SMatrix{3,3}(f( 1. / ω )),geom.fχ⁽²⁾s)
# 	εₛ(ngvds,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε tensor array
# end
#
# function εₛ_nngₛ_ngvdₛ(ω,geom::Geometry,grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	minds = geom.material_inds
# 	es = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fεs),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 	nngs = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fnn̂gs),[εᵥ,])
# 	ngvds = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fnĝvds),[εᵥ,])
# 	return εₛ(es,dropgrad(ps),dropgrad(minds),Srvol), εₛ(nngs,dropgrad(ps),dropgrad(minds),Srvol), εₛ(ngvds,dropgrad(ps),dropgrad(minds),Srvol)
# end
#
# function εₛ⁻¹(ω,geom::Geometry;ms::ModeSolver)
# 	Srvol = S_rvol(geom;ms)
# 	es = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fεs),[εᵥ,])
# 	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# 	ei_new = εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function εₛ⁻¹(ω,geom::Geometry,grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	minds = geom.material_inds
# 	es = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fεs),[εᵥ,])
# 	eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# 	ei_new = εₛ⁻¹(es,eis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function nngₛ⁻¹(ω,geom::Geometry,grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	minds = geom.material_inds
# 	nngs = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fnn̂gs),[εᵥ,])
# 	nngis = inv.(nngs)	# corresponding list of inverse dielectric tensors for each material
# 	εₛ⁻¹(nngs,nngis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function nngₛ⁻¹(ω,geom::Geometry;ms::ModeSolver)
# 	Srvol = S_rvol(geom;ms)
# 	nngs = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fnn̂gs),[εᵥ,])
# 	nngis = inv.(nngs)	# corresponding list of inverse dielectric tensors for each material
# 	ei_new = εₛ⁻¹(nngs,nngis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function ngvdₛ⁻¹(ω,geom::Geometry,grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	minds = geom.material_inds
# 	ngvds = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fnĝvds),[εᵥ,])
# 	ngvdis = inv.(ngvds)	# corresponding list of inverse dielectric tensors for each material
# 	εₛ⁻¹(ngvds,ngvdis,dropgrad(ps),dropgrad(minds),Srvol)  # new spatially smoothed ε⁻¹ tensor array
# end
#
# function εₛ⁻¹_nngₛ⁻¹_ngvdₛ⁻¹(ω,geom::Geometry,grid::Grid)
# 	Srvol = S_rvol(geom,grid)
# 	ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 	minds = geom.material_inds
# 	es = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fεs),[εᵥ,])
# 	nngs = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fnn̂gs),[εᵥ,])
# 	ngvds = vcat(map(f->SMatrix{3,3}(f( 1. / ω )),geom.fnĝvds),[εᵥ,])
# 	return εₛ⁻¹(es,inv.(es),dropgrad(ps),dropgrad(minds),Srvol), εₛ⁻¹(nngs,inv.(nngs),dropgrad(ps),dropgrad(minds),Srvol), εₛ⁻¹(ngvds,inv.(ngvds),dropgrad(ps),dropgrad(minds),Srvol)
# end
#
# # smoothed parameter functions using Geometry function
#
# function εₛ(ω::T1,p::AbstractVector{T2},f_geom::F,grid::Grid{ND}) where {ND,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	# om_p = [promote(ω,p...)...]
# 	om_p = vcat(ω,p)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		Srvol = S_rvol(geom,grid)
# 		ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 		minds = geom.material_inds
# 		es = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fεs),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor appended
# 		return flat(εₛ(es,dropgrad(ps),dropgrad(minds),Srvol))  # new spatially smoothed ε tensor array
# 	end
# 	# return arr_flat
# 	return parent(parent(arr_flat))
# end
#
# function nngₛ(ω::T1,p::AbstractVector{T2},f_geom::F,grid::Grid{ND}) where {ND,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	# om_p = [promote(ω,p...)...]
# 	om_p = vcat(ω,p)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		Srvol = S_rvol(geom,grid)
# 		ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 		minds = geom.material_inds
# 		nngs = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fnn̂gs),[εᵥ,])
# 		return flat(εₛ(nngs,dropgrad(ps),dropgrad(minds),Srvol))  # new spatially smoothed ε tensor array
# 	end
# 	return reinterpret(SMatrix{3,3,Float64,9},reshape(arr_flat,(9*grid.Nx,grid.Ny)))
# end
#
# function ngvdₛ(ω::T1,p::AbstractVector{T2},f_geom::F,grid::Grid{ND}) where {ND,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	# om_p = [promote(ω,p...)...]
# 	om_p = vcat(ω,p)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		Srvol = S_rvol(geom,grid)
# 		ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 		minds = geom.material_inds
# 		ngvds = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fnĝvds),[εᵥ,])
# 		return flat(εₛ(ngvds,dropgrad(ps),dropgrad(minds),Srvol))  # new spatially smoothed ε tensor array
# 	end
# 	return reinterpret(SMatrix{3,3,Float64,9},reshape(arr_flat,(9*grid.Nx,grid.Ny)))
# end
#
# function χ⁽²⁾ₛ(ω::T1,p::AbstractVector{T2},f_geom::F,grid::Grid{ND}) where {ND,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	# om_p = [promote(ω,p...)...]
# 	om_p = vcat(ω,p)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		Srvol = S_rvol(geom,grid)
# 		ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 		minds = geom.material_inds
# 		χ⁽²⁾s = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fχ⁽²⁾s),[εᵥ,])
# 		return flat(εₛ(ngvds,dropgrad(ps),dropgrad(minds),Srvol))  # new spatially smoothed ε tensor array
# 	end
# 	return reinterpret(SMatrix{3,3,Float64,9},reshape(arr_flat,(9*grid.Nx,grid.Ny)))
# end
#
# function εₛ_nngₛ_ngvdₛ(ω::T1,p::AbstractVector{T2},f_geom::F,grid::Grid{ND}) where {ND,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	# om_p = [promote(ω,p...)...]
# 	om_p = vcat(ω,p)
# 	Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		Srvol = S_rvol(geom,grid)
# 		ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 		minds = geom.material_inds
# 		es = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fεs),[εᵥ,])		# dielectric tensors for each material, vacuum permittivity tensor app
# 		nngs = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fnn̂gs),[εᵥ,])
# 		ngvds = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fnĝvds),[εᵥ,])
# 		return εₛ(es,dropgrad(ps),dropgrad(minds),Srvol), εₛ(nngs,dropgrad(ps),dropgrad(minds),Srvol), εₛ(ngvds,dropgrad(ps),dropgrad(minds),Srvol)
# 	end
# 	return reinterpret(SMatrix{3,3,Float64,9},reshape(arr_flat,(9*grid.Nx,grid.Ny)))
# end
#
# function εₛ⁻¹(ω::T1,p::AbstractVector{T2},f_geom::F;ms::ModeSolver) where {T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	# om_p = [promote(ω,p...)...]
# 	om_p = vcat(ω,p)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		Srvol = S_rvol(geom;ms)
# 		es = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fεs),[εᵥ,])
# 		eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# 		return flat(εₛ⁻¹(es,eis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol))  # new spatially smoothed ε⁻¹ tensor array
# 	end
# 	return reinterpret(SMatrix{3,3,Float64,9},reshape(arr_flat,(9*grid.Nx,grid.Ny)))
# end
#
# function εₛ⁻¹(ω::T1,p::AbstractVector{T2},f_geom::F,grid::Grid{ND}) where {ND,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	# om_p = [promote(ω,p...)...]
# 	om_p = vcat(ω,p)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		Srvol = S_rvol(geom,grid)
# 		ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 		minds = geom.material_inds
# 		es = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fεs),[εᵥ,])
# 		eis = inv.(es)	# corresponding list of inverse dielectric tensors for each material
# 		return flat(εₛ⁻¹(es,eis,dropgrad(ps),dropgrad(minds),Srvol))  # new spatially smoothed ε⁻¹ tensor array
# 	end
# 	return reinterpret(SMatrix{3,3,Float64,9},reshape(arr_flat,(9*grid.Nx,grid.Ny)))
# end
#
# function nngₛ⁻¹(ω::T1,p::AbstractVector{T2},f_geom::F,grid::Grid{ND}) where {ND,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	# om_p = [promote(ω,p...)...]
# 	om_p = vcat(ω,p)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		Srvol = S_rvol(geom,grid)
# 		ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 		minds = geom.material_inds
# 		nngs = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fnn̂gs),[εᵥ,])
# 		nngis = inv.(nngs)	# corresponding list of inverse dielectric tensors for each material
# 		return flat(εₛ⁻¹(nngs,nngis,dropgrad(ps),dropgrad(minds),Srvol))  # new spatially smoothed ε⁻¹ tensor array
# 	end
# 	return reinterpret(SMatrix{3,3,Float64,9},reshape(arr_flat,(9*grid.Nx,grid.Ny)))
# end
#
# function nngₛ⁻¹(ω::T1,p::AbstractVector{T2},f_geom::F;ms::ModeSolver) where {T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	# om_p = [promote(ω,p...)...]
# 	om_p = vcat(ω,p)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		Srvol = S_rvol(geom;ms)
# 		nngs = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fnn̂gs),[εᵥ,])
# 		nngis = inv.(nngs)	# corresponding list of inverse dielectric tensors for each material
# 		return flat(εₛ⁻¹(nngs,nngis,dropgrad(ms.sinds_proc),dropgrad(ms.minds),Srvol))  # new spatially smoothed ε⁻¹ tensor array
# 	end
# 	return reinterpret(SMatrix{3,3,Float64,9},reshape(arr_flat,(9*grid.Nx,grid.Ny)))
# end
#
# function ngvdₛ⁻¹(ω::T1,p::AbstractVector{T2},f_geom::F,grid::Grid{ND}) where {ND,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	# om_p = [promote(ω,p...)...]
# 	om_p = vcat(ω,p)
# 	arr_flat = Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		Srvol = S_rvol(geom,grid)
# 		ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 		minds = geom.material_inds
# 		ngvds = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fnĝvds),[εᵥ,])
# 		ngvdis = inv.(ngvds)	# corresponding list of inverse dielectric tensors for each material
# 		return flat(εₛ⁻¹(ngvds,ngvdis,dropgrad(ps),dropgrad(minds),Srvol))  # new spatially smoothed ε⁻¹ tensor array
# 	end
# 	return reinterpret(SMatrix{3,3,Float64,9},reshape(arr_flat,(9*grid.Nx,grid.Ny)))
# end
#
# function εₛ⁻¹_nngₛ⁻¹_ngvdₛ⁻¹(ω::T1,p::AbstractVector{T2},f_geom::F,grid::Grid{ND}) where {ND,T1<:Real,T2<:Real,F}
# 	n_p = length(p)
# 	# om_p = [promote(ω,p...)...]
# 	om_p = vcat(ω,p)
# 	Zygote.forwarddiff(om_p) do om_p
# 		geom = f_geom(om_p[2:n_p+1])
# 		Srvol = S_rvol(geom,grid)
# 		ps = Zygote.@ignore(proc_sinds(geom.shapes,grid))
# 		minds = geom.material_inds
# 		es = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fεs),[εᵥ,])
# 		nngs = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fnn̂gs),[εᵥ,])
# 		ngvds = vcat(map(f->SMatrix{3,3}(f( inv(first(om_p)) )),geom.fnĝvds),[εᵥ,])
# 		return εₛ⁻¹(es,inv.(es),dropgrad(ps),dropgrad(minds),Srvol), εₛ⁻¹(nngs,inv.(nngs),dropgrad(ps),dropgrad(minds),Srvol), εₛ⁻¹(ngvds,inv.(ngvds),dropgrad(ps),dropgrad(minds),Srvol)
# 	end
# 	return reinterpret(SMatrix{3,3,Float64,9},reshape(arr_flat,(9*grid.Nx,grid.Ny)))
# end


# make_KDTree(shapes::AbstractVector{<:Shape}) = (tree = @ignore (KDTree(shapes)); tree)::KDTree
#
# function εₛ(shapes::AbstractVector{<:GeometryPrimitives.Shape{2}},x::Real,y::Real; δx::Real,δy::Real,npix_sm::Int=1)
# 	tree = make_KDTree(shapes)
# 	s1 = @ignore(findfirst(SVector(x+δx/2.,y+δy/2.),tree))
#     s2 = @ignore(findfirst(SVector(x+δx/2.,y-δy/2.),tree))
#     s3 = @ignore(findfirst(SVector(x-δx/2.,y-δy/2.),tree))
#     s4 = @ignore(findfirst(SVector(x-δx/2.,y+δy/2.),tree))
#
#     ε1 = isnothing(s1) ? εᵥ : s1.data
#     ε2 = isnothing(s2) ? εᵥ : s2.data
#     ε3 = isnothing(s3) ? εᵥ : s3.data
#     ε4 = isnothing(s4) ? εᵥ : s4.data
#
#     if (ε1==ε2==ε3==ε4)
#         return ε1
#     else
#         sinds = @ignore ( [ isnothing(ss) ? length(shapes)+1 : findfirst(isequal(ss),shapes) for ss in [s1,s2,s3,s4]] )
#         n_unique = @ignore( length(unique(sinds)) )
#         if n_unique==2
#             s_fg = @ignore(shapes[minimum(dropgrad(sinds))])
#             r₀,nout = surfpt_nearby([x; y], s_fg)
#             rvol = volfrac((SVector{2}(x-δx/2.,y-δy/2.), SVector{2}(x+δx/2.,y+δy/2.)),nout,r₀)
#             sind_bg = @ignore(maximum(dropgrad(sinds))) #max(sinds...)
#             ε_bg = sind_bg > length(shapes) ? εᵥ : shapes[sind_bg].data
#             return avg_param(
#                     s_fg.data,
#                     ε_bg,
#                     [nout[1];nout[2];0],
#                     rvol,)
#         else
#             return +(ε1,ε2,ε3,ε4)/4.
#         end
#     end
# end
#
#
#
# function make_εₛ⁻¹(shapes::Vector{<:Shape{N}};Δx::Real,Δy::Real,Δz::Real,Nx::Int,Ny::Int,Nz::Int,
# 	 	δx=Δx/Nx, δy=Δy/Ny, δz=Δz/Nz, x=( ( δx .* (0:(Nx-1))) .- Δx/2. ),
# 		y=( ( δy .* (0:(Ny-1))) .- Δy/2. ), z=( ( δz .* (0:(Nz-1))) .- Δz/2. ) ) where N
#     tree = make_KDTree(shapes)
#     eibuf = Buffer(bounds(shapes[1])[1],3,3,Nx,Ny,Nz)
# 	# eibuf = Buffer(bounds(shapes[1])[1],3,3,Nx,Ny,Nz)
#     for i=1:Nx,j=1:Ny,kk=1:Nz
# 		# eps = εₛ(shapes,Zygote.dropgrad(tree),Zygote.dropgrad(g.x[i]),Zygote.dropgrad(g.y[j]),Zygote.dropgrad(g.δx),Zygote.dropgrad(g.δy))
# 		eps = εₛ(shapes,x[i],y[j];tree,δx,δy)
# 		epsi = inv(eps) # inv( (eps' + eps) / 2) # Hermitian(inv(eps))  # inv(Hermitian(eps)) #   # inv(eps)
#         eibuf[:,:,i,j,kk] = epsi #(epsi' + epsi) / 2
#     end
#     # return HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},T,5,5,Array{T,5}}( real(copy(eibuf)) )
# 	return HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()}}( real(copy(eibuf)) )
# end
#
# function make_εₛ⁻¹_fwd(shapes::Vector{<:Shape{N}};Δx::Real,Δy::Real,Δz::Real,Nx::Int,Ny::Int,Nz::Int,
# 	 	δx=Δx/Nx, δy=Δy/Ny, δz=Δz/Nz, x=( ( δx .* (0:(Nx-1))) .- Δx/2. ),
# 		y=( ( δy .* (0:(Ny-1))) .- Δy/2. ), z=( ( δz .* (0:(Nz-1))) .- Δz/2. ) ) where N
#     Zygote.forwarddiff(shapes) do shapes
# 		make_εₛ⁻¹(shapes;Δx,Δy,Δz,Nx,Ny,Nz,δx,δy,δz,x,y,z)
# 	end
# end
