export ε_init, εₛ, εₛ⁻¹, ε_tensor, test_εs, εₘₐₓ, ñₘₐₓ, nₘₐₓ, surfpt_nearby2, make_εₛ⁻¹, make_εₛ⁻¹_fwd, make_KDTree

using Zygote: dropgrad, Buffer, @ignore

εᵥ = @SMatrix	[	1. 	0. 	0.
                	0. 	1. 	0.
                	0. 	0. 	1.  ]

function ε_tensor(n::T)::SMatrix{3,3,T,9} where T<:Real
    n² = n^2
	SMatrix{3,3,T,9}( 	n², 	0., 	0.,
						0., 	n², 	0.,
						0., 	0., 	n²	)
end

function ε_tensor(n₁::Real,n₂::Real,n₃::Real)::SMatrix{3,3}
	SMatrix{3,3}( 	n₁^2, 	0., 	0.,
					0., 	n₂^2, 	0.,
					0., 	0., 	n₃^2	)
end

ε_tensor(fₙ::Function) = λ -> ε_tensor(fₙ(λ))
ε_tensor(fₙ₁::Function,fₙ₂::Function,fₙ₃::Function) = λ -> ε_tensor(fₙ₁(λ),fₙ₂(λ),fₙ₃(λ))

function test_εs(n₁::T,n₂::T,n₃::T) where T<:Real
    ε₁ = ε_tensor(n₁)
    ε₂ = ε_tensor(n₂)
    ε₃ = ε_tensor(n₃)
    return ε₁, ε₂, ε₃
end


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

function avg_param(param1, param2, n12, rvol1)
	n = n12 / norm(n12)
	# n = normalize(n12) #n12 / norm(n12) #sqrt(sum2(abs2,n12))
    # Pick a vector that is not along n.
    h = any(iszero.(n)) ? n × normalize(iszero.(n)) :  n × SVector(1., 0. , 0.)
	v = n × h
    # Create a local Cartesian coordinate system.
    S = [n h v]  # unitary
    τ1 = τ_trans(transpose(S) * param1 * S)  # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans(transpose(S) * param2 * S)  # express param2 in S coordinates, and apply τ transform
    τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)  # volume-weighted average
    return S * τ⁻¹_trans(τavg) * transpose(S)  # apply τ⁻¹ and transform back to global coordinates
end

function update_corner_sinds!(ms::Modesolver,shapes::AbstractVector{<:GeometryPrimitives.Shape{2}})
	n_shapes = length(shapes)
	tree = KDTree(shapes)
	a = 0
	unq = [0,0]
	@avx for I ∈ eachindex(ms.M̂.xyzc)
		a = findfirst(isequal(findfirst(ms.M̂.xyzc[I],tree)),shapes)
		ms.M̂.corner_sinds[I] = isnothing(a) ? (n_shapes+1) : a
	end
	@avx for I ∈ eachindex(ms.M̂.xyz)
		unq .= [	ms.M̂.corner_sinds[I],
					ms.M̂.corner_sinds[I+CartesianIndex(1,0,0)],
					ms.M̂.corner_sinds[I+CartesianIndex(0,1,0)],
					ms.M̂.corner_sinds[I+CartesianIndex(1,1,0)],
					ms.M̂.corner_sinds[I+CartesianIndex(0,0,1)],
					ms.M̂.corner_sinds[I+CartesianIndex(1,0,1)],
					ms.M̂.corner_sinds[I+CartesianIndex(0,1,1)],
					ms.M̂.corner_sinds[I+CartesianIndex(1,1,1)],
		  		]
		unq = unique!( unq )
		a = length(unq)
		ms.M̂.corner_sinds_proc[I] = a==1 ? (unq[1],0,0,0,0,0,0,0) :
			( a==2 ?  (minimum(unq),maximum(unq),0,0,0,0,0,0)  :
				( 	ms.M̂.corner_sinds[I],
					ms.M̂.corner_sinds[I+CartesianIndex(1,0,0)],
					ms.M̂.corner_sinds[I+CartesianIndex(0,1,0)],
					ms.M̂.corner_sinds[I+CartesianIndex(1,1,0)],
					ms.M̂.corner_sinds[I+CartesianIndex(0,0,1)],
					ms.M̂.corner_sinds[I+CartesianIndex(1,0,1)],
					ms.M̂.corner_sinds[I+CartesianIndex(0,1,1)],
					ms.M̂.corner_sinds[I+CartesianIndex(1,1,1)],
				)
		)
	end
end

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


function εₘₐₓ(shapes::AbstractVector{<:GeometryPrimitives.Shape})
    maximum(vec([shapes[i].data[j,j] for j=1:3,i=1:size(shapes)[1]]))
end

ñₘₐₓ(ε⁻¹)::Float64 = √(maximum(3 ./ tr.(ε⁻¹)))
nₘₐₓ(ε)::Float64 = √(maximum(reinterpret(Float64,ε)))

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




# function avg_param(param1, param2, n12, rvol1)
#     n = n12 / norm(n12) #sqrt(sum2(abs2,n12))
#
#     # Pick a vector that is not along n.
#     if any(n .== 0)
#     	htemp1 = (n .== 0)
#     else
#     	htemp1 = SVector(1., 0. , 0.)
#     end
#
#     # Create two vectors that are normal to n and normal to each other.
#     htemp2 = n × htemp1
#     h = htemp2 / norm(htemp2) #sqrt(sum2(abs2,htemp2))
#     vtemp = n × h
#     v = vtemp / norm(vtemp) #sqrt(sum2(abs2,vtemp))
#     # Create a local Cartesian coordinate system.
#     S = [n h v]  # unitary
#
#     τ1 = τ_trans(transpose(S) * param1 * S)  # express param1 in S coordinates, and apply τ transform
#     τ2 = τ_trans(transpose(S) * param2 * S)  # express param2 in S coordinates, and apply τ transform
#
#     τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)  # volume-weighted average
#
#     return S * τ⁻¹_trans(τavg) * transpose(S)  # apply τ⁻¹ and transform back to global coordinates
# end



# function ε_init(shapes::AbstractVector{<:GeometryPrimitives.Shape};Δx=6.,Δy=4.,Nx=64,Ny=64)::Array{Float64,5}
#     g = MaxwellGrid(Δx,Δy,Nx,Ny)
#     tree = KDTree(shapes)
#     return Float64[ isnothing(findfirst([xx,yy],tree)) ? εᵥ[i,j] : findfirst([xx,yy],tree).data[i,j] for xx=g.x,yy=g.y,i=1:3,j=1:3 ]
# end
#
# function ε_init(shapes::AbstractVector{<:GeometryPrimitives.Shape},g::MaxwellData)::Array{Float64,5}
#     tree = KDTree(shapes)
#     return Float64[ isnothing(findfirst([xx,yy],tree)) ? εᵥ[i,j] : findfirst([xx,yy],tree).data[i,j] for xx=g.x,yy=g.y,i=1:3,j=1:3 ]
# end
#
# function ε_init(shapes::AbstractVector{<:GeometryPrimitives.Shape}, x::AbstractVector{Float64}, y::AbstractVector{Float64})::Array{Float64,5}
#     tree = KDTree(shapes)
#     return Float64[ isnothing(findfirst([xx,yy],tree)) ? εᵥ[i,j] : findfirst([xx,yy],tree).data[i,j] for xx=x,yy=y,i=1:3,j=1:3 ]
# end
#
#
#
# function surfpt_nearby2(x::AbstractVector, s::GeometryPrimitives.Sphere{2})
#     nout = x==s.c ? SVector(1.0,0.0) : # nout = e₁ for x == s.c
#                     normalize(x-s.c)
#     return s.c+s.r*nout, nout
# end
#
#
# f_onbnd(bin,absdin) =  (b = Zygote.@ignore bin; absd = Zygote.@ignore absdin; abs.(b.r.-absd) .≤ Base.rtoldefault(Float64) .* b.r)  # basically b.r .≈ absd but faster
# f_isout(b,absd) =  (isout = Zygote.@ignore ((b.r.<absd) .| f_onbnd(b,absd) ); isout)
# # isout(bin) =  (b = Zygote.@ignore bin; (b.r.<absd) .| (abs.(b.r.-absd) .≤ Base.rtoldefault(Float64) .* b.r))
# f_signs(d) =  (signs = Zygote.@ignore (copysign.(1.0,d)'); signs)
#
#
# # function surfpt_nearby2(x::Vector{Float64}, s::Sphere{2})
# #     nout = x==s.c ? SVector(1.0,0.0) : # nout = e₁ for x == s.c
# #                     normalize(x-s.c)
# #     return s.c+s.r*nout, nout
# # end
#
#
#
# function surfpt_nearby2(x, b::GeometryPrimitives.Box{2})
#     ax = inv(b.p)
#     n0 = b.p ./  [ sqrt(b.p[1,1]^2 + b.p[1,2]^2) sqrt(b.p[2,1]^2 + b.p[2,2]^2)  ]
#     d = Array(b.p * (x - b.c))
#     cosθ = diag(n0*ax)
#
#     n = n0 .* f_signs(d)
#     absd = abs.(d)
#     ∆ = (b.r .- absd) .* cosθ
#     # onbnd = abs.(dropgrad(b.r).-dropgrad(absd)) .≤ Base.rtoldefault(Float64) .* dropgrad(b.r)  # basically b.r .≈ absd but faster
#     # isout = (dropgrad(b.r).<dropgrad(absd)) .| dropgrad(onbnd)
#     onbnd = f_onbnd(b,absd)
#     isout = f_isout(b,absd)
#     projbnd =  all(.!isout .| onbnd)
#     onbnd_float =  Float64.(onbnd)
#     isout_float =  Float64.(isout)
#     # if ( ( abs(b.r[1]-absd[1]) ≤ Base.rtoldefault(Float64) * b.r[1] ) | (b.r[1]<absd[1]) ) | ( ( abs(b.r[2]-absd[2]) ≤ Base.rtoldefault(Float64) * b.r[2] ) | ( b.r[2] < absd[2] ) )
#     if count(isout) == 0
#         # = not(isout[1]) &  not(isout[2]) case, point is inside box
#         l∆x, i = findmin(∆)  # find closest face
#         nout = n[i,:]
#         ∆x = l∆x * nout
#     else
#         ∆x = n' * (∆ .* isout_float)
#         # = isout[1] | isout[2] case 1: at least one dimension outside or on boundry
#         if all(.!isout .| onbnd)
#             nout0 = n' * onbnd_float
#         else
#             nout0 = -∆x
#         end
#         nout = nout0 / norm(nout0)
#     end
#
#     return SVector{2,Float64}(x+∆x), SVector{2,Float64}(nout)
# end
#
# # surfpt_nearby2(x, p::Polygon) = surfpt_nearby(x, p::Polygon)
#
# f_onbnd_poly(pin,abs∆xe) =  (p = Zygote.@ignore pin; onbnd = Zygote.@ignore ( abs∆xe .≤ Base.rtoldefault(Float64) * maximum(abs.((-)(bounds(p)...))) ); onbnd)
# f_isout_poly(p,∆xe) =  (isout = Zygote.@ignore ( (∆xe.>0) .| f_onbnd_poly(p,abs.(∆xe))); isout)
# function surfpt_nearby2(x, s::Polygon{K}) where {K}
#     ∆xe = sum(s.n .* (x' .- s.v), dims=2)[:,1]  # Calculate the signed distances from x to edge lines.
#     abs∆xe = abs.(∆xe)
#     onbnd = f_onbnd_poly(s,abs∆xe) # abs∆xe .≤ Base.rtoldefault(Float64) * max(sz.data...)  # SVector{K}
#     isout = f_isout_poly(s,∆xe) #(∆xe.>0) .| onbnd  # SVector{K}
#     cout = count(isout)
#     if cout == 2  # x is outside two edges
#         ∆xv = x' .- s.v
#         l∆xv = hypot.(∆xv[:,1], ∆xv[:,2])
#         imin = argmin(l∆xv)
#         surf = s.v[imin,:]
#         imin₋₁ = mod1(imin-1,K)
#         if onbnd[imin] && onbnd[imin₋₁]  # x is very close to vertex imin
#             nout = s.n[imin,:] + s.n[imin₋₁,:]
#         else
#             nout = x - s.v[imin,:]
#         end
#         nout = normalize(nout)
#     else  # cout ≤ 1 or cout ≥ 3
#         imax = argmax(∆xe)
#         vmax, nmax = s.v[imax,:], s.n[imax,:]
#
#         ∆x = (nmax⋅(vmax-x)) .* nmax
#         surf = x + ∆x
#         nout = nmax
#     end
#     return surf, nout
# end
#
#
#
# function τ_trans(ε)
#     ε₁₁, ε₂₁, ε₃₁, ε₁₂, ε₂₂, ε₃₂, ε₁₃, ε₂₃, ε₃₃ = ε
#     return SMatrix{3,3,Float64,9}(
#         -1/ε₁₁, ε₂₁/ε₁₁, ε₃₁/ε₁₁,
#         ε₁₂/ε₁₁, ε₂₂ - ε₂₁*ε₁₂/ε₁₁, ε₃₂ - ε₃₁*ε₁₂/ε₁₁,
#         ε₁₃/ε₁₁, ε₂₃ - ε₂₁*ε₁₃/ε₁₁, ε₃₃ - ε₃₁*ε₁₃/ε₁₁
#     )
# end
#
# function τ⁻¹_trans(τ)
#     τ₁₁, τ₂₁, τ₃₁, τ₁₂, τ₂₂, τ₃₂, τ₁₃, τ₂₃, τ₃₃ = τ
#     return SMatrix{3,3,Float64,9}(
#         -1/τ₁₁, -τ₂₁/τ₁₁, -τ₃₁/τ₁₁,
#         -τ₁₂/τ₁₁, τ₂₂ - τ₂₁*τ₁₂/τ₁₁, τ₃₂ - τ₃₁*τ₁₂/τ₁₁,
#         -τ₁₃/τ₁₁, τ₂₃ - τ₂₁*τ₁₃/τ₁₁, τ₃₃ - τ₃₁*τ₁₃/τ₁₁
#     )
# end
#
# # function kottke_avg_param(param1::SMat3Complex, param2::SMat3Complex, n12::SVec3Float, rvol1::Real)
# function avg_param(param1, param2, n12, rvol1)
#     n = n12 / norm(n12) #sqrt(sum2(abs2,n12))
#
#     # Pick a vector that is not along n.
#     if any(n .== 0)
#     	htemp1 = (n .== 0)
#     else
#     	htemp1 = SVector(1., 0. , 0.)
#     end
#
#     # Create two vectors that are normal to n and normal to each other.
#     htemp2 = n × htemp1
#     h = htemp2 / norm(htemp2) #sqrt(sum2(abs2,htemp2))
#     vtemp = n × h
#     v = vtemp / norm(vtemp) #sqrt(sum2(abs2,vtemp))
#     # Create a local Cartesian coordinate system.
#     S = [n h v]  # unitary
#
#     τ1 = τ_trans(transpose(S) * param1 * S)  # express param1 in S coordinates, and apply τ transform
#     τ2 = τ_trans(transpose(S) * param2 * S)  # express param2 in S coordinates, and apply τ transform
#
#     τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)  # volume-weighted average
#
#     return S * τ⁻¹_trans(τavg) * transpose(S)  # apply τ⁻¹ and transform back to global coordinates
# end
#
# function εₛ2(shapes::AbstractVector{<:GeometryPrimitives.Shape},tree::KDTree,x::Real,y::Real,δx::Real,δy::Real)::Array{Float64,2} #;npix_sm::Int=1)::Array{Float64,2}
#     # x1,y1 = x+npix_sm*δx/2.,y+npix_sm*δy/2.
#     # x2,y2 = x+npix_sm*δx/2.,y-npix_sm*δy/2.
#     # x3,y3 = x-npix_sm*δx/2.,y-npix_sm*δy/2.
#     # x4,y4 = x-npix_sm*δx/2.,y+npix_sm*δy/2.
#
#     x1,y1 = x+δx/2.,y+δy/2.
#     x2,y2 = x+δx/2.,y-δy/2.
#     x3,y3 = x-δx/2.,y-δy/2.
#     x4,y4 = x-δx/2.,y+δy/2.
#
#     s1 = findfirst([x1,y1],tree)
#     s2 = findfirst([x2,y2],tree)
#     s3 = findfirst([x3,y3],tree)
#     s4 = findfirst([x4,y4],tree)
#
#     ε1 = isnothing(s1) ? εᵥ : s1.data
#     ε2 = isnothing(s2) ? εᵥ : s2.data
#     ε3 = isnothing(s3) ? εᵥ : s3.data
#     ε4 = isnothing(s4) ? εᵥ : s4.data
#
#     if (ε1==ε2==ε3==ε4)
#         return ε1
#     else
#         sinds = Zygote.@ignore ( [ isnothing(ss) ? length(shapes)+1 : findfirst(isequal(ss),shapes) for ss in [s1,s2,s3,s4]] )
#         n_unique = Zygote.@ignore( length(unique(sinds)) )
#         if n_unique==2
#             s_fg = shapes[minimum(dropgrad(sinds))]
#             r₀,nout = surfpt_nearby2([x; y], s_fg)
#             # bndry_pxl[i,j] = 1
#             # nouts[i,j,:] = nout
#             vxl = (SVector{2,Float64}(x3,y3), SVector{2,Float64}(x1,y1))
#             rvol = volfrac(vxl,nout,r₀)
#             sind_bg = maximum(dropgrad(sinds)) #max(sinds...)
#             ε_bg = sind_bg > length(shapes) ? εᵥ : shapes[sind_bg].data
#             return avg_param(
#                     s_fg.data,
#                     ε_bg,
#                     [nout[1];nout[2];0],
#                     rvol,)
#         else
#             return (ε1+ε2+ε3+ε4)/4.
#         end
#     end
# end
#
# function εₛ(shapes::AbstractVector{<:GeometryPrimitives.Shape},Δx=6.,Δy=4.,Nx=64,Ny=64;npix_sm::Int=1)
#     g=MaxwellGrid(Δx,Δy,Nx,Ny)
#     tree = KDTree(shapes)
#     ε_sm = zeros(Float64,g.Nx,g.Ny,3,3)
#     for ix=1:g.Nx, iy=1:g.Ny
#         ε_sm[ix,iy,:,:] = εₛ(shapes,tree,g.x[ix],g.y[iy],g.δx,g.δy;npix_sm)
#     end
#     return ε_sm
# end
#
# function εₛ(shapes::AbstractVector{<:GeometryPrimitives.Shape},g::MaxwellGrid;npix_sm::Int=1)
#     tree = KDTree(shapes)
#     # ε_sm = copy(reshape(  [εₛ(shapes,tree,g.x[i],g.y[j],g.δx,g.δy) for i=1:g.Nx,j=1:g.Ny] , (g.Nx,g.Ny,1)) )
#     ε_sm = [εₛ(shapes,tree,g.x[ix],g.y[iy],g.δx,g.δy;npix_sm)[a,b] for ix=1:g.Nx,iy=1:g.Ny,a=1:3,b=1:3]
# end
#
# function εₛ⁻¹(shapes::AbstractVector{<:GeometryPrimitives.Shape},Δx=6.,Δy=4.,Nx=64,Ny=64;npix_sm::Int=1)
#     g=MaxwellGrid(Δx,Δy,Nx,Ny)
#     tree = KDTree(shapes)
#     # ε_sm_inv = copy(reshape( [inv(εₛ(shapes,tree,g.x[i],g.y[j],g.δx,g.δy)) for i=1:g.Nx,j=1:g.Ny], (g.Nx,g.Ny,1)) )
#     # ε_sm_inv = [inv(εₛ(shapes,tree,g.x[ix],g.y[iy],g.δx,g.δy))[a,b] for a=1:3,b=1:3,ix=1:g.Nx,iy=1:g.Ny,iz=1:g.Nz]
#     ε_sm_inv = [inv(εₛ(shapes,tree,g.x[ix],g.y[iy],g.δx,g.δy;npix_sm))[a,b] for ix=1:g.Nx,iy=1:g.Ny,a=1:3,b=1:3]
# end
#
# # function εₛ⁻¹(shapes::AbstractVector{T},g::MaxwellGrid) where T <: GeometryPrimitives.Shape{2,4,D} where D
# function εₛ⁻¹(shapes::AbstractVector{<:GeometryPrimitives.Shape},g::MaxwellGrid;npix_sm::Int=1)
#     tree = KDTree(shapes)
#     # ε_sm_inv = copy(reshape( [inv(εₛ(shapes,tree,g.x[i],g.y[j],g.δx,g.δy)) for i=1:g.Nx,j=1:g.Ny], (g.Nx,g.Ny,1)) )
#     # ε_sm_inv = [inv(εₛ(shapes,tree,g.x[ix],g.y[iy],g.δx,g.δy))[a,b] for a=1:3,b=1:3,ix=1:g.Nx,iy=1:g.Ny,iz=1:g.Nz]
#     ε_sm_inv = [inv(εₛ(shapes,tree,g.x[ix],g.y[iy],g.δx,g.δy;npix_sm))[a,b] for ix=1:g.Nx,iy=1:g.Ny,a=1:3,b=1:3]
# end
#
