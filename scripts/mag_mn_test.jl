using LinearAlgebra, StaticArrays, ChainRulesCore, ChainRules, Zygote, ForwardDiff, 
    FiniteDifferences, FFTW, Tullio, Test, BenchmarkTools
using OptiMode: Grid, N, g⃗, _fftaxes
using Zygote: Buffer
# mutating version without Zygote Buffer
function mag_m_n0(k⃗::SVector{3,T1},g⃗s::AbstractArray{SVector{3,T2}}) where {T1<:Real,T2<:Real}
	T = promote_type(T1,T2)
	local ẑ = SVector{3,T}(0,0,1)
	local ŷ = SVector{3,T}(0,1,0)
	n = Array{SVector{3,T}}(undef,size(g⃗s)) #similar(g⃗s,size(g⃗s))
	m = Array{SVector{3,T}}(undef,size(g⃗s)) #similar(g⃗s,size(g⃗s))
	mag = Array{T}(undef,size(g⃗s)) #similar(zeros(T,size(g⃗s)),size(g⃗s))
	@fastmath @inbounds for i ∈ eachindex(g⃗s)
		@inbounds kpg::SVector{3,T} = k⃗ - g⃗s[i]
		@inbounds mag[i] = norm(kpg)
		@inbounds n[i] =   ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : ŷ
		@inbounds m[i] =  normalize( cross( n[i], kpg )  )
	end
	return mag, m, n
end
function mag_m_n0(kz::T,g⃗s::AbstractArray) where T <: Real
	mag_m_n0(SVector{3,T}(0.,0.,kz),g⃗s)
end

# mutating version with Zygote Buffer
function mag_m_n1(k⃗::SVector{3,T1},g⃗s::AbstractArray{SVector{3,T2}}) where {T1<:Real,T2<:Real}
	# for iz ∈ axes(g⃗s,3), iy ∈ axes(g⃗s,2), ix ∈ axes(g⃗s,1) #, l in 0:0
    T = promote_type(T1,T2)
	local ẑ = SVector{3,T}(0.,0.,1.)
	local ŷ = SVector{3,T}(0.,1.,0.)
	n = Buffer(zeros(SVector{3,T},2),size(g⃗s))
	m = Buffer(zeros(SVector{3,T},2),size(g⃗s))
	mag = Buffer(zeros(T,size(g⃗s)),size(g⃗s))
	@fastmath @inbounds for i ∈ eachindex(g⃗s)
		@inbounds kpg::SVector{3,T} = k⃗ - g⃗s[i]
		@inbounds mag[i] = norm(kpg)
		@inbounds n[i] =  ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : ŷ
		# @inbounds n[i] =   !iszero(kpg[1]) || !iszero(kpg[2]) ?  normalize( cross( ẑ, kpg ) ) : ŷ
		@inbounds m[i] =  normalize( cross( n[i], kpg )  )
	end
	return copy(mag), copy(m), copy(n) # HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,Float64,copy(m))), HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,Float64,copy(n)))
end
function mag_m_n1(kz::T,g⃗s::AbstractArray) where T <: Real
	mag_m_n1(SVector{3,T}(0.,0.,kz),g⃗s)
end

# Tullio version
function mag_m_n2(k⃗::SVector{3,T1},g⃗s::AbstractArray{SVector{3,T2},2}) where {T1<:Real,T2<:Real}
	# g⃗ₜ_zero_mask = Zygote.@ignore(  sum(abs2,g⃗s[1:2,:,:];dims=1)[1,:,:] .> 0. );
    T = promote_type(T1,T2)
    g⃗s_flat = copy(reinterpret(reshape,T2,g⃗s))
	g⃗ₜ_zero_mask = Zygote.@ignore(  sum(abs2,g⃗s_flat[1:2,:,:];dims=1)[1,:,:] .> 0. );
	g⃗ₜ_zero_mask! = Zygote.@ignore( .!(g⃗ₜ_zero_mask) );
	local ŷ = [0.; 1. ;0.]
	local zxinds = [2; 1; 3]
	local zxscales = [-1; 1. ;0.]
	local xinds1 = [2; 3; 1]
	local xinds2 = [3; 1; 2]
	@tullio kpg[a,ix,iy] := k⃗[a] - g⃗s_flat[a,ix,iy] fastmath=false
	@tullio mag[ix,iy] := sqrt <| kpg[a,ix,iy]^2 fastmath=false
	@tullio nt[a,ix,iy] := zxscales[a] * kpg[zxinds[a],ix,iy] * g⃗ₜ_zero_mask[ix,iy] + ŷ[a] * g⃗ₜ_zero_mask![ix,iy]  nograd=(zxscales,zxinds,ŷ,g⃗ₜ_zero_mask,g⃗ₜ_zero_mask!) fastmath=false
	@tullio nmag[ix,iy] := sqrt <| nt[a,ix,iy]^2 fastmath=false
	@tullio n[a,ix,iy] := nt[a,ix,iy] / nmag[ix,iy] fastmath=false
	@tullio mt[a,ix,iy] := n[xinds1[a],ix,iy] * kpg[xinds2[a],ix,iy] - kpg[xinds1[a],ix,iy] * n[xinds2[a],ix,iy] nograd=(xinds1,xinds2) fastmath=false
	@tullio mmag[ix,iy] := sqrt <| mt[a,ix,iy]^2 fastmath=false
	@tullio m[a,ix,iy] := mt[a,ix,iy] / mmag[ix,iy] fastmath=false
	# return mag, m, n
    return mag, reinterpret(reshape,SVector{3,T},m), reinterpret(reshape,SVector{3,T},n)
end
function mag_m_n2(k⃗::SVector{3,T1},g⃗s::AbstractArray{SVector{3,T2},3}) where {T1<:Real,T2<:Real}
	# g⃗ₜ_zero_mask = Zygote.@ignore(  sum(abs2,g⃗s[1:2,:,:,:];dims=1)[1,:,:,:] .> 0. );
    T = promote_type(T1,T2)
    g⃗s_flat = copy(reinterpret(reshape,T2,g⃗s))
	g⃗ₜ_zero_mask = Zygote.@ignore(  sum(abs2,g⃗s_flat[1:2,:,:,:];dims=1)[1,:,:,:] .> 0. );
	g⃗ₜ_zero_mask! = Zygote.@ignore( .!(g⃗ₜ_zero_mask) );
	local ŷ = [0.; 1. ;0.]
	local zxinds = [2; 1; 3]
	local zxscales = [-1; 1. ;0.]
	local xinds1 = [2; 3; 1]
	local xinds2 = [3; 1; 2]
	@tullio kpg[a,ix,iy,iz] := k⃗[a] - g⃗s_flat[a,ix,iy,iz] fastmath=false
	@tullio mag[ix,iy,iz] := sqrt <| kpg[a,ix,iy,iz]^2 fastmath=false
	@tullio nt[a,ix,iy,iz] := zxscales[a] * kpg[zxinds[a],ix,iy,iz] * g⃗ₜ_zero_mask[ix,iy,iz] + ŷ[a] * g⃗ₜ_zero_mask![ix,iy,iz]  nograd=(zxscales,zxinds,ŷ,g⃗ₜ_zero_mask,g⃗ₜ_zero_mask!) fastmath=false
	@tullio nmag[ix,iy,iz] := sqrt <| nt[a,ix,iy,iz]^2 fastmath=false
	@tullio n[a,ix,iy,iz] := nt[a,ix,iy,iz] / nmag[ix,iy,iz] fastmath=false
	@tullio mt[a,ix,iy,iz] := n[xinds1[a],ix,iy,iz] * kpg[xinds2[a],ix,iy,iz] - kpg[xinds1[a],ix,iy,iz] * n[xinds2[a],ix,iy,iz] nograd=(xinds1,xinds2) fastmath=false
	@tullio mmag[ix,iy,iz] := sqrt <| mt[a,ix,iy,iz]^2 fastmath=false
	@tullio m[a,ix,iy,iz] := mt[a,ix,iy,iz] / mmag[ix,iy,iz] fastmath=false
	# return mag, m, n
    return mag, reinterpret(reshape,SVector{3,T},m), reinterpret(reshape,SVector{3,T},n)
end
function mag_m_n2(kz::T,g⃗s::AbstractArray) where T <: Real
	mag_m_n2(SVector{3,T}(0.,0.,kz),g⃗s)
end

# map-based mag_m_n
function mag_m_n3(k⃗::SVector{3,T1},g⃗s::AbstractArray{SVector{3,T2}}) where {T1<:Real,T2<:Real}
    T = promote_type(T1,T2)
	local ẑ = SVector{3,T}(0,0,1)
	local ŷ = SVector{3,T}(0,1,0)
	magmn = mapreduce(hcat,g⃗s) do gg
		kpg = k⃗ - gg
		mag = norm(kpg)
		n =  !iszero(kpg[1]) || !iszero(kpg[2])  ?  normalize( cross( ẑ, kpg ) ) : ŷ
		m =  normalize( cross( n, kpg )  )
        return vcat(mag,m,n) #(mag,m,n) #
	end
    mag = reshape(magmn[1,:],size(g⃗s))
    m =  reinterpret(SVector{3,T},magmn[2:4,:])
    n =  reinterpret(SVector{3,T},magmn[5:7,:])
    # m =  map(x->SVector{3,T}(x),eachcol(magmn[2:4,:]))
    # n =  map(x->SVector{3,T}(x),eachcol(magmn[5:7,:]))
    return mag, reshape(m,size(g⃗s)), reshape(n,size(g⃗s))
end
function mag_m_n3(kz::T,g⃗s::AbstractArray) where T <: Real
	mag_m_n3(SVector{3,T}(0.,0.,kz),g⃗s)
end

# map-based mag_m_n with fn barrier
function mag_m_n_single(k⃗::SVector{3,T1},g::SVector{3,T2}) where {T1<:Real,T2<:Real}
    T = promote_type(T1,T2)
    local ẑ = SVector{3,T}(0,0,1)
	local ŷ = SVector{3,T}(0,1,0)
    kpg = k⃗ - g
	mag = norm(kpg)
	n =   !iszero(kpg[1]) || !iszero(kpg[2]) ?  normalize( cross( ẑ, kpg ) ) : ŷ
	m =  normalize( cross( n, kpg )  )
    return vcat(mag,m,n)
end
function mag_m_n4(k⃗::SVector{3,T1},g⃗s::AbstractArray{SVector{3,T2}}) where {T1<:Real,T2<:Real}
    T = promote_type(T1,T2)
	magmn = mapreduce(gg->mag_m_n_single(k⃗,gg),hcat,g⃗s) 
    mag = reshape(magmn[1,:],size(g⃗s))
    m =  reinterpret(SVector{3,T},magmn[2:4,:])
    n =  reinterpret(SVector{3,T},magmn[5:7,:])
    # m =  map(x->SVector{3,T}(x),eachcol(magmn[2:4,:]))
    # n =  map(x->SVector{3,T}(x),eachcol(magmn[5:7,:]))
    return mag, reshape(m,size(g⃗s)), reshape(n,size(g⃗s))
end
function mag_m_n4(kz::T,g⃗s::AbstractArray) where T <: Real
	mag_m_n4(SVector{3,T}(0.,0.,kz),g⃗s)
end

##
kz = 2.2
grid = Grid(6.0,4.0,256,128)
mag0, m0, n0 = mag_m_n0(kz,g⃗(grid))
mag1, m1, n1 = mag_m_n1(kz,g⃗(grid))
mag2, m2, n2 = mag_m_n2(kz,g⃗(grid))
mag3, m3, n3 = mag_m_n3(kz,g⃗(grid))
mag4, m4, n4 = mag_m_n4(kz,g⃗(grid))

@test (mag1 ≈ mag0) && (m1 ≈ m0) && (n1 ≈ n0)
@test (mag2 ≈ mag0) && (m2 ≈ m0) && (n2 ≈ n0)
@test (mag3 ≈ mag0) && (m3 ≈ m0) && (n3 ≈ n0)
@test (mag4 ≈ mag0) && (m4 ≈ m0) && (n4 ≈ n0)

gs1 = g⃗(grid)
ForwardDiff.derivative(kk->mag_m_n0(kk,g⃗(grid))[1],kz)
Zygote.gradient(kk->sum(Zygote.forwarddiff(x->copy(mag_m_n1(x,g⃗(grid))[1]),kk)),kz)
Zygote.gradient(kk->sum(mag_m_n1(kk,g⃗(grid))[1]),kz)
Zygote.gradient(kk->sum(mag_m_n2(kk,gs1)[1]),kz)
Zygote.gradient(kk->sum(mag_m_n3(kk,gs1)[1]),kz)
Zygote.gradient(kk->sum(mag_m_n4(kk,gs1)[1]),kz)
ForwardDiff.derivative(kk->mag_m_n0(kk,g⃗(grid)),kz)

Zygote.gradient(kk->sum(mag_m_n(kk,gs1)[1]),kz)
Zygote.gradient((kk,Dx,Dy)->sum(mag_m_n(kk,g⃗(Grid(Dx,Dy,128,64)))[1]),kz,6.0,4.0)
Zygote.gradient((kk,Dx,Dy)->sum(mag_m_n2(kk,g⃗(Grid(Dx,Dy,128,64)))[1]),kz,6.0,4.0)
ff_magmn2(kk_Dx_Dy) = sum(mag_m_n2(kk_Dx_Dy[1],g⃗(Grid(kk_Dx_Dy[2],kk_Dx_Dy[3],128,64)))[1])
@test Zygote.gradient(ff_magmn2,[2.2,6.0,4.0])[1] ≈ ForwardDiff.gradient(ff_magmn2,[2.2,6.0,4.0])


typeof(m0)
typeof(m1)
typeof(m2)
typeof(m3)
typeof(m4)

eltype(m0)
eltype(m1)
eltype(m2)
eltype(m3)
eltype(m4)












########################################################## 
##           old stuff
##########################################################

# Tullio version

# function mag_m_n2(k⃗::SVector{3,T},g⃗::AbstractArray) where T <: Real
# 	# g⃗ₜ_zero_mask = Zygote.@ignore(  sum(abs2,g⃗[1:2,:,:,:];dims=1)[1,:,:,:] .> 0. );
# 	g⃗ₜ_zero_mask = Zygote.@ignore(  sum(abs2,g⃗[1:2,:,:,:];dims=1)[1,:,:,:] .> 0. );
# 	g⃗ₜ_zero_mask! = Zygote.@ignore( .!(g⃗ₜ_zero_mask) );
# 	local ŷ = [0.; 1. ;0.]
# 	local zxinds = [2; 1; 3]
# 	local zxscales = [-1; 1. ;0.]
# 	local xinds1 = [2; 3; 1]
# 	local xinds2 = [3; 1; 2]
# 	@tullio kpg[ix,iy,iz] := k⃗[a] - g⃗[a,ix,iy,iz] fastmath=false
# 	@tullio mag[ix,iy,iz] := sqrt <| kpg[a,ix,iy,iz]^2 fastmath=false
# 	@tullio nt[ix,iy,iz,a] := zxscales[a] * kpg[zxinds[a],ix,iy,iz] * g⃗ₜ_zero_mask[ix,iy,iz] + ŷ[a] * g⃗ₜ_zero_mask![ix,iy,iz]  nograd=(zxscales,zxinds,ŷ,g⃗ₜ_zero_mask,g⃗ₜ_zero_mask!) fastmath=false
# 	@tullio nmag[ix,iy,iz] := sqrt <| nt[a,ix,iy,iz]^2 fastmath=false
# 	@tullio n[a,ix,iy,iz] := nt[a,ix,iy,iz] / nmag[ix,iy,iz] fastmath=false
# 	@tullio mt[a,ix,iy,iz] := n[xinds1[a],ix,iy,iz] * kpg[xinds2[a],ix,iy,iz] - kpg[xinds1[a],ix,iy,iz] * n[xinds2[a],ix,iy,iz] nograd=(xinds1,xinds2) fastmath=false
# 	@tullio mmag[ix,iy,iz] := sqrt <| mt[a,ix,iy,iz]^2 fastmath=false
# 	@tullio m[a,ix,iy,iz] := mt[a,ix,iy,iz] / mmag[ix,iy,iz] fastmath=false
# 	return mag, m, n
# end

# function mag_m_n2(kz::T,g⃗::AbstractArray) where T <: Real
# 	mag_m_n2(SVector{3,T}(0.,0.,kz),g⃗)
# end

# mutating version with Zygote Buffer

# function mag_m_n(k⃗::SVector{3,T},g⃗::AbstractArray{SVector{3,T2}}) where {T<:Real,T2<:Real}
# 	# for iz ∈ axes(g⃗,3), iy ∈ axes(g⃗,2), ix ∈ axes(g⃗,1) #, l in 0:0
# 	local ẑ = SVector{3,T}(0.,0.,1.)
# 	local ŷ = SVector{3,T}(0.,1.,0.)
# 	n = Buffer(g⃗,size(g⃗))
# 	m = Buffer(g⃗,size(g⃗))
# 	mag = Buffer(zeros(T,size(g⃗)),size(g⃗))
# 	@fastmath @inbounds for i ∈ eachindex(g⃗)
# 		@inbounds kpg::SVector{3,T} = k⃗ - g⃗[i]
# 		@inbounds mag[i] = norm(kpg)
# 		@inbounds n[i] =  ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : ŷ
# 		# @inbounds n[i] =   !iszero(kpg[1]) || !iszero(kpg[2]) ?  normalize( cross( ẑ, kpg ) ) : ŷ
# 		@inbounds m[i] =  normalize( cross( n[i], kpg )  )
# 	end
# 	return copy(mag), copy(m), copy(n) # HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,Float64,copy(m))), HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},T}(reinterpret(reshape,Float64,copy(n)))
# end

# mutating version without Zygote Buffer
# function mag_m_n(k⃗::SVector{3,T1},g⃗s::AbstractArray{SVector{3,T2}}) where {T1<:Real,T2<:Real}
# 	T = promote_type(T1,T2)
# 	local ẑ = SVector{3,T}(0,0,1)
# 	local ŷ = SVector{3,T}(0,1,0)
# 	n = similar(g⃗s,size(g⃗s))
# 	m = similar(g⃗s,size(g⃗s))
# 	mag = Array{T}(undef,size(g⃗s)) #similar(zeros(T,size(g⃗s)),size(g⃗s))
# 	@fastmath @inbounds for i ∈ eachindex(g⃗s)
# 		@inbounds kpg::SVector{3,T} = k⃗ - g⃗s[i]
# 		@inbounds mag[i] = norm(kpg)
# 		@inbounds n[i] =   ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : ŷ
# 		@inbounds m[i] =  normalize( cross( n[i], kpg )  )
# 	end
# 	return mag, m, n
# end

# # map-based mag_m_n

# function mag_m_n(k⃗::SVector{3,T1},g⃗s::AbstractArray{SVector{3,T2}}) where {T1<:Real,T2<:Real}
#     T = promote_type(T1,T2)
# 	local ẑ = SVector{3,T}(0,0,1)
# 	local ŷ = SVector{3,T}(0,1,0)
# 	magmn = mapreduce(hcat,g⃗s) do gg
# 		kpg = k⃗ - gg
# 		mag = norm(kpg)
# 		n =  !iszero(kpg[1]) || !iszero(kpg[2])  ?  normalize( cross( ẑ, kpg ) ) : ŷ
# 		m =  normalize( cross( n, kpg )  )
#         return (mag,m,n) #vcat(mag,m,n)
# 	end
#     mag = reshape(magmn[1,:],size(g⃗s))
#     m =  reinterpret(SVector{3,T},magmn[2:4,:])
#     n =  reinterpret(SVector{3,T},magmn[5:7,:])
#     # m =  map(x->SVector{3,T}(x),eachcol(magmn[2:4,:]))
#     # n =  map(x->SVector{3,T}(x),eachcol(magmn[5:7,:]))
#     return mag, reshape(m,size(g⃗s)), reshape(n,size(g⃗s))
# end

# function mag_m_n_single(k⃗::SVector{3,T1},g::SVector{3,T2}) where {T1<:Real,T2<:Real}
#     T = promote_type(T1,T2)
#     local ẑ = SVector{3,T}(0,0,1)
# 	local ŷ = SVector{3,T}(0,1,0)
#     kpg = k⃗ - g
# 	mag = norm(kpg)
# 	n =   !iszero(kpg[1]) || !iszero(kpg[2]) ?  normalize( cross( ẑ, kpg ) ) : ŷ
# 	m =  normalize( cross( n, kpg )  )
#     return vcat(mag,m,n)
# end

# map-based mag_mn version

# function mag_mn5(k⃗::SVector{3,T1},g⃗s::AbstractArray{SVector{3,T2}}) where {T1<:Real,T2<:Real}
#     T = promote_type(T1,T2)
# 	magmn = mapreduce(gg->mag_m_n_single(k⃗,gg),hcat,g⃗s) 
#     mag = reshape(magmn[1,:],size(g⃗s))
#     mn =  reshape(magmn[2:7,:],(3,2,size(g⃗s)...))
#     # m = reshape(reinterpret(SVector{3,T},copy(magmn[2:4,:])),(3,size(g⃗s)...))
#     # n = reshape(reinterpret(SVector{3,T},copy(magmn[5:7,:])),(3,size(g⃗s)...))
#     # m   = reshape(reinterpret(reshape,SVector{3,T},copy(magmn[2:4,:])),size(g⃗s))
#     # n   = reshape(reinterpret(reshape,SVector{3,T},copy(magmn[5:7,:])),size(g⃗s))
#     return mag,mn
# end

# map-based mag_m_n with fn barrier
# function mag_m_n(k⃗::SVector{3,T1},g⃗s::AbstractArray{SVector{3,T2}}) where {T1<:Real,T2<:Real}
#     T = promote_type(T1,T2)
# 	magmn = mapreduce(gg->mag_m_n_single(k⃗,gg),hcat,g⃗s) 
#     mag = reshape(magmn[1,:],size(g⃗s))
#     m =  reinterpret(SVector{3,T},magmn[2:4,:])
#     n =  reinterpret(SVector{3,T},magmn[5:7,:])
#     # m =  map(x->SVector{3,T}(x),eachcol(magmn[2:4,:]))
#     # n =  map(x->SVector{3,T}(x),eachcol(magmn[5:7,:]))
#     return mag, reshape(m,size(g⃗s)), reshape(n,size(g⃗s))
# end
