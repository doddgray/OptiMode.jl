# using GeometryPrimitives: orthoaxes
export MaxwellGrid, MaxwellData, KVec, t2c, c2t, kcross_c2t, kcross_t2c, ucross_t2c, d_from_H, e_from_d, h_from_H, H_from_e, flatten, unflatten, M̂ₖ, ∂ₖM̂ₖ, e_from_H, d_from_e, H_from_d, P̂ₖ

twopi = 1.                  # still not sure whether to include factors of 2π, for now use this global to control all instances

function k_mn(k::SArray{Tuple{3},Float64,1,3})
    mag = sqrt(k[1]^2 + k[2]^2 + k[3]^2)
    if mag==0
        n = SVector(0.,1.,0.)
        m = SVector(0.,0.,1.)
    else
        if k[1]==0. && k[2]==0.    # put n in the y direction if k+G is in z
            n = SVector(0.,1.,0.)
        else                                # otherwise, let n = z x (k+G), normalized
            temp = SVector(0.,0.,1.) × k
            n = temp / sqrt( temp[1]^2 + temp[2]^2 + temp[3]^2 )
        end
    end

    # m = n x (k+G), normalized
    mtemp = n × k
    m = mtemp / sqrt( mtemp[1]^2 + mtemp[2]^2 + mtemp[3]^2 )
    return m,n
end

struct KVec
    k::SVector{3,Float64}   # 3-vector
    mag::Float64            # vector magnitude
    m::SVector{3,Float64}   # k vector normal with even parity in y 
    n::SVector{3,Float64}   # k vector normal with even parity in x
end

KVec(k::SVector{3,Float64}) = KVec( 
    k,
    sqrt(sum(ki^2 for ki in k)),
    k_mn(k)...,
)

struct MaxwellGrid
    Δx::Float64
    Δy::Float64
    Nx::Int32
    Ny::Int32
    δx::Float64
    δy::Float64
    Gx::SVector{3,Float64}
    Gy::SVector{3,Float64}
    x::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    y::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    gx::Array{SArray{Tuple{3},Float64,1,3},1}
    gy::Array{SArray{Tuple{3},Float64,1,3},1}
    𝓕::FFTW.cFFTWPlan
    𝓕⁻¹::AbstractFFTs.ScaledPlan
end

MaxwellGrid(Δx::Real,Δy::Real,Nx::Int,Ny::Int) = MaxwellGrid( 
    Δx,
    Δy,
    Nx,
    Ny,
    Δx / Nx,    # δx
    Δy / Ny,    # δy
    SVector(1., 0., 0.),      # Gx
    SVector(0., 1., 0.),      # Gy
    ( ( Δx / Nx ) .* (0:(Nx-1))) .- Δx/2.,  # x
    ( ( Δy / Ny ) .* (0:(Ny-1))) .- Δy/2.,  # y
    [SVector(ggx, 0., 0.) for ggx in fftfreq(Nx,Nx/Δx)],     # gx
    [SVector(0., ggy, 0.) for ggy in fftfreq(Ny,Ny/Δy)],     # gy
    plan_fft(rand(ComplexF64, (3,Nx,Ny)),(2,3)),    # planned DFT operator 𝓕
    plan_ifft(rand(ComplexF64, (3,Nx,Ny)),(2,3)),   # planned inverse DFT operator 𝓕⁻¹
)

struct MaxwellData
    k::SVector{3,Float64}
    grid::MaxwellGrid
    Δx::Float64
    Δy::Float64
    Nx::Int32
    Ny::Int32
    δx::Float64
    δy::Float64
    Gx::SVector{3,Float64}
    Gy::SVector{3,Float64}
    x::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    y::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    gx::Array{SArray{Tuple{3},Float64,1,3},1}
    gy::Array{SArray{Tuple{3},Float64,1,3},1}
    kpG::Array{KVec,2}
    𝓕::FFTW.cFFTWPlan
    𝓕⁻¹::AbstractFFTs.ScaledPlan
end

function kpG(k::SVector{3,Float64},g::MaxwellGrid)::Array{KVec,2}
    [KVec(k-gx-gy) for gx=g.gx, gy=g.gy]
end

MaxwellData(k::SVector{3,Float64},g::MaxwellGrid) = MaxwellData(   
    k,
    g,
    g.Δx,
    g.Δy,
    g.Nx,
    g.Ny,
    g.δx,       # δx
    g.δy,       # δy
    g.Gx,       # Gx
    g.Gy,       # Gy
    g.x,        # x
    g.y,        # y
    g.gx,
    g.gy,
    kpG(k,g),  # kpG
    g.𝓕,
    g.𝓕⁻¹,
)

MaxwellData(kz::Float64,g::MaxwellGrid) = MaxwellData(SVector(0.,0.,kz),g)
MaxwellData(k::SVector{3,Float64},Δx::Real,Δy::Real,Nx::Int,Ny::Int) = MaxwellData(k,MaxwellGrid(Δx,Δy,Nx,Ny))


"""
    t2c: v (transverse vector) → a (cartesian vector)
"""
function t2c(v::SVector{2,ComplexF64},k::KVec)::SVector{3,ComplexF64}
    return v[1] * k.m + v[2] * k.n
end


"""
    c2t: a (cartesian vector) → v (transverse vector)
"""
function c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{2,ComplexF64}
    v0 = a ⋅ k.m
    v1 = a ⋅ k.n
    return SVector(v0,v1)
end

"""
    kcross_t2c: a (cartesian vector) = k × v (transverse vector) 
"""
function kcross_t2c(v::SVector{2,ComplexF64},k::KVec)::SVector{3,ComplexF64}
    return ( v[1] * k.n - v[2] * k.m ) * k.mag
end

"""
    kcross_c2t: v (transverse vector) = k × a (cartesian vector) 
"""
function kcross_c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{2,ComplexF64}
    at1 = a ⋅ k.m
    at2 = a ⋅ k.n
    v0 = -at2 * k.mag
    v1 = at1 * k.mag
    return SVector(v0,v1)
end


"""
    kcrossinv_t2c: compute a⃗ (cartestion vector) st. v⃗ (cartesian vector from two trans. vector components) ≈ k⃗ × a⃗
    This neglects the component of a⃗ parallel to k⃗ (not available by inverting this cross product)  
"""
function kcrossinv_t2c(v::SVector{2,ComplexF64},k::KVec)::SVector{3,ComplexF64}
    return ( v[1] * k.n - v[2] * k.m ) * ( -1 / k.mag )
end

"""
    kcrossinv_c2t: compute  v⃗ (transverse 2-vector) st. a⃗ (cartestion 3-vector) = k⃗ × v⃗
    This cross product inversion is exact because v⃗ is transverse (perp.) to k⃗ 
"""
function kcrossinv_c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{2,ComplexF64}
    at1 = a ⋅ k.m
    at2 = a ⋅ k.n
    v0 = -at2 * (-1 / k.mag )
    v1 = at1 * ( -1 / k.mag )
    return SVector(v0,v1)
end

"""
    ucross_t2c: a (cartesian vector) = u × v (transverse vector) 
"""
function ucross_t2c(u::SVector{3,ComplexF64},v::SVector{2,ComplexF64},k::KVec)::SVector{3,ComplexF64}
    return cross(u,t2c(v,k))
end


# """
#     fft: run fftw on 2D or 3D field of 3-vectors, stored as 2D or 3D Array of SVector{3,ComplexF64} entries
# """
# function fft(vf::Array{SArray{Tuple{3},Complex{Float64},1,3},N}) where N
#     reshape( reinterpret( ComplexF64, vf ), (  ))
# end


"""
    d_from_H(H,k): cartesian position space d vector field from transverse, PW basis H vector field
"""
function d_from_H(H::Array{SVector{2,ComplexF64},2},ds::MaxwellData)::Array{SVector{3,ComplexF64},2}
    # d_recip = [ kcross_t2c(H[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
    # temp =  (-1/(2π)) .* fft( reinterpret( ComplexF64, reshape( d_recip , (1,ds.Nx,ds.Ny) )), (2,3))
    # return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
    return reshape(reinterpret(SVector{3,ComplexF64}, (-1/(2π)) .* ( ds.𝓕 * reinterpret( ComplexF64, reshape( kcross_t2c.(H,ds.kpG), (1,ds.Nx,ds.Ny) )) ) ),(ds.Nx,ds.Ny))
end


"""
    e_from_d(d,ε⁻¹): e-field from d-field in cartesian position space, from division by local ε tensor 
"""
function e_from_d(d::Array{SVector{3,ComplexF64},2},ε⁻¹::Array{SHM3,2})::Array{SVector{3,ComplexF64},2}
    return ε⁻¹ .* d
end


"""
    H_from_e(e,k): reciprocal space (transverse basis) H vector field from position space cartesian basis e vector field
"""
function H_from_e(e::Array{SVector{3,ComplexF64},2},ds::MaxwellData)::Array{SVector{2,ComplexF64},2}
    # temp = (1/(2π)) .* conj.(ifft(reinterpret(ComplexF64,reshape(e,(1,ds.Nx,ds.Ny))), (2,3)))
    # e_recip = reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
    # return [ kcross_c2t(e_recip[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
    return kcross_c2t.( reshape(reinterpret(SVector{3,ComplexF64}, (1/(2π)) .* conj.( ds.𝓕⁻¹ * reinterpret(ComplexF64,reshape(e,(1,ds.Nx,ds.Ny))) ) ) ,(ds.Nx,ds.Ny)) , ds.kpG)
end

"""
    d_from_e(d,ε): e-field from d-field in cartesian position space, from division by local ε tensor 
"""
function d_from_e(d::Array{SVector{3,ComplexF64},2},ε::Array{SHM3,2})::Array{SVector{3,ComplexF64},2}
    return ε .* d
end

"""
    h_from_H(e,k):  position space cartesian basis h vector field from reciprocal space (transverse basis) H vector field
"""
function h_from_H(H,ds::MaxwellData)
    h_recip = [ t2c(H[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
    temp =   fft( reinterpret( ComplexF64, reshape( h_recip , (1,ds.Nx,ds.Ny) )), (2,3))
    return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
end

function flatten(H::Array{SVector{2,ComplexF64},2})
    # return reinterpret(ComplexF64,vec(H))
    return reinterpret(ComplexF64,vec(permutedims(H,[2,1])))
end

function unflatten(Hvec,ds::MaxwellData)::Array{SVector{2,ComplexF64},2}
    # return reshape(reinterpret(SVector{2,ComplexF64},Hvec),(ds.Nx,ds.Ny))
    return permutedims(reshape(reinterpret(SVector{2,ComplexF64},Hvec),(ds.Ny,ds.Nx)), [2,1])
end

function M̂ₖ(ε⁻¹::Array{SHM3,2},ds::MaxwellData)::LinearMaps.FunctionMap{ComplexF64}
    N = 2 * ds.Nx * ds.Ny
    f = H -> flatten(H_from_e( e_from_d( d_from_H( unflatten(H,ds), ds ), ε⁻¹ ), ds))
    fc = H -> flatten(H_from_e( d_from_e( d_from_H( unflatten(H,ds), ds ), SHM3.(inv.(ε⁻¹)) ), ds))
    return LinearMap{ComplexF64}(f,fc,N,N,ishermitian=true)
end

function z_cross_H(H::Array{SVector{2,ComplexF64},2},ds::MaxwellData)::Array{SVector{3,ComplexF64},2}
    z_cross_h_recip = [ ucross_t2c(SVector{3,ComplexF64}(0.,0.,1.),H[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
    temp =  (-1/(2π)) .* fft( reinterpret( ComplexF64, reshape( z_cross_h_recip , (1,ds.Nx,ds.Ny) )), (2,3))
    return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
end

function ∂ₖM̂ₖ(ε⁻¹::Array{SHM3,2},ds::MaxwellData)::LinearMaps.FunctionMap{ComplexF64}
    N = 2 * ds.Nx * ds.Ny
    f = H -> flatten(H_from_e( e_from_d( z_cross_H( unflatten(H,ds), ds ), ε⁻¹ ), ds))
    return LinearMap{ComplexF64}(f,f,N,N,ishermitian=true)
end


# Define approximate inversion operator P̂ₖ ≈ M̂ₖ⁻¹ to use as preconditioner 

"""
    e_from_H_approx(H,k): cartesian position space d vector field from transverse, PW basis H vector field
"""
function e_from_H_approx(H::Array{SVector{2,ComplexF64},2},ds::MaxwellData)::Array{SVector{3,ComplexF64},2}
    # d_recip = [ kcross_t2c(H[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
    # temp =  (-1/(2π)) .* fft( reinterpret( ComplexF64, reshape( d_recip , (1,ds.Nx,ds.Ny) )), (2,3))
    # return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
    return reshape(reinterpret(SVector{3,ComplexF64}, (-1/(2π)) .* ( ds.𝓕 * reinterpret( ComplexF64, reshape( kcrossinv_t2c.(H,ds.kpG), (1,ds.Nx,ds.Ny) )) ) ),(ds.Nx,ds.Ny))
end

"""
    H_from_d(e,k): reciprocal space (transverse basis) H vector field from position space cartesian basis e vector field
"""
function H_from_d(e::Array{SVector{3,ComplexF64},2},ds::MaxwellData)::Array{SVector{2,ComplexF64},2}
    # temp = (1/(2π)) .* conj.(ifft(reinterpret(ComplexF64,reshape(e,(1,ds.Nx,ds.Ny))), (2,3)))
    # e_recip = reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
    # return [ kcross_c2t(e_recip[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
    return kcrossinv_c2t.( reshape(reinterpret(SVector{3,ComplexF64}, (1/(2π)) .* conj.( ds.𝓕⁻¹ * reinterpret(ComplexF64,reshape(e,(1,ds.Nx,ds.Ny))) ) ) ,(ds.Nx,ds.Ny)) , ds.kpG)
end

"""
approximate inversion operator P̂ₖ ≈ M̂ₖ⁻¹ to use as preconditioner
"""
function P̂ₖ(ε::Array{SHM3,2},ds::MaxwellData)::LinearMaps.FunctionMap
    N = 2 * ds.Nx * ds.Ny
    f = H -> flatten(H_from_d( d_from_e( e_from_H_approx( unflatten(H,ds), ds ), ε ), ds))
    fc = H -> flatten(H_from_d( e_from_d( e_from_H_approx( unflatten(H,ds), ds ), SHM3.(inv.(ε)) ), ds))
    return LinearMap{ComplexF64}(f,fc,N,N)
end

