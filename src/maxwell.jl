# using GeometryPrimitives: orthoaxes
export MaxwellGrid, MaxwellData, KVec, kpG, t2c, c2t, kcross_c2t, kcross_t2c, zcross_t2c, kcrossinv_c2t, kcrossinv_t2c, ε⁻¹_dot, ε_dot_approx, M, M̂, P, P̂, Mₖ, M̂ₖ, t2c!, c2t!, kcross_c2t!, kcross_t2c!, zcross_t2c!, kcrossinv_c2t!, kcrossinv_t2c!, ε⁻¹_dot!, ε_dot_approx!, M!, M̂!, P!, P̂!

# twopi = 1.                  # still not sure whether to include factors of 2π, for now use this global to control all instances

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
    sqrt(sum(abs2(ki) for ki in k)), # sqrt(sum(ki^2 for ki in k)),
    k_mn(k)...,
)

struct MaxwellGrid
    Δx::Float64
    Δy::Float64
    Δz::Float64
    Nx::Int64
    Ny::Int64
    Nz::Int64
    δx::Float64
    δy::Float64
    δz::Float64
    Gx::SVector{3,Float64}
    Gy::SVector{3,Float64}
    Gz::SVector{3,Float64}
    x::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    y::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    z::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    gx::Array{SArray{Tuple{3},Float64,1,3},1}
    gy::Array{SArray{Tuple{3},Float64,1,3},1}
    gz::Array{SArray{Tuple{3},Float64,1,3},1}
    𝓕::FFTW.cFFTWPlan
    𝓕⁻¹::AbstractFFTs.ScaledPlan
    𝓕!::FFTW.cFFTWPlan
    𝓕⁻¹!::AbstractFFTs.ScaledPlan
end

MaxwellGrid(Δx::Real,Δy::Real,Nx::Int,Ny::Int) = MaxwellGrid( 
    Δx,
    Δy,
    0.,
    Nx,
    Ny,
    1,
    Δx / Nx,    # δx
    Δy / Ny,    # δy
    1.,    # δz
    SVector(1., 0., 0.),      # Gx
    SVector(0., 1., 0.),      # Gy
    SVector(0., 0., 1.),      # Gz
    ( ( Δx / Nx ) .* (0:(Nx-1))) .- Δx/2.,  # x
    ( ( Δy / Ny ) .* (0:(Ny-1))) .- Δy/2.,  # y
    0.0:1.0:0.0,  # z
    [SVector(ggx, 0., 0.) for ggx in fftfreq(Nx,Nx/Δx)],     # gx
    [SVector(0., ggy, 0.) for ggy in fftfreq(Ny,Ny/Δy)],     # gy
    [SVector(0., 0., 0.),],                                  # gz
    plan_fft(randn(ComplexF64, (3,Nx,Ny,1)),(2:4)),    # planned DFT operator 𝓕
    plan_ifft(randn(ComplexF64, (3,Nx,Ny,1)),(2:4)),   # planned inverse DFT operator 𝓕⁻¹
    plan_fft!(randn(ComplexF64, (3,Nx,Ny,1)),(2:4)),    # planned DFT operator 𝓕
    plan_ifft!(randn(ComplexF64, (3,Nx,Ny,1)),(2:4)),   # planned inverse DFT operator 𝓕⁻¹
)

MaxwellGrid(Δx::Real,Δy::Real,Δz::Real,Nx::Int,Ny::Int,Nz::Int) = MaxwellGrid( 
    Δx,
    Δy,
    Δz,
    Nx,
    Ny,
    Nz,
    Δx / Nx,    # δx
    Δy / Ny,    # δy
    Δz / Nz,    # δz
    SVector(1., 0., 0.),      # Gx
    SVector(0., 1., 0.),      # Gy
    SVector(0., 0., 1.),      # Gz
    ( ( Δx / Nx ) .* (0:(Nx-1))) .- Δx/2.,  # x
    ( ( Δy / Ny ) .* (0:(Ny-1))) .- Δy/2.,  # y
    ( ( Δz / Nz ) .* (0:(Nz-1))) .- Δz/2.,  # z
    [SVector(ggx, 0., 0.) for ggx in fftfreq(Nx,Nx/Δx)],     # gx
    [SVector(0., ggy, 0.) for ggy in fftfreq(Ny,Ny/Δy)],     # gy
    [SVector(0., 0., ggz) for ggz in fftfreq(Nz,Nz/Δz)],     # gz
    plan_fft(randn(ComplexF64, (3,Nx,Ny,Nz)),(2:4)),    # planned DFT operator 𝓕
    plan_ifft(randn(ComplexF64, (3,Nx,Ny,Nz)),(2:4)),   # planned inverse DFT operator 𝓕⁻¹
    plan_fft!(randn(ComplexF64, (3,Nx,Ny,Nz)),(2:4)),    # planned DFT operator 𝓕
    plan_ifft!(randn(ComplexF64, (3,Nx,Ny,Nz)),(2:4)),   # planned inverse DFT operator 𝓕⁻¹
)

mutable struct MaxwellData
    k::SVector{3,Float64}
    ω²::Float64
    ω²ₖ::Float64
    ω::Float64
    ωₖ::Float64
    H⃗::Array{ComplexF64,2}
    H::Array{ComplexF64,4}
    e::Array{ComplexF64,4}
    d::Array{ComplexF64,4}
    grid::MaxwellGrid
    Δx::Float64
    Δy::Float64
    Δz::Float64
    Neigs::Int64
    Nx::Int64
    Ny::Int64
    Nz::Int64
    δx::Float64
    δy::Float64
    δz::Float64
    Gx::SVector{3,Float64}
    Gy::SVector{3,Float64}
    Gz::SVector{3,Float64}
    x::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    y::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    z::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    gx::Array{SArray{Tuple{3},Float64,1,3},1}
    gy::Array{SArray{Tuple{3},Float64,1,3},1}
    gz::Array{SArray{Tuple{3},Float64,1,3},1}
    kpG::Array{KVec,3}
    𝓕::FFTW.cFFTWPlan
    𝓕⁻¹::AbstractFFTs.ScaledPlan
    𝓕!::FFTW.cFFTWPlan
    𝓕⁻¹!::AbstractFFTs.ScaledPlan
end

function kpG(k::SVector{3,Float64},g::MaxwellGrid)::Array{KVec,3}
    [KVec(k-gx-gy-gz) for gx=g.gx, gy=g.gy, gz=g.gz]
    # [KVec(k+gx+gy+gz) for gx=g.gx, gy=g.gy, gz=g.gz]
end

MaxwellData(k::SVector{3,Float64},g::MaxwellGrid,Neigs::Int64) = MaxwellData(   
    k,
    0.0,
    0.0,
    0.0,
    0.0,
    randn(ComplexF64,(2*g.Nx*g.Ny*g.Nz,Neigs)),
    randn(ComplexF64,(2,g.Nx,g.Ny,g.Nz)),
    randn(ComplexF64,(3,g.Nx,g.Ny,g.Nz)),
    randn(ComplexF64,(3,g.Nx,g.Ny,g.Nz)),
    g,
    g.Δx,
    g.Δy,
    g.Δz,
    Neigs,
    g.Nx,
    g.Ny,
    g.Nz,
    g.δx,       # δx
    g.δy,       # δy
    g.δz,       # δz
    g.Gx,       # Gx
    g.Gy,       # Gy
    g.Gz,       # Gz
    g.x,        # x
    g.y,        # y
    g.z,        # z
    g.gx,
    g.gy,
    g.gz,
    kpG(k,g),  # kpG
    g.𝓕,
    g.𝓕⁻¹,
    g.𝓕!,
    g.𝓕⁻¹!,
)

MaxwellData(k::SVector{3,Float64},g::MaxwellGrid) = MaxwellData(k,g,1)
MaxwellData(kz::Float64,g::MaxwellGrid) = MaxwellData(SVector(0.,0.,kz),g)
MaxwellData(k::SVector{3,Float64},Δx::Real,Δy::Real,Δz::Real,Nx::Int,Ny::Int,Nz::Int) = MaxwellData(k,MaxwellGrid(Δx,Δy,Δz,Nx,Ny,Nz))
MaxwellData(k::SVector{3,Float64},Δx::Real,Δy::Real,Nx::Int,Ny::Int) = MaxwellData(k,MaxwellGrid(Δx,Δy,Nx,Ny))

# non-Mutating Operators

function t2c(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    Hout = Array{ComplexF64}(undef,(3,size(Hin)[2:end]...))
    for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @inbounds scale = ds.kpG[i,j,k].mag
        @inbounds Hout[1,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[1] ) * scale  
        @inbounds Hout[2,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].m[2] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[2] ) * scale  
        @inbounds Hout[3,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].m[3] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[3] ) * scale 
    end
    return Hout
end

function c2t(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    Hout = Array{ComplexF64}(undef,(2,size(Hin)[2:end]...))
    for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @inbounds Hout[1,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].m[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].m[3] 
        @inbounds Hout[2,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].n[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].n[3] 
    end
    return Hout
end

function zcross_t2c(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    Hout = zeros(ComplexF64,(3,size(Hin)[2:end]...))
    for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @inbounds Hout[1,i,j,k] = -Hin[1,i,j,k] * ds.kpG[i,j,k].m[2] - Hin[2,i,j,k] * ds.kpG[i,j,k].n[2]   
        @inbounds Hout[2,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[1]  
    end
    return Hout
end

function kcross_t2c(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    Hout = Array{ComplexF64}(undef,(3,size(Hin)[2:end]...))
    for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @inbounds scale = -ds.kpG[i,j,k].mag
        @inbounds Hout[1,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].n[1] - Hin[2,i,j,k] * ds.kpG[i,j,k].m[1] ) * scale  
        @inbounds Hout[2,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].n[2] - Hin[2,i,j,k] * ds.kpG[i,j,k].m[2] ) * scale  
        @inbounds Hout[3,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].n[3] - Hin[2,i,j,k] * ds.kpG[i,j,k].m[3] ) * scale 
    end
    return Hout
end

function kcross_c2t(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    Hout = Array{ComplexF64}(undef,(2,size(Hin)[2:end]...))
    for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @inbounds scale = ds.kpG[i,j,k].mag
        @inbounds at1 = Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].m[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].m[3]
        @inbounds at2 = Hin[1,i,j,k] * ds.kpG[i,j,k].n[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].n[3]
        @inbounds Hout[1,i,j,k] =  -at2 * scale 
        @inbounds Hout[2,i,j,k] =  at1 * scale 
    end
    return Hout
end

function kcrossinv_t2c(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    Hout = Array{ComplexF64}(undef,(3,size(Hin)[2:end]...))
    for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @inbounds scale = 1 / ds.kpG[i,j,k].mag
        @inbounds Hout[1,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].n[1] - Hin[2,i,j,k] * ds.kpG[i,j,k].m[1] ) * scale  
        @inbounds Hout[2,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].n[2] - Hin[2,i,j,k] * ds.kpG[i,j,k].m[2] ) * scale  
        @inbounds Hout[3,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].n[3] - Hin[2,i,j,k] * ds.kpG[i,j,k].m[3] ) * scale 
    end
    return Hout
end

function kcrossinv_c2t(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    Hout = Array{ComplexF64}(undef,(2,size(Hin)[2:end]...))
    for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @inbounds scale = -1 / ds.kpG[i,j,k].mag
        @inbounds at1 = Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].m[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].m[3]
        @inbounds at2 = Hin[1,i,j,k] * ds.kpG[i,j,k].n[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].n[3]
        @inbounds Hout[1,i,j,k] =  -at2 * scale 
        @inbounds Hout[2,i,j,k] =  at1 * scale 
    end
    return Hout
end

function ε⁻¹_dot(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    Hout = similar(Hin)
    for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @inbounds Hout[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,1]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,1]*Hin[3,i,j,k]
        @inbounds Hout[2,i,j,k] =  ε⁻¹[i,j,k][1,2]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,2]*Hin[3,i,j,k]
        @inbounds Hout[3,i,j,k] =  ε⁻¹[i,j,k][1,3]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,3]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,3]*Hin[3,i,j,k]
        # @inbounds Hout[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][1,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][1,3]*Hin[3,i,j,k]
        # @inbounds Hout[2,i,j,k] =  ε⁻¹[i,j,k][2,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][2,3]*Hin[3,i,j,k]
        # @inbounds Hout[3,i,j,k] =  ε⁻¹[i,j,k][3,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][3,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,3]*Hin[3,i,j,k]
    end
    return Hout
end

function ε_dot_approx(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    Hout = similar(Hin)
    for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @inbounds ε_ave = 3 / tr(ε⁻¹[i,j,k])
        @inbounds Hout[1,i,j,k] =  ε_ave * Hin[1,i,j,k]
        @inbounds Hout[2,i,j,k] =  ε_ave * Hin[2,i,j,k]
        @inbounds Hout[3,i,j,k] =  ε_ave * Hin[3,i,j,k]
    end
    return Hout
end

function M(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
    d = ds.𝓕 * kcross_t2c(Hin,ds);
    e = ε⁻¹_dot(d,ε⁻¹,ds); # (-1/(π)) .*
    kcross_c2t(ds.𝓕⁻¹ * e,ds)
end

function M(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
    HinA = reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
    HoutA = M(HinA,ε⁻¹,ds)
    return vec(HoutA)
end

M̂(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> M(H,ε⁻¹,ds)::AbstractArray{ComplexF64,1},(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=false)

function P(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
    e = ds.𝓕⁻¹ * kcrossinv_t2c(Hin,ds);
    d = ε_dot_approx(e,ε⁻¹,ds); # (-1/(π)) .*
    kcrossinv_c2t(ds.𝓕 * d,ds)
end

function P(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
    HinA = reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
    HoutA = P(HinA,ε⁻¹,ds)
    return vec(HoutA)
end

P̂(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> P(H,ε⁻¹,ds)::AbstractArray{ComplexF64,1},(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=false)

function Mₖ(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
    d = ds.𝓕 * zcross_t2c(Hin,ds);
    e = ε⁻¹_dot(d,ε⁻¹,ds); # (-1/(π)) .*
    kcross_c2t(ds.𝓕⁻¹ * e,ds)
end

function Mₖ(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
    HinA = reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
    HoutA = Mₖ(HinA,ε⁻¹,ds)
    return -vec(HoutA)
end

M̂ₖ(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> Mₖ(H,ε⁻¹,ds)::AbstractArray{ComplexF64,1},(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=false)

# Mutating Operators

function t2c!(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @inbounds scale = ds.kpG[i,j,k].mag
        @inbounds ds.e[1,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[1] ) * scale  
        @inbounds ds.e[2,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].m[2] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[2] ) * scale  
        @inbounds ds.e[3,i,j,k] = ( Hin[1,i,j,k] * ds.kpG[i,j,k].m[3] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[3] ) * scale 
    end
    return ds.e
end

function c2t!(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @inbounds ds.e[1,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].m[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].m[3] 
        @inbounds ds.e[2,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].n[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[2] + Hin[3,i,j,k] * ds.kpG[i,j,k].n[3]
        # @inbounds ds.e[3,i,j,k] =  0.0 
    end
    return ds.e
end

function zcross_t2c!(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        @inbounds ds.e[1,i,j,k] = -Hin[1,i,j,k] * ds.kpG[i,j,k].m[2] - Hin[2,i,j,k] * ds.kpG[i,j,k].n[2]   
        @inbounds ds.e[2,i,j,k] =  Hin[1,i,j,k] * ds.kpG[i,j,k].m[1] + Hin[2,i,j,k] * ds.kpG[i,j,k].n[1]  
    end
    return ds.e
end

function kcross_t2c!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
    @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        scale = -ds.kpG[i,j,k].mag #-ds.kpG[i,j,k].mag
        ds.d[1,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[1] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[1] ) * scale  
        ds.d[2,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[2] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[2] ) * scale  
        ds.d[3,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[3] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[3] ) * scale 
    end
    return ds.d
end

function kcross_c2t!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
    @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        scale = ds.kpG[i,j,k].mag
        at1 = ds.e[1,i,j,k] * ds.kpG[i,j,k].m[1] + ds.e[2,i,j,k] * ds.kpG[i,j,k].m[2] + ds.e[3,i,j,k] * ds.kpG[i,j,k].m[3]
        at2 = ds.e[1,i,j,k] * ds.kpG[i,j,k].n[1] + ds.e[2,i,j,k] * ds.kpG[i,j,k].n[2] + ds.e[3,i,j,k] * ds.kpG[i,j,k].n[3]
        ds.H[1,i,j,k] =  -at2 * scale 
        ds.H[2,i,j,k] =  at1 * scale
    end
    return ds.H
end

function kcrossinv_t2c!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
    @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        scale = 1 / ds.kpG[i,j,k].mag
        ds.e[1,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[1] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[1] ) * scale  
        ds.e[2,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[2] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[2] ) * scale  
        ds.e[3,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[3] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[3] ) * scale 
    end
    return ds.e
end

function kcrossinv_c2t!(ds::MaxwellData)::AbstractArray{ComplexF64,4}
    @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        scale = -1 / ds.kpG[i,j,k].mag
        at1 = ds.d[1,i,j,k] * ds.kpG[i,j,k].m[1] + ds.d[2,i,j,k] * ds.kpG[i,j,k].m[2] + ds.d[3,i,j,k] * ds.kpG[i,j,k].m[3]
        at2 = ds.d[1,i,j,k] * ds.kpG[i,j,k].n[1] + ds.d[2,i,j,k] * ds.kpG[i,j,k].n[2] + ds.d[3,i,j,k] * ds.kpG[i,j,k].n[3]
        ds.H[1,i,j,k] =  -at2 * scale 
        ds.H[2,i,j,k] =  at1 * scale 
    end
    return ds.H
end

function ε⁻¹_dot!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        ds.e[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*ds.d[1,i,j,k] + ε⁻¹[i,j,k][2,1]*ds.d[2,i,j,k] + ε⁻¹[i,j,k][3,1]*ds.d[3,i,j,k]
        ds.e[2,i,j,k] =  ε⁻¹[i,j,k][1,2]*ds.d[1,i,j,k] + ε⁻¹[i,j,k][2,2]*ds.d[2,i,j,k] + ε⁻¹[i,j,k][3,2]*ds.d[3,i,j,k]
        ds.e[3,i,j,k] =  ε⁻¹[i,j,k][1,3]*ds.d[1,i,j,k] + ε⁻¹[i,j,k][2,3]*ds.d[2,i,j,k] + ε⁻¹[i,j,k][3,3]*ds.d[3,i,j,k]
        # ds.e[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][1,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][1,3]*Hin[3,i,j,k]
        # ds.e[2,i,j,k] =  ε⁻¹[i,j,k][2,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][2,3]*Hin[3,i,j,k]
        # ds.e[3,i,j,k] =  ε⁻¹[i,j,k][3,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][3,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,3]*Hin[3,i,j,k]
    end
    return ds.e
end

function ε_dot_approx!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::AbstractArray{ComplexF64,4}
    @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
        ε_ave = 3 / tr(ε⁻¹[i,j,k])
        ds.d[1,i,j,k] =  ε_ave * ds.e[1,i,j,k]
        ds.d[2,i,j,k] =  ε_ave * ds.e[2,i,j,k]
        ds.d[3,i,j,k] =  ε_ave * ds.e[3,i,j,k]
    end
    return ds.d
end

function M!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
    kcross_t2c!(ds);
    # ds.𝓕! * ds.d;
    mul!(ds.d,ds.𝓕!,ds.d);
    ε⁻¹_dot!(ε⁻¹,ds);
    # ds.𝓕⁻¹! * ds.e;
    mul!(ds.e,ds.𝓕⁻¹!,ds.e)
    kcross_c2t!(ds)
end

function M!(Hout::AbstractArray{ComplexF64,1},Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
    # copyto!(ds.H,reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz)))
    @inbounds ds.H .= reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
    M!(ε⁻¹,ds);
    # copyto!(Hout,vec(ds.H))
    @inbounds Hout .= vec(ds.H)
end

function M̂!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)
    function f!(y::AbstractArray{ComplexF64,1},x::AbstractArray{ComplexF64,1})::AbstractArray{ComplexF64,1}
        M!(y,x,ε⁻¹,ds)    
    end
    return LinearMap{ComplexF64}(f!,(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=true)
end

function P!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
    kcrossinv_t2c!(ds);
    # ds.𝓕⁻¹! * ds.e;
    mul!(ds.e,ds.𝓕⁻¹!,ds.e)
    ε_dot_approx!(ε⁻¹,ds);
    # ds.𝓕! * ds.d;
    mul!(ds.d,ds.𝓕!,ds.d);
    kcrossinv_c2t!(ds)
end

function P!(Hout::AbstractArray{ComplexF64,1},Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
    # copyto!(ds.H,reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz)))
    @inbounds ds.H .= reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
    P!(ε⁻¹,ds);
    # copyto!(Hout,vec(ds.H))
    @inbounds Hout .= vec(ds.H)
end

function P̂!(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)
    function fp!(y::AbstractArray{ComplexF64,1},x::AbstractArray{ComplexF64,1})::AbstractArray{ComplexF64,1}
        P!(y,x,ε⁻¹,ds)    
    end
    return LinearMap{ComplexF64}(fp!,(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=true)
end



#########################
#
#   old stuff
#
#########################


# """
#     t2c: v (transverse vector) → a (cartesian vector)
# """
# function t2c(v::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     return v[1] * k.m + v[2] * k.n
# end


# """
#     c2t: a (cartesian vector) → v (transverse vector)
# """
# function c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     v0 = a ⋅ k.m
#     v1 = a ⋅ k.n
#     return SVector(v0,v1)
# end

# """
#     kcross_t2c: a (cartesian vector) = k × v (transverse vector) 
# """
# function kcross_t2c(v::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     return ( v[1] * k.n - v[2] * k.m ) * k.mag
# end

# """
#     kcross_t2c!: a (cartesian vector) = k × v (transverse vector) 
# """
# function kcross_t2c!(v::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     return v = ( v[1] * k.n - v[2] * k.m ) * k.mag
# end



# """
#     kcross_c2t: v (transverse vector) = k × a (cartesian vector) 
# """
# function kcross_c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     at1 = a ⋅ k.m
#     at2 = a ⋅ k.n
#     v0 = -at2 * k.mag
#     v1 = at1 * k.mag
#     return SVector(v0,v1,0.0)
# end


# """
#     kcrossinv_t2c: compute a⃗ (cartestion vector) st. v⃗ (cartesian vector from two trans. vector components) ≈ k⃗ × a⃗
#     This neglects the component of a⃗ parallel to k⃗ (not available by inverting this cross product)  
# """
# function kcrossinv_t2c(v::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     return ( v[1] * k.n - v[2] * k.m ) * ( -1 / k.mag )
# end

# """
#     kcrossinv_c2t: compute  v⃗ (transverse 2-vector) st. a⃗ (cartestion 3-vector) = k⃗ × v⃗
#     This cross product inversion is exact because v⃗ is transverse (perp.) to k⃗ 
# """
# function kcrossinv_c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     at1 = a ⋅ k.m
#     at2 = a ⋅ k.n
#     v0 = -at2 * (-1 / k.mag )
#     v1 = at1 * ( -1 / k.mag )
#     return SVector(v0,v1,0.0)
# end

# """
#     ucross_t2c: a (cartesian vector) = u × v (transverse vector) 
# """
# function ucross_t2c(u::SVector{3,ComplexF64},v::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     return cross(u,t2c(v,k))
# end

# """
#     d_from_H(H,k): cartesian position space d vector field from transverse, PW basis H vector field
# """
# function d_from_H(H::Array{SVector{3,ComplexF64},3},ds::MaxwellData)::Array{SVector{3,ComplexF64},3}
#     # d_recip = [ kcross_t2c(H[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
#     # temp =  (-1/(2π)) .* fft( reinterpret( ComplexF64, reshape( d_recip , (1,ds.Nx,ds.Ny) )), (2,3))
#     # return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
#     return reshape(reinterpret(SVector{3,ComplexF64}, (-1/(2π)) .* ( ds.𝓕 * reinterpret( ComplexF64, reshape( kcross_t2c.(H,ds.kpG), (1,ds.Nx,ds.Ny) )) ) ),(ds.Nx,ds.Ny))
# end


# """
#     e_from_d(d,ε⁻¹): e-field from d-field in cartesian position space, from division by local ε tensor 
# """
# function e_from_d(d::Array{SVector{3,ComplexF64},3},ε⁻¹::Array{SHM3,2})::Array{SVector{3,ComplexF64},3}
#     return ε⁻¹ .* d
# end


# """
#     H_from_e(e,k): reciprocal space (transverse basis) H vector field from position space cartesian basis e vector field
# """
# function H_from_e(e::Array{SVector{3,ComplexF64},3},ds::MaxwellData)::Array{SVector{3,ComplexF64},3}
#     # temp = (1/(2π)) .* conj.(ifft(reinterpret(ComplexF64,reshape(e,(1,ds.Nx,ds.Ny))), (2,3)))
#     # e_recip = reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
#     # return [ kcross_c2t(e_recip[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
#     return kcross_c2t.( reshape(reinterpret(SVector{3,ComplexF64}, (1/(2π)) .* conj.( ds.𝓕⁻¹ * reinterpret(ComplexF64,reshape(e,(1,ds.Nx,ds.Ny))) ) ) ,(ds.Nx,ds.Ny)) , ds.kpG)
# end

# """
#     d_from_e(d,ε): e-field from d-field in cartesian position space, from division by local ε tensor 
# """
# function d_from_e(d::Array{SVector{3,ComplexF64},3},ε::Array{SHM3,2})::Array{SVector{3,ComplexF64},3}
#     return ε .* d
# end

# """
#     h_from_H(e,k):  position space cartesian basis h vector field from reciprocal space (transverse basis) H vector field
# """
# function h_from_H(H,ds::MaxwellData)
#     h_recip = [ t2c(H[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
#     temp =   fft( reinterpret( ComplexF64, reshape( h_recip , (1,ds.Nx,ds.Ny) )), (2,3))
#     return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
# end

# function flatten(H::Array{SVector{3,ComplexF64},3})
#     # return reinterpret(ComplexF64,vec(H))
#     return reinterpret(ComplexF64,vec(permutedims(H,[2,1])))
# end

# function unflatten(Hvec,ds::MaxwellData)::Array{SVector{3,ComplexF64},3}
#     # return reshape(reinterpret(SVector{3,ComplexF64},Hvec),(ds.Nx,ds.Ny))
#     return permutedims(reshape(reinterpret(SVector{3,ComplexF64},Hvec),(ds.Ny,ds.Nx)), [2,1])
# end


# Define Maxwell operator function and LinearMap instantiator objects

# function M!(H::Array{SVector{3,ComplexF64}},ε⁻¹::Array{SHM3,2},ds::MaxwellData)::Array{SVector{3,ComplexF64}}
#     H .= kcross_t2c.(H,k);
#     ds.𝓕 * H;
#     H .= ε⁻¹ .* H;
#     ds.𝓕⁻¹ * H;
#     H .= kcross_c2t.(H,k)
# end

# function M!(Hv::Vector{SVector{3,ComplexF64}},ε⁻¹::Array{SHM3,2},ds::MaxwellData)::Vector{SVector{3,ComplexF64}}
#     H = reshape(Hv,size(k))
#     H .= kcross_t2c.(H,k);
#     ds.𝓕 * H;
#     H .= ε⁻¹ .* H;
#     ds.𝓕⁻¹ * H;
#     H .= kcross_c2t.(H,k)
#     return Hv
# end

# function M!(Hv::Vector{ComplexF64},ε⁻¹::Array{SHM3,2},ds::MaxwellData)::Vector{ComplexF64}
#     HSv = copy(reinterpret(SVector{3,ComplexF64}, Hv))
#     M!(HSv,ε⁻¹,ds);
#     Hv .= copy(reinterpret(ComplexF64, HSv))
# end

# function M!(Hv::Vector{ComplexF64},ε⁻¹::Array{SHM3,2},ds::MaxwellData,Hw::Array{SVector{3,ComplexF64}})::Vector{ComplexF64}
#     copyto!(Hw, reinterpret(SVector{3,ComplexF64},reshape(Hv,(3*size(k)[1],size(k)[2:end]...))) )
#     M!(Hw,ε⁻¹,ds);
#     copyto!(Hv, vec( reinterpret(ComplexF64,Hw) ) )
#     return Hv
# end

# function M̂ₖ!(ε⁻¹::Array{SHM3,3},ds::MaxwellData,Hw::Array{SVector{3,ComplexF64}})::LinearMaps.FunctionMap{ComplexF64}
#     N = 3 * ds.Nx * ds.Ny * ds.Nz
#     f = H::Vector{ComplexF64} -> M!(H,ε⁻¹,ds,Hw)
#     return LinearMap{ComplexF64}(f,f,N,N,ishermitian=true,ismutating=true)
# end

# function M̂ₖ(ε⁻¹::Array{SHM3,2},ds::MaxwellData)::LinearMaps.FunctionMap{ComplexF64}
#     N = 2 * ds.Nx * ds.Ny * ds.Nz
#     f = H -> flatten(H_from_e( e_from_d( d_from_H( unflatten(H,ds), ds ), ε⁻¹ ), ds))
#     fc = H -> flatten(H_from_e( d_from_e( d_from_H( unflatten(H,ds), ds ), SHM3.(inv.(ε⁻¹)) ), ds))
#     return LinearMap{ComplexF64}(f,fc,N,N,ishermitian=true)
# end

# function z_cross_H(H::Array{SVector{3,ComplexF64},3},ds::MaxwellData)::Array{SVector{3,ComplexF64},3}
#     z_cross_h_recip = [ ucross_t2c(SVector{3,ComplexF64}(0.,0.,1.),H[i,j,k],ds.kpG[i,j,k]) for i=1:ds.Nx, j=1:ds.Ny, k=1:ds.Nz]
#     temp =  (-1/(2π)) .* fft( reinterpret( ComplexF64, reshape( z_cross_h_recip , (1,ds.Nx,ds.Ny,ds.Nz) )), (2,3))
#     return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny,ds.Nz))
# end

# function ∂ₖM̂ₖ(ε⁻¹::Array{SHM3,3},ds::MaxwellData)::LinearMaps.FunctionMap{ComplexF64}
#     N = 2 * ds.Nx * ds.Ny * ds.Nz
#     f = H -> flatten(H_from_e( e_from_d( z_cross_H( unflatten(H,ds), ds ), ε⁻¹ ), ds))
#     return LinearMap{ComplexF64}(f,f,N,N,ishermitian=true)
# end


# Define approximate inversion operator P̂ₖ ≈ M̂ₖ⁻¹ to use as preconditioner 


# function P!(H::Array{SVector{3,ComplexF64}},ε::Array{SHM3,3},ds::MaxwellData)::Array{SVector{3,ComplexF64}}
#     H .= kcrossinv_t2c.(H,k);
#     ds.𝓕⁻¹ * H;
#     H .= ε .* H;
#     ds.𝓕 * H;
#     H .= kcrossinv_c2t.(H,k)
# end

# function P!(Hv::Vector{ComplexF64},ε::Array{SHM3,3},ds::MaxwellData,Hw::Array{SVector{3,ComplexF64}})::Vector{ComplexF64}
#     copyto!(Hw, reinterpret(SVector{3,ComplexF64},reshape(Hv,(3*size(k)[1],size(k)[2:end]...))) )
#     P!(Hw,ε,ds);
#     copyto!(Hv, vec( reinterpret(ComplexF64,Hw) ) )
#     return Hv
# end

# function P̂ₖ!(ε::Array{SHM3,3},ds::MaxwellData,Hw::Array{SVector{3,ComplexF64}})::LinearMaps.FunctionMap{ComplexF64}
#     N = 3 * ds.Nx * ds.Ny * ds.Nz
#     f = H::Vector{ComplexF64} -> P!(H,ε,ds,Hw)
#     return LinearMap{ComplexF64}(f,f,N,N,ismutating=true)
# end


# """
#     e_from_H_approx(H,k): cartesian position space d vector field from transverse, PW basis H vector field
# """
# function e_from_H_approx(H::Array{SVector{3,ComplexF64},3},ds::MaxwellData)::Array{SVector{3,ComplexF64},3}
#     # d_recip = [ kcross_t2c(H[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
#     # temp =  (-1/(2π)) .* fft( reinterpret( ComplexF64, reshape( d_recip , (1,ds.Nx,ds.Ny) )), (2,3))
#     # return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
#     return reshape(reinterpret(SVector{3,ComplexF64}, (-1/(2π)) .* ( ds.𝓕 * reinterpret( ComplexF64, reshape( kcrossinv_t2c.(H,ds.kpG), (1,ds.Nx,ds.Ny,ds.Nz) )) ) ),(ds.Nx,ds.Ny,ds.Nz))
# end

# """
#     H_from_d(e,k): reciprocal space (transverse basis) H vector field from position space cartesian basis e vector field
# """
# function H_from_d(e::Array{SVector{3,ComplexF64},3},ds::MaxwellData)::Array{SVector{3,ComplexF64},3}
#     # temp = (1/(2π)) .* conj.(ifft(reinterpret(ComplexF64,reshape(e,(1,ds.Nx,ds.Ny))), (2,3)))
#     # e_recip = reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
#     # return [ kcross_c2t(e_recip[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
#     return kcrossinv_c2t.( reshape(reinterpret(SVector{3,ComplexF64}, (1/(2π)) .* conj.( ds.𝓕⁻¹ * reinterpret(ComplexF64,reshape(e,(1,ds.Nx,ds.Ny,ds.Nz))) ) ) ,(ds.Nx,ds.Ny,ds.Nz)) , ds.kpG)
# end

# """
# approximate inversion operator P̂ₖ ≈ M̂ₖ⁻¹ to use as preconditioner
# """
# function P̂ₖ(ε::Array{SHM3,3},ds::MaxwellData)::LinearMaps.FunctionMap
#     N = 2 * ds.Nx * ds.Ny * ds.Nz
#     f = H -> flatten(H_from_d( d_from_e( e_from_H_approx( unflatten(H,ds), ds ), ε ), ds))
#     fc = H -> flatten(H_from_d( e_from_d( e_from_H_approx( unflatten(H,ds), ds ), SHM3.(inv.(ε)) ), ds))
#     return LinearMap{ComplexF64}(f,fc,N,N)
# end

