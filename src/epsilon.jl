export ε_init, εₛ, εₛ⁻¹, ε_tensor, test_εs, εₘₐₓ, ñₘₐₓ, nₘₐₓ

εᵥ = [	1. 	0. 	0.
        0. 	1. 	0.
        0. 	0. 	1.  ]

function ε_tensor(n::Float64)
    n² = n^2
    ε =     [	n²      0. 	    0.
                0. 	    n² 	    0.
                0. 	    0. 	    n²  ]
end

function test_εs(n₁::Float64,n₂::Float64,n₃::Float64)
    ε₁ = ε_tensor(n₁)
    ε₂ = ε_tensor(n₂)
    ε₃ = ε_tensor(n₃)
    return ε₁, ε₂, ε₃
end

function ε_init(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D;Δx=6.,Δy=4.,Nx=64,Ny=64)::Array{Float64,5}
    g = MaxwellGrid(Δx,Δy,Nx,Ny)
    tree = KDTree(shapes)
    return Float64[ isnothing(findfirst([xx,yy],tree)) ? εᵥ[i,j] : findfirst([xx,yy],tree).data[i,j] for xx=g.x,yy=g.y,i=1:3,j=1:3 ]
end

function ε_init(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D,g::MaxwellData)::Array{Float64,5}
    tree = KDTree(shapes)
    return Float64[ isnothing(findfirst([xx,yy],tree)) ? εᵥ[i,j] : findfirst([xx,yy],tree).data[i,j] for xx=g.x,yy=g.y,i=1:3,j=1:3 ]
end

function ε_init(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D, x::AbstractVector{Float64}, y::AbstractVector{Float64})::Array{Float64,5}
    tree = KDTree(shapes)
    return Float64[ isnothing(findfirst([xx,yy],tree)) ? εᵥ[i,j] : findfirst([xx,yy],tree).data[i,j] for xx=x,yy=y,i=1:3,j=1:3 ]
end



function surfpt_nearby2(x::Vector{Float64}, s::Sphere{2})
    nout = x==s.c ? SVector(1.0,0.0) : # nout = e₁ for x == s.c
                    normalize(x-s.c)
    return s.c+s.r*nout, nout
end


f_onbnd(bin,absdin) =  (b = Zygote.@ignore bin; absd = Zygote.@ignore absdin; abs.(b.r.-absd) .≤ Base.rtoldefault(Float64) .* b.r)  # basically b.r .≈ absd but faster
f_isout(b,absd) =  (isout = Zygote.@ignore ((b.r.<absd) .| f_onbnd(b,absd) ); isout)
# isout(bin) =  (b = Zygote.@ignore bin; (b.r.<absd) .| (abs.(b.r.-absd) .≤ Base.rtoldefault(Float64) .* b.r))
f_signs(d) =  (signs = Zygote.@ignore (copysign.(1.0,d)'); signs)


function surfpt_nearby2(x::Vector{Float64}, s::Sphere{2})
    nout = x==s.c ? SVector(1.0,0.0) : # nout = e₁ for x == s.c
                    normalize(x-s.c)
    return s.c+s.r*nout, nout
end

function surfpt_nearby2(x, b::Box{2})
    ax = inv(b.p)
    n0 = b.p ./  [ sqrt(b.p[1,1]^2 + b.p[1,2]^2) sqrt(b.p[2,1]^2 + b.p[2,2]^2)  ]
    d = Array(b.p * (x - b.c))
    cosθ = diag(n0*ax)

    n = n0 .* f_signs(d)
    absd = abs.(d)
    ∆ = (b.r .- absd) .* cosθ
    # onbnd = abs.(dropgrad(b.r).-dropgrad(absd)) .≤ Base.rtoldefault(Float64) .* dropgrad(b.r)  # basically b.r .≈ absd but faster
    # isout = (dropgrad(b.r).<dropgrad(absd)) .| dropgrad(onbnd)
    onbnd = f_onbnd(b,absd)
    isout = f_isout(b,absd)
    projbnd =  all(.!isout .| onbnd)
    onbnd_float =  Float64.(onbnd)
    isout_float =  Float64.(isout)
    # if ( ( abs(b.r[1]-absd[1]) ≤ Base.rtoldefault(Float64) * b.r[1] ) | (b.r[1]<absd[1]) ) | ( ( abs(b.r[2]-absd[2]) ≤ Base.rtoldefault(Float64) * b.r[2] ) | ( b.r[2] < absd[2] ) )
    if count(isout) == 0
        # = not(isout[1]) &  not(isout[2]) case, point is inside box
        l∆x, i = findmin(∆)  # find closest face
        nout = n[i,:]
        ∆x = l∆x * nout
    else
        ∆x = n' * (∆ .* isout_float)
        # = isout[1] | isout[2] case 1: at least one dimension outside or on boundry
        if all(.!isout .| onbnd)
            nout0 = n' * onbnd_float
        else
            nout0 = -∆x
        end
        nout = nout0 / norm(nout0)
    end

    return SVector{2,Float64}(x+∆x), SVector{2,Float64}(nout)
end

function τ_trans(ε)
    ε₁₁, ε₂₁, ε₃₁, ε₁₂, ε₂₂, ε₃₂, ε₁₃, ε₂₃, ε₃₃ = ε
    return SMatrix{3,3,Float64,9}(
        -1/ε₁₁, ε₂₁/ε₁₁, ε₃₁/ε₁₁,
        ε₁₂/ε₁₁, ε₂₂ - ε₂₁*ε₁₂/ε₁₁, ε₃₂ - ε₃₁*ε₁₂/ε₁₁,
        ε₁₃/ε₁₁, ε₂₃ - ε₂₁*ε₁₃/ε₁₁, ε₃₃ - ε₃₁*ε₁₃/ε₁₁
    )
end

function τ⁻¹_trans(τ)
    τ₁₁, τ₂₁, τ₃₁, τ₁₂, τ₂₂, τ₃₂, τ₁₃, τ₂₃, τ₃₃ = τ
    return SMatrix{3,3,Float64,9}(
        -1/τ₁₁, -τ₂₁/τ₁₁, -τ₃₁/τ₁₁,
        -τ₁₂/τ₁₁, τ₂₂ - τ₂₁*τ₁₂/τ₁₁, τ₃₂ - τ₃₁*τ₁₂/τ₁₁,
        -τ₁₃/τ₁₁, τ₂₃ - τ₂₁*τ₁₃/τ₁₁, τ₃₃ - τ₃₁*τ₁₃/τ₁₁
    )
end

# function kottke_avg_param(param1::SMat3Complex, param2::SMat3Complex, n12::SVec3Float, rvol1::Real)
function avg_param(param1, param2, n12, rvol1)
    n = n12 / norm(n12) #sqrt(sum2(abs2,n12))

    # Pick a vector that is not along n.
    if any(n .== 0)
    	htemp1 = (n .== 0)
    else
    	htemp1 = SVector(1., 0. , 0.)
    end

    # Create two vectors that are normal to n and normal to each other.
    htemp2 = n × htemp1
    h = htemp2 / norm(htemp2) #sqrt(sum2(abs2,htemp2))
    vtemp = n × h
    v = vtemp / norm(vtemp) #sqrt(sum2(abs2,vtemp))
    # Create a local Cartesian coordinate system.
    S = [n h v]  # unitary

    τ1 = τ_trans(transpose(S) * param1 * S)  # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans(transpose(S) * param2 * S)  # express param2 in S coordinates, and apply τ transform

    τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)  # volume-weighted average

    return S * τ⁻¹_trans(τavg) * transpose(S)  # apply τ⁻¹ and transform back to global coordinates
end

function εₛ(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D,tree::KDTree,x::Real,y::Real,δx::Real,δy::Real)::Array{Float64,2}
    x1,y1 = x+δx/2.,y+δy/2
    x2,y2 = x+δx/2.,y-δy/2
    x3,y3 = x-δx/2.,y-δy/2
    x4,y4 = x-δx/2.,y+δy/2

    s1 = findfirst([x1,y1],tree)
    s2 = findfirst([x2,y2],tree)
    s3 = findfirst([x3,y3],tree)
    s4 = findfirst([x4,y4],tree)

    ε1 = isnothing(s1) ? εᵥ : s1.data
    ε2 = isnothing(s2) ? εᵥ : s2.data
    ε3 = isnothing(s3) ? εᵥ : s3.data
    ε4 = isnothing(s4) ? εᵥ : s4.data

    if (ε1==ε2==ε3==ε4)
        return ε1
    else
        sinds = [ isnothing(ss) ? length(shapes)+1 : findfirst(isequal(ss),shapes) for ss in [s1,s2,s3,s4]]
        s_fg = shapes[min(sinds...)]
        r₀,nout = surfpt_nearby2([x; y], s_fg)
        # bndry_pxl[i,j] = 1
        # nouts[i,j,:] = nout
        vxl = (SVector{2,Float64}(x3,y3), SVector{2,Float64}(x1,y1))
        rvol = volfrac(vxl,nout,r₀)
        sind_bg = max(sinds...)
        ε_bg = sind_bg > length(shapes) ? εᵥ : shapes[sind_bg].data
        return avg_param(
                s_fg.data,
                ε_bg,
                [nout[1];nout[2];0],
                rvol,)
    end
end

function εₛ(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D,Δx=6.,Δy=4.,Nx=64,Ny=64)
    g=MaxwellGrid(Δx,Δy,Nx,Ny)
    tree = KDTree(shapes)
    ε_sm = zeros(Float64,g.Nx,g.Ny,3,3)
    for i=1:g.Nx, j=1:g.Ny
        ε_sm[i,j,:,:] = εₛ(shapes,tree,g.x[i],g.y[j],g.δx,g.δy)
    end
    return ε_sm
end

function εₛ(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D,g::MaxwellGrid)
    tree = KDTree(shapes)
    # ε_sm = copy(reshape(  [εₛ(shapes,tree,g.x[i],g.y[j],g.δx,g.δy) for i=1:g.Nx,j=1:g.Ny] , (g.Nx,g.Ny,1)) )
    ε_sm = [εₛ(shapes,tree,g.x[ix],g.y[iy],g.δx,g.δy)[a,b] for a=1:3,b=1:3,ix=1:g.Nx,iy=1:g.Ny]
end

# function εₛ⁻¹(shapes::AbstractVector{T},g::MaxwellGrid) where T <: GeometryPrimitives.Shape{2,4,D} where D
function εₛ⁻¹(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D,g::MaxwellGrid)
    tree = KDTree(shapes)
    # ε_sm_inv = copy(reshape( [inv(εₛ(shapes,tree,g.x[i],g.y[j],g.δx,g.δy)) for i=1:g.Nx,j=1:g.Ny], (g.Nx,g.Ny,1)) )
    # ε_sm_inv = [inv(εₛ(shapes,tree,g.x[ix],g.y[iy],g.δx,g.δy))[a,b] for a=1:3,b=1:3,ix=1:g.Nx,iy=1:g.Ny,iz=1:g.Nz]
    ε_sm_inv = [inv(εₛ(shapes,tree,g.x[ix],g.y[iy],g.δx,g.δy))[a,b] for a=1:3,b=1:3,ix=1:g.Nx,iy=1:g.Ny]
end

function εₘₐₓ(shapes::AbstractVector{T} where T <: GeometryPrimitives.Shape{2,4,D} where D)
    maximum(vec([shapes[i].data[j,j] for j=1:3,i=1:size(shapes)[1]]))
end

ñₘₐₓ(ε⁻¹)::Float64 = √(maximum(3 ./ tr.(ε⁻¹)))
nₘₐₓ(ε)::Float64 = √(maximum(reinterpret(Float64,ε)))
