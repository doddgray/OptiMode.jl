using Revise
using LinearAlgebra, StaticArrays, GeometryPrimitives, Zygote, ForwardDiff #ChainRules, ReverseDiff
##
Zygote.@adjoint (T::Type{<:SArray})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
Zygote.@adjoint (T::Type{<:SArray})(x::AbstractArray) = T(x), dv -> (nothing, dv)
Zygote.@adjoint (T::Type{<:SMatrix})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
Zygote.@adjoint (T::Type{<:SMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
Zygote.@adjoint (T::Type{<:SVector})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
Zygote.@adjoint (T::Type{<:SVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)


Zygote.@adjoint enumerate(xs) = enumerate(xs), diys -> (map(last, diys),)
_ndims(::Base.HasShape{d}) where {d} = d
_ndims(x) = Base.IteratorSize(x) isa Base.HasShape ? _ndims(Base.IteratorSize(x)) : 1
Zygote.@adjoint function Iterators.product(xs...)
                    d = 1
                    Iterators.product(xs...), dy -> ntuple(length(xs)) do n
                        nd = _ndims(xs[n])
                        dims = ntuple(i -> i<d ? i : i+nd, ndims(dy)-nd)
                        d += nd
                        func = sum(y->y[n], dy; dims=dims)
                        ax = axes(xs[n])
                        reshape(func, ax)
                    end
                end

Zygote.refresh()

##
function ε_tensor(n::Float64)
    n² = n^2
    ε =     [	n²      0. 	    0.
                0. 	    n² 	    0.
                0. 	    0. 	    n²  ]
end

# volfrac(vxl::NTuple{2,SVector{2,<:Number}}, nout::SVector{2,<:Real}, r₀::SVector{2,<:Real}) =
#     volfrac((SVector(vxl[N][1],vxl[N][2],0), SVector(vxl[P][1],vxl[P][2],1)),
#             SVector(nout[1], nout[2], 0),
#             SVector(r₀[1], r₀[2], 0))

function surfpt_nearby2(x::Vector{Float64}, s::Sphere{2})
    nout = x==s.c ? SVector(1.0,0.0) : # nout = e₁ for x == s.c
                    normalize(x-s.c)
    return s.c+s.r*nout, nout
end


f_onbnd(bin,absdin) =  (b = Zygote.@ignore bin; absd = Zygote.@ignore absdin; abs.(b.r.-absd) .≤ Base.rtoldefault(Float64) .* b.r)  # basically b.r .≈ absd but faster
f_isout(b,absd) =  (isout = Zygote.@ignore ((b.r.<absd) .| f_onbnd(b,absd) ); isout)
# isout(bin) =  (b = Zygote.@ignore bin; (b.r.<absd) .| (abs.(b.r.-absd) .≤ Base.rtoldefault(Float64) .* b.r))
f_signs(d) =  (signs = Zygote.@ignore (copysign.(1.0,d)'); signs)
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

# Equation (4) of the paper by Kottke et al.
function τ_trans(ε)
    ε₁₁, ε₂₁, ε₃₁, ε₁₂, ε₂₂, ε₃₂, ε₁₃, ε₂₃, ε₃₃ = ε
    return SMatrix{3,3,Float64,9}(
        -1/ε₁₁, ε₂₁/ε₁₁, ε₃₁/ε₁₁,
        ε₁₂/ε₁₁, ε₂₂ - ε₂₁*ε₁₂/ε₁₁, ε₃₂ - ε₃₁*ε₁₂/ε₁₁,
        ε₁₃/ε₁₁, ε₂₃ - ε₂₁*ε₁₃/ε₁₁, ε₃₃ - ε₃₁*ε₁₃/ε₁₁
    )
end

# Equation (23) of the paper by Kottke et al.
function τ⁻¹_trans(τ)
    τ₁₁, τ₂₁, τ₃₁, τ₁₂, τ₂₂, τ₃₂, τ₁₃, τ₂₃, τ₃₃ = τ
    return SMatrix{3,3,Float64,9}(
        -1/τ₁₁, -τ₂₁/τ₁₁, -τ₃₁/τ₁₁,
        -τ₁₂/τ₁₁, τ₂₂ - τ₂₁*τ₁₂/τ₁₁, τ₃₂ - τ₃₁*τ₁₂/τ₁₁,
        -τ₁₃/τ₁₁, τ₂₃ - τ₂₁*τ₁₃/τ₁₁, τ₃₃ - τ₃₁*τ₁₃/τ₁₁
    )
end

# function kottke_avg_param(param1::SMat3Complex, param2::SMat3Complex, n12::SVec3Float, rvol1::Real)
function kottke_avg_param(param1, param2, n12, rvol1)
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

fign(x) = (y = Zygote.@ignore x; x * y)
fign2(x) = (y = x; x * y)
fign(3.0)
fign2(3.0)
Zygote.gradient(fign,3.0)
Zygote.gradient(fign2,3.0)

function ftest2(a,b,c)
    Zygote.ignore() do
        d = a + 2
        println("d: $d")
    end
    a * b * c * d
end

ftest2(1.,2.,3.)


function foo(p1,p2,p3,p4,p5,p6,p7)
    s = Sphere(					# Instantiate N-D sphere, here N=2 (circle)
        [p1,p2],				# c: center
        p3,						# r: "radii" (half span of each axis)
        ε_tensor(p4),		    # data: any type, data associated with circle shape
        )
    ε_bg = ε_tensor(p5)
    # sum(abs2.(bounds(s)[1])) #*p1
    # bounds(s)[1][1] #*p1
    # r₀,nout = surfpt_nearby2([p6,p7],s)
    r₀,nout = surfpt_nearby2([p6,p7],s)
    vxl1 = SVector(1.3,2.4)
    vxl2 = SVector(-1.2,6.4)
    rvol = volfrac((vxl1,vxl2),nout,r₀)
    kottke_avg_param(s.data, ε_bg, [nout[1];nout[2];0.0], rvol)[1,2]
end

foo(0.13,0.12,1.3,2.24,3.5,0.34,0.41)
@show Zygote.gradient(foo,0.13,0.12,1.3,2.24,3.5,0.34,0.41)

##
@show p8
@show p9
@show p10
@show p11
@show [[1  2];  [3  4]]


ax = [      ax11     ax21
            ax12     ax22      ]
               # d: size of the box in axis directions (cols of ax)
SMatrix{2,2,Float64,4}(ax)

function boo(ax11_v,ax12_v,ax21_v,ax22_v,c1_v,c2_v,d1_v,d2_v,n1_v,n2_v,xx_v,yy_v,dx_v,dy_v)
    ## columns of ax are the axes of the box
    b_ax = [      ax11_v     ax21_v
                ax12_v     ax22_v      ]
    #ax1b,ax2b = GeometryPrimitives.normalize.(([p8,p9], [p10,p11]))
    b = Box(					# Instantiate N-D sphere, here N=2 (circle)
        [c1_v,c2_v],				# c: center
        [d1_v,d2_v],               # d: size of the box in axis directions (cols of ax)
        b_ax,						# r: "radii" (half span of each axis)
        ε_tensor(n1_v),		    # data: any type, data associated with circle shape
        )
    ε_bg = ε_tensor(n2_v)
    # sum(abs2.(bounds(s)[1])) #*p1
    # bounds(s)[1][1] #*p1
    # r₀,nout = surfpt_nearby2([p6,p7],s)
    # r₀,nout = surfpt_nearby2([xx_v; yy_v],b)
    r₀,nout = surfpt_nearby2([xx_v;yy_v],b)
    vxl2 = SVector{2}(xx_v+dx_v/2.,yy_v+dy_v/2.)
    vxl1 = SVector{2}(xx_v-dx_v/2.,yy_v-dy_v/2.)
    rvol = volfrac((vxl1,vxl2),nout,r₀)
    kottke_avg_param(b.data, ε_bg, [nout[1];nout[2];0.0], rvol)[1,1]
end

##
ax11    =   1.1
ax12    =   0.02
ax21    =   0.2
ax22    =   1.05
c1      =   0.1
c2      =   0.
d1      =   1.0
d2      =   1.0
n1      =   2.5
n2      =   3.5
xx      =   0.6
yy      =   0.
dx      =   0.03
dy      =   0.03
##
boo(ax11,ax12,ax21,ax22,c1,c2,d1,d2,n1,n2,xx,yy,dx,dy)
boox(q) = q -> boo(ax11,ax12,ax21,ax22,c1,c2,d1,d2,n1,n2,q,yy,dx,dy)
dboox(q) = q -> Zygote.gradient(boo,ax11,ax12,ax21,ax22,c1,c2,d1,d2,n1,n2,q,yy,dx,dy)[end-3]
using Plots: plot, plot!
x = 0.55:0.001:0.65
plot(x,boox(x))
plot(x,dboox(x))
# 3×3 Array{Float64,2}:
#  9.37196  -0.0           0.0
#  0.0      10.7091        2.66454e-15
#  0.0       2.66454e-15  10.7091
Zygote.gradient(boo,ax11,ax12,ax21,ax22,c1,c2,d1,d2,n1,n2,xx,yy,dx,dy)

ax1b,ax2b = GeometryPrimitives.normalize.(([1.,0.], [0.,1.]))

# SVector(
ntuple(k -> k==1 ? 1.0 : 0.0, 2)

function surfpt_nearby2(x::Vector{Float64}, s::Sphere{2})
    nout = x==s.c ? SVector(1.0,0.0) : # nout = e₁ for x == s.c
                    normalize(x-s.c)
    return s.c+s.r*nout, nout
end

p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12 = 0.13,0.12,1.3,2.24,3.5,0.34,0.41,1.0,0.2,0.0,1.0,1.4
ax1b,ax2b = GeometryPrimitives.normalize.(([p8,p9], [p10,p11]))
b = Box(					# Instantiate N-D sphere, here N=2 (circle)
    [p1,p2],				# c: center
    [p3,p12],
    [ax1b ax2b],						# r: "radii" (half span of each axis)
    ε_tensor(p4),		    # data: any type, data associated with circle shape
    )
for xx=0.4:0.01:5.2,yy=0.4:0.01:5.2

    x = SVector(xx,yy)
    ax = inv(b.p)
    ##
    n = (b.p ./ sqrt.(sum(abs2,b.p,dims=2)[:,1]))  # b.p normalized in row direction.
    cosθ = sum(ax.*n', dims=1)[1,:]  # equivalent to diag(n*ax)
    d = b.p * (x - b.c)
    n = n .* copysign.(1.0,d)'  # operation returns SMatrix (reason for leaving n untransposed)
    absd = abs.(d)
    onbnd = abs.(b.r.-absd) .≤ Base.rtoldefault(Float64) .* b.r  # basically b.r .≈ absd but faster
    isout = (b.r.<absd) .| onbnd
    ∆ = (b.r .- absd) .* cosθ  # entries can be negative
    if count(isout) == 0  # x strictly inside box; ∆ all positive
        case_str_old = "case1: in the box"
        l∆x, i = findmin(∆)  # find closest face
        nout = n[i,:]
        ∆x = l∆x * nout
        xout = x + ∆x
    else  # x outside box or on boundary in one or multiple directions
        ∆x = n' * (∆ .* isout)  # project out .!isout directions
        nout = all(.!isout .| onbnd) ? n'*onbnd : -∆x
        if all(.!isout .| onbnd)
            case_str =  "case2.1: proj onto boundry"
        else
            case_str = "case2.2: just outside "
        end
        nout = normalize(nout)
        xout = x + ∆x
    end

    nout_old = nout
    xout_old = xout


    n0 = b.p ./  [ sqrt(b.p[1,1]^2 + b.p[1,2]^2) sqrt(b.p[2,1]^2 + b.p[2,2]^2)  ]
    d = Array(b.p * (x - b.c))
    cosθ = diag(n0*ax)
    n = n0 .* copysign.(1.0,d)'
    absd = abs.(d)
    ∆ = (b.r .- absd) .* cosθ
    # onbnd = abs.(dropgrad(b).r.-dropgrad(absd)) .≤ Base.rtoldefault(Float64) .* dropgrad(b.r)  # basically b.r .≈ absd but faster
    # isout = (dropgrad(b.r).<dropgrad(absd)) .| dropgrad(onbnd)
    onbnd = Zygote.@ignore abs.(b.r.-absd) .≤ Base.rtoldefault(Float64) .* b.r  # basically b.r .≈ absd but faster
    isout = Zygote.@ignore (b.r.<absd) .| onbnd
    projbnd = Zygote.@ignore all(.!isout .| onbnd)
    onbnd_float = Zygote.@ignore Float64.(onbnd)
    isout_float = Zygote.@ignore Float64.(isout)
    # if ( ( abs(b.r[1]-absd[1]) ≤ Base.rtoldefault(Float64) * b.r[1] ) | (b.r[1]<absd[1]) ) | ( ( abs(b.r[2]-absd[2]) ≤ Base.rtoldefault(Float64) * b.r[2] ) | ( b.r[2] < absd[2] ) )
    if count(isout) == 0
        # = not(isout[1]) &  not(isout[2]) case, point is inside box
        case_str = "case1: in the box"
        l∆x, i = findmin(∆)  # find closest face
        nout = n[i,:]
        ∆x = l∆x * nout
    else
        ∆x = n' * (∆ .* isout_float)
        # = isout[1] | isout[2] case 1: at least one dimension outside or on boundry
        if all(.!isout .| onbnd)
            case_str =  "case2.1: proj onto boundry"
            nout0 = n' * onbnd_float
        else
            case_str = "case2.2: just outside "
            nout0 = -∆x
        end
        nout = nout0 / norm(nout0)
    end
    xout = x + ∆x

    if !((nout_old[1] ≈ nout[1]) & (nout_old[2] ≈ nout[2]) & (xout_old[1] ≈ xout[1]) & (xout_old[2] ≈ xout[2]))
        println("### outputs don't match at x,y: $xx,   $yy")
        println("# old code:")
        println(case_str_old)
        println("nout_old: $nout_old")
        println("xout_old: $xout_old")
        println("# new code:")
        println(case_str_old)
        println("nout: $nout")
        println("xout: $xout")
    end
    # @assert nout_old[1] ≈ nout[1]
    # @assert nout_old[2] ≈ nout[2]
    # @assert xout_old[1] ≈ xout[1]
    # @assert xout_old[2] ≈ xout[2]
end
##
xx1 = 1.3; yy1 = 3.4;
xx2 = 0.2; yy2 = 2.43;
println("### vectors:   x1,y1: $xx1,   $yy1")
println("###            x2,y2: $xx2,   $yy2")
@show v1, v2 =  normalize.(([xx1,yy1], [xx2,yy2]))

dot(v1,v2)


##


sum(abs2,b.p,dims=2)
s = Sphere(					# Instantiate N-D sphere, here N=2 (circle)
    [0.1,0.2],				# c: center
    0.3,						# r: "radii" (half span of each axis)
    ε_tensor(2.2),		    # data: any type, data associated with circle shape
    )
s.c

s = Sphere(					# Instantiate N-D sphere, here N=2 (circle)
    [0.1,0.1],				# c: center
    0.5,						# r: "radii" (half span of each axis)
    ε_tensor(2.2),		    # data: any type, data associated with circle shape
    )
surfpt_nearby([0.2,0.25],s)

surfpt_nearby([0.2,0.1],s)

sum(abs2.(bounds(s)[1]))
