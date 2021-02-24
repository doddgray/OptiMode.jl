using ModelingToolkit, IfElse, LinearAlgebra, StaticArrays
import LinearAlgebra.norm
norm(x::AbstractVector{Num}) = sqrt(sum(x.^2))

# attempt to create symbolic version of GeometryPrimitives.Polygon

# nv vertices in 2D plane
nv = 4  # first try trapazoid like in ridge waveguide
@parameters w, t, θ # compare with wₜₒₚ,t_core,θ
# derived parameters for vertices
wt_half = w / 2
wb_half = wt_half + ( t * tan(θ) )
tc_half = t / 2
# vertices
v =    	[   wt_half     tc_half
            -wt_half    tc_half
            -wb_half    -tc_half
            wb_half     -tc_half    ]
vs = SMatrix{4,2}(v)
Δv = v - circshift(v,1)
Δvs = SMatrix{4,2}(Δv)

one_mone = [1, -1]
one_mones = SVector(1, -1)
@tullio n[i,j] := Δv[i,3-j] * $one_mone[j] / sqrt(Δv[i,1]^2 + Δv[i,2]^2 ) nograd=one_mone
ns = SMatrix{4,2}(n)
lbnd = transpose(minimum(v;dims=1))
ubnd = transpose(maximum(v;dims=1))

lbnds = transpose(minimum(vs;dims=1))
ubnds = transpose(maximum(vs;dims=1))
bnds = lbnd, ubnd

function Δxe(x,v,n)
    @tullio res[k] := n[k,a] * ( x[a] - v[k,a] )
end

minbool(x) = map(a->(a <= minimum(x)), x)
maxbool(x) = map(a->(a >= maximum(x)), x)
minone(x) = IfElse.ifelse.(x .<= minimum(x), 1, 0)
maxone(x) = IfElse.ifelse.(x .>= maximum(x), 1, 0)
atmin(x,y) = sum(IfElse.ifelse.(y .<= minimum(y), x, 0))
atmax(x,y) = sum(IfElse.ifelse.(y .>= maximum(y), x, 0))

# test atmin, atmax
# @parameters u[1:10], q[1:10]
# fminbool,fminbool! = eval.(build_function(minbool(q),q))
# fmaxbool,fmaxbool! = eval.(build_function(maxbool(q),q))
# fatmin = eval(build_function(atmin(u,q),u,q))
# fatmax = eval(build_function(atmax(u,q),u,q))
# un = [1,2,3,4,5,6,7,8,9,10]
# qn = [3,4,3,2,10,5,3,5,1,2]
# fmaxbool(un)
# fatmin(un,qn)
# fatmax(un,qn)

function pgon_cout2(x,v,n)
	Δxv = x' .- v
	lΔxv = sqrt.(sum(x->x^2,dxv,dims=2))
	imin = minone(lΔxv)
	surf = sum(v.*imin,dims=1)[1,:]
	# IfElse.ifelse(obd(x)&&)
	# imin₋₁ = circshift(imin,-1) #for ifelse if needed
	n0 = x-surf
	# return surf, n0./norm(n0)
	hcat(surf, n0./norm(n0))
end

function pgon_cout_not2(x,v,n,Δxe)
	imax = maxone(Δxe)
	vmax = sum(imax.*v,dims=1)[1,:]
	nout = sum(imax.*n,dims=1)[1,:]
	Δx = (nout⋅(vmax-x)) .* nout
    surf = x + Δx
	# return surf, nout
	hcat(surf, nout)
end

function spn_pgon(x,v,n)
    # @tullio Δxe[k] := n[k,a] * ( x[a] - v[k,a] )
	Δxe = sum(n .* (x' .- v), dims=2)[:,1]
    # cout = sum( 1 .- signbit.( Δxe ) )
	# cout = sum( Δxe .> 0 )
	cout = sum(IfElse.ifelse.(Δxe .> 0, 1, 0))
	res = IfElse.ifelse(
		isequal(cout,2),
		pgon_cout2(x,v,n),
		pgon_cout_not2(x,v,n,Δxe)
		)
	return res #::Matrix
end

# fspn_pgon,fspn_pgon! = eval.(build_function(spn_pgon(x,v,n),x,w,t,θ))
fspn_pgon,fspn_pgon! = eval.(build_function(spn_pgon(x,v,n),x,ps))
fDspn_pgon = eval(build_function(Dspn_pgon(x,v,n),x,ps))
fspn_pgons,fspn_pgons! = eval.(build_function(spn_pgon(xs,vs,ns),x,w,t,θ))

fpgon_cout2,fpgon_cout2! = eval.(build_function(pgon_cout2(x,v,n),x,w,t,θ))
fpgon_cout_not2,fpgon_cout_not2! = eval.(build_function(pgon_cout_not2(x,v,n,Δxe(x,v,n)),x,w,t,θ))

xnum = [0.3,0.5]
@parameters x[1:2]
spn_pgon(xnum,v,n)
spn_pgon(x,v,n)

f1 = eval(build_function(spn_pgon(xnum,v,n),w,t,θ))
f2,f2! = eval.(build_function(Δxe(xnum,v,n),w,t,θ))
f3,f3! = eval.(build_function(Δxe(x,v,n),x,w,t,θ))
f4 = eval(build_function(spn_pgon(x,v,n),x,w,t,θ))
flbnd,flbnd! = eval.(build_function(lbnd,w,t,θ))
fubnd,fubnd! = eval.(build_function(ubnd,w,t,θ))
fbnd(a,b,c) = flbnd(a,b,c),fubnd(a,b,c)
# @variables v[nv,2]

# compare with original functions
using GeometryPrimitives, StaticArrays, LinearAlgebra
function pg(w,t,θ)
    wt_half = w / 2
    wb_half = wt_half + ( t * tan(θ) )
    tc_half = t / 2
    # vertices
    verts = SMatrix{4,2}( [  wt_half     tc_half
                            -wt_half    tc_half
                            -wb_half    -tc_half
                            wb_half     -tc_half    ] )
    GeometryPrimitives.Polygon(					                        # Instantiate 2D polygon, here a trapazoid
                    verts,			                            # v: polygon vertices in counter-clockwise order
                    2.4,					                                    # data: any type, data associated with box shape
                )
end

bounds(pg1)
function _Δxe_poly(x::SVector{2},v::SMatrix{K,2},n::SMatrix{K,2})::SVector{K}  where {K} #,T<:Real}
	@tullio out[k] := n[k,a] * ( x[a] - v[k,a] ) # edge line eqns for a K-point Polygon{K} `s`
end
function Δxe_poly(x::AbstractVector{<:Real},s::Polygon{K})::SVector{K} where K
	_Δxe_poly(x,s.v,s.n)
end
function onbnd(Δxe::SVector{K},s::Polygon{K})::SVector{K,Bool} where K
	map(x->abs(x)≤s.rbnd,Δxe)
end
function onbnd(x::SVector{2},s::Polygon)::SVector{K,Bool} where K
	map(x->abs(x)≤s.rbnd,Δxe_poly(x,s))
end
function cout(x::SVector{2},s::Polygon)::Int
	mapreduce((a,b)->Int(a|b),+,onbnd(x,s),map(isposdef,Δxe_poly(x,s)))
end
function cout(Δxe::SVector{K,<:Real},obd::SVector{K,Bool})::Int where K
	mapreduce((a,b)->Int(a|b),+,obd,map(isposdef,Δxe))
end

pg1 = pg(1.7,0.7,π/14.0)
# x = SVector(0.2,0.3)
∆xe1 = _Δxe_poly(SVector{2}(xnum),pg1.v,pg1.n)
obd1 = onbnd(∆xe1,pg1)
co1 = cout(∆xe1,obd1)


## reference code

p = [
       1.7,                #   top ridge width         `w_top`         [μm]
       0.7,                #   ridge thickness         `t_core`        [μm]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
       2.4,                #   core index              `n_core`        [1]
       1.4,                #   substrate index         `n_subs`        [1]
       0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
               ];
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
rwg(p) = ridge_wg(p[1],p[2],p[3],p[6],p[4],p[5],Δx,Δy)
ms = ModeSolver(1.45, rwg(p), Δx, Δy, Δz, Nx, Ny, Nz)

function ridge_wg2(wₜₒₚ::T1,t_core::T1,θ::T1,edge_gap::T1,ε_core,ε_subs,Δx::T2,Δy::T2)::Vector{<:GeometryPrimitives.Shape} where {T1<:Real,T2<:Real}
    t_subs = (Δy -t_core - edge_gap )/2.
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
    # ε_core = ε_tensor(n_core)
    # ε_subs = ε_tensor(n_subs)
    wt_half = wₜₒₚ / 2
    wb_half = wt_half + ( t_core * tan(θ) )
    tc_half = t_core / 2
    verts =     [   wt_half     -wt_half     -wb_half    wb_half
                    tc_half     tc_half    -tc_half      -tc_half    ]'
    core = GeometryPrimitives.Polygon(					                        # Instantiate 2D polygon, here a trapazoid
                    SMatrix{4,2}(verts),			                            # v: polygon vertices in counter-clockwise order
                    ε_core,					                                    # data: any type, data associated with box shape
                )
    ax = [      1.     0.
                0.     1.      ]
    b_subs = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
                    [0. , c_subs_y],            	# c: center
                    [Δx - edge_gap, t_subs ],	# r: "radii" (half span of each axis)
                    ax,	    		        # axes: box axes
                    ε_subs,					        # data: any type, data associated with box shape
                )
    return [core,b_subs]
end

rwg2(p) = ridge_wg2(p[1],p[2],p[3],p[6],MgO_LiNbO₃,SiO₂,Δx,Δy)
shapes = rwg(p)
shapes2 = rwg2(p)
