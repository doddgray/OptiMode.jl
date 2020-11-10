using Revise, StaticArrays, ChainRules, FiniteDifferences, ForwardDiff, Zygote
using OptiMode
using GeometryPrimitives
SHM3 = SHermitianCompact{3,Float64,6}   # static Hermitian 3×3 matrix Type, renamed for brevity

###############################################################################
#                                                                             #
#                                  Rules                                      #
#                                                                             #
###############################################################################

# @Zygote.adjoint (T::Type{<:SMatrix})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
# @Zygote.adjoint (T::Type{<:SMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
# @Zygote.adjoint (T::Type{<:SVector})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
# @Zygote.adjoint (T::Type{<:SVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)

@Zygote.adjoint reinterpret(::Type{T}, x::T) where T = x, Δ->(nothing, Δ)
@Zygote.adjoint reinterpret(::Type{NTuple{N,T}}, xs::AbstractArray{T}) where {N,T} = reinterpret(NTuple{N,T}, xs), Δ->(nothing, reinterpret(T, Δ))
@Zygote.adjoint reinterpret(::Type{T}, xs::AbstractArray{NTuple{N,T}}) where {N,T} = reinterpret(T, xs), Δ->(nothing, reinterpret(NTuple{N,T}, Δ))
@Zygote.adjoint reinterpret(::Type{T}, x) where T = reinterpret(T, x), _ -> error("Non-trivial reinterpreting is not supported")

Zygote._zero(xs::AbstractArray{<:AbstractArray}, T=eltype(xs)) = eltype(xs)[_zero(x,eltype(T)) for x in xs]

Zygote.refresh()

###############################################################################
#                                                                             #
#                                 Parameters                                  #
#                                                                             #
###############################################################################

w               =               1.7
t_core          =               0.7
edge_gap        =               0.5             # μm
n_core          =               2.4
n_subs          =               1.4

Δx              =               6.              # μm
Δy              =               4.              # μm
Nx              =               60
Ny              =               40


###############################################################################
#                                                                             #
#                               Generate Data                                 #
#                                                                             #
###############################################################################

g   =   MaxwellGrid(Δx,Δy,Nx,Ny)
s   =   ridge_wg(w,t_core,edge_gap,n_core,n_subs,g)
ε   =   εₛ(s,g);
plot_ε(s,g)



##

function circ_wg2(w,t_core,edge_gap::Float64,n_core::Float64,n_subs::Float64,g::MaxwellGrid)::Array{GeometryPrimitives.Shape{2,4,SHM3},1}
    t_subs = (g.Δy -t_core - edge_gap )/2.
    c_subs_y = -g.Δy/2. + edge_gap/2. + t_subs/2.
    ε_core = ε_tensor(n_core)
    ε_subs = ε_tensor(n_subs)
    ax1,ax2 = GeometryPrimitives.normalize.(([1.,0.], [0.,1.]))
    b_core = GeometryPrimitives.Sphere(					# Instantiate N-D sphere, here N=2 (circle)
                    SVector(0.,t_core),			# c: center
                    w,						# r: "radii" (half span of each axis)
                    ε_core,					        # data: any type, data associated with box shape
                )
    return [b_core,b_subs]
end

function foo(p) #;g=MaxwellGrid(6.,4.,30,20))
    s   =   ridge_wg(p[1],p[2],edge_gap,n_core,n_subs,g)
    tree = KDTree(s)
    ε   =    [εₛ(s,tree,g.x[i],g.y[j],g.δx,g.δy)[k,l] for i=1:g.Nx,j=1:g.Ny,k=1:3,l=1:3]
    return vec(ε)
end

function goo(r) #;g=MaxwellGrid(6.,4.,30,20))
    s   =   [ circ_wg2(r,t_core,edge_gap,n_core,n_subs,g)[1] ]
    tree = KDTree(s)
    ε   =    [εₛ(s,tree,g.x[i],g.y[j],g.δx,g.δy)[k,l] for i=1:g.Nx,j=1:g.Ny,k=1:3,l=1:3]
    return vec(ε)
end

function unflatten(v,Nx,Ny)
    v_rs = reshape(v,(Nx,Ny,3,3))
    return [SMatrix{3,3,Float64,9}(v_rs[i,j,:,:]) for i=1:Nx,j=1:Ny]
end

##
ε = unflatten(foo((1.5,0.8)),Nx,Ny); # returns Array{Float64,4} with size(out)= (Nx,Ny,3,3);
δε_fd = FiniteDifferences.jacobian(central_fdm(3,1),foo,(1.5,0.8))[1];          
δw_ε_fd = unflatten(δε_fd[:,1],Nx,Ny);
δt_ε_fd = unflatten(δε_fd[:,2],Nx,Ny);
##

plot_ε(δw_ε_fd,g)
# plot_ε(δt_ε_fd,g)
# plot_ε(ε,g)

##
using ForwardDiff
# ForwardDiff.jacobian(foo,[1.5,0.8])
ForwardDiff.derivative(goo,1.1)

## 

ReverseDiff.jacobian(foo,[1.5,0.8])
##
using Zygote
Zygote.gradient(1.1) do a
    sum(goo(a))
end
# Zygote.gradient(1.1) do a
#     Zygote.forwarddiff(a) do a
#         goo(a)
#     end
# end

##

##
ε = unflatten(goo(1.1),Nx,Ny); # returns Array{Float64,4} with size(out)= (Nx,Ny,3,3);
δε_fd = FiniteDifferences.jacobian(central_fdm(3,1),goo,1.1)[1];          
δr_ε_fd = unflatten(δε_fd[:,1],Nx,Ny);
##

plot_ε(δr_ε_fd,g)