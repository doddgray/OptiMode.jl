# Dispersion models for NLO-relevant materials adapted from
# https://github.com/doddgray/optics_modeling/blob/master/nlo/NLO_tools.py

################################################################################
#            Temperature Dependent Index, Group Index and GVD models           #
#                        for phase-matching calculations                       #
################################################################################
# ng(fₙ::Function) = λ -> ( (n,n_pb) = Zygote.pullback(fₙ,λ); ( n - λ * n_pb(1)[1] ) )
export n, ng, gvd, λ, Material

c = Unitful.c0      # Unitful.jl speed of light
@parameters λ, T
Dλ = Differential(λ)
DT = Differential(T)
ng(n_sym::Num) = n_sym - λ * expand_derivatives(Dλ(n_sym))
gvd(n_sym::Num) = λ^3 * expand_derivatives(Dλ(Dλ(n_sym))) # gvd = uconvert( ( 1 / ( 2π * c^2) ) * _gvd(lm_um,T_C)u"μm", u"fs^2 / mm" )

struct Material{T}
	ε::SMatrix{3,3,T,9}
end

n(mat::Material) = sqrt.(diag(mat.ε))
n(mat::Material,axind::Int) = sqrt(mat.ε[axind,axind])
ng(mat::Material) = ng.(n(mat))
ng(mat::Material,axind::Int) = ng(n(mat,axind))
gvd(mat::Material) = gvd.(n(mat))
gvd(mat::Material,axind::Int) = gvd(n(mat,axind))

materials(shapes) = unique!(getfield.(shapes,:data))

################################################################################
#                                Load Materials                                #
################################################################################
include("material_lib/MgO_LiNbO3.jl")
include("material_lib/SiO2.jl")
include("material_lib/Si3N4.jl")
# include("material_lib/LiB3O5.jl")
# include("material_lib/silicon.jl")
# include("material_lib/GaAs.jl")
# include("material_lib/MgF2.jl")
# include("material_lib/HfO2.jl")
