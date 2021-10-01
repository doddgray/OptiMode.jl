using OptiMode
using LinearAlgebra, Statistics
using StaticArrays
using GeometryPrimitives
using ChainRules
using Zygote #, ForwardDiff, FiniteDifferences
using Optim, Interpolations
using Zygote: @ignore, dropgrad
using Rotations
using StaticArrays: Dynamic
using Rotations: RotY, MRP
using FFTW
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

function E_relpower_xyz(ms::ModeSolver{ND,T},ω²H) where {ND,T<:Real}
    E = 1im * ε⁻¹_dot( fft( kx_tc( reshape(ω²H[2],(2,size(ms.grid)...)),mn(ms),ms.M̂.mag), (2:1+ND) ), flat( ms.M̂.ε⁻¹ ))
    Es = reinterpret(reshape, SVector{3,Complex{T}},  E)
    Pₑ_xyz_rel = normalize([mapreduce((ee,epss)->(abs2(ee[a])*inv(epss)[a,a]),+,Es,ms.M̂.ε⁻¹) for a=1:3],1)
    return Pₑ_xyz_rel
end

TE_filter = (ms,ω²H)->E_relpower_xyz(ms,ω²H)[1]>0.7
TM_filter = (ms,ω²H)->E_relpower_xyz(ms,ω²H)[2]>0.7
oddX_filter = (ms,αX)->sum(abs2,𝓟x̄(ms.grid)*αX[2])>0.7
evenX_filter = (ms,αX)->sum(abs2,𝓟x(ms.grid)*αX[2])>0.7


LNx = rotate(MgO_LiNbO₃,Matrix(MRP(RotY(π/2))))
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
grid = Grid(Δx,Δy,Nx,Ny)
# p_pe = [
#        1.7,                #   top ridge width         `w_top`         [μm]
#        0.7,                #   ridge thickness         `t_core`        [μm]
#        0.5,                #   top layer etch fraction `etch_frac`     [1]
#        π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
#                ];
p_pe = [
    1.85,        # 700 nm top width of angle sidewall ridge
    0.7,        # 600 nm MgO:LiNbO₃ ridge thickness
    3.4 / 7.0,    # etch fraction (they say etch depth of 500 nm, full thickness 600 nm)
    0.5236,      # 30° sidewall angle in radians (they call it 60° , in our params 0° is vertical)
]
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy)
rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).

geom_pe = rwg_pe(p_pe)
# ms = ModeSolver(0.9, rwg_pe(p_pe), grid)
p_pe_lower = [0.8, 0.3, 0., 0.01]
p_pe_upper = [2.5, 2., 1., π/4.]


# λs = reverse(1.45:0.02:1.65)
# ωs = 1 ./ λs

nω_jank = 20
ωs_jank = collect(range(1/2.25,1/1.95,length=nω_jank)) # collect(0.416:0.01:0.527)
λs_jank = 1 ./ ωs_jank

# ωs = ωs_jank #collect(range(0.6,0.7,length=10))
# λs = λs_jank #inv.(ωs)

# n1F,ng1F = solve_n(ms,ωs,rwg_pe(p_pe)); n1S,ng1S = solve_n(ms,2*ωs,rwg_pe(p_pe))
##

function sum_Δng²_FHSH(ωs,p)
    ms = Zygote.@ignore(ModeSolver(kguess(1/2.25,rwg_pe(p)), rwg(p), Grid(6.0,4.0,128,128); nev=4)) #ModeSolver(1.45, rwg_pe(p), gr)
	nω = length(ωs)
    # ngs_FHSH = solve_n(
	# 	ms,
	# 	vcat(ωs, 2*ωs ),
	# 	rwg_pe(p);
	# 	f_filter=(ms,ω²H)->E_relpower_xyz(ms,ω²H)[1]>0.6)[2]
	# ms = Zygote.@ignore ModeSolver(kguess(ωs[1],rwg_pe(p)), rwg_pe(p), grid; nev=4)

	nFS,ngFS,gvdFS,EFS = solve_n(
	    ms,
	    vcat(ωs,2*ωs),
	    rwg_pe(p);
	    f_filter=(ms,ω²H)->E_relpower_xyz(ms,ω²H)[1]>0.6)

    ngs_FH = ngFS[1:nω]
	ngs_SH = ngFS[nω+1:2*nω]
    Δng² = abs2.(ngs_SH .- ngs_FH)
    sum(Δng²)
end

# sum_Δng_FHSH(ωs,p_pe)

# warmup
println("warmup function runs")
p0 = copy(p_pe)
@show sum_Δng²_FHSH(ωs_jank,p_pe)
# @show vng0, vng0_pb = Zygote.pullback(x->var_ng(ωs,x),p0)
# @show grad_vng0 = vng0_pb(1)

# define function that computes value and gradient of function `f` to be optimized
# according to https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/
function fg!(F,G,x)
    value, value_pb = Zygote.pullback(x) do x
       # var_ng(ωs,x)
	   nω = 20
	   ωs = collect(range(1/2.25,1/1.95,length=nω))
	   sum_Δng²_FHSH(ωs,x)
    end
    if G != nothing
        G .= value_pb(1)[1]
    end
    if F != nothing
        # F = value
        return value
    end
end

# G0 = [0.,0.,0.,0.]
# @show fg!(0.,G0,p0)
# println("G0 = $G0")
##
rand_p0() = p_pe_lower .+ [rand()*(p_pe_upper[i]-p_pe_lower[i]) for i=1:4]

opts =  Optim.Options(
                        outer_iterations = 4,
                        iterations = 6,
                        # time_limit = 3*3600,
                        store_trace = true,
                        show_trace = true,
                        show_every = 1,
                        extended_trace = true,
                        x_tol = 1e-4, # Absolute tolerance in changes of the input vector x, in infinity norm. Defaults to 0.0.
                        f_tol = 1e-5, # Relative tolerance in changes of the objective value. Defaults to 0.0.
                        g_tol = 1e-5, # Absolute tolerance in the gradient, in infinity norm. Defaults to 1e-8. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
                    )

println("########################### Opt 1 ##########################")

# p_opt1 = 0.62929, 0.71422,0.7658459,0.125366
res1 = optimize( Optim.only_fg!(fg!),
                p_pe_lower,
                p_pe_upper,
                rand_p0(),
                Fminbox(Optim.BFGS()),
                opts,
            )


println("########################### Opt 2 ##########################")

# res2 = optimize( Optim.only_fg!(fg!),
#                 rand_p0(),
#                 Optim.BFGS(),
#                 opts,
#             )

res2 = optimize( Optim.only_fg!(fg!),
                p_pe_lower,
                p_pe_upper,
                rand_p0(),
                Fminbox(Optim.BFGS();mu0=1e-7),
                opts,
            )

println("########################### Opt 3 ##########################")

# res3 = optimize( Optim.only_fg!(fg!),
#                 rand_p0(),
#                 Optim.BFGS(),
#                 opts,
#             )

res3 = optimize( Optim.only_fg!(fg!),
                p_pe_lower,
                p_pe_upper,
                rand_p0(),
                Fminbox(Optim.BFGS();mu0=1e-6),
                opts,
            )
