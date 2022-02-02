## Playing around with new array functionality in Symbolics.jl after a long time away from material model code
using Symbolics, LinearAlgebra
using BenchmarkTools
# includet("rules.jl")
using Symbolics: get_variables, wrap, unwrap, MakeTuple, toexpr, substitute, value
# using ReversePropagation: gradient_expr
using SymbolicUtils: @rule, @acrule, @slots, RuleSet
using SymbolicUtils.Rewriters: Chain, RestartedChain, PassThrough, Prewalk, Postwalk
using Symbolics: unwrap, wrap
r_sqrt_pow = @rule sqrt(~x) --> (~x)^(1//2)
r_pow_sqrt = Prewalk( PassThrough( @rule (~x)^(1//2) --> sqrt(~x) ); threaded=true )
# r_neg_exp = Prewalk(PassThrough( @rule  ( 1 / ~x )^(~a) => (~x)^(-(~a)) ) )
# r_lor_inv = Prewalk(PassThrough( @acrule ~a / ( ~b + (~x)^-2 ) =>  ~a * (~x)^2 / ( ~b * (~x)^2 + 1 ) ) )
r_neg_exp = Prewalk(PassThrough( @rule  ( 1 / ~x )^(~a) =>  1 / (~x)^(~a)  ); threaded=true )
r_lor_inv = Prewalk(PassThrough( @acrule ~a / ( ~b + 1 / (~x)^2  ) =>  ~a * (~x)^2 / ( ~b * (~x)^2 + 1 ) ); threaded=true )
r_exp_prod = @acrule((~a)^(~x) * (~a)^(~y) => (~a)^(~x + ~y))
my_rules = Chain([r_neg_exp,r_lor_inv])
# my_simp(x) = simplify( flatten_fractions( simplify( x, rewriter=my_rules ) ), rewriter=r_pow_sqrt )
my_simp(x) = flatten_fractions( simplify( x, rewriter=my_rules ) )
# function RP_simp_grad(x::Num)
#     vars = (Symbolics.wrap.(Symbolics.get_variables(x))...,)
#     fx_simp, grad_f = ReversePropagation.gradient(x,vars)(vars)
#     return fx_simp, grad_f
# end
# println("before: ",nₒ²);println("after:  ",simplify(nₒ², my_rules))


get_array_vars(A) = mapreduce(x->wrap.(get_variables(x)),union,A)

# function pb_code(x,vars)
#     code, final_var, gradient_vars = gradient_expr(x,vars)
#     input_vars = toexpr(Symbolics.MakeTuple(vars))
#     final = toexpr(final_var)
#     gradient = toexpr(Symbolics.MakeTuple(gradient_vars))

#     full_code = quote
#         ($input_vars, ) -> begin
#             $code
#             return $(final), $(gradient)
#         end
#     end

#     return full_code
# end

# function array_pb_code(A)
#     vars = get_array_vars(A)
#     return pb_code.(A,(vars,))
# end

function n²_MgO_LiNbO₃_sym(λ, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    f = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    λ² = λ^2
    a₁ + b₁*f + (a₂ + b₂*f) / (λ² - (a₃ + b₃*f)^2) + (a₄ + b₄*f) / (λ² - a₅^2) - a₆*λ²
end

pₑ = (
    a₁ = 5.756,
    a₂ = 0.0983,
    a₃ = 0.202,
    a₄ = 189.32,
    a₅ = 12.52,
    a₆ = 1.32e-2,
    b₁ = 2.86e-6,
    b₂ = 4.7e-8,
    b₃ = 6.113e-8,
    b₄ = 1.516e-4,
    T₀ = 24.5,      # reference temperature in [Deg C]
)
pₒ = (
    a₁ = 5.653,
    a₂ = 0.1185,
    a₃ = 0.2091,
    a₄ = 89.61,
    a₅ = 10.85,
    a₆ = 1.97e-2,
    b₁ = 7.941e-7,
    b₂ = 3.134e-8,
    b₃ = -4.641e-9,
    b₄ = -2.188e-6,
    T₀ = 24.5,      # reference temperature in [Deg C]
)


function n²_sym_fmt1( λ ; A₀=1, B₁=0, C₁=0, B₂=0, C₂=0, B₃=0, C₃=0, kwargs...)
    λ² = λ^2
    A₀  + ( B₁ * λ² ) / ( λ² - C₁ ) + ( B₂ * λ² ) / ( λ² - C₂ ) + ( B₃ * λ² ) / ( λ² - C₃ )
end

p_n²_Si₃N₄= (
    A₀ = 1,
    B₁ = 3.0249,
    C₁ = (0.1353406)^2,         #                           [μm²]
    B₂ = 40314,
    C₂ = (1239.842)^2,          #                           [μm²]
    dn_dT = 2.96e-5,            # thermo-optic coefficient  [K⁻¹]
    T₀ = 24.5,                  # reference temperature     [°C]
)

# n²_Si₃N₄(λ,T) = (sqrt(n²_sym_fmt1( λ ; p_n²_Si₃N₄...)) + p_n²_Si₃N₄.dn_dT  *  ( T - p_n²_Si₃N₄.T₀  ))^2
n²_Si₃N₄(λ,T) = ( n²_sym_fmt1( λ ; p_n²_Si₃N₄...)^(1//2) + p_n²_Si₃N₄.dn_dT  *  ( T - p_n²_Si₃N₄.T₀ ) )^2

"""
Create and return a local Cartesian coordinate system `S` (an ortho-normal 3x3 matrix) from a 3-vector `n0`.
`n0` inputs will be outward-pointing surface-normal vectors from shapes in a geometry, and `S` matrix outputs
will be used to rotate dielectric tensors into a coordinate system with two transverse axes and one perpendicular
axis w.r.t a (locally) planar dielectric interface. This allows transverse and perpendicular tensor components
to be smoothed differently, see Kottke Phys. Rev. E paper.

The input 3-vector `n` is assumed to be normalized such that `sum(abs2,n) == 1` 
"""
function normcart(n)
    ### assume input n is normalized, but if not normalize input vector `n0` to get `n` 
	# n = n0 / (n0[1]^2 + n0[2]^2 + n0[3]^2)^(1//2) # = norm(n0)   
	### Pick `h` to be a vector that is perpendicular to n, some arbitrary choices to make here
	# h = any(iszero.(n)) ? n × normalize(iszero.(n)) :  n × [1, 0 , 0]
    h = [0, n[3], -n[2] ] # = n × [ 1, 0, 0 ]  for now ignore edge case where n = [1,0,0]
	v = n × h   # the third unit vector `v` is just the cross of `n` and `h`
    S = [ n h v ]  # S is a unitary 3x3 matrix
    return S
end

function τ_trans(ε)
    return               [     -1/ε[1,1]            ε[1,2]/ε[1,1]                           ε[1,3]/ε[1,1]
                                ε[2,1]/ε[1,1]       ε[2,2] - ε[2,1]*ε[1,2]/ε[1,1]           ε[2,3] - ε[2,1]*ε[1,3]/ε[1,1]
                                ε[3,1]/ε[1,1]       ε[3,2] - ε[3,1]*ε[1,2]/ε[1,1]           ε[3,3] - ε[3,1]*ε[1,3]/ε[1,1]       ]
end

function τ⁻¹_trans(τ)
    return           [      -1/τ[1,1]           -τ[1,2]/τ[1,1]                          -τ[1,3]/τ[1,1]
                            -τ[2,1]/τ[1,1]       τ[2,2] - τ[2,1]*τ[1,2]/τ[1,1]           τ[2,3] - τ[2,1]*τ[1,3]/τ[1,1]
                            -τ[3,1]/τ[1,1]       τ[3,2] - τ[3,1]*τ[1,2]/τ[1,1]           τ[3,3] - τ[3,1]*τ[1,3]/τ[1,1]          ]
end

function avg_param(ε_fg, ε_bg, S, rvol1)
    τ1 = τ_trans(transpose(S) * ε_fg * S)  # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans(transpose(S) * ε_bg * S)  # express param2 in S coordinates, and apply τ transform
    τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)   # volume-weighted average
    return S * τ⁻¹_trans(τavg) * transpose(S)  # apply τ⁻¹ and transform back to global coordinates
end

@variables ω, T, r, λ
Dom = Differential(ω)

# Use this once Symbolics Array variables work well
# @variables ε₁[1:3,1:3](ω), ε₂[1:3,1:3](ω), S[1:3,1:3]
# @variables ε₁[1:3,1:3], ε₂[1:3,1:3], S[1:3,1:3]

# Until Symbolics Array variables work out some kinks, just make the 3x3 matrices out of scalar real variables

@variables ε₁_11, ε₁_12, ε₁_13, ε₁_21, ε₁_22, ε₁_23, ε₁_31, ε₁_32, ε₁_33, ε₂_11, ε₂_12, ε₂_13, ε₂_21, ε₂_22, ε₂_23, ε₂_31, ε₂_32, ε₂_33 
ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_21  ε₁_22  ε₁_23 ; ε₁_31  ε₁_32  ε₁_33 ] 
ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_21  ε₂_22  ε₂_23 ; ε₂_31  ε₂_32  ε₂_33 ]
@variables n_1, n_2 #, n_3    try enforcing normalization of 3-vector `n` by building it with two real scalar variables
n = [ n_1, n_2, (1 - n_1^2 - n_2^2)^(1//2)] # [ n_1, n_2, n_3 ] / sqrt( n_1^2 + n_2^2 + n_3^2 )  # simplification works better with (...)^(1//2) than sqrt(...)
S = normcart(n)


nₒ² = n²_MgO_LiNbO₃_sym(1/ω, pₒ.T₀; pₒ...)
nₑ² = n²_MgO_LiNbO₃_sym(1/ω, pₑ.T₀; pₑ...)
εLN 	= diagm([nₒ², nₒ², nₑ²])
εSN 	= n²_Si₃N₄(1/ω,p_n²_Si₃N₄.T₀) * diagm([1,1,1])

# nₒ² = n²_MgO_LiNbO₃_sym(λ, pₒ.T₀; pₒ...)
# nₑ² = n²_MgO_LiNbO₃_sym(λ, pₑ.T₀; pₑ...)
# εLN 	= diagm([nₒ², nₒ², nₑ²])
# εSN 	= n²_Si₃N₄(λ,p_n²_Si₃N₄.T₀) * diagm([1,1,1])

# εₑ = avg_param(ε₁, ε₂, S, r);
# εₑ_subs = substitute.(
#     εₑ,
#     (Dict(
#         ε₁[1,1] => εLN[1,1],ε₁[1,2] => εLN[1,2],ε₁[1,3] => εLN[1,3],
#         ε₁[2,1] => εLN[2,1],ε₁[2,2] => εLN[2,2],ε₁[2,3] => εLN[2,3],
#         ε₁[3,1] => εLN[3,1],ε₁[3,2] => εLN[3,2],ε₁[3,3] => εLN[3,3],
#         ε₂[1,1] => εSN[1,1],ε₂[1,2] => εSN[1,2],ε₂[1,3] => εSN[1,3],
#         ε₂[2,1] => εSN[2,1],ε₂[2,2] => εSN[2,2],ε₂[2,3] => εSN[2,3],
#         ε₂[3,1] => εSN[3,1],ε₂[3,2] => εSN[3,2],ε₂[3,3] => εSN[3,3],
#     ),)
# )

# εₑ = simplify.(avg_param(εLN, εSN, S, r));
εₑ = simplify.(avg_param(my_simp.(εLN), my_simp.(εSN), my_simp.(S), r));
∂ωεₑ = simplify.(expand_derivatives.(Dom.(εₑ)));
∂²ωεₑ = simplify.(expand_derivatives.((Dom*Dom).(εₑ)));
# ∂ω_nₒ² = ReversePropagation.gradient(nₒ², ω)(ω)[2][1]
##
# εₑ_pb_codes = array_pb_code(simplify(εₑ, rewriter=r_pow_sqrt ))
# ∂ωεₑ_pb_codes = array_pb_code(simplify(∂ωεₑ, rewriter=r_pow_sqrt ))
# ∂²ωεₑ_pb_codes = array_pb_code(simplify(∂²ωεₑ, rewriter=r_pow_sqrt ))

# εₑ_pb_fns = eval.(εₑ_pb_codes)
# ∂ωεₑ_pb_fns = eval.(∂ωεₑ_pb_codes)
# ∂²ωεₑ_pb_fns = eval.(∂²ωεₑ_pb_codes)


# εₑ_pb_fn(x) = [ ff->ff(x) for ff in εₑ_pb_fns ]
# ∂ωεₑ_pb_fn(x) = [ ff->ff(x) for ff in ∂ωεₑ_pb_fns ]
# ∂²ωεₑ_pb_fn(x) = [ ff->ff(x) for ff in ∂²ωεₑ_pb_fns ]

##
mats = [MgO_LiNbO₃,Si₃N₄,SiO₂,LiB₃O₅];
ε_mats = map(mm->substitute.(get_model(mm,:ε,:λ),(Dict([(λ=>1/ω),]),)),mats);
εₑ_mats = map(subsets(1:length(mats),2)) do mat_inds
    simplify.(avg_param(my_simp.(ε_mats[mat_inds[1]]), my_simp.(ε_mats[mat_inds[2]]), my_simp.(S), r))
end
∂ωε_mats = map(eps->simplify.(expand_derivatives.(Dom.(eps))),ε_mats) ;
∂²ωε_mats = map(eps->simplify.(expand_derivatives.((Dom*Dom).(eps))),ε_mats) ;
∂ωεₑ_mats = map(eps->simplify.(expand_derivatives.(Dom.(eps))),εₑ_mats) ;
∂²ωεₑ_mats = map(eps->simplify.(expand_derivatives.((Dom*Dom).(eps))),εₑ_mats) ;

##
# using Symbolics: MultithreadedForm, SerialForm
genfn_dir = pwd()

fεₑ_fpath = "fεₑ.jl"
fεₑ!_fpath = "fεₑ_inplace.jl"
if isfile(fεₑ_fpath)
    println("loading εₑ functions")
    fεₑ     =   include(joinpath(genfn_dir,fεₑ_fpath))
    fεₑ!    =   include(joinpath(genfn_dir,fεₑ!_fpath))
else    
    println("generating and saving εₑ functions")
    fεₑ_expr, fεₑ!_expr = build_function(simplify(εₑ, rewriter=r_pow_sqrt ), ω, r, n_1, n_2) #; parallel=MultithreadedForm())
    write(joinpath(genfn_dir,fεₑ_fpath), string(fεₑ_expr))
    write(joinpath(genfn_dir,fεₑ!_fpath),string(fεₑ!_expr))
    fεₑ  = eval(fεₑ_expr)
    fεₑ!  = eval(fεₑ!_expr)
end
fεₑ(1.0,0.3,normalize(rand(3))[1:2]...)
fεₑ!(rand(3,3),1.0,0.3,normalize(rand(3))[1:2]...)
@btime fεₑ(1.0,0.3,$(normalize(rand(3))[1:2])...)
@btime fεₑ!($(rand(3,3)),1.0,0.3,$(normalize(rand(3))[1:2])...)

f∂ωεₑ_fpath = "f∂ωεₑ.jl"
f∂ωεₑ!_fpath = "f∂ωεₑ_inplace.jl"
if isfile(f∂ωεₑ_fpath)
    println("loading ∂ωεₑ functions")
    f∂ωεₑ   =   include(joinpath(genfn_dir,f∂ωεₑ_fpath))
    f∂ωεₑ!  =   include(joinpath(genfn_dir,f∂ωεₑ!_fpath))
else    
    println("generating and saving ∂ωεₑ functions")
    f∂ωεₑ_expr, f∂ωεₑ!_expr = build_function(simplify(∂ωεₑ, rewriter=r_pow_sqrt ), ω, r, n_1, n_2) #; parallel=MultithreadedForm())
    write(joinpath(genfn_dir,f∂ωεₑ_fpath), string(f∂ωεₑ_expr))
    write(joinpath(genfn_dir,f∂ωεₑ!_fpath),string(f∂ωεₑ!_expr))
    f∂ωεₑ  = eval(f∂ωεₑ_expr)
    f∂ωεₑ!  = eval(f∂ωεₑ!_expr)
end
f∂ωεₑ(1.0,0.3,normalize(rand(3))[1:2]...)
f∂ωεₑ!(rand(3,3),1.0,0.3,normalize(rand(3))[1:2]...)
@btime f∂ωεₑ(1.0,0.3,$(normalize(rand(3))[1:2])...)
@btime f∂ωεₑ!($(rand(3,3)),1.0,0.3,$(normalize(rand(3))[1:2])...)

f∂²ωεₑ_fpath = "f∂²ωεₑ.jl"
f∂²ωεₑ!_fpath = "f∂²ωεₑ_inplace.jl"
if isfile(f∂²ωεₑ_fpath)
    println("loading ∂²ωεₑ functions")
    f∂²ωεₑ  =   include(joinpath(genfn_dir,f∂²ωεₑ_fpath))
    f∂²ωεₑ! =   include(joinpath(genfn_dir,f∂²ωεₑ!_fpath))
else    
    println("generating and saving ∂²ωεₑ functions")
    f∂²ωεₑ_expr, f∂²ωεₑ!_expr = build_function(simplify(∂²ωεₑ, rewriter=r_pow_sqrt ), ω, r, n_1, n_2) #; parallel=MultithreadedForm())
    write(joinpath(genfn_dir,f∂²ωεₑ_fpath), string(f∂²ωεₑ_expr))
    write(joinpath(genfn_dir,f∂²ωεₑ!_fpath),string(f∂²ωεₑ!_expr))
    f∂²ωεₑ  = eval(f∂²ωεₑ_expr)
    f∂²ωεₑ!  = eval(f∂²ωεₑ!_expr)
end
f∂²ωεₑ(1.0,0.3,normalize(rand(3))[1:2]...)
f∂²ωεₑ!(rand(3,3),1.0,0.3,normalize(rand(3))[1:2]...)
@btime f∂²ωεₑ(1.0,0.3,$(normalize(rand(3))[1:2])...)
@btime f∂²ωεₑ!($(rand(3,3)),1.0,0.3,$(normalize(rand(3))[1:2])...)

# 194.969 ns (14 allocations: 336 bytes)
# 136.610 ns (5 allocations: 96 bytes)
# 249.731 ns (14 allocations: 336 bytes)
# 190.826 ns (5 allocations: 96 bytes)
# 513.876 ns (14 allocations: 336 bytes)
# 441.338 ns (5 allocations: 96 bytes)

##
include("grad_test_utils.jl")

function εₑ_num(om,rr,nn1,nn2)
    invom = inv(om)
    nn = [ nn1, nn2, sqrt(1 - nn1^2 - nn2^2) ] #^(1//2)]
    SS = normcart(nn)
    no_sq = n²_MgO_LiNbO₃_sym(invom, pₒ.T₀; pₒ...)
    ne_sq = n²_MgO_LiNbO₃_sym(invom, pₑ.T₀; pₑ...)
    epsLN 	= diagm([no_sq, no_sq, ne_sq])
    epsSN 	= n²_Si₃N₄(invom,p_n²_Si₃N₄.T₀) * diagm([1,1,1])
    return avg_param(epsLN, epsSN, SS, rr)
end
εₑ_num(rand(Float64,2)...,normalize(rand(Float64,3))[1:2]...)
##
function test_εₑ_grads()
    args = vcat(rand(Float64,2),normalize(rand(Float64,3))[1:2])  # four random Float64s with the last two taken from a random normalized 3-vector
    println("args: ", args,"\n")
    
    εₑ_of_ω = oo->εₑ_num(oo,args[2:4]...)
    ∂ωεₑ_of_ω = oo->f∂ωεₑ(oo,args[2:4]...)

    εₑ_val = εₑ_num(args...)
    println("Primal εₑ val: ", εₑ_val,"\n")
    
    println("\n ∂ω∂εₑ calculations: ","\n")
    ∂ω∂εₑ_sym = f∂ωεₑ(args...)
    println("\t∂ω∂εₑ_sym: ", ∂ω∂εₑ_sym,"\n")
    
    ∂ω∂εₑ_FD = jacFD(εₑ_of_ω,args[1],out_shape=(3,3))
    println("\t∂ω∂εₑ_FD: ", ∂ω∂εₑ_FD,"\n")
    ∂ω∂εₑ_FM = derivFM(εₑ_of_ω,args[1])
    println("\t∂ω∂εₑ_FM: ", ∂ω∂εₑ_FM,"\n")
    ∂ω∂εₑ_RM = jacRM(εₑ_of_ω,args[1])
    println("\t∂ω∂εₑ_RM: ", ∂ω∂εₑ_RM,"\n")

    println("\n ∂²ω∂εₑ calculations: ","\n")
    ∂²ω∂εₑ_sym = f∂²ωεₑ(args...)
    println("\t∂²ω∂εₑ_sym: ", ∂²ω∂εₑ_sym,"\n")
    ∂²ω∂εₑ_FDFD = reshape.(eachcol(FiniteDifferences.jacobian(central_fdm(3,2),x->vec(εₑ_of_ω(x...)),args[1])[1]),((3,3),))
    println("\t∂²ω∂εₑ_FDFD: ", ∂²ω∂εₑ_FDFD,"\n")
    ∂²ω∂εₑ_FDSym = jacFD(∂ωεₑ_of_ω,args[1],out_shape=(3,3))
    println("\t∂²ω∂εₑ_FDSym: ", ∂²ω∂εₑ_FDSym,"\n")
    ∂²ω∂εₑ_FMSym = derivFM(∂ωεₑ_of_ω,args[1])
    println("\t∂²ω∂εₑ_FMSym: ", ∂²ω∂εₑ_FMSym,"\n")
    # ∂²ω∂εₑ_RMSym = jacRM(∂ωεₑ_of_ω,args[1])
    # println("\t∂²ω∂εₑ_RMSym: ", ∂²ω∂εₑ_RMSym,"\n")

    # println("\n ∂(∂εₑ_∂ω)_∂p calculations for other parameters {p}: ", εₑ_val,"\n")
    # # ∂εₑ_∂ω_sym = f∂ωεₑ(args...)
    # # println("\t∂εₑ_∂ω_sym: ", ∂εₑ_∂ω_sym,"\n")
    # ∂ω∂ω_εₑ_FD,∂ω∂r_εₑ_FD,∂ω∂n1_εₑ_FD,∂ω∂n2_εₑ_FD = jacFD(f∂ωεₑ,args)
    # ∂ω∂ω_εₑ_FM,∂ω∂r_εₑ_FM,∂ω∂n1_εₑ_FM,∂ω∂n2_εₑ_FM = jacFM(f∂ωεₑ,args)
    # println("\t∂ω∂ω_εₑ_FD: ", ∂ω∂ω_εₑ_FD,"\n")
    # ∂εₑ_∂ω_FM = derivFM(εₑ_of_ω,args[1])
    # println("\t∂εₑ_∂ω_FM: ", ∂εₑ_∂ω_FM,"\n")
    # ∂εₑ_∂ω_RM = jacRM(εₑ_of_ω,args[1])
    # println("\t∂εₑ_∂ω_RM: ", ∂εₑ_∂ω_RM,"\n")

end
test_εₑ_grads()

##
@show args =  vcat(rand(Float64,2),normalize(rand(Float64,3))[1:2])
om, r, nn1, nn2 = copy(args)
epse = zeros(Float64,3,3) 
# fepse_out = fεₑ!(epse,args...)
# @assert isnothing(fepse_out)

using Enzyme

@show ∂z_∂epse = rand(size(epse)...)  # Some gradient/tangent passed to us

∂z_∂om  =   0.0
∂z_∂r   =   0.0
∂z_∂nn1  =   0.0
∂z_∂nn2  =   0.0
# fεₑ!(epse,args...)
# Enzyme.autodiff(fεₑ!, Const, Duplicated(epse, ∂z_∂epse), Duplicated(om, ∂z_∂om), Duplicated(r, ∂z_∂r), Duplicated(nn1, ∂z_∂nn1), Duplicated(nn2, ∂z_∂nn2))
@show (∂z_∂om,∂z_∂r,∂z_∂nn1,∂z_∂nn2) = Enzyme.autodiff(fεₑ!, Const, Duplicated(epse, ∂z_∂epse), Active(om), Active(r), Active(nn1), Active(nn2))

@show epse
@show ∂z_∂epse
@show fεₑ(args...)
@show εₑ_num(args...)
@assert fεₑ(args...) ≈ εₑ_num(args...)

println("Enzyme grad:")
@show ∂z_∂om
@show ∂z_∂r
@show ∂z_∂nn1
@show ∂z_∂nn2

∂ω_εₑ_FD,∂r_εₑ_FD,∂n1_εₑ_FD,∂n2_εₑ_FD = jacFD(fεₑ,args)
@show ∂ω_εₑ_FD
@show ∂r_εₑ_FD
@show ∂n1_εₑ_FD
@show ∂n2_εₑ_FD


##

using ReverseDiff

