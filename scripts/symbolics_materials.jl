## Playing around with new array functionality in Symbolics.jl after a long time away from material model code
using Symbolics, LinearAlgebra
using BenchmarkTools
# includet("rules.jl")
using Symbolics: get_variables, wrap, unwrap, MakeTuple, toexpr, substitute, value, Sym, Num
# using ReversePropagation: gradient_expr
using SymbolicUtils: @rule, @acrule, RuleSet, numerators, denominators, flatten_pows
using SymbolicUtils: PolyForm, get_pvar2sym, get_sym2term, unpolyize, numerators, denominators #, toexpr
using SymbolicUtils.Rewriters: Chain, RestartedChain, PassThrough, Prewalk, Postwalk
using SymbolicUtils.Code: toexpr, cse #, cse!, _cse
using Symbolics: unwrap, wrap, toexpr

#### attempt to add basic SymPy -> Symbolics expression conversion
### from https://github.com/JuliaSymbolics/Symbolics.jl/issues/201  (April 2021)
import PyCall
const sympy = PyCall.pyimport_conda("sympy", "sympy")
function PyCall.PyObject(ex::Num)
    sympy.sympify(string(ex)) # lose type detail of variables
end;
Symbolics.Num(o::PyCall.PyObject) = eval(Meta.parse(sympy.julia_code(o)))
###

#### Define term-rewriting rules useful for combinations of Sellmeier-type (or Cauchy-type) equations
r_sqrt_pow = @rule sqrt(~x) --> (~x)^(1//2)
r_pow_sqrt = Prewalk( PassThrough( @rule (~x)^(1//2) --> sqrt(~x) ); threaded=true )
# r_neg_exp = Prewalk(PassThrough( @rule  ( 1 / ~x )^(~a) => (~x)^(-(~a)) ) )
# r_lor_inv = Prewalk(PassThrough( @acrule ~a / ( ~b + (~x)^-2 ) =>  ~a * (~x)^2 / ( ~b * (~x)^2 + 1 ) ) )

# r_lor_inv = Prewalk(PassThrough( @acrule ~a / ( ~b + 1 / (~x)^2  ) =>  ~a * (~x)^2 / ( ~b * (~x)^2 + 1 ) ); threaded=true )
r_lor_inv2 = Prewalk(PassThrough( @acrule ( ~a * ( ( 1 / ~x )^2 ) ) / ( (1 / ~x)^2 - ~b) =>  ~a  / ( 1 - ~b*(~x)^2 ) ) ; threaded=true )
r_lor_inv5 = Prewalk(PassThrough( @acrule ( ~a*( 1 /  (~x)^2 )) / ((1 / (~x)^2) - ~b) =>  ~a  / ( 1 - ~b*(~x)^2 ) ) ; threaded=true )

r_lor_inv3 = Prewalk(PassThrough( @acrule ( ~a * ( 1 / (~x)^2 ) ) / ( (1 / ~x)^2 - ~b) =>  ~a  / ( 1 - ~b*(~x)^2 ) ); threaded=true )
r_lor_inv4 = Prewalk(PassThrough( @acrule ( ~a / (~x)^2 ) / ( (1 / ~x)^2 - ~b) =>  ~a  / ( 1 - ~b*(~x)^2 ) ); threaded=true )
r_exp_prod = @acrule((~a)^(~x) * (~a)^(~y) => (~a)^(~x + ~y))
r_neg_exp = Prewalk(PassThrough( @rule  ( 1 / ~x )^(~a) =>  1 / (~x)^(~a)  ); threaded=true )
r_neg_exp2 = Prewalk(PassThrough( @rule  ~b * ( 1 / ~x )^(~a) =>  ~b / (~x)^(~a)  ); threaded=true )
# my_rules = Chain([r_neg_exp,r_lor_inv2,r_lor_inv3,r_lor_inv4,r_lor_inv,])

r_lor_inv = Prewalk(PassThrough( @acrule ~a / ( ~b + ( 1 / (~x) )^2  ) =>  ~a * (~x)^2 / ( ~b * (~x)^2 + 1 ) ); threaded=true )
my_rules = Chain([r_lor_inv2,r_lor_inv3,r_lor_inv4,r_lor_inv,r_neg_exp,])



function as_polynomial(f, exprs...; polyform=false, T=Real)
    @assert length(exprs) >= 1 "At least one expression must be passed to `multivariatepolynomial`."

    pvar2sym, sym2term = get_pvar2sym(), get_sym2term()
    ps = map(x->PolyForm(x, pvar2sym, sym2term), exprs)
    convert_back = polyform ? x -> PolyForm{T}(x, pvar2sym, sym2term) :
        x -> unpolyize(PolyForm{T}(x, pvar2sym, sym2term)) # substitute back
    res = f(convert_back, map(x->x.p, ps)...)
end

my_simp(x) = wrap(flatten_fractions(unwrap(simplify( x, rewriter=my_rules ) ) ) )

depends_on(eqn,var) = any(isequal.((unwrap(var),),Symbolics.get_variables(eqn)))

function rational_poly_coeffs(x,var)
    # x_ff = flatten_fractions(simplify( unwrap(x), rewriter=my_rules ) )
    x_ff = flatten_fractions( unwrap(x) )
    num = first(numerators(x_ff))
    denom = prod(denominators(x_ff))
    num_poly_dict, num_poly_rem = polynomial_coeffs(num,(var,))
    denom_poly_dict, denom_poly_rem = polynomial_coeffs(denom,(var,))
    return num_poly_dict, num_poly_rem, denom_poly_dict, denom_poly_rem
end

using Symbolics: pdegree, tosymbol

# function polynomial_ratio(x,var;simplify_coeffs=true)
#     num_poly_dict, num_poly_rem, denom_poly_dict, denom_poly_rem = rational_poly_coeffs(x,var)
#     maxdeg_num::Int = mapreduce(kv->pdegree(first(kv)),max,num_poly_dict)
#     maxdeg_denom::Int = mapreduce(kv->pdegree(first(kv)),max,denom_poly_dict)
#     degrees_num = zeros(Num,maxdeg_num+1)
#     degrees_denom = zeros(Num,maxdeg_denom+1)
#     if simplify_coeffs
#         degrees_num[1] = simplify(num_poly_rem,expand=true)
#         degrees_denom[1] = simplify(denom_poly_rem,expand=true)
#         _ = [degrees_num[pdegree(first(kv))+1]=simplify(wrap(last(kv)),expand=true) for kv in num_poly_dict]
#         _ = [degrees_denom[pdegree(first(kv))+1]=simplify(wrap(last(kv)),expand=true) for kv in denom_poly_dict]
#     else
#         degrees_num[1] = num_poly_rem
#         degrees_denom[1] = denom_poly_rem
#         _ = [degrees_num[pdegree(first(kv))+1]=wrap(last(kv)) for kv in num_poly_dict]
#         _ = [degrees_denom[pdegree(first(kv))+1]=wrap(last(kv)) for kv in denom_poly_dict]
#     end
#     # return Polynomial(degrees_num,tosymbol(var)),  Polynomial(degrees_denom,tosymbol(var))
#     poly_ratio = Polynomial(degrees_num,tosymbol(var)) // Polynomial(degrees_denom,tosymbol(var))
#     return poly_ratio
# end

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

##

function n²_MgO_LiNbO₃_sym(λ, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    f = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    λ² = λ^2
    a₁ + b₁*f + (a₂ + b₂*f) / (λ² - (a₃ + b₃*f)^2) + (a₄ + b₄*f) / (λ² - a₅^2) - a₆*λ²
end

function n²_MgO_LiNbO₃_sym_ω(ω, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    f = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    a₁ + b₁*f + (a₂ + b₂*f)*ω^2 / (1 - (a₃ + b₃*f)^2*ω^2) + (a₄ + b₄*f)*ω^2 / (1 - a₅^2*ω^2) - a₆ / ω^2
end

function n²_MgO_LiNbO₃_sym_ω2(ω, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    # ff = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    ff = T^2 - T₀^2 + 2*(T + T₀) * 273.16  # so-called 'temperature dependent parameter'
    a₁ + b₁*ff + ( (a₂ + b₂*ff) * ω^2 ) / ( 1.0 - (a₃ + b₃*ff)^2 * ω^2 ) + (a₄ + b₄*ff)*ω^2 / ( 1.0 - a₅^2 * ω^2 ) - (a₆ / ω^2)
end

function n²_sym_fmt1( λ ; A₀=1, B₁=0, C₁=0, B₂=0, C₂=0, B₃=0, C₃=0, kwargs...)
    λ² = λ^2
    A₀  + ( B₁ * λ² ) / ( λ² - C₁ ) + ( B₂ * λ² ) / ( λ² - C₂ ) + ( B₃ * λ² ) / ( λ² - C₃ )
end

function n²_sym_fmt1_ω( ω ; A₀=1, B₁=0, C₁=0, B₂=0, C₂=0, B₃=0, C₃=0, kwargs...)
    A₀  + B₁ / ( 1 - C₁*ω^2 ) + B₂ / ( 1 - C₂*ω^2 ) + B₃ / ( 1 - C₃*ω^2 )
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

p_n²_Si₃N₄= (
    A₀ = 1,
    B₁ = 3.0249,
    C₁ = (0.1353406)^2,         #                           [μm²]
    B₂ = 40314.0,
    C₂ = (1239.842)^2,          #                           [μm²]
    dn_dT = 2.96e-5,            # thermo-optic coefficient  [K⁻¹]
    T₀ = 24.5,                  # reference temperature     [°C]
    dn²_dT = 2*sqrt(n²_sym_fmt1( 1.55 ; A₀ = 1, B₁ = 3.0249, C₁ = (0.1353406)^2, B₂ = 40314, C₂ = (1239.842)^2,))*2.96e-5,
)   
# # The last term is just 2n₀*dn_dT where n₀=n(λ₀,T₀) is index at the wavelength and temperature where 
# # the thermo-optic coefficient `dn_dT` was measured. `dn²_dT` is the lowest order (linear) thermo-optic
# # coefficient for n²(λ,T) corresponding to `dn_dT`, and avoids square roots which complicate computer algebra.
# # This neglects the wavelength/frequency dependence of thermo-optic coupling, just like `dn_dT`. 


# pₑ = (
#     a₁ = big(5.756),
#     a₂ = big(0.0983),
#     a₃ = big(0.202),
#     a₄ = big(189.32),
#     a₅ = big(12.52),
#     a₆ = big(1.32e-2),
#     b₁ = big(2.86e-6),
#     b₂ = big(4.7e-8),
#     b₃ = big(6.113e-8),
#     b₄ = big(1.516e-4),
#     T₀ = big(24.5),      # reference temperature in [Deg C]
# )
# pₒ = (
#     a₁ = big(5.653),
#     a₂ = big(0.1185),
#     a₃ = big(0.2091),
#     a₄ = big(89.61),
#     a₅ = big(10.85),
#     a₆ = big(1.97e-2),
#     b₁ = big(7.941e-7),
#     b₂ = big(3.134e-8),
#     b₃ = big(-4.641e-9),
#     b₄ = big(-2.188e-6),
#     T₀ = big(24.5),      # reference temperature in [Deg C]
# )

# p_n²_Si₃N₄= (
#     A₀ = 1,
#     B₁ = big(3.0249),
#     C₁ = big((0.1353406)^2),         #                           [μm²]
#     B₂ = big(40314),
#     C₂ = big((1239.842)^2),          #                           [μm²]
#     dn_dT = big(2.96e-5),            # thermo-optic coefficient  [K⁻¹]
#     T₀ = big(24.5),                  # reference temperature     [°C]
#     dn²_dT = big(2*sqrt(n²_sym_fmt1( 1.55 ; A₀ = 1, B₁ = 3.0249, C₁ = (0.1353406)^2, B₂ = 40314, C₂ = (1239.842)^2,))*2.96e-5),
# )   
# The last term is just 2n₀*dn_dT where n₀=n(λ₀,T₀) is index at the wavelength and temperature where 
# the thermo-optic coefficient `dn_dT` was measured. `dn²_dT` is the lowest order (linear) thermo-optic
# coefficient for n²(λ,T) corresponding to `dn_dT`, and avoids square roots which complicate computer algebra.
# This neglects the wavelength/frequency dependence of thermo-optic coupling, just like `dn_dT`. 


n²_Si₃N₄(λ,T) = n²_sym_fmt1( λ ; p_n²_Si₃N₄...) + p_n²_Si₃N₄.dn²_dT  *  ( T - p_n²_Si₃N₄.T₀ ) 
n²_Si₃N₄_ω(ω,T) = n²_sym_fmt1_ω( ω ; p_n²_Si₃N₄...) + p_n²_Si₃N₄.dn²_dT  *  ( T - p_n²_Si₃N₄.T₀ )

##

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
    h_temp = n × [ 0, 0, 1 ] # for now ignore edge case where n = [1,0,0]
    h = h_temp ./ (h_temp[1]^2 + h_temp[2]^2)^(1//2)
	v = n × h   # the third unit vector `v` is just the cross of `n` and `h`
    S = [ n h v ]  # S is a unitary 3x3 matrix
    return S
end

function τ_trans(ε)
    return               [     -inv(ε[1,1])            ε[1,2]/ε[1,1]                           ε[1,3]/ε[1,1]
                                ε[2,1]/ε[1,1]       ε[2,2] - ε[2,1]*ε[1,2]/ε[1,1]           ε[2,3] - ε[2,1]*ε[1,3]/ε[1,1]
                                ε[3,1]/ε[1,1]       ε[3,2] - ε[3,1]*ε[1,2]/ε[1,1]           ε[3,3] - ε[3,1]*ε[1,3]/ε[1,1]       ]
end

function τ⁻¹_trans(τ)
    return           [      -inv(τ[1,1])           -τ[1,2]/τ[1,1]                          -τ[1,3]/τ[1,1]
                            -τ[2,1]/τ[1,1]       τ[2,2] - τ[2,1]*τ[1,2]/τ[1,1]           τ[2,3] - τ[2,1]*τ[1,3]/τ[1,1]
                            -τ[3,1]/τ[1,1]       τ[3,2] - τ[3,1]*τ[1,2]/τ[1,1]           τ[3,3] - τ[3,1]*τ[1,3]/τ[1,1]          ]
end

"""
Kottke averaging of two dielectric tensors `ε₁`, `ε₂` across a planar material interface whose normal vector `n₁₂` pointing from material 1 to material 2.
`n₁₂` is the first column of the unitary rotation matrix `S` which rotates `ε₁` and `ε₂` into the "frame of the interface" so that dielectric tensor 
elements interacting with E-fields parallel and perpendicular to the interface can be averaged differently.

`r₁` is the ratio of the averaging volume filled by material 1. The normal vector mentioned above should point out from material 1 to material 2.
"""
function avg_param(ε₁, ε₂, S, r₁)
    S_tp = transpose(S)
    τ1 = τ_trans(S_tp * (ε₁ * S))  # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans(S_tp * (ε₂ * S))  # express param2 in S coordinates, and apply τ transform
    τavg = ( r₁ * τ1 ) + ( (1-r₁) * τ2 ) # volume-weighted average
    return (S * τ⁻¹_trans(τavg)) * S_tp  # apply τ⁻¹ and transform back to global coordinates
end

##
@variables ω, T, r₁, λ
Dom = Differential(ω)

# Use this once Symbolics Array variables work well
# @variables ε₁[1:3,1:3](ω), ε₂[1:3,1:3](ω), S[1:3,1:3]
# @variables ε₁[1:3,1:3], ε₂[1:3,1:3], S[1:3,1:3]

# Until Symbolics Array variables work out some kinks, just make the 3x3 matrices out of scalar real variables

@variables ε₁_11, ε₁_12, ε₁_13, ε₁_21, ε₁_22, ε₁_23, ε₁_31, ε₁_32, ε₁_33, ε₂_11, ε₂_12, ε₂_13, ε₂_21, ε₂_22, ε₂_23, ε₂_31, ε₂_32, ε₂_33 
ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_21  ε₁_22  ε₁_23 ; ε₁_31  ε₁_32  ε₁_33 ] 
ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_21  ε₂_22  ε₂_23 ; ε₂_31  ε₂_32  ε₂_33 ]
@variables n_1, n_2, n_3 #    try enforcing normalization of 3-vector `n` by building it with two real scalar variables
n = [ n_1, n_2, n_3 ] # (1 - n_1^2 - n_2^2)^(1//2)] # [ n_1, n_2, n_3 ] / sqrt( n_1^2 + n_2^2 + n_3^2 )  # simplification works better with (...)^(1//2) than sqrt(...)
S = normcart(n)



# nₒ² = simplify_fractions(n²_MgO_LiNbO₃_sym_ω(ω, pₒ.T₀; pₒ...), polyform=true)
# nₑ² = simplify_fractions(n²_MgO_LiNbO₃_sym_ω(ω, pₑ.T₀; pₑ...), polyform=true)
nₒ² = simplify_fractions(n²_MgO_LiNbO₃_sym_ω(ω, pₒ.T₀; pₒ...))
nₑ² = simplify_fractions(n²_MgO_LiNbO₃_sym_ω(ω, pₑ.T₀; pₑ...))
εLN 	= diagm([nₒ², nₒ², nₑ²])
# n²_SN 	= simplify_fractions(n²_Si₃N₄_ω(ω,p_n²_Si₃N₄.T₀), polyform=true)
n²_SN 	= simplify_fractions(n²_Si₃N₄_ω(ω,p_n²_Si₃N₄.T₀))
εSN 	= n²_SN * diagm([1,1,1])


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

# εₑ = avg_param(my_simp.(εLN), my_simp.(εSN), S, r);
εₑ = avg_param(εLN, εSN, S, r₁);
εₑ2 = avg_param(ε₁, ε₂, S, r₁);
# εₑ = simplify.(avg_param(εLN, εSN, S, r));
# εₑ = simplify.(avg_param(my_simp.(εLN), my_simp.(εSN), my_simp.(S), r));
∂ωεₑ = expand_derivatives.(Dom.(εₑ));
∂²ωεₑ = expand_derivatives.(Dom.(∂ωεₑ));

εₑ_jac = Symbolics.jacobian(vec(εₑ),[ω,r₁,n_1,n_2,n_3]);
∂ωεₑ_jac = Symbolics.jacobian(vec(∂ωεₑ),[ω,r₁,n_1,n_2,n_3]);
∂²ωεₑ_jac = Symbolics.jacobian(vec(∂²ωεₑ),[ω,r₁,n_1,n_2,n_3]);

εₑ_∂ωεₑ_∂²ωεₑ = hcat(vec(εₑ),vec(∂ωεₑ),vec(∂²ωεₑ));
εₑ_∂ωεₑ_∂²ωεₑ_jac = Symbolics.jacobian(vec(εₑ_∂ωεₑ_∂²ωεₑ),[ω,r₁,n_1,n_2,n_3]);
εₑ_∂ωεₑ_∂²ωεₑ_and_jac = hcat(vec(εₑ_∂ωεₑ_∂²ωεₑ),εₑ_∂ωεₑ_∂²ωεₑ_jac);
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

p = @variables r₁, n_1, n_2, n_3, ε₁_11, ε₁_12, ε₁_13, ε₁_21, ε₁_22, ε₁_23, ε₁_31, ε₁_32, ε₁_33, ε₂_11, ε₂_12, ε₂_13, ε₂_21, ε₂_22, ε₂_23, ε₂_31, ε₂_32, ε₂_33 
# @variables r₁, n[1:3], ε₁[1:3,1:3], ε₂[1:3,1:3]
ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_21  ε₁_22  ε₁_23 ; ε₁_31  ε₁_32  ε₁_33 ] 
ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_21  ε₂_22  ε₂_23 ; ε₂_31  ε₂_32  ε₂_33 ]
n = [ n_1, n_2, n_3 ] 
f_εₑ_sym = avg_param(ε₁, ε₂, normcart(n), r₁); # 3 x 3
j_εₑ_sym = Symbolics.jacobian(vec(f_εₑ_sym),p); # 9 x 22
fj_εₑ_sym = hcat(vec(f_εₑ_sym),f_εₑ_sym);
# h_εₑ_sym = mapreduce(x->Symbolics.hessian(x,p),hcat,vec(εₑ_sym)); # 9 x 22 x 22
f_εₑ_ex, f_εₑ!_ex = build_function(εₑ_sym, p ; expression=Val{true});
f_εₑ, f_εₑ! = eval(f_εₑ_ex), eval(f_εₑ!_ex);
fj_εₑ_ex, fj_εₑ!_ex = build_function(fj_εₑ_sym, p ; expression=Val{true});
fj_εₑ, fj_εₑ! = eval(fj_εₑ_ex), eval(fj_εₑ!_ex);
# fjh_εₑ_ex, fjh_εₑ!_ex = build_function(εₑ_hes, p ; expression=Val{true});
# fjh_εₑ, fjh_εₑ! = eval(fjh_εₑ_ex), eval(fjh_εₑ!_ex);

rand_p_εₑ() = [rand(), normalize(rand(3))..., (rand()+1.0)^2, 0.0, 0.0, 0.0, (rand()+1.0)^2, 0.0, 0.0, 0.0, (rand()+1.0)^2, (rand()+1.0)^2, 0.0, 0.0, 0.0, (rand()+1.0)^2, 0.0, 0.0, 0.0, (rand()+1.0)^2 ]

function _f_εₑ_sym()
	p = @variables r₁, n_1, n_2, n_3, ε₁_11, ε₁_12, ε₁_13, ε₁_21, ε₁_22, ε₁_23, ε₁_31, ε₁_32, ε₁_33, ε₂_11, ε₂_12, ε₂_13, ε₂_21, ε₂_22, ε₂_23, ε₂_31, ε₂_32, ε₂_33 
    # @variables r₁, n[1:3], ε₁[1:3,1:3], ε₂[1:3,1:3]
    ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_21  ε₁_22  ε₁_23 ; ε₁_31  ε₁_32  ε₁_33 ] 
    ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_21  ε₂_22  ε₂_23 ; ε₂_31  ε₂_32  ε₂_33 ]
    n = [ n_1, n_2, n_3 ] 
    f_εₑ_sym = avg_param(ε₁, ε₂, normcart(n), r₁); # 3 x 3
    return f_εₑ_sym, p
end

function _f_εₑ()
	# p = @variables r₁, n_1, n_2, n_3, ε₁_11, ε₁_12, ε₁_13, ε₁_21, ε₁_22, ε₁_23, ε₁_31, ε₁_32, ε₁_33, ε₂_11, ε₂_12, ε₂_13, ε₂_21, ε₂_22, ε₂_23, ε₂_31, ε₂_32, ε₂_33 
    # # @variables r₁, n[1:3], ε₁[1:3,1:3], ε₂[1:3,1:3]
    # ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_21  ε₁_22  ε₁_23 ; ε₁_31  ε₁_32  ε₁_33 ] 
    # ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_21  ε₂_22  ε₂_23 ; ε₂_31  ε₂_32  ε₂_33 ]
    # n = [ n_1, n_2, n_3 ] 
    # f_εₑ_sym = avg_param(ε₁, ε₂, normcart(n), r₁); # 3 x 3
    f_εₑ_sym, p = _f_εₑ_sym()
    f_εₑ_ex, f_εₑ!_ex = build_function(f_εₑ_sym, p ; expression=Val{true});
    return f_εₑ, f_εₑ! = eval(f_εₑ_ex), eval(f_εₑ!_ex);
end

function _fj_εₑ()
	# p = @variables r₁, n_1, n_2, n_3, ε₁_11, ε₁_12, ε₁_13, ε₁_21, ε₁_22, ε₁_23, ε₁_31, ε₁_32, ε₁_33, ε₂_11, ε₂_12, ε₂_13, ε₂_21, ε₂_22, ε₂_23, ε₂_31, ε₂_32, ε₂_33 
    # # @variables r₁, n[1:3], ε₁[1:3,1:3], ε₂[1:3,1:3]
    # ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_21  ε₁_22  ε₁_23 ; ε₁_31  ε₁_32  ε₁_33 ] 
    # ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_21  ε₂_22  ε₂_23 ; ε₂_31  ε₂_32  ε₂_33 ]
    # n = [ n_1, n_2, n_3 ] 
    # f_εₑ_sym = avg_param(ε₁, ε₂, normcart(n), r₁); # 3 x 3
    f_εₑ_sym, p = _f_εₑ_sym()
    j_εₑ_sym = Symbolics.jacobian(vec(f_εₑ_sym),p); # 9 x 22
    fj_εₑ_sym = hcat(vec(f_εₑ_sym),f_εₑ_sym); # 10 x 22
    # h_εₑ_sym = mapreduce(x->Symbolics.hessian(x,p),hcat,vec(εₑ_sym)); # 9 x 22 x 22
    f_εₑ_ex, f_εₑ!_ex = build_function(εₑ_sym, p ; expression=Val{true});
    f_εₑ, f_εₑ! = eval(f_εₑ_ex), eval(f_εₑ!_ex);
    fj_εₑ_ex, fj_εₑ!_ex = build_function(fj_εₑ_sym, p ; expression=Val{true});
    fj_εₑ, fj_εₑ! = eval(fj_εₑ_ex), eval(fj_εₑ!_ex);
end

# function _fjh_εₑ()
# 	# p = @variables r₁, n_1, n_2, n_3, ε₁_11, ε₁_12, ε₁_13, ε₁_21, ε₁_22, ε₁_23, ε₁_31, ε₁_32, ε₁_33, ε₂_11, ε₂_12, ε₂_13, ε₂_21, ε₂_22, ε₂_23, ε₂_31, ε₂_32, ε₂_33 
#     # # @variables r₁, n[1:3], ε₁[1:3,1:3], ε₂[1:3,1:3]
#     # ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_21  ε₁_22  ε₁_23 ; ε₁_31  ε₁_32  ε₁_33 ] 
#     # ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_21  ε₂_22  ε₂_23 ; ε₂_31  ε₂_32  ε₂_33 ]
#     # n = [ n_1, n_2, n_3 ] 
#     # f_εₑ_sym = avg_param(ε₁, ε₂, normcart(n), r₁); # 3 x 3
#     f_εₑ_sym, p = _f_εₑ_sym()
#     j_εₑ_sym = Symbolics.jacobian(vec(f_εₑ_sym),p); # 9 x 22
#     fj_εₑ_sym = hcat(vec(f_εₑ_sym),f_εₑ_sym); # 10 x 22
#     h_εₑ_sym = mapreduce(x->Symbolics.hessian(x,p),hcat,vec(εₑ_sym)); # 9 x 22 x 22
#     fjh_εₑ_sym = 
#     fjh_εₑ_ex, fjh_εₑ!_ex = build_function(fjh_εₑ_sym, p ; expression=Val{true});
#     fjh_εₑ, fjh_εₑ! = eval(fjh_εₑ_ex), eval(fjh_εₑ!_ex);
# end

##
using OptiMode
using Symbolics, LinearAlgebra
using BenchmarkTools
using RuntimeGeneratedFunctions
using IterTools: subsets
RuntimeGeneratedFunctions.init(@__MODULE__)
mats1 = [MgO_LiNbO₃,Si₃N₄];
mats2 = [MgO_LiNbO₃,Si₃N₄,SiO₂];
mats3 = [MgO_LiNbO₃,Si₃N₄,SiO₂,LiB₃O₅];

function _ε_fn(mats)
	@variables ω, λ
	Dom = Differential(ω)
	ε_mats = mapreduce(mm->vec(get_model(mm,:ε,:ω)),hcat,mats);
    # ε_mats = mapreduce(mm->vec(substitute.(get_model(mm,:ε,:λ),(Dict([(λ=>1/ω),]),))),hcat,mats);
	∂ωε_mats = expand_derivatives.(Dom.(ε_mats));
	∂²ωε_mats = expand_derivatives.(Dom.(∂ωε_mats));
	ε_∂ωε_∂²ωε_mats = hcat(ε_mats,∂ωε_mats,∂²ωε_mats)
	fε_∂ωε_∂²ωε, fε_∂ωε_∂²ωε! = build_function(ε_∂ωε_∂²ωε_mats, ω ; expression=Val{true})
    return eval(fε_∂ωε_∂²ωε), eval(fε_∂ωε_∂²ωε!)
end

function _ε_fns(mats,matvars=())
	@variables ω, λ
	Dom = Differential(ω)
	ε_mats = mapreduce(mm->vec(get_model(mm,:ε,:ω,matvars...)),hcat,mats);
    # ε_mats = mapreduce(mm->vec(substitute.(get_model(mm,:ε,:λ,matvars...),(Dict([(λ=>1/ω),]),))),hcat,mats);
    ε_deps = [ω, (Num(Sym{Real}(matvar)) for matvar in matvars)...]
    smoothing_vars = @variables n_1, n_2, n_3, r 
    # n = [ n_1, n_2, n_3 ]
    # S = normcart(n)
    εₑ_mats = mapreduce(hcat,subsets(1:length(mats),2)) do mat_inds
        vec(avg_param(reshape(ε_mats[:,mat_inds[1]],(3,3)), reshape(ε_mats[:,mat_inds[2]],(3,3)), normcart([n_1,n_2,n_3]), r))
    end
    ε_mats = hcat(ε_mats,εₑ_mats)
    
    # matvar_nums = [ω, (Num(Sym{Real}(matvar)) for matvar in matvars)...]
	∂ωε_mats = expand_derivatives.(Dom.(ε_mats));
	ε_∂ωε_mats = hcat(ε_mats,∂ωε_mats);
	fε_∂ωε, fε_∂ωε! = build_function(ε_∂ωε_mats, ε_deps..., smoothing_vars... ; expression=Val{true})
    return eval(fε_∂ωε), eval(fε_∂ωε!)
    # ∂²ωε_mats = expand_derivatives.(Dom.(∂ωε_mats));
    # ε_∂ωε_∂²ωε_mats = hcat(ε_mats,∂ωε_mats,∂²ωε_mats)
	# fε_∂ωε_∂²ωε, fε_∂ωε_∂²ωε! = build_function(ε_∂ωε_∂²ωε_mats, ω, (Num(Sym{Real}(matvar)) for matvar in matvars)... ; expression=Val{true})
    # return fε_∂ωε_∂²ωε, fε_∂ωε_∂²ωε!
end

function _ε_fns2(mats,matvars=())
	@variables ω, λ
	Dom = Differential(ω)
	ε_mats = mapreduce(mm->vec(get_model(mm,:ε,:ω,matvars...)),hcat,mats);
    # ε_mats = mapreduce(mm->vec(substitute.(get_model(mm,:ε,:λ,matvars...),(Dict([(λ=>1/ω),]),))),hcat,mats);
    ε_deps = [ω, (Num(Sym{Real}(matvar)) for matvar in matvars)...]
    smoothing_vars = @variables n_1, n_2, n_3, r 
    # n = [ n_1, n_2, n_3 ]
    # S = normcart(n)
    εₑ_mats = mapreduce(hcat,subsets(1:length(mats),2)) do mat_inds
        vec(avg_param(reshape(ε_mats[:,mat_inds[1]],(3,3)), reshape(ε_mats[:,mat_inds[2]],(3,3)), normcart([n_1,n_2,n_3]), r))
    end
    ε_mats = hcat(ε_mats,εₑ_mats)
    
    # matvar_nums = [ω, (Num(Sym{Real}(matvar)) for matvar in matvars)...]
	∂ωε_mats = expand_derivatives.(Dom.(ε_mats));
	# ε_∂ωε_mats = hcat(ε_mats,∂ωε_mats);
	# fε_∂ωε, fε_∂ωε! = build_function(ε_∂ωε_mats, ε_deps..., smoothing_vars... ; expression=Val{true})
    # return eval(fε_∂ωε), eval(fε_∂ωε!)
    ∂²ωε_mats = expand_derivatives.(Dom.(∂ωε_mats));
    ε_∂ωε_∂²ωε_mats = hcat(ε_mats,∂ωε_mats,∂²ωε_mats)
	fε_∂ωε_∂²ωε, fε_∂ωε_∂²ωε! = build_function(ε_∂ωε_∂²ωε_mats, ε_deps..., smoothing_vars... ; expression=Val{true})
    return eval(fε_∂ωε_∂²ωε), eval(fε_∂ωε_∂²ωε!)
end

function _ε_fjs(mats,matvars=())
	@variables ω, λ
	Dom = Differential(ω)
	ε_mats = mapreduce(mm->vec(get_model(mm,:ε,:ω,matvars...)),hcat,mats);
    # ε_mats = mapreduce(mm->vec(substitute.(get_model(mm,:ε,:λ,matvars...),(Dict([(λ=>1/ω),]),))),hcat,mats);
    ε_deps = [ω, (Num(Sym{Real}(matvar)) for matvar in matvars)...]

    smoothing_vars = @variables n_1, n_2, n_3, r 
    εₑ_mats = mapreduce(hcat,subsets(1:length(mats),2)) do mat_inds
        vec(avg_param(reshape(ε_mats[:,mat_inds[1]],(3,3)), reshape(ε_mats[:,mat_inds[2]],(3,3)), normcart([n_1,n_2,n_3]), r))
    end
    ε_mats = mapreduce(vec,vcat,(ε_mats,εₑ_mats,))

	∂ωε_mats = expand_derivatives.(Dom.(ε_mats));
	
    ε_∂ωε_mats = vcat(ε_mats,∂ωε_mats);
    jac_ε_∂ωε_mats = Symbolics.jacobian(ε_∂ωε_mats,vcat(ε_deps,smoothing_vars));
	fε_∂ωε_and_jac, fε_∂ωε_and_jac! = build_function(hcat(ε_∂ωε_mats,jac_ε_∂ωε_mats), ε_deps..., smoothing_vars... ; expression=Val{true})
    return eval(fε_∂ωε_and_jac), eval(fε_∂ωε_and_jac!)

    # ∂²ωε_mats = expand_derivatives.(Dom.(∂ωε_mats));
	# ε_∂ωε_∂²ωε_mats = vcat(ε_mats,∂ωε_mats,∂²ωε_mats);
    # jac_ε_∂ωε_∂²ωε_mats = Symbolics.jacobian(ε_∂ωε_∂²ωε_mats,ε_deps);
    # fε_∂ωε_∂²ωε_and_jac, fε_∂ωε_∂²ωε_and_jac! = build_function(hcat(ε_∂ωε_∂²ωε_mats,jac_ε_∂ωε_∂²ωε_mats), ε_deps... , smoothing_vars... ; expression=Val{true})
    # return fε_∂ωε_∂²ωε_and_jac, fε_∂ωε_∂²ωε_and_jac!
end

function _ε_fjs2(mats,matvars=())
	@variables ω, λ
	Dom = Differential(ω)
	ε_mats = mapreduce(mm->vec(get_model(mm,:ε,:ω,matvars...)),hcat,mats);
    # ε_mats = mapreduce(mm->vec(substitute.(get_model(mm,:ε,:λ,matvars...),(Dict([(λ=>1/ω),]),))),hcat,mats);
    ε_deps = [ω, (Num(Sym{Real}(matvar)) for matvar in matvars)...]

    smoothing_vars = @variables n_1, n_2, n_3, r 
    εₑ_mats = mapreduce(hcat,subsets(1:length(mats),2)) do mat_inds
        vec(avg_param(reshape(ε_mats[:,mat_inds[1]],(3,3)), reshape(ε_mats[:,mat_inds[2]],(3,3)), normcart([n_1,n_2,n_3]), r))
    end
    ε_mats = mapreduce(vec,vcat,(ε_mats,εₑ_mats,))

	∂ωε_mats = expand_derivatives.(Dom.(ε_mats));
	
    # ε_∂ωε_mats = vcat(ε_mats,∂ωε_mats);
    # jac_ε_∂ωε_mats = Symbolics.jacobian(ε_∂ωε_mats,vcat(ε_deps,smoothing_vars));
	# fε_∂ωε_and_jac, fε_∂ωε_and_jac! = build_function(hcat(ε_∂ωε_mats,jac_ε_∂ωε_mats), ε_deps..., smoothing_vars... ; expression=Val{true})
    # return eval(fε_∂ωε_and_jac), eval(fε_∂ωε_and_jac!)

    ∂²ωε_mats = expand_derivatives.(Dom.(∂ωε_mats));
	ε_∂ωε_∂²ωε_mats = vcat(ε_mats,∂ωε_mats,∂²ωε_mats);
    jac_ε_∂ωε_∂²ωε_mats = Symbolics.jacobian(ε_∂ωε_∂²ωε_mats,ε_deps);
    fε_∂ωε_∂²ωε_and_jac, fε_∂ωε_∂²ωε_and_jac! = build_function(hcat(ε_∂ωε_∂²ωε_mats,jac_ε_∂ωε_∂²ωε_mats), ε_deps... , smoothing_vars... ; expression=Val{true})
    return eval(fε_∂ωε_∂²ωε_and_jac), eval(fε_∂ωε_∂²ωε_and_jac!)
end

# @variables ω, λ
# smoothing_vars = @variables n_1, n_2, n_3, r 
# ε_mats1 = mapreduce(mm->vec(get_model(mm,:ε,:ω)),hcat,mats1);
# εₑ_mats1 = mapreduce(hcat,subsets(1:length(mats1),2)) do mat_inds
#     vec(avg_param(reshape(ε_mats1[:,mat_inds[1]],(3,3)), reshape(ε_mats1[:,mat_inds[2]],(3,3)), normcart([n_1,n_2,n_3]), r))
# end

f_ε_∂ωε_mats1, f_ε_∂ωε_mats1! = _ε_fns(mats1)
ε_∂ωε_mats1 = f_ε_∂ωε_mats1(1.0,normalize(rand(3))...,0.3)
f_ε_∂ωε_mats1_T, f_ε_∂ωε_mats1_T! = _ε_fns(mats1,(:T,))
ε_∂ωε_mats1_T = f_ε_∂ωε_mats1_T(1.0,31.4,normalize(rand(3))...,0.3)
f_ε_∂ωε_mats2, f_ε_∂ωε_mats2! = _ε_fns(mats2)
ε_∂ωε_mats2 = f_ε_∂ωε_mats2(1.0,normalize(rand(3))...,0.3)
f_ε_∂ωε_mats2_T, f_ε_∂ωε_mats2_T! = _ε_fns(mats2,(:T,))
ε_∂ωε_mats2_T = f_ε_∂ωε_mats2_T(1.0,31.4,normalize(rand(3))...,0.3)
f_ε_∂ωε_mats3, f_ε_∂ωε_mats3! = _ε_fns(mats3)
ε_∂ωε_mats3 = f_ε_∂ωε_mats3(1.0,normalize(rand(3))...,0.3)
f_ε_∂ωε_mats3_T, f_ε_∂ωε_mats3_T! = _ε_fns(mats3,(:T,))
ε_∂ωε_mats3_T = f_ε_∂ωε_mats3_T(1.0,31.4,normalize(rand(3))...,0.3)

f_ε_∂ωε_∂²ωε_mats1, f_ε_∂ωε_∂²ωε_mats1! = _ε_fns2(mats1)
ε_∂ωε_∂²ωε_mats1 = f_ε_∂ωε_∂²ωε_mats1(1.0,normalize(rand(3))...,0.3)
f_ε_∂ωε_∂²ωε_mats1_T, f_ε_∂ωε_∂²ωε_mats1_T! = _ε_fns2(mats1,(:T,))
ε_∂ωε_∂²ωε_mats1_T = f_ε_∂ωε_∂²ωε_mats1_T(1.0,31.4,normalize(rand(3))...,0.3)
f_ε_∂ωε_∂²ωε_mats2, f_ε_∂ωε_∂²ωε_mats2! = _ε_fns2(mats2)
ε_∂ωε_∂²ωε_mats2 = f_ε_∂ωε_∂²ωε_mats2(1.0,normalize(rand(3))...,0.3)
f_ε_∂ωε_∂²ωε_mats2_T, f_ε_∂ωε_∂²ωε_mats2_T! = _ε_fns2(mats2,(:T,))
ε_∂ωε_∂²ωε_mats2_T = f_ε_∂ωε_∂²ωε_mats2_T(1.0,31.4,normalize(rand(3))...,0.3)
f_ε_∂ωε_∂²ωε_mats3, f_ε_∂ωε_∂²ωε_mats3! = _ε_fns2(mats3)
ε_∂ωε_∂²ωε_mats3 = f_ε_∂ωε_∂²ωε_mats3(1.0,normalize(rand(3))...,0.3)
f_ε_∂ωε_∂²ωε_mats3_T, f_ε_∂ωε_∂²ωε_mats3_T! = _ε_fns2(mats3,(:T,))
ε_∂ωε_∂²ωε_mats3_T = f_ε_∂ωε_∂²ωε_mats3_T(1.0,31.4,normalize(rand(3))...,0.3)

fj_ε_∂ωε_mats1, fj_ε_∂ωε_mats1! = _ε_fjs(mats1)
ε_∂ωε_mats1 = fj_ε_∂ωε_mats1(1.0,normalize(rand(3))...,0.3)
fj_ε_∂ωε_mats1_T, fj_ε_∂ωε_mats1_T! = _ε_fjs(mats1,(:T,))
ε_∂ωε_mats1_T = fj_ε_∂ωε_mats1_T(1.0,31.4,normalize(rand(3))...,0.3)
fj_ε_∂ωε_mats2, fj_ε_∂ωε_mats2! = _ε_fjs(mats2)
ε_∂ωε_mats2 = fj_ε_∂ωε_mats2(1.0,normalize(rand(3))...,0.3)
fj_ε_∂ωε_mats2_T, fj_ε_∂ωε_mats2_T! = _ε_fjs(mats2,(:T,))
ε_∂ωε_mats2_T = fj_ε_∂ωε_mats2_T(1.0,31.4,normalize(rand(3))...,0.3)
fj_ε_∂ωε_mats3, fj_ε_∂ωε_mats3! = _ε_fjs(mats3)
ε_∂ωε_mats3 = fj_ε_∂ωε_mats3(1.0,normalize(rand(3))...,0.3)
fj_ε_∂ωε_mats3_T, fj_ε_∂ωε_mats3_T! = _ε_fjs(mats3,(:T,))
ε_∂ωε_mats3_T = fj_ε_∂ωε_mats3_T(1.0,31.4,normalize(rand(3))...,0.3)

fj_ε_∂ωε_∂²ωε_mats1, fj_ε_∂ωε_∂²ωε_mats1! = _ε_fjs2(mats1)
ε_∂ωε_∂²ωε_mats1 = fj_ε_∂ωε_∂²ωε_mats1(1.0,normalize(rand(3))...,0.3)
fj_ε_∂ωε_∂²ωε_mats1_T, fj_ε_∂ωε_∂²ωε_mats1_T! = _ε_fjs2(mats1,(:T,))
ε_∂ωε_∂²ωε_mats1_T = fj_ε_∂ωε_∂²ωε_mats1_T(1.0,31.4,normalize(rand(3))...,0.3)
fj_ε_∂ωε_∂²ωε_mats2, fj_ε_∂ωε_∂²ωε_mats2! = _ε_fjs2(mats2)
ε_∂ωε_∂²ωε_mats2 = fj_ε_∂ωε_∂²ωε_mats2(1.0,normalize(rand(3))...,0.3)
fj_ε_∂ωε_∂²ωε_mats2_T, fj_ε_∂ωε_∂²ωε_mats2_T! = _ε_fjs2(mats2,(:T,))
ε_∂ωε_∂²ωε_mats2_T = fj_ε_∂ωε_∂²ωε_mats2_T(1.0,31.4,normalize(rand(3))...,0.3)
fj_ε_∂ωε_∂²ωε_mats3, fj_ε_∂ωε_∂²ωε_mats3! = _ε_fjs2(mats3)
ε_∂ωε_∂²ωε_mats3 = fj_ε_∂ωε_∂²ωε_mats3(1.0,normalize(rand(3))...,0.3)
fj_ε_∂ωε_∂²ωε_mats3_T, fj_ε_∂ωε_∂²ωε_mats3_T! = _ε_fjs2(mats3,(:T,))
ε_∂ωε_∂²ωε_mats3_T = fj_ε_∂ωε_∂²ωε_mats3_T(1.0,31.4,normalize(rand(3))...,0.3)

function test_ε_fns()
    n1,n2,n3 = normalize(rand(3));
    om = 1.0;
    rr = 0.3;
    TT = 31.5;

    res96 = zeros(9,6);
    res912 = zeros(9,12);
    res920 = zeros(9,20);
    println("Compute ε & ∂ωε vs. (ω, T, n̂, r):")
    println("f_ε_∂ωε_mats1")
    @btime f_ε_∂ωε_mats11($om,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_mats1!")
    @btime f_ε_∂ωε_mats11!($res96,$om,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_mats1_T")
    @btime f_ε_∂ωε_mats1_T($om,$TT,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_mats1_T!")
    @btime f_ε_∂ωε_mats1_T!($res96,$om,$TT,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_mats2")
    @btime f_ε_∂ωε_mats2($om,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_mats2!")
    @btime f_ε_∂ωε_mats2!($res912,$om,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_mats2_T")
    @btime f_ε_∂ωε_mats2_T($om,$TT,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_mats2_T!")
    @btime f_ε_∂ωε_mats2_T!($res912,$om,$TT,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_mats3")
    @btime f_ε_∂ωε_mats3($om,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_mats3!")
    @btime f_ε_∂ωε_mats3!($res920,$om,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_mats3_T")
    @btime f_ε_∂ωε_mats3_T($om,$TT,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_mats3_T!")
    @btime f_ε_∂ωε_mats3_T!($res920,$om,$TT,$n1,$n2,$n3,$rr);

    res99 = zeros(9,9);
    res918 = zeros(9,18);
    res930 = zeros(9,30);
    println("Compute ε, ∂ωε & ∂²ωε vs. (ω, T, n̂, r):")
    println("f_ε_∂ωε_∂²ωε_mats1")
    @btime f_ε_∂ωε_∂²ωε_mats11($om,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_∂²ωε_mats1!")
    @btime f_ε_∂ωε_∂²ωε_mats11!($res99,$om,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_∂²ωε_mats1_T")
    @btime f_ε_∂ωε_∂²ωε_mats1_T($om,$TT,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_∂²ωε_mats1_T!")
    @btime f_ε_∂ωε_∂²ωε_mats1_T!($res99,$om,$TT,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_∂²ωε_mats2")
    @btime f_ε_∂ωε_∂²ωε_mats2($om,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_∂²ωε_mats2!")
    @btime f_ε_∂ωε_∂²ωε_mats2!($res918,$om,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_∂²ωε_mats2_T")
    @btime f_ε_∂ωε_∂²ωε_mats2_T($om,$TT,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_∂²ωε_mats2_T!")
    @btime f_ε_∂ωε_∂²ωε_mats2_T!($res918,$om,$TT,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_∂²ωε_mats3")
    @btime f_ε_∂ωε_∂²ωε_mats3($om,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_∂²ωε_mats3!")
    @btime f_ε_∂ωε_∂²ωε_mats3!($res930,$om,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_∂²ωε_mats3_T")
    @btime f_ε_∂ωε_∂²ωε_mats3_T($om,$TT,$n1,$n2,$n3,$rr);
    println("f_ε_∂ωε_∂²ωε_mats3_T!")
    @btime f_ε_∂ωε_∂²ωε_mats3_T!($res930,$om,$TT,$n1,$n2,$n3,$rr);

    # res96 = zeros(9,6);
    # res912 = zeros(9,12);
    # res920 = zeros(9,20);
    # println("Compute ε, ∂ωε & their jacobians vs. (ω, T, n̂, r):")
    # println("fj_ε_∂ωε_mats1")
    # @btime fj_ε_∂ωε_mats11($om,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_mats1!")
    # @btime fj_ε_∂ωε_mats11!($res96,$om,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_mats1_T")
    # @btime fj_ε_∂ωε_mats1_T($om,$TT,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_mats1_T!")
    # @btime fj_ε_∂ωε_mats1_T!($res96,$om,$TT,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_mats2")
    # @btime fj_ε_∂ωε_mats2($om,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_mats2!")
    # @btime fj_ε_∂ωε_mats2!($res912,$om,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_mats2_T")
    # @btime fj_ε_∂ωε_mats2_T($om,$TT,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_mats2_T!")
    # @btime fj_ε_∂ωε_mats2_T!($res912,$om,$TT,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_mats3")
    # @btime fj_ε_∂ωε_mats3($om,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_mats3!")
    # @btime fj_ε_∂ωε_mats3!($res920,$om,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_mats3_T")
    # @btime fj_ε_∂ωε_mats3_T($om,$TT,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_mats3_T!")
    # @btime fj_ε_∂ωε_mats3_T!($res920,$om,$TT,$n1,$n2,$n3,$rr);

    # res96 = zeros(9,6);
    # res912 = zeros(9,12);
    # res920 = zeros(9,20);
    # println("Compute ε, ∂ωε, ∂²ωε & their jacobians vs. (ω, T, n̂, r):")
    # println("fj_ε_∂ωε_∂²ωε_mats1")
    # @btime fj_ε_∂ωε_∂²ωε_mats11($om,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_∂²ωε_mats1!")
    # @btime fj_ε_∂ωε_∂²ωε_mats11!($res96,$om,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_∂²ωε_mats1_T")
    # @btime fj_ε_∂ωε_∂²ωε_mats1_T($om,$TT,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_∂²ωε_mats1_T!")
    # @btime fj_ε_∂ωε_∂²ωε_mats1_T!($res96,$om,$TT,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_∂²ωε_mats2")
    # @btime fj_ε_∂ωε_∂²ωε_mats2($om,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_∂²ωε_mats2!")
    # @btime fj_ε_∂ωε_∂²ωε_mats2!($res912,$om,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_∂²ωε_mats2_T")
    # @btime fj_ε_∂ωε_∂²ωε_mats2_T($om,$TT,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_∂²ωε_mats2_T!")
    # @btime fj_ε_∂ωε_∂²ωε_mats2_T!($res912,$om,$TT,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_∂²ωε_mats3")
    # @btime fj_ε_∂ωε_∂²ωε_mats3($om,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_∂²ωε_mats3!")
    # @btime fj_ε_∂ωε_∂²ωε_mats3!($res920,$om,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_∂²ωε_mats3_T")
    # @btime fj_ε_∂ωε_∂²ωε_mats3_T($om,$TT,$n1,$n2,$n3,$rr);
    # println("fj_ε_∂ωε_∂²ωε_mats3_T!")
    # @btime fj_ε_∂ωε_∂²ωε_mats3_T!($res920,$om,$TT,$n1,$n2,$n3,$rr);
    return nothing
end
test_ε_fns()
# fε_∂ωε_mats1
#   37.790 μs (95 allocations: 2.84 KiB)
# fε_∂ωε_mats1!
#   194.942 ns (5 allocations: 80 bytes)
# fε_∂ωε_mats1_T
#   38.982 μs (96 allocations: 2.86 KiB)
# fε_∂ωε_mats1_T!
#   218.544 ns (6 allocations: 96 bytes)
# fε_∂ωε_mats2
#   120.748 μs (191 allocations: 5.72 KiB)
# fε_∂ωε_mats2!
#   427.212 ns (5 allocations: 80 bytes)
# fε_∂ωε_mats2_T
#   115.933 μs (192 allocations: 5.73 KiB)
# fε_∂ωε_mats2_T!
#   406.800 ns (6 allocations: 96 bytes)
# fε_∂ωε_mats3
#   261.958 μs (323 allocations: 9.55 KiB)
# fε_∂ωε_mats3!
#   827.223 ns (5 allocations: 80 bytes)
# fε_∂ωε_mats3_T
#   257.068 μs (324 allocations: 9.56 KiB)
# fε_∂ωε_mats3_T!
#   727.197 ns (6 allocations: 96 bytes)


@btime fε_∂ωε_mats32(1.0,31.4,$(normalize(rand(3)))...,0.3)

fε_∂ωε_mats21, fε_∂ωε_mats21! = _ε_fns(mats2)
fε_∂ωε_mats22, fε_∂ωε_mats22! = _ε_fns(mats2,(:T,))
ε_∂ωε_mats21 = fε_∂ωε_mats21(1.0,normalize(rand(3))...,0.3)
ε_∂ωε_mats22 = fε_∂ωε_mats22(1.0,31.4,normalize(rand(3))...,0.3)

fε_∂ωε_mats31, fε_∂ωε_mats31! = _ε_fns(mats3)
fε_∂ωε_mats32, fε_∂ωε_mats32! = _ε_fns(mats3,(:T,))
ε_∂ωε_mats31 = fε_∂ωε_mats31(1.0,normalize(rand(3))...,0.3)
ε_∂ωε_mats32 = fε_∂ωε_mats32(1.0,31.4,normalize(rand(3))...,0.3)

@btime fε_∂ωε_mats32(1.0,31.4,$(normalize(rand(3)))...,0.3)

fε_∂ωε_∂²ωε_and_jac, fε_∂ωε_∂²ωε_and_jac! = _ε_and_jac_fn(mats2,(:T,))
ε_∂ωε_∂²ωε_and_jac2 = fε_∂ωε_∂²ωε_and_jac(1.0,31.4,normalize(rand(3))...,0.3)

# εₑ_mats = map(subsets(1:length(mats),2)) do mat_inds
#     avg_param(ε_mats[mat_inds[1]], ε_mats[mat_inds[2]], S, r)
# end

# ε_mats = map(mm->substitute.(get_model(mm,:ε,:λ),(Dict([(λ=>1/ω),]),)),mats);
# εₑ_mats = map(subsets(1:length(mats),2)) do mat_inds
#     simplify.(avg_param(my_simp.(ε_mats[mat_inds[1]]), my_simp.(ε_mats[mat_inds[2]]), my_simp.(S), r))
# end
# ∂ωε_mats = map(eps->simplify.(expand_derivatives.(Dom.(eps))),ε_mats) ;
# ∂²ωε_mats = map(eps->simplify.(expand_derivatives.((Dom*Dom).(eps))),ε_mats) ;
# ∂ωεₑ_mats = map(eps->simplify.(expand_derivatives.(Dom.(eps))),εₑ_mats) ;
# ∂²ωεₑ_mats = map(eps->simplify.(expand_derivatives.((Dom*Dom).(eps))),εₑ_mats) ;

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
    fεₑ_expr, fεₑ!_expr = build_function(εₑ, ω, r₁, n_1, n_2, n_3) #; parallel=MultithreadedForm())
    write(joinpath(genfn_dir,fεₑ_fpath), string(fεₑ_expr))
    write(joinpath(genfn_dir,fεₑ!_fpath),string(fεₑ!_expr))
    fεₑ  = eval(fεₑ_expr)
    fεₑ!  = eval(fεₑ!_expr)
end
fεₑ(1.0,0.3,normalize(rand(3))...)
fεₑ!(rand(3,3),1.0,0.3,normalize(rand(3))...)
@btime fεₑ(1.0,0.3,$(normalize(rand(3)))...)
@btime fεₑ!($(rand(3,3)),1.0,0.3,$(normalize(rand(3)))...)

f∂ωεₑ_fpath = "f∂ωεₑ.jl"
f∂ωεₑ!_fpath = "f∂ωεₑ_inplace.jl"
if isfile(f∂ωεₑ_fpath)
    println("loading ∂ωεₑ functions")
    f∂ωεₑ   =   include(joinpath(genfn_dir,f∂ωεₑ_fpath))
    f∂ωεₑ!  =   include(joinpath(genfn_dir,f∂ωεₑ!_fpath))
else    
    println("generating and saving ∂ωεₑ functions")
    f∂ωεₑ_expr, f∂ωεₑ!_expr = build_function(∂ωεₑ, ω, r₁, n_1, n_2, n_3) #; parallel=MultithreadedForm())
    write(joinpath(genfn_dir,f∂ωεₑ_fpath), string(f∂ωεₑ_expr))
    write(joinpath(genfn_dir,f∂ωεₑ!_fpath),string(f∂ωεₑ!_expr))
    f∂ωεₑ  = eval(f∂ωεₑ_expr)
    f∂ωεₑ!  = eval(f∂ωεₑ!_expr)
end
f∂ωεₑ(1.0,0.3,normalize(rand(3))...)
f∂ωεₑ!(rand(3,3),1.0,0.3,normalize(rand(3))...)
@btime f∂ωεₑ(1.0,0.3,$(normalize(rand(3)))...)
@btime f∂ωεₑ!($(rand(3,3)),1.0,0.3,$(normalize(rand(3)))...)

f∂²ωεₑ_fpath = "f∂²ωεₑ.jl"
f∂²ωεₑ!_fpath = "f∂²ωεₑ_inplace.jl"
if isfile(f∂²ωεₑ_fpath)
    println("loading ∂²ωεₑ functions")
    f∂²ωεₑ  =   include(joinpath(genfn_dir,f∂²ωεₑ_fpath))
    f∂²ωεₑ! =   include(joinpath(genfn_dir,f∂²ωεₑ!_fpath))
else    
    println("generating and saving ∂²ωεₑ functions")
    f∂²ωεₑ_expr, f∂²ωεₑ!_expr = build_function(∂²ωεₑ, ω, r₁, n_1, n_2, n_3) #; parallel=MultithreadedForm())
    write(joinpath(genfn_dir,f∂²ωεₑ_fpath), string(f∂²ωεₑ_expr))
    write(joinpath(genfn_dir,f∂²ωεₑ!_fpath),string(f∂²ωεₑ!_expr))
    f∂²ωεₑ  = eval(f∂²ωεₑ_expr)
    f∂²ωεₑ!  = eval(f∂²ωεₑ!_expr)
end
f∂²ωεₑ(1.0,0.3,normalize(rand(3))...)
f∂²ωεₑ!(rand(3,3),1.0,0.3,normalize(rand(3))...)
@btime f∂²ωεₑ(1.0,0.3,$(normalize(rand(3)))...)
@btime f∂²ωεₑ!($(rand(3,3)),1.0,0.3,$(normalize(rand(3)))...)

##
fεₑ_∂ωεₑ_∂²ωεₑ_and_jac_fpath = "fεₑ_∂ωεₑ_∂²ωεₑ_and_jac.jl"
fεₑ_∂ωεₑ_∂²ωεₑ_and_jac!_fpath = "fεₑ_∂ωεₑ_∂²ωεₑ_and_jac_inplace.jl"
if isfile(fεₑ_∂ωεₑ_∂²ωεₑ_and_jac_fpath)
    println("loading εₑ_∂ωεₑ_∂²ωεₑ_and_jac functions")
    fεₑ_∂ωεₑ_∂²ωεₑ_and_jac  =   include(joinpath(genfn_dir,fεₑ_∂ωεₑ_∂²ωεₑ_and_jac_fpath))
    fεₑ_∂ωεₑ_∂²ωεₑ_and_jac! =   include(joinpath(genfn_dir,fεₑ_∂ωεₑ_∂²ωεₑ_and_jac!_fpath))
else    
    println("generating and saving εₑ_∂ωεₑ_∂²ωεₑ_and_jac functions")
    fεₑ_∂ωεₑ_∂²ωεₑ_and_jac_expr, fεₑ_∂ωεₑ_∂²ωεₑ_and_jac!_expr = build_function(εₑ_∂ωεₑ_∂²ωεₑ_and_jac, ω, r₁, n_1, n_2, n_3) #; parallel=MultithreadedForm())
    write(joinpath(genfn_dir,fεₑ_∂ωεₑ_∂²ωεₑ_and_jac_fpath), string(fεₑ_∂ωεₑ_∂²ωεₑ_and_jac_expr))
    write(joinpath(genfn_dir,fεₑ_∂ωεₑ_∂²ωεₑ_and_jac!_fpath),string(fεₑ_∂ωεₑ_∂²ωεₑ_and_jac!_expr))
    fεₑ_∂ωεₑ_∂²ωεₑ_and_jac  = eval(fεₑ_∂ωεₑ_∂²ωεₑ_and_jac_expr)
    fεₑ_∂ωεₑ_∂²ωεₑ_and_jac!  = eval(fεₑ_∂ωεₑ_∂²ωεₑ_and_jac!_expr)
end
fεₑ_∂ωεₑ_∂²ωεₑ_and_jac(1.0,0.3,normalize(rand(3))...)
fεₑ_∂ωεₑ_∂²ωεₑ_and_jac!(rand(3,3),1.0,0.3,normalize(rand(3))...)
@btime fεₑ_∂ωεₑ_∂²ωεₑ_and_jac(1.0,0.3,$(normalize(rand(3)))...)
@btime fεₑ_∂ωεₑ_∂²ωεₑ_and_jac!($(rand(3,3)),1.0,0.3,$(normalize(rand(3)))...)


# 194.969 ns (14 allocations: 336 bytes)
# 136.610 ns (5 allocations: 96 bytes)
# 249.731 ns (14 allocations: 336 bytes)
# 190.826 ns (5 allocations: 96 bytes)
# 513.876 ns (14 allocations: 336 bytes)
# 441.338 ns (5 allocations: 96 bytes)

##
include("grad_test_utils.jl")

function εₑ_num(om,rr,nn1,nn2,nn3)
    invom = inv(om)
    nn = normalize([ nn1, nn2, nn3 ]) #^(1//2)]
    SS = normcart(nn)
    no_sq = n²_MgO_LiNbO₃_sym(invom, pₒ.T₀; pₒ...)
    ne_sq = n²_MgO_LiNbO₃_sym(invom, pₑ.T₀; pₑ...)
    epsLN 	= diagm([no_sq, no_sq, ne_sq])
    epsSN 	= n²_Si₃N₄(invom,p_n²_Si₃N₄.T₀) * diagm([1,1,1])
    return avg_param(epsLN, epsSN, SS, rr)
end
εₑ_num(rand(Float64,2)...,normalize(rand(Float64,3))...)
##
function test_εₑ_grads()
    args = vcat(rand(Float64,2),normalize(rand(Float64,3)))  # four random Float64s with the last two taken from a random normalized 3-vector
    println("args: ", args,"\n")
    
    εₑ_of_ω = oo->εₑ_num(oo,args[2:5]...)
    ∂ωεₑ_of_ω = oo->f∂ωεₑ(oo,args[2:5]...)

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

