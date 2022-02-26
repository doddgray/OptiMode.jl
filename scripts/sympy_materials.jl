using LinearAlgebra
using SymPy

# basic SymPy -> julia code conversion from
# from https://github.com/JuliaSymbolics/Symbolics.jl/issues/201
# (April 2021)
# 
#
using Symbolics
import PyCall
# const sympy = PyCall.pyimport_conda("sympy", "sympy")

function PyCall.PyObject(ex::Num)
    # sympy.sympify(string(ex)) # lose type detail of variables
    sympy.sympify(string(ex))
end;
# Symbolics.Num(o::PyCall.PyObject) = eval(Meta.parse(sympy.julia_code(o)))
Symbolics.Num(o::PyCall.PyObject) = eval(Meta.parse(sympy.julia_code(o)))
#

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
    h = [0.0, n[3], -n[2] ] # = n × [ 1, 0, 0 ]  for now ignore edge case where n = [1,0,0]
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

@syms ω::(real,nonnegative) T::(real,nonnegative) θ::real
@syms ε₁[1:3,1:3]::real ε₂[1:3,1:3]::real
# @syms ε₁_11::real ε₁_12::real ε₁_13::real ε₁_21::real ε₁_22::real ε₁_23::real ε₁_31::real ε₁_32::real ε₁_33::real ε₂_11::real ε₂_12::real ε₂_13::real ε₂_21::real ε₂_22::real ε₂_23::real ε₂_31::real ε₂_32::real ε₂_33::real 
# ε₁ = [ ε₁_11  ε₁_12  ε₁_13 ;  ε₁_21  ε₁_22  ε₁_23 ; ε₁_31  ε₁_32  ε₁_33 ] 
# ε₂ = [ ε₂_11  ε₂_12  ε₂_13 ;  ε₂_21  ε₂_22  ε₂_23 ; ε₂_31  ε₂_32  ε₂_33 ]
@syms r::(real,nonnegative) n[1:3]::real # n_1::real n_2::real n_3::real   # try enforcing normalization of 3-vector `n` by building it with two real scalar variables
# n = [ n_1, n_2, n_3 ] # [ n_1, n_2, n_3 ] / sqrt( n_1^2 + n_2^2 + n_3^2 )  # simplification works better with (...)^(1//2) than sqrt(...)
S = normcart(n)

nₒ² = n²_MgO_LiNbO₃_sym(1/ω, pₒ.T₀; pₒ...)
nₑ² = n²_MgO_LiNbO₃_sym(1/ω, pₑ.T₀; pₑ...)
εLN 	= diagm([nₒ², nₒ², nₑ²])
εSN 	= n²_Si₃N₄(1/ω,p_n²_Si₃N₄.T₀) * diagm([1,1,1])


εₑ = avg_param(εLN, εSN, S, r);
# ∂ωεₑ = differentiate(εₑ,ω);
# ∂²ωεₑ = differentiate(εₑ,ω,Val{2}());