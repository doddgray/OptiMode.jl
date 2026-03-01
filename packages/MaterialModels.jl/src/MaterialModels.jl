"""
    MaterialModels

A Julia package for dielectric material dispersion modeling.

Provides:
- Symbolic and numerical dispersion models (Sellmeier, Cauchy, NASA thermo-optic)
- Dielectric tensor representations and conversions
- Material rotation and anisotropy
- Group index and group velocity dispersion (GVD) computation
- Symbolic Jacobian/Hessian generation for efficient numerical evaluation
- ChainRulesCore-compatible reverse-mode AD rules
- Extensions for Mooncake.jl and Enzyme.jl automatic differentiation
"""
module MaterialModels

using LinearAlgebra
using LinearAlgebra: diag
using StaticArrays
using StaticArrays: SMatrix, SVector
using Rotations
using Symbolics
using SymbolicUtils
using Symbolics: Sym, Num, scalarize, build_function, expand_derivatives, substitute
using SymbolicUtils: @rule, @acrule, RuleSet, numerators, denominators, flatten_pows
using SymbolicUtils.Rewriters: Chain, RestartedChain, PassThrough, Prewalk, Postwalk
using SymbolicUtils.Code: toexpr, MakeArray
using Symbolics: get_variables, make_array, SerialForm, Func, toexpr, MultithreadedForm,
    tosymbol, Sym, wrap, unwrap, MakeTuple, substitute, value
using RuntimeGeneratedFunctions
using Tullio
using ChainRulesCore
using ChainRulesCore: @non_differentiable, NoTangent, ZeroTangent

RuntimeGeneratedFunctions.init(@__MODULE__)

import Base: nameof

include("cse.jl")
include("epsilon.jl")
include("materials.jl")

end # module MaterialModels
