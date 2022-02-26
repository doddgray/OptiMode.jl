using SymbolicUtils
using SymbolicUtils: @rule, @acrule, @ordered_acrule, isnotflat, needs_sorting, is_literal_number, hasrepeats, merge_repeats, _iszero, _isone, _isreal, _isinteger, is_operation, needs_sorting, has_trig_exp
using SymbolicUtils.Rewriters

const PLUS_RULES = [
    @rule(~x::isnotflat(+) => flatten_term(+, ~x))
    @rule(~x::needs_sorting(+) => sort_args(+, ~x))
    @ordered_acrule(~a::is_literal_number + ~b::is_literal_number => ~a + ~b)

    @acrule(*(~~x) + *(~β, ~~x) => *(1 + ~β, (~~x)...))
    @acrule(*(~α, ~~x) + *(~β, ~~x) => *(~α + ~β, (~~x)...))
    @acrule(*(~~x, ~α) + *(~~x, ~β) => *(~α + ~β, (~~x)...))

    @acrule(~x + *(~β, ~x) => *(1 + ~β, ~x))
    @acrule(*(~α::is_literal_number, ~x) + ~x => *(~α + 1, ~x))
    @rule(+(~~x::hasrepeats) => +(merge_repeats(*, ~~x)...))

    @ordered_acrule((~z::_iszero + ~x) => ~x)
    @rule(+(~x) => ~x)
]

const TIMES_RULES = [
    @rule(~x::isnotflat(*) => flatten_term(*, ~x))
    @rule(~x::needs_sorting(*) => sort_args(*, ~x))

    @ordered_acrule(~a::is_literal_number * ~b::is_literal_number => ~a * ~b)
    @rule(*(~~x::hasrepeats) => *(merge_repeats(^, ~~x)...))

    @acrule((~y)^(~n) * ~y => (~y)^(~n+1))
    @ordered_acrule((~x)^(~n) * (~x)^(~m) => (~x)^(~n + ~m))

    @ordered_acrule((~z::_isone  * ~x) => ~x)
    @ordered_acrule((~z::_iszero *  ~x) => ~z)
    @rule(*(~x) => ~x)
]


const POW_RULES = [
    @rule(^(*(~~x), ~y::_isinteger) => *(map(a->pow(a, ~y), ~~x)...))
    @rule((((~x)^(~p::_isinteger))^(~q::_isinteger)) => (~x)^((~p)*(~q)))
    @rule(^(~x, ~z::_iszero) => 1)
    @rule(^(~x, ~z::_isone) => ~x)
    @rule(inv(~x) => 1/(~x))
]

const ASSORTED_RULES = [
    @rule(identity(~x) => ~x)
    @rule(-(~x) => -1*~x)
    @rule(-(~x, ~y) => ~x + -1(~y))
    @rule(~x::_isone \ ~y => ~y)
    @rule(~x \ ~y => ~y / (~x))
    @rule(one(~x) => one(symtype(~x)))
    @rule(zero(~x) => zero(symtype(~x)))
    @rule(conj(~x::_isreal) => ~x)
    @rule(real(~x::_isreal) => ~x)
    @rule(imag(~x::_isreal) => zero(symtype(~x)))
    @rule(ifelse(~x::is_literal_number, ~y, ~z) => ~x ? ~y : ~z)
]

const TRIG_EXP_RULES = [
    @acrule(~r*~x::has_trig_exp + ~r*~y => ~r*(~x + ~y))
    @acrule(~r*~x::has_trig_exp + -1*~r*~y => ~r*(~x - ~y))
    @acrule(sin(~x)^2 + cos(~x)^2 => one(~x))
    @acrule(sin(~x)^2 + -1        => -1*cos(~x)^2)
    @acrule(cos(~x)^2 + -1        => -1*sin(~x)^2)

    @acrule(cos(~x)^2 + -1*sin(~x)^2 => cos(2 * ~x))
    @acrule(sin(~x)^2 + -1*cos(~x)^2 => -cos(2 * ~x))
    @acrule(cos(~x) * sin(~x) => sin(2 * ~x)/2)

    @acrule(tan(~x)^2 + -1*sec(~x)^2 => one(~x))
    @acrule(-1*tan(~x)^2 + sec(~x)^2 => one(~x))
    @acrule(tan(~x)^2 +  1 => sec(~x)^2)
    @acrule(sec(~x)^2 + -1 => tan(~x)^2)

    @acrule(cot(~x)^2 + -1*csc(~x)^2 => one(~x))
    @acrule(cot(~x)^2 +  1 => csc(~x)^2)
    @acrule(csc(~x)^2 + -1 => cot(~x)^2)

    @acrule(exp(~x) * exp(~y) => _iszero(~x + ~y) ? 1 : exp(~x + ~y))
    @rule(exp(~x)^(~y) => exp(~x * ~y))
]

const BOOLEAN_RULES = [
    @rule((true | (~x)) => true)
    @rule(((~x) | true) => true)
    @rule((false | (~x)) => ~x)
    @rule(((~x) | false) => ~x)
    @rule((true & (~x)) => ~x)
    @rule(((~x) & true) => ~x)
    @rule((false & (~x)) => false)
    @rule(((~x) & false) => false)

    @rule(!(~x) & ~x => false)
    @rule(~x & !(~x) => false)
    @rule(!(~x) | ~x => true)
    @rule(~x | !(~x) => true)
    @rule(xor(~x, !(~x)) => true)
    @rule(xor(~x, ~x) => false)

    @rule(~x == ~x => true)
    @rule(~x != ~x => false)
    @rule(~x < ~x => false)
    @rule(~x > ~x => false)

    # simplify terms with no symbolic arguments
    # e.g. this simplifies term(isodd, 3, type=Bool)
    # or term(!, false)
    @rule((~f)(~x::is_literal_number) => (~f)(~x))
    # and this simplifies any binary comparison operator
    @rule((~f)(~x::is_literal_number, ~y::is_literal_number) => (~f)(~x, ~y))
]

function number_simplifier()
    rule_tree = [If(istree, Chain(ASSORTED_RULES)),
                    If(is_operation(+),
                    Chain(PLUS_RULES)),
                    If(is_operation(*),
                    Chain(TIMES_RULES)),
                    If(is_operation(^),
                    Chain(POW_RULES))] |> RestartedChain

    rule_tree
end

trig_exp_simplifier(;kw...) = Chain(TRIG_EXP_RULES)


const nobool_simplifier(; kw...) = Postwalk(Chain((number_simplifier(),trig_exp_simplifier())); kw...)
const nobool_notrig_simplifier(; kw...) = Postwalk(Chain((number_simplifier(),)); kw...)

const serial_nobool_simplifier = If(istree, Fixpoint(nobool_simplifier()))

const threaded_nobool_simplifier(cutoff) = Fixpoint(nobool_simplifier(threaded=true,
                                                            thread_cutoff=cutoff))

const serial_expand_nobool_simplifier = If(istree,
                                Fixpoint(Chain((expand,
                                                Fixpoint(nobool_simplifier())))))

const serial_nobool_notrig_simplifier = If(istree, Fixpoint(nobool_notrig_simplifier()))

const threaded_nobool_notrig_simplifier(cutoff) = Fixpoint(nobool_notrig_simplifier(threaded=true,
                                                            thread_cutoff=cutoff))

const serial_expand_nobool_notrig_simplifier = If(istree,
                                Fixpoint(Chain((expand,
                                                Fixpoint(nobool_notrig_simplifier())))))

bool_simplifier() = Chain(BOOLEAN_RULES)

global default_simplifier
global serial_simplifier
global threaded_simplifier
global serial_simplifier
global serial_expand_simplifier

function default_simplifier(; kw...)
    IfElse(has_trig_exp,
            Postwalk(IfElse(x->symtype(x) <: Number,
                            Chain((number_simplifier(),
                                    trig_exp_simplifier())),
                            If(x->symtype(x) <: Bool,
                                bool_simplifier()))
                    ; kw...),
            Postwalk(Chain((If(x->symtype(x) <: Number,
                                number_simplifier()),
                            If(x->symtype(x) <: Bool,
                                bool_simplifier())))
                    ; kw...))
end

# reduce overhead of simplify by defining these as constant
serial_simplifier = If(istree, Fixpoint(default_simplifier()))

threaded_simplifier(cutoff) = Fixpoint(default_simplifier(threaded=true,
                                                            thread_cutoff=cutoff))

serial_expand_simplifier = If(istree,
                                Fixpoint(Chain((expand,
                                                Fixpoint(default_simplifier())))))
end



#r_sqrt_pow = @rule sqrt(~x) --> (~x)^(1/2)

# # example simplify rules from 
# # https://discourse.julialang.org/t/symbolic-computations-with-functions-having-a-variable-number-of-arguments/65140/14
# using SymbolicUtils
# using SymbolicUtils: Add, Mul, Term, Symbolic

# const f = SymbolicUtils.Sym{(SymbolicUtils.FnType){Tuple, Number}}(:f)

# function expand_multilinear_arg(t::Add, i::Int)
#     sum(c * expand_multilinear_arg(x, i) for (x, c) in t.dict)
# end

# function expand_multilinear_arg(t::Mul, i::Int)
#     a, _ = first(t.dict)
#     # by construction, t has only one factor, and that has exponent 1
#     t.coeff * expand_multilinear_arg(a, i)
# end

# function expand_multilinear_arg(t::Term{T}, i::Int) where T
#     f = t.f
#     b = convert(Vector{Any}, t.arguments)
#     # this is needed for the assignment of one(T) to b[i] later one
#     # note that the type of FnType.arguments is Any anyway
#     x = b[i]
#     if x isa Add
#         d = Dict((b[i] = y; f(b...)) => c for (y, c) in x.dict)   
#         c = x.coeff
#         if !iszero(c)
#             b[i] = one(T)
#             d[f(b...)] = c
#         end
#         Add(T, zero(T), d)
#     elseif x isa Mul
#         p = x.dict
#         b[i] = length(p) == 1 ? first(p).first : Mul(T, one(T), p)
#         Mul(T, x.coeff, Dict(f(b...) => one(T)))
#     else
#         t
#     end
# end

# function expand_multilinear_allargs(t)
# # the function f is hard-coded here
#     if t isa Term && t.f == f
#         foldl(expand_multilinear_arg, 1:length(t.arguments); init = t)
#     else
#         t
#     end
# end

# expand_multilinear = Rewriters.Postwalk(expand_multilinear_allargs)

# ##

# using SymbolicUtils, SymbolicUtils.Rewriters

# using BenchmarkTools

# const f = SymbolicUtils.Sym{SymbolicUtils.FnType{Tuple, Number}}(:f)

# r1 = @rule(f(~~a, +(~~b), ~~c) => sum([f(~~a..., x, ~~c...) for x in ~~b]))
# r2 = @rule(f(~~a, ~c::(x -> x isa Int) * ~x, ~~b) => ~c * f(~~a..., ~x, ~~b...))

# expand_multilinear = Fixpoint(Postwalk(Chain([r1, r2])))

# @syms u v w x y z

# @btime expand_multilinear(f(u+2v+3w+4x+5y+6z, u+2v+3w+4x+5y+6z, u+2v+3w+4x+5y+6z, u+2v+3w+4x+5y+6z, u+2v+3w+4x+5y+6z));

# ##

# ### Examples of rule writing 

# # from 
# # https://discourse.julialang.org/t/why-wont-symbolics-jl-simplify-sqrt-1-to-1/65921/4
# r1 = @rule sqrt(1) => 1
# r2 = @rule cos(~x)^2+sin(~x)^2 => 1
# simplify(Symbolics.Term(sqrt, 1), RuleSet([r1]))
# simplify(sqrt(cos(x)^2 + sin(x)^2) , RuleSet([r1,r2]))

# # from
# # https://discourse.julialang.org/t/can-symbolics-jl-simplify-subexpressions/66731
# using Symbolics.Rewriters
# sqrtrule = @rule sqrt((~x)^2) => ~x
# simplify(t2, Rewriters.Prewalk(Rewriters.PassThrough(sqrtrule)))

# # from the Symbolics wrapping of SymbolicUtils functions at the bottom of 
# # https://github.com/JuliaSymbolics/Symbolics.jl/blob/master/src/Symbolics.jl
# SymbolicUtils.simplify_fractions(n::Symbolics.Num; kw...) = wrap(SymbolicUtils.simplify_fractions(unwrap(n); kw...))