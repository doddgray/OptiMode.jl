# Copied/adapted from David P. Sanders' (@dpsanders) ReversePropagation.jl:
# https://github.com/dpsanders/ReversePropagation.jl
# which is currently incompatible with Symbolics.jl & SymbolicUtils.jl

# module
using OrderedCollections
using SymbolicUtils
using SymbolicUtils: Sym, Term
using SymbolicUtils.Rewriters

using Symbolics
using Symbolics: value, unwrap, wrap, toexpr, variable, 
                make_array, destructure_arg, build_function, _build_function,
                MakeArray, Func, DestructuredArgs, Assignment, MakeTuple,
                istree, operation, arguments, SerialForm, MultithreadedForm
               



const symbol_numbers = Dict{Symbol, Int}()

"""Return a new, unique symbol like _z3.
Updates the global dict `symbol_numbers`"""
function make_symbol(s::Symbol)

    i = get(symbol_numbers, s, 0)
    symbol_numbers[s] = i + 1

    if i == 0
        return Symbol("_", s)
    else
        return Symbol("_", s, i)
    end
end

make_symbol(c::Char) = make_symbol(Symbol(c))

make_variable(s;T=Float64) = variable(make_symbol(s);T=T)# Variable(make_symbol(s))

let current_symbol = 'a'

    """Make a new symbol like `_c`. 
    Cycles through the alphabet and adds numbers if necessary.
    """
    global function make_variable()
        current_sym = current_symbol

        if current_sym < 'z'
            current_symbol += 1
        else
            current_symbol = 'a'
        end

        return make_variable(current_sym)

    end
end

## Common Subexpression Elimination (CSE)

"""Do common subexpression elimination on the expression `ex`, 
by traversing it recursively and reducing to binary operations.
Modifies the `OrderedDict` `dict`.
"""
function cse!(dict, ex)

    if istree(ex)
        
        args = arguments(ex)
        op = operation(ex)

        if length(args) == 1 

            left  = cse!(dict, args[1])

            ex = op(left)

        elseif length(args) == 2 

            left  = cse!(dict, args[1])
            right = cse!(dict, args[2])

            ex = op(left, right)

        else
            left  = cse!(dict, args[1])
            right = cse!(dict, op(args[2:end]...))   # use similarterm?

            ex = op(left, right)

        end

        if haskey(dict, ex)
            return dict[ex]

        else
            val = make_variable()
            push!(dict, ex => val)
        end

        return val
    
    else  # not a tree
        return ex
    end

end

"Do CSE on an expression"
function cse(ex)
    dict = OrderedDict()
    final = cse!(dict, ex)

    return dict, final 
end

"Version of CSE returing a vector of equations"
function cse_equations(ex) 
    dict, final = cse(ex) 

    return [Assignment(rhs, lhs) for (lhs, rhs) in pairs(dict)], final
end

cse(ex::Num) = cse(Symbolics.value(ex))


# @syms x y 

# ex = exp(3x^2 + 4x^2 * y)

# dict, final = cse(ex)

# cse_equations(ex)

# dict
# final

# end

### New code extending this for generating code from arrays of expression with global CSE

function cse(A::AbstractArray{<:Num})
    dict = OrderedDict() 
    A_final = cse!.((dict,),Symbolics.unwrap.(A));
    return dict, A_final
end

function cse_equations(A::AbstractArray{<:Num})
    dict, A_final = cse(A)
    [Assignment(rhs, lhs) for (lhs, rhs) in pairs(dict)], A_final
end

function fn_expr(A::AbstractArray{<:Num},vars;T=Float64)
    assigns, final = cse_equations(A)
    input_vars = toexpr(Symbolics.MakeTuple((vars...,)))
    code = Expr(:block, toexpr.(assigns)..., )
    # final_ex = toexpr(MakeArray
    full_code = quote
        (v_in, ) -> begin
            $input_vars = v_in
            $code
            return reshape( $T[ $(toexpr.(final)...) ] , $(size(final)) )
        end
    end
    return full_code
end

### testing various code-gen approaches
function DA_MA_exp1(A,args...;parallel=SerialForm(),similarto=nothing)
    dargs = map((x) -> destructure_arg(x[2], !false,Symbol("ˍ₋arg$(x[1])")), enumerate([args...]))
    if isnothing(similarto)
        i = findfirst(x->x isa DestructuredArgs, dargs)
        similarto = i === nothing ? Array : dargs[i].name
    end
    MA = make_array(parallel,dargs,A,similarto)
    return dargs, MA
end

function DA_MA_exp2(A,args...;similarto=nothing,T=Float64)
    dargs = map((x) -> destructure_arg(x[2], !false,Symbol("ˍ₋arg$(x[1])")), enumerate([args...]))
    if isnothing(similarto)
        i = findfirst(x->x isa DestructuredArgs, dargs)
        similarto = i === nothing ? Array : dargs[i].name
    end
    MA = MakeArray(unwrap.(A),similarto,T)
    return dargs, MA
end