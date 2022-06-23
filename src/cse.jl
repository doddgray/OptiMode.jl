export make_dargs, oop_fn_op, oop_fn_expr, eval_fn_oop, rgf_fn_oop, ip_fn_op, ip_fn_expr, eval_fn_ip, rgf_fn_ip

# Copied/adapted from David P. Sanders' (@dpsanders) ReversePropagation.jl:
# https://github.com/dpsanders/ReversePropagation.jl
# which is currently incompatible with Symbolics.jl & SymbolicUtils.jl

# module
using OrderedCollections
using SymbolicUtils
using SymbolicUtils: Sym, Term
using SymbolicUtils.Rewriters
# using SymbolicUtils.Code: SetArray, AtIndex, Func, MakeArray, DestructuredArgs, Let, LazyState, Assignment, MakeTuple, LiteralExpr
using Symbolics: SetArray, AtIndex, Func, MakeArray, DestructuredArgs, Let, LazyState, Assignment, MakeTuple, LiteralExpr, SerialForm, MultithreadedForm
using Symbolics
using Symbolics: value, unwrap, wrap, toexpr, variable, 
                make_array, destructure_arg, build_function, _build_function,
                istree, operation, arguments, set_array, _build_and_inject_function
               
is_rathalf(x::Number) = isequal(x, 1//2)
is_rathalf(x) = false
r_ratsqrt1 = @rule ^(~x,~y::is_rathalf) => sqrt(~x)
r_ratsqrt2 = @rule ^(~~xs,~y::is_rathalf) => sqrt(~~xs)
rs_ratsqrt = RuleSet([r_ratsqrt1,r_ratsqrt2])
rw_ratsqrt = PassThrough(Postwalk(Chain([r_ratsqrt1,r_ratsqrt2])))
# To replace (...)^(1//2) with sqrt(...) in a large expression `x` use either
#      ∘  simplify(x,rewriter=rw_ratsqrt,simplify_fractions=false)
#      ∘  simplify(x,rs_ratsqrt,simplify_fractions=false)


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


####################################################################################




# function fn_expr(A::AbstractArray{<:Num},vars;T=Float64)
#     assigns, final = cse_equations(A)
#     input_vars = toexpr(Symbolics.MakeTuple((vars...,)))
#     code = Expr(:block, toexpr.(assigns)..., )
#     # final_ex = toexpr(MakeArray
#     full_code = quote
#         (v_in, ) -> begin
#             $input_vars = v_in
#             $code
#             return reshape( $T[ $(toexpr.(final)...) ] , $(size(final)) )
#         end
#     end
#     return full_code
# end

### testing various code-gen approaches
# using SymbolicUtils.Code: Let, LazyState

make_dargs(args...;checkbounds=false) = map((x) -> destructure_arg(x[2], !checkbounds, Symbol("ˍ₋arg$(x[1])")), enumerate([args...]))

function oop_fn_op(A::AbstractArray, args...;
        checkbounds = false,
        simp_fn = ex->simplify(ex,rewriter=rw_ratsqrt,simplify_fractions=false),
        let_block = false,
        parallel=SerialForm(), kwargs...)
    dargs = make_dargs(args...;checkbounds)
    i = findfirst(x->x isa DestructuredArgs, dargs)
    similarto = i === nothing ? Array : dargs[i].name
    assigns, final = cse_equations(simp_fn.(A))
    return Func(dargs, [], Let(assigns,make_array(parallel, dargs, final, similarto,false),let_block))
end
oop_fn_expr(A, args...;states=LazyState(),kwargs...) = toexpr(oop_fn_op(A, args...; kwargs...), states)
eval_fn_oop(A,args...;kwargs...) = eval(oop_fn_expr(A, args...;kwargs...))
rgf_fn_oop(A,args...;expression_module = @__MODULE__(), kwargs...) = _build_and_inject_function(expression_module,oop_fn_expr(A, args...;kwargs...))

function ip_fn_op(A::AbstractArray, args...;
        checkbounds = false,
        linenumbers = false,
        outputidxs=nothing,
        skipzeros = false,
        fillzeros = skipzeros && !(A isa SparseMatrixCSC),
        simp_fn = ex->simplify(ex,rewriter=rw_ratsqrt,simplify_fractions=false),
        let_block = false,
        parallel=SerialForm(), kwargs...)
    dargs = make_dargs(args...;checkbounds)
    i = findfirst(x->x isa DestructuredArgs, dargs)
    similarto = i === nothing ? Array : dargs[i].name
    assigns, final = cse_equations(simp_fn.(A))
    out = Sym{Any}(:ˍ₋out)
    body = set_array(parallel, dargs, out, outputidxs, final, checkbounds, skipzeros)
    return Func([out, dargs...], [], Let(assigns,body,let_block) )
end
ip_fn_expr(A, args...;states=LazyState(),kwargs...) = toexpr(ip_fn_op(A, args...; kwargs...), states)
eval_fn_ip(A,args...;kwargs...) = eval(ip_fn_expr(A, args...;kwargs...))
rgf_fn_ip(A,args...;expression_module = @__MODULE__(), kwargs...) = _build_and_inject_function(expression_module,ip_fn_expr(A, args...;kwargs...))

