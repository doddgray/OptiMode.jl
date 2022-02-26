# using ChainRules
# using FiniteDifferences
# using ForwardDiff
# using Zygote
# using ReverseDiff
# using ReversePropagation
# using Enzyme

using ChainRules, FiniteDifferences, ForwardDiff, Diffractor
using ReverseDiff, Tracker, Enzyme
using BenchmarkTools

function _jacDiffractor(f)
    function jacf(z)
        primal, vjp_fn = ∂⃖(x->vec(f(x)),z)
        return mapreduce(y->getindex(vjp_fn(y),2),hcat,eachcol(diagm(ones(length(primal)))))'
    end
    return jacf
end

_jacFD(f) = (y;n=3)->first(FiniteDifferences.jacobian(central_fdm(n,1),x->vec(f(x)),y)) 
_jacFM(f) = y->ForwardDiff.jacobian(x->vec(f(x)),y);
_jacRM(f) = _jacDiffractor(f);
_jacs(f) = (_jacFD(f),_jacFM(f),_jacRM(f));

function _jacCheck(f)
    jacFD,jacFM,jacRM = _jacs(f)
    function jacCheck(x;do_benchmarks=false,nFD=3)
        jFD,jFM,jRM = jacFD(x;n=nFD),jacFM(x),jacRM(x)
        println("###################################################################################\n")
        println("Maximum discrepancy magnitude between Forward-Mode and Finite-Difference Jacobians")
        @show maximum(abs,(jFD - jFM))
        println("")
        println("Maximum discrepancy magnitude between Reverse-Mode and Finite-Difference Jacobians")
        @show maximum(abs,(jFD - jRM))
        println("\n###################################################################################")

        # if do_benchmarks
        #     @benchmark $(jacFD)($x)
        #     @benchmark $(jacFM)($x)
        #     @benchmark $(jacRM)($x)
        # end

        return jFD,jFM,jRM
    end
end

function _jacCheck(f,j)
    jacFD,jacFM,jacRM = _jacs(f)
    function jacCheck(x;do_benchmarks=false,nFD=3)
        jFD,jFM,jRM,jSym = jacFD(x;n=nFD),jacFM(x),jacRM(x),j(x)
        println("###################################################################################\n")
        println("Maximum discrepancy magnitude between Forward-Mode and Finite-Difference Jacobians")
        @show maximum(abs,(jFD - jFM))
        println("")
        println("Maximum discrepancy magnitude between Reverse-Mode and Finite-Difference Jacobians")
        @show maximum(abs,(jFD - jRM))
        println("")
        println("Maximum discrepancy magnitude between Symbolic and Finite-Difference Jacobians")
        @show maximum(abs,(jFD - jSym))
        println("\n###################################################################################")

        # if do_benchmarks
        #     @benchmark $(jacFD)($x)
        #     @benchmark $(jacFM)($x)
        #     @benchmark $(jacRM)($x)
        #     @benchmark $(j)($x)
        # end
        return jFD,jFM,jRM,jSym
    end
end

# gradRM(fn,in) 			= 	Zygote.gradient(fn,in)[1]
# gradFM(fn,in) 			= 	ForwardDiff.gradient(fn,in)
# gradFD(fn,in;n=3)		=	FiniteDifferences.grad(central_fdm(n,1),fn,in)[1]
# # gradFD2(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_gradient(fn,in;relstep=rs)

# derivRM(fn,in) 			= 	Zygote.gradient(fn,in)[1]
# derivFM(fn,in) 			= 	ForwardDiff.derivative(fn,in)
# derivFD(fn,in;n=3)		=	FiniteDifferences.grad(central_fdm(n,1),fn,in)[1]
# # derivFD2(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_derivative(fn,in;relstep=rs)

# jacRM(fn,in;out_shape=size(fn(in))) 			= 	reshape(Zygote.jacobian(x->vec(fn(x)),in)[1],out_shape)
# jacFM(fn,in) 			= 	ForwardDiff.jacobian(x->vec(fn(x)),in)
# # jacFD(fn,in::Real;out_shape=size(fn(in)),n=3)		    =	reshape(FiniteDifferences.jacobian(central_fdm(n,1),x->vec(fn(x...)),in)[1],out_shape)
# jacFD(fn,in::AbstractArray;n=3)  =	FiniteDifferences.jacobian(central_fdm(n,1),x->vec(fn(x)),in)[1]
# # jacFD2(fn,in;rs=1e-2)	=	FiniteDiff.finite_difference_jacobian(fn,in;relstep=rs)