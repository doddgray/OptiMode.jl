# Vendored ChainRules→Enzyme rule importers.
#
# `Enzyme.@import_rrule`/`@import_frule` call `Enzyme._import_rrule`/`_import_frule`, whose
# *methods* live in Enzyme's `EnzymeChainRulesCoreExt`. That extension is not reliably
# loaded while a package's own Enzyme extension precompiles or runs `__init__` (Julia loads
# the two extensions in an unspecified order once both become loadable, and
# `Base.retry_load_extensions()` is a no-op mid-load), so the macros hit the empty
# `_import_rrule` generic and the bridge silently fails to register (observed with Enzyme
# 0.13.168 on Julia 1.11). These vendored generators reproduce Enzyme's importers verbatim
# (Enzyme 0.13.x) — they use only Enzyme-core / `EnzymeRules` / `ChainRulesCore` APIs, all
# available whenever Enzyme is loaded — so the rules register regardless of extension order.
# Only the bare `same_or_one` reference is interpolated (`$(Enzyme.same_or_one)`) since it
# is an Enzyme internal, not exported. Use `@vendored_import_rrule`/`@vendored_import_frule`.

function _vendored_import_frule(fn, tys...)
    vals = []
    exprs = []
    primals = []
    tangents = []
    tangentsi = []
    anns = []
    for (i, ty) in enumerate(tys)
        val = Symbol("arg_$i")
        TA = Symbol("AN_$i")
        e = :($val::$TA)
        push!(anns, :($TA <: Annotation{<:$(esc(ty))}))
        push!(vals, val)
        push!(exprs, e)
        push!(primals, :($val.val))
        push!(tangents, :($val isa Const ? $ChainRulesCore.NoTangent() : $val.dval))
        push!(tangentsi, :($val isa Const ? $ChainRulesCore.NoTangent() : $val.dval[i]))
    end

    quote
        function EnzymeRules.forward(config, fn::FA, ::Type{RetAnnotation}, $(exprs...); kwargs...) where {RetAnnotation, FA<:Annotation{<:$(esc(fn))}, $(anns...)}
            batchsize = $(Enzyme.same_or_one)(1, $(vals...))
            if batchsize == 1
                dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval
                cres = $ChainRulesCore.frule((dfn, $(tangents...),), fn.val, $(primals...); kwargs...)
                if RetAnnotation <: Const
                    if EnzymeRules.needs_primal(config)
                       return cres[1]::eltype(RetAnnotation)
                    else
                       return nothing
                    end
                elseif RetAnnotation <: Duplicated
                    return Duplicated(cres[1], cres[2])
                elseif RetAnnotation <: DuplicatedNoNeed
                    return cres[2]::eltype(RetAnnotation)
                else
                    @assert false
                end
            else
                if RetAnnotation <: Const
                    cres = ntuple(Val(batchsize)) do i
                        Base.@_inline_meta
                        dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval[i]
                        $ChainRulesCore.frule((dfn, $(tangentsi...),), fn.val, $(primals...); kwargs...)
                    end
                    if EnzymeRules.needs_primal(config)
                       return cres[1][1]::eltype(RetAnnotation)
                    else
                       return nothing
                    end
                elseif RetAnnotation <: BatchDuplicated
                    cres1 = begin
                        i = 1
                        dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval[i]
                        $ChainRulesCore.frule((dfn, $(tangentsi...),), fn.val, $(primals...); kwargs...)
                    end
                    batches = ntuple(Val(batchsize-1)) do j
                        Base.@_inline_meta
                        i = j+1
                        dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval[i]
                        $ChainRulesCore.frule((dfn, $(tangentsi...),), fn.val, $(primals...); kwargs...)[2]
                    end
                    return BatchDuplicated(cres1[1], (cres1[2], batches...))
                elseif RetAnnotation <: BatchDuplicatedNoNeed
                    ntuple(Val(batchsize)) do i
                        Base.@_inline_meta
                        dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval[i]
                        $ChainRulesCore.frule((dfn, $(tangentsi...),), fn.val, $(primals...); kwargs...)[2]
                    end
                else
                    @assert false
                end
            end
        end
    end # quote
end

macro vendored_import_frule(args...)
    return _vendored_import_frule(args...)
end

function _vendored_import_rrule(fn, tys...)
    vals = []
    valtys = []
    exprs = []
    primals = []
    tangents = []
    tangentsi = []
    anns = []
    nothings = []
    ntys = length(tys)
    act_res = Expr[:(fn isa Active ? res[1] : nothing)]
    invertcomb = Expr[]

    ptys = []
    for (i, ty) in enumerate(tys)
        push!(nothings, :(nothing))
        val = Symbol("arg_$i")
        TA = Symbol("AN_$i")
        e = :($val::$TA)
        push!(ptys, :(::$(esc(ty))))
        push!(anns, :($TA <: Annotation{<:$(esc(ty))}))
        push!(vals, val)
        push!(exprs, e)
        primal = Symbol("primcopy_$i")
        push!(primals, primal)
        push!(valtys, :($primal = $(EnzymeRules.overwritten)(config)[$i+1] ? deepcopy($val.val) : $val.val))
        push!(tangents, :($val isa $Enzyme.Const ? $ChainRulesCore.NoTangent() : $val.dval))
        push!(tangentsi, :($val isa  $Enzyme.Const ? $ChainRulesCore.NoTangent() : $val.dval[i]))
        push!(act_res, :($val isa Active ? (res[$i+1] isa $ChainRulesCore.NoTangent ? zero($val) : $ChainRulesCore.unthunk(res[$i+1]) ) : nothing))
        push!(invertcomb, quote
        $val isa Active ? (
            (EnzymeRules.width(config) == 1) ? tcomb[1][$i+1] :
            ntuple(Val(EnzymeRules.width(config))) do batch_i
                Base.@_inline_meta
                tcomb[batch_i][$i+1]
            end
           ) : nothing
        end)
    end

    quote
        EnzymeRules.has_easy_rule(::$(esc(fn)), $(ptys...)) = true

        function EnzymeRules.augmented_primal(config, fn::FA, ::Type{RetAnnotation}, $(exprs...); kwargs...) where {RetAnnotation, FA<:Annotation{<:$(esc(fn))}, $(anns...)}
            $(valtys...)

            @assert !(RetAnnotation <: Const)
            res, pullback = $ChainRulesCore.rrule(fn.val, $(primals...); kwargs...)

            primal = if EnzymeRules.needs_primal(config)
                res
            else
                nothing
            end

            shadow, byref = if !EnzymeRules.needs_shadow(config)
                nothing, Val(false)
            elseif !Enzyme.Compiler.guaranteed_nonactive(Core.Typeof(res))
                (if EnzymeRules.width(config) == 1
                    Ref(Enzyme.make_zero(res))
                else
                    ntuple(Val(EnzymeRules.width(config))) do j
                        Base.@_inline_meta
                        Ref(Enzyme.make_zero(res))
                    end
                end, Val(true))
            else
                (if EnzymeRules.width(config) == 1
                    Enzyme.make_zero(res)
                else
                    ntuple(Val(EnzymeRules.width(config))) do j
                        Base.@_inline_meta
                        Enzyme.make_zero(res)
                    end
                end, Val(false))
            end

            cache = (shadow, pullback, byref)
            return EnzymeRules.augmented_rule_return_type(config, RetAnnotation){typeof(cache)}(primal, shadow, cache)
        end

        function EnzymeRules.reverse(config, fn::FA, ::Type{RetAnnotation}, tape::TapeTy, $(exprs...); kwargs...) where {RetAnnotation, TapeTy, FA<:Annotation{<:$(esc(fn))}, $(anns...)}
            if !(RetAnnotation <: Const)
                shadow, pullback, byref = tape

                tcomb = ntuple(Val(EnzymeRules.width(config))) do batch_i
                    Base.@_inline_meta
                    shad = EnzymeRules.width(config)  == 1 ? shadow : shadow[batch_i]
                    if byref === Val(true)
                        shad = shad[]
                    end
                    res = pullback(shad)

                    for (cr, en) in zip(res, (fn, $(vals...),))
                        if en isa Const || cr isa $ChainRulesCore.NoTangent
                            continue
                        end
                        if en isa Active
                            continue
                        end
                        if EnzymeRules.width(config)  == 1
                            en.dval .+= cr
                        else
                            en.dval[batch_i] .+= cr
                        end
                    end

                    ($(act_res...),)
                end

                return ($(invertcomb...),)
            end

            return ($(nothings...),)
        end

        function EnzymeRules.reverse(config, fn::FA, dval::Active{RetAnnotation}, tape::TapeTy, $(exprs...); kwargs...) where {RetAnnotation, TapeTy, FA<:Annotation{<:$(esc(fn))}, $(anns...)}
            oldshadow, pullback = tape

            shadow = dval.val

            tcomb = ntuple(Val(EnzymeRules.width(config))) do batch_i
                Base.@_inline_meta
                shad = EnzymeRules.width(config)  == 1 ? shadow : shadow[batch_i]
                res = pullback(shad)

                for (cr, en) in zip(res, (fn, $(vals...),))
                    if en isa Const || cr isa $ChainRulesCore.NoTangent
                        continue
                    end
                    if en isa Active
                        continue
                    end
                    if EnzymeRules.width(config)  == 1
                        en.dval .+= cr
                    else
                        en.dval[batch_i] .+= cr
                    end
                end

                ($(act_res...),)
            end

            return ($(invertcomb...),)
        end
    end
end

macro vendored_import_rrule(args...)
    return _vendored_import_rrule(args...)
end
