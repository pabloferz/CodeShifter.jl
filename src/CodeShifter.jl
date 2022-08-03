module CodeShifter


# Dependencies
# ============

import Base.Meta: isexpr
import Core:
    GlobalRef, GotoIfNot, GotoNode, NewvarNode, QuoteNode, ReturnNode, SlotNumber, SSAValue
import Core.Compiler:
    MethodInstance, compute_basic_blocks, quoted, retrieve_code_info, specialize_method

import OrderedCollections: OrderedSet
import Setfield: @set

const IRTag = Union{SSAValue, SlotNumber}
const var"@ns" = var"@nospecialize"


# Exports
# =======

export FunTransform, TransformContext, process_inputs, transform!


# Type definitions
# ================

abstract type TransformContext end
struct BaseContext <: TransformContext end

struct FunTransform{T <: Tuple, F} <: Function
    f::F
end

FunTransform(ft::FunTransform) = ft
FunTransform(f::F) where {F} = FunTransform{Tuple{}, F}(f)
FunTransform(f::F, ::Type{T}) where {F, T <: TransformContext} = FunTransform{Tuple{T}, F}(f)
function FunTransform(ft::FunTransform{S, F}, ::Type{T}) where {S, F, T <: TransformContext}
    return FunTransform{to_tuple(T, S), F}(ft.f)
end

struct StatementTransform
    id::Int
    code::Vector{Any}
end


# Methods
# =======

# ## Public interface

# The next two methods need to be implemented for user defined `TransformContext`s.

function transform! end

process_inputs(::Type{BaseContext}, Args...) = Args

function signature(::Type{C}, @ns(T::Type{F}), @ns(Args...)) where {C, F}
    return _to_tuple(F, process_inputs(C, Args...)...)
end

function code_info(::Type{C}, sig, method_instance) where {C, F}
    if is_primitive(C, sig.parameters...)
        return retrieve_code_info(method_instance)
    end
    return retrieve_code_info(method_instance)
end

"""
    transform_code_info(ft::FunTransform, args...; show_reference = false)

Similar to `Base.code_lowered`, but returns the `CodeInfo` after applying the code
transformation `ft`.  To compare the result with the original `CodeInfo` instance, set
`show_reference = true`.
"""
function transform_code_info(@ns(ft::FunTransform), @ns(args...); show_reference = false)
    return transform_generator(typeof(ft), args; show_reference)
end

# ## Internals

function transform_generator(
    @ns(ft::Type{FunTransform{T, F}}), @ns(args); show_reference = false
) where {T, F}
    sigctx = T === Tuple{} ? BaseContext : last(T.parameters)
    sig = signature(BaseContext, F, args...)
    methods = Base._methods_by_ftype(sig, -1, typemax(UInt))
    match = only(methods)

    mi = specialize_method(match)
    ci₀ = retrieve_code_info(mi)

    show_reference && println(ci₀)

    ci = copy(ci₀)
    if isdefined(ci, :edges)
        ci.edges = MethodInstance[mi]
    end

    offset = 1
    nargs = length(sig.parameters)
    for ctx in prune_contexts(T === Tuple ? Core.svec() : T.parameters)
        ci, nargs, offset = transform!(ctx, ci, mi.def, nargs, offset, match.sparams)
    end

    return ci
end

function prune_contexts(list)
    revinds = (reverse ∘ eachindex)(list)
    return (BaseContext(), (list[i]() for i = revinds if list[i] !== BaseContext)...)
end

function transform!(::BaseContext, ci, meth, nargs, offset, sparams)
    if meth.isva
        pushfirst!(ci.code, mkarg(Base.rest, 2, meth.nargs - 1))
        prepend!(ci.code, (mkarg(getfield, 2, i) for i = 1:meth.nargs-2))
        nargs = meth.nargs
    else
        prepend!(ci.code, (mkarg(getfield, 2, i) for i = 1:nargs-1))
    end
    pushfirst!(ci.code, mkarg(getproperty, 1, quoted(:f)))
    prepend!(ci.codelocs, (0 for i = 1:nargs))
    prepend!(ci.ssaflags, (0x00 for i = 1:nargs))
    ci.ssavaluetypes += nargs
    tags_map = Dict{IRTag, IRTag}(SlotNumber(i) => SSAValue(i) for i in 1:nargs)
    for id = (nargs+1:length(ci.code))
        ci.code[id] = transform_stmt(ci.code[id], tags_map, nargs, sparams)
    end
    local_slots = length(tags_map) - nargs
    ci.slotnames = Symbol[
        Symbol("#self#"), :args,
        (Symbol(:a, i) for i = 1:nargs-2)...,
        (Symbol(:v, i) for i = 1:local_slots)...,
    ]
    ci.slotflags = UInt8[
        0x00, 0x00,
        (0x00 for i = 1:nargs-2)...,
        (0x08 for i = 1:local_slots)...,
    ]
    ci.slottypes = Any[FunTransform, Any]
    return ci, nargs, offset
end

mkarg(getf, n, p) = Expr(:call, getf, SlotNumber(n), p)

function transform_stmt(stmt, tags_map, offset, sparams)
    transform(stmt) = transform_stmt(stmt, tags_map, offset, sparams)
    if isexpr(stmt, :(=))
        key = stmt.args[1]
        args = Tuple(transform(s) for s in @view stmt.args[2:end])
        sn = haskey(tags_map, key) ? tags_map[key] : push_tag!(tags_map, key)
        return Expr(:(=), sn, args...)
    elseif isexpr(stmt, :static_parameter)
        return quoted(sparams[stmt.args[1]])
    elseif isa(stmt, Expr)
        return Expr(stmt.head, map(transform, stmt.args)...)
    elseif isa(stmt, GlobalRef)
        return getproperty(stmt.mod, stmt.name)
    elseif isa(stmt, SlotNumber)
        return tags_map[stmt]
    elseif isa(stmt, SSAValue)
        return SSAValue(stmt.id + offset)
    elseif isa(stmt, NewvarNode)
        return NewvarNode(push_tag!(tags_map, stmt.slot))
    elseif isa(stmt, GotoIfNot)
        return GotoIfNot(transform(stmt.cond), stmt.dest + offset)
    elseif isa(stmt, GotoNode)
        return GotoNode(stmt.label + offset)
    elseif isa(stmt, ReturnNode)
        return ReturnNode(transform(stmt.val))
    elseif isa(stmt, QuoteNode)
        return stmt.value
    else
        return stmt
    end
end

function push_tag!(tags_map, key)
    sn = SlotNumber(length(tags_map) + 1)
    tags_map[key] = sn
    return sn
end

function __init__()
    # This is the transforms entry point
    @eval function (ft::FunTransform)(args...)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated,
            Expr(:new,
                Core.GeneratedFunctionStub,
                :transform_generator,
                Any[:ft, :args],
                Any[],
                @__LINE__,
                QuoteNode(Symbol(@__FILE__)),
                true
            )
        ))
    end
end


# Utils
# =====

_parameters(::Type{T}) where {T} = (T,)
_parameters(::Type{T}) where {T <: Tuple} = T.parameters
# _parameters(::Type{Type{T}}) where {T} = (T,)
# _parameters(::Type{Type{T}}) where {T <: Tuple} = T.parameters

@generated function to_tuple(Ts::Type...)
    T = _to_tuple(Ts...)
    return :($T)
end

function _to_tuple(@ns(Ts...))
    Types = Any[]
    foreach(T -> append!(Types, _parameters(T)), Ts)
    return Tuple{Types...}
end

function Base.show(io::IO, ct::FunTransform{T}) where {T <: Tuple}
    print(io, "FunTransform")
    if T !== Tuple{}
        print(io, "{")
        join(io, T.parameters, ", ")
        print(io, "}")
    end
    print(io, "(", ct.f, ")")
end

Base.show(io::IO, ::Type{T}) where {T <: TransformContext} = print(io, T.name.name)


end  # module CodeShifter
