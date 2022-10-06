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


const kCallOnlyPrimitives = Set([
    Base.flush,
    Base.print,
    Base.println,
    Base.show,
    Core.throw,
])


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

# The next three methods need to be implemented for user defined `TransformContext`s.
function process_inputs end
function is_primitive end
function transform! end

function signature(::Type{C}, @ns(T::Type{F}), @ns(Args...)) where {C, F}
    return _to_tuple(F, process_inputs(C, Args...)...)
end

function code_info(::Type{C}, sig, method_instance) where {C}
    ci = retrieve_code_info(method_instance)
    ci === nothing && throw("Cannot transform builtin function $(method_instance.def.name)")
    if is_primitive(C, sig.parameters...)
        return primitive_code_info!(ci)
    end
    empty!(ci.linetable)
    return ci
end

function primitive_code_info!(ci)
    meth = ci.parent.def
    nargs = meth.nargs
    f = getproperty(meth.module, meth.name)
    ci.code = [
        Expr(:call, f, (SlotNumber(i) for i = 2:nargs)...),
        ReturnNode(SSAValue(1))
    ]
    ci.codelocs = UInt32[0, 0]
    empty!(ci.linetable)
    deleteat!(ci.slotflags, (nargs + 1):length(ci.slotflags))
    deleteat!(ci.slotnames, (nargs + 1):length(ci.slotnames))
    ci.ssavaluetypes = 1
    return ci
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

process_inputs(::Type{BaseContext}, Args...) = Args
is_primitive(::Type{BaseContext}, Args...) = false

# ## Internals

function transform_generator(
    @ns(ft::Type{FunTransform{T, F}}), @ns(args); show_reference = false
) where {T, F}
    SigCtx = T === Tuple{} ? BaseContext : last(T.parameters)
    sig = signature(BaseContext, F, args...)
    methods = Base._methods_by_ftype(sig, -1, typemax(UInt))
    match = only(methods)

    mi = specialize_method(match)
    ci = code_info(SigCtx, sig, mi)

    show_reference && println(ci)

    if isdefined(ci, :edges)
        ci.edges = MethodInstance[mi]
    end

    code = Any[]
    nargs = length(sig.parameters)
    for ctx in prune_contexts(T === Tuple ? Core.svec() : T.parameters)
        ci, code, nargs = transform!(ctx, ci, code, mi.def, nargs, match.sparams)
    end

    return ci
end

function prune_contexts(list)
    revinds = (reverse ∘ eachindex)(list)
    return (BaseContext(), (list[i]() for i = revinds if list[i] !== BaseContext)...)
end

function transform!(::BaseContext, ci, code, meth, nargs, sparams)
    empty!(code)
    ssa_mapping = Int[]

    if meth.isva
        pushfirst!(code, mkarg(Base.rest, 2, meth.nargs, meth.nargs - 1))
        prepend!(code, (mkarg(getfield, 2, meth.nargs, i) for i = 1:meth.nargs-2))
        nargs = meth.nargs
    else
        prepend!(code, (mkarg(getfield, 2, nargs, i) for i = 1:nargs-1))
    end

    pushfirst!(code, Expr(:call, getproperty, SlotNumber(1), quoted(:f)))
    ci.codelocs = UInt32[0 for i = 1:nargs]
    prepend!(ci.ssaflags, (0x00 for i = 1:nargs))
    tags_map = Dict{IRTag, IRTag}(SlotNumber(i) => SlotNumber(i + 1) for i in 1:nargs)
    offset = nargs
    for (id, stmt) in enumerate(ci.code)
        stmt = transform_stmt(stmt, id, tags_map, sparams)
        push!(ssa_mapping, id + offset)
        if (stmt isa NewvarNode || stmt isa SlotNumber)
            offset -= 1
            continue
        end
        push!(code, stmt)
    end
    for (id, stmt) in enumerate(code)
        if stmt isa GotoIfNot
            code[id] = GotoIfNot(stmt.cond, ssa_mapping[stmt.dest])
        elseif stmt isa GotoNode
            code[id] = GotoNode(ssa_mapping[stmt.label])
        end
    end
    local_slots = maximum(sn.id for sn in values(tags_map) if sn isa SlotNumber) - nargs
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
    ci.ssavaluetypes = length(code)
    append!(ci.codelocs, (0 for i = 1:length(code)))
    ci.code, code = code, ci.code
    return ci, code, nargs
end

mkarg(getf, n, o, p) = Expr(:(=), SlotNumber(o + p), Expr(:call, getf, SlotNumber(n), p))

function transform_stmt(stmt, id, tags_map, sparams; inner = false)
    transform(stmt) = transform_stmt(stmt, id, tags_map, sparams; inner = true)
    if isexpr(stmt, :(=))
        key = stmt.args[1]
        args = Tuple(transform(s) for s in @view stmt.args[2:end])
        sn = push_tag!(tags_map, key)
        return Expr(:(=), sn, args...)
    elseif !inner && isexpr(stmt, :call)
        ex = Expr(:call, map(transform, stmt.args)...)
        if ex.args[1] in kCallOnlyPrimitives
            return ex
        end
        sn = push_tag!(tags_map, SSAValue(id))
        return Expr(:(=), sn, ex)
    elseif isexpr(stmt, :static_parameter)
        return quoted(sparams[stmt.args[1]])
    elseif isa(stmt, Expr)
        return Expr(stmt.head, map(transform, stmt.args)...)
    elseif isa(stmt, GlobalRef)
        return getproperty(stmt.mod, stmt.name)
    elseif isa(stmt, SlotNumber)
        if inner
            return tags_map[stmt]
        else
            return tags_map[SSAValue(id)] = tags_map[stmt]
        end
    elseif isa(stmt, SSAValue)
        return haskey(tags_map, stmt) ? tags_map[stmt] : push_tag!(tags_map, stmt)
    elseif isa(stmt, NewvarNode)
        return stmt
    elseif isa(stmt, GotoIfNot)
        return GotoIfNot(transform(stmt.cond), stmt.dest)
    elseif isa(stmt, GotoNode)
        return GotoNode(stmt.label)
    elseif isa(stmt, ReturnNode)
        return ReturnNode(transform(stmt.val))
    elseif isa(stmt, QuoteNode)
        return stmt.value
    else
        stmt
    end
end

function push_tag!(tags_map, key)
    sn = SlotNumber(maximum(sn.id for sn in values(tags_map) if sn isa SlotNumber) + 1)
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
