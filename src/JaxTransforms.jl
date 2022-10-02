module JaxTransforms


# Dependencies
# ============

using ChainRulesCore
using ChainRules

include("CodeShifter.jl")
include("ForwardRules.jl")

import Base.Meta: isexpr
import Core: ReturnNode, SlotNumber, SSAValue
import .CodeShifter:
    @ns, FunTransform, TransformContext,
    is_primitive, process_inputs, transform!
import .ForwardRules: c_InactivePrimitives


# Type definitions
# ================

struct JVP <: TransformContext end


const IRTag = Union{SlotNumber, SSAValue}


# Methods
# =======

# TODO:
#
# - [ ] Leverage basic blocks info (iterate over blocks then over statements)
# - [ ] Leverage reachability analysis tags
# - [ ] Operate over a separate collection of transforms for reachable statements
# - [ ] Assemble CodeInfo


jvp(f) = FunTransform(f, JVP)

process_inputs(::Type{JVP}, primals::Type{<:Tuple}, tangents::Type{<:Tuple}) = (primals,)
process_inputs(::Type{JVP}, primals, tangents) = (primals,)

function is_primitive(::Type{JVP}, F, Args...)
    T = Core.Compiler.return_type(frule, Tuple{Any, F, Args...})
    return T !== Nothing && !(T <: Tuple{Any, NoTangent})
end

function transform!(::JVP, ci, meth, nargs, offset, sparams)
    code = Any[]
    ssa_mapping = Int[]

    function emit!(stmt)
        (isexpr(stmt, :call) || isexpr(stmt, :(=)) || isexpr(stmt, :new)) || return stmt
        push!(code, stmt)
        # push!(new_codelocs, isempty(new_codelocs) ? 0 : new_codelocs[end])
        SSAValue(length(code))
    end

    tags_set = transformable_ids(ci.code, nargs)

    for i in 2:nargs
        # ci.code[i] = Expr(:call, ci.code[i].args[1], SlotNumber(3), i - 1)
    end

    for (id, stmt) in enumerate(ci.code)
        if SSAValue(id) in tags_set
            push!(code, stmt)
        else
            push!(code, stmt)
        end
    end

    ci.code = code
    # insert!(ci.code, 2, Expr(:(=), SlotNumber(4), Expr(:call, getfield, SlotNumber(2), 2)))
    # insert!(ci.code, 2, Expr(:(=), SlotNumber(3), Expr(:call, getfield, SlotNumber(2), 1)))

    # insert!(ci.slotnames, 3, :tangents)
    # insert!(ci.slotnames, 3, :primals)

    # insert!(ci.slotflags, 3, 0x00)
    # insert!(ci.slotflags, 3, 0x00)

    return ci, nargs, offset + 1
end

function transformable_ids(code, nargs)
    tags_set = Set{IRTag}()
    for (id, stmt) in Iterators.reverse(pairs(code))
        collect_tags_rev!(tags_set, stmt, id)
    end
    intersect!(tags_set, (SlotNumber(i) for i = nargs+1:2nargs-1))
    for (id, stmt) in Iterators.drop(pairs(code), nargs)
        collect_tags_fwd!(tags_set, stmt, id)
    end
    return filter!(t -> t isa SSAValue, tags_set)
end

function collect_tags_rev!(ids_set, stmt, id)
    if isa(stmt, ReturnNode) && isa(stmt.val, IRTag)
        push!(ids_set, stmt.val)
    elseif isa(stmt, IRTag)
        push!(ids_set, stmt)
    elseif isexpr(stmt, :call) && !(stmt.args[1] in c_InactivePrimitives)
        ind = isa(stmt.args[1], IRTag) ? 1 : 2
        foreach(stmt -> collect_tags_rev!(ids_set, stmt, id), @view stmt.args[ind:end])
    elseif isexpr(stmt, :(=)) && stmt.args[1] in ids_set
        foreach(stmt -> collect_tags_rev!(ids_set, stmt, id), @view stmt.args[2:end])
    end
end

function collect_tags_fwd!(ids_set, stmt, id)
    if isa(stmt, IRTag) && stmt in ids_set
        push!(ids_set, SSAValue(id))
    elseif isexpr(stmt, :call) && !(stmt.args[1] in c_InactivePrimitives)
        ind = isa(stmt.args[1], IRTag) ? 1 : 2
        for stmt in @view stmt.args[ind:end]
            collect_tags_fwd!(ids_set, stmt, id)
            SSAValue(id) in ids_set && break
        end
    elseif isexpr(stmt, :(=))
        foreach(stmt -> collect_tags_fwd!(ids_set, stmt, id), @view stmt.args[2:end])
        SSAValue(id) in ids_set && push!(ids_set, stmt.args[1])
    end
end


end  # module JaxTransforms
