module JaxTransforms


# Dependencies
# ============

using ChainRulesCore
using ChainRules

include("CodeShifter.jl")
include("ForwardRules.jl")

import Base.Meta: isexpr
import Core: ReturnNode, SlotNumber, SSAValue
import .CodeShifter: @ns, FunTransform, TransformContext, process_inputs, transform!
import .ForwardRules: c_InactivePrimitives

const Itrs = Iterators


# Type definitions
# ================

struct JVP <: TransformContext end


const IRTag = Union{SlotNumber, SSAValue}


# Methods
# =======

# TODO:
#
# - [ ] CodeShifter: Define a fallback `is_primitive`
# - [ ] CodeShifter: Wrap `retrieve_code_info` to take primitives into account
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
    new_code = Any[]
    new_codelocs = Any[]
    ssa_mapping = Int[]
    loc_mapping = Int[]

    function emit!(stmt)
        (isexpr(stmt, :call) || isexpr(stmt, :(=)) || isexpr(stmt, :new)) || return stmt
        push!(new_code, stmt)
        push!(new_codelocs, isempty(new_codelocs) ? 0 : new_codelocs[end])
        SSAValue(length(new_code))
    end

    tags_set = transformable_ids(ci.code, nargs)

    for i in 2:nargs
        ci.code[i] = Expr(:call, ci.code[i].args[1], SlotNumber(3), i - 1)
    end

    insert!(ci.code, 2, Expr(:(=), SlotNumber(4), Expr(:call, getfield, SlotNumber(2), 2)))
    insert!(ci.code, 2, Expr(:(=), SlotNumber(3), Expr(:call, getfield, SlotNumber(2), 1)))

    insert!(ci.slotnames, 3, :tangents)
    insert!(ci.slotnames, 3, :primals)

    insert!(ci.slotflags, 3, 0x00)
    insert!(ci.slotflags, 3, 0x00)

    # for (i, stmt) in enumerate(new_code)
    #     if isa(stmt, GotoNode)
    #         new_code[i] = GotoNode(loc_mapping[stmt.label])
    #     elseif isa(stmt, GotoIfNot)
    #         new_code[i] = GotoIfNot(stmt.cond, loc_mapping[stmt.dest])
    #     end
    # end

    return ci, nargs, offset + 1
end

function transformable_ids(code, nargs)
    tags_set = Set{IRTag}()
    for (id, stmt) in Itrs.reverse(pairs(code))
        collect_tags_bwd!(tags_set, stmt, id)
    end
    intersect!(tags_set, (SSAValue(i) for i = 1:nargs))
    for (id, stmt) in Itrs.drop(pairs(code), nargs)
        collect_tags_fwd!(tags_set, stmt, id)
    end
    return tags_set
end

function collect_tags_bwd!(ids_set, stmt, id)
    if isa(stmt, ReturnNode) && isa(stmt.val, IRTag)
        push!(ids_set, stmt.val)
    elseif isa(stmt, IRTag) && SSAValue(id) in ids_set
        push!(ids_set, stmt)
    elseif isexpr(stmt, :call) && !(stmt.args[1] in c_InactivePrimitives) && SSAValue(id) in ids_set
        ind = isa(stmt.args[1], IRTag) ? 1 : 2
        foreach(stmt -> collect_tags_bwd!(ids_set, stmt, id), @view stmt.args[ind:end])
    elseif isexpr(stmt, :(=)) && stmt.args[1] in ids_set
        push!(ids_set, SSAValue(id))
        foreach(stmt -> collect_tags_bwd!(ids_set, stmt, id), @view stmt.args[2:end])
    end
end

function collect_tags_fwd!(ids_set, stmt, id)
    if isa(stmt, ReturnNode) && stmt.val in ids_set
        push!(ids_set, SSAValue(id))
    elseif isa(stmt, IRTag) && stmt in ids_set
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
