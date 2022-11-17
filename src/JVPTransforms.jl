module JVPTransforms


# Dependencies
# ============

using ChainRulesCore
using ChainRules

include("CodeShifter.jl")
include("JVPRules.jl")

import Base.Meta: isexpr
import Core: GotoIfNot, GotoNode, ReturnNode, SlotNumber, SSAValue
import Core.Compiler: return_type
import .CodeShifter:
    @ns, FunTransform, TransformContext,
    is_primitive, mkarg, next_slot, process_inputs, transform!, transform_stmt
import .JVPRules: kInactivePrimitives, kJVPRules

export jvp


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
    # T = Core.Compiler.return_type(frule, Tuple{Any, F, Args...})
    # return T !== Nothing && !(T <: Tuple{Any, NoTangent})
    # return Symbol(F.instance) in keys(kJVPRules)
    return JVPRules.jvprule(F, Args...) !== nothing
end

function transform!(::JVP, ci, code, sig, meth, nargs, sparams)
    empty!(code)
    primal_tangent_pairs = Dict{Int, Int}()
    tags_set = transformable_ids(ci.code, nargs)
    ssa_mapping = collect(1:nargs)

    push!(code, ci.code[1])
    push!(code, mkarg(getfield, 2, 2, 1))
    push!(code, mkarg(getfield, 2, 2, 2))
    for i in 2:nargs
        push!(code, mkarg(getfield, 3, 4, i - 1))
    end
    for i in 2:nargs
        push!(code, mkarg(getfield, 4, nargs + 3, i - 1))
        primal_tangent_pairs[i + 3] = nargs + i + 2
    end

    tags_map = Dict{IRTag, IRTag}(SlotNumber(i + 1) => SlotNumber(i + 3) for i in 2:nargs)
    tags_map[SSAValue(length(code))] = last(code).args[1]
    types_map = Dict{SlotNumber, Any}(SlotNumber(i) => ci.slottypes[i] for i in 3:length(ci.slottypes))
    for i in 3:length(ci.slottypes)
        types_map[SlotNumber(i + 2)] = ci.slottypes[i]
    end

    for i in (nargs + 1):length(ci.code)
        stmt = ci.code[i]
        new_stmt = transform_stmt(stmt, i, tags_map, sparams)
        if SSAValue(i) in tags_set
            append!(code,
                transform_active_stmt(
                    new_stmt, stmt, i, tags_map, types_map, primal_tangent_pairs, sparams;
                    isfirst = (i == (nargs + 1))
                )
            )
        else
            push!(code, new_stmt)
        end
        push!(ssa_mapping, length(code) - isa(stmt, ReturnNode))
    end

    for (id, stmt) in enumerate(code)
        if stmt isa GotoIfNot
            code[id] = GotoIfNot(stmt.cond, ssa_mapping[stmt.dest])
        elseif stmt isa GotoNode
            code[id] = GotoNode(ssa_mapping[stmt.label])
        end
    end

    max_id = maximum(sn.id for sn in values(tags_map) if sn isa SlotNumber)
    local_slots = 1:(max_id - 2nargs - 1)

    deleteat!(ci.slotnames, 3:lastindex(ci.slotnames))
    append!(ci.slotnames, (:primals, :tangents))
    append!(ci.slotnames, (Symbol(:p, i - 1) for i = 2:nargs))
    append!(ci.slotnames, (Symbol(:t, i - 1) for i = 2:nargs))
    append!(ci.slotnames, (Symbol(:v, i) for i = local_slots))

    deleteat!(ci.slotflags, 3:lastindex(ci.slotflags))
    append!(ci.slotflags, (0x08, 08))
    append!(ci.slotflags, (0x18 for i = 1:(2nargs - 2)))
    append!(ci.slotflags, (0x18 for i = local_slots))

    empty!(ci.codelocs)
    append!(ci.codelocs, (0 for i = 1:length(code)))

    deleteat!(ci.ssaflags, (length(code) + 1):lastindex(ci.ssaflags))
    append!(ci.ssaflags, (0x00 for i = (length(ci.ssaflags) + 1):length(code)))
    ci.ssavaluetypes = length(code)

    ci.code, code = code, ci.code
    return ci, code, nargs
end

function transform_active_stmt(stmt, prev_stmt, id, tags_map, types_map, pt_pairs, sparams; isfirst = false)
    code = Any[]
    skip = isa(stmt, Union{GotoIfNot, GotoNode})
    if isa(stmt, Union{GotoIfNot, GotoNode})
        push!(code, stmt)
    elseif isa(stmt, ReturnNode)
        tid = pt_pairs[stmt.val.id]
        sn = next_slot(tags_map)
        tup = Expr(:(=), sn, Expr(:call, :tuple, stmt.val, SlotNumber(tid)))
        push!(code, tup)
        push!(code, ReturnNode(sn))
        tags_map[SSAValue(id + 2)] = sn
    else
        f = stmt.args[2].args[1]
        sn = stmt.args[1]
        args = @views(stmt.args[2].args[2:end])
        nargs = length(args)
        sn_args = ((i, sn) for (i, sn) in enumerate(args) if isa(sn, SlotNumber))
        pid, tid = if nargs == 1
            pid = only(args).id
            pid, pt_pairs[pid]
        else
            primals_tuple = Expr(:(=), sn, Expr(:call, :tuple, args...))
            tangents_tuple = Expr(:(=), SlotNumber(sn.id + 1),
                Expr(:call, :tuple, (SlotNumber(pt_pairs[sn.id]) for (_, sn) in sn_args)...)
            )
            push!(code, primals_tuple)
            push!(code, tangents_tuple)
            sn.id .+ (0, 1)
        end
        args_types = (get(types_map, arg, typeof(arg)) for arg in args)
        offset = isfirst ? 0 : Int(nargs == 1)
        jvp_code, jvp_pid, jvp_tid = JVPRules.jvprule(typeof(f), args_types...)(pid, tid; o = offset)
        last_tuple = jvp_code[end - 1]
        if nargs > 1
            jvp_code[end - 1] = Expr(:(=), last_tuple.args[1],
                Expr(:call, :tuple, (last_tuple.args[2].args[i + 1] for (i, _) in sn_args)...)
            )
        end
        pt_pairs[jvp_pid] = jvp_tid
        prev_sn = prev_stmt.args[1]
        tags_map[prev_sn] = SlotNumber(jvp_pid)
        tags_map[last(jvp_code).args[1]] = last(jvp_code).args[1]
        types_map[tags_map[prev_sn]] = return_type(f, Tuple{args_types...})
        append!(code, jvp_code)
    end
    return code
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
    return tags_set
end

function collect_tags_rev!(ids_set, stmt, id)
    if isa(stmt, ReturnNode) && isa(stmt.val, IRTag)
        push!(ids_set, stmt.val)
    elseif isa(stmt, IRTag)
        push!(ids_set, stmt)
    elseif isexpr(stmt, :call) && !(stmt.args[1] in kInactivePrimitives)
        ind = isa(stmt.args[1], IRTag) ? 1 : 2
        foreach(stmt -> collect_tags_rev!(ids_set, stmt, id), @view stmt.args[ind:end])
    elseif isexpr(stmt, :(=)) && stmt.args[1] in ids_set
        foreach(stmt -> collect_tags_rev!(ids_set, stmt, id), @view stmt.args[2:end])
    end
end

function collect_tags_fwd!(ids_set, stmt, id)
    if isa(stmt, ReturnNode) && stmt.val in ids_set
        push!(ids_set, SSAValue(id))
    elseif isa(stmt, IRTag) && stmt in ids_set
        push!(ids_set, SSAValue(id))
    elseif isexpr(stmt, :call) && !(stmt.args[1] in kInactivePrimitives)
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


end  # module JVPTransforms
