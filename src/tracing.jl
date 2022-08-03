module CodeShifter


import Base.Meta: isexpr
import Core: GlobalRef, GotoIfNot, GotoNode, QuoteNode, ReturnNode, SlotNumber, SSAValue
import Core.Compiler: compute_basic_blocks


export trace


const BranchNode = Union{GotoIfNot, GotoNode, ReturnNode}
const IRTag = Union{SSAValue, SlotNumber}


struct Variable
    id::Int
end

struct Const{T}
  value::T
  Const(x) = new{Core.Typeof(x)}(x)
end

struct Inputs
    ids::Vector{Int}
    types::Vector{Any}
end

Inputs() = Inputs(Int[], [])

struct Statement
    expr::Any
    line::Int
end

struct Branch
    condition::Any
    block::Int
    args::Vector{Any}
end

struct Block
    args::Vector{Any}
    argtypes::Vector{Any}
    stmts::Vector{Statement}
    branches::Vector{Branch}
end

Block() = Block([], [], Statement[], Branch[])

struct BaseCtx end

struct Tape{C}
    inputs::Inputs
    blocks::Vector{Block}
    ids_map::Dict{IRTag, Any}
    ctx::C
end

Tape(; ctx = BaseCtx()) = Tape(Inputs(), Block[], Dict{IRTag, Any}(), ctx)


function trace(f, argtypes; ctx = BaseCtx())
    types = canonicalize(argtypes)
    ci = code_lowered(f, types)[1]

    tape = Tape(; ctx)

    nargs = length(types) + 1
    append!(tape.inputs.ids, 1:nargs)
    append!(tape.inputs.types, (Const(f),), types)
    foreach(i -> (tape.ids_map[SlotNumber(i)] = Variable(i)), 1:nargs)

    trace!(tape, f, types, ci.code)

    return tape
end

canonicalize(types::Tuple) = types
canonicalize(tuple::Type{<: Tuple}) = tuple.types

function trace!(
    tape, f, @nospecialize(argtypes), code; id = Ref(length(argtypes) + 1), stmts = Statement[]
)
    nargs = length(argtypes) + 1
    sparams = get_static_params(f, argtypes)
    basic_blocks = compute_basic_blocks(code).blocks
    sink_blocks = code[basic_blocks[1].stmts[end]] isa BranchNode ? BitSet(1) : BitSet()
    foreach(bb -> union!(sink_blocks, bb.succs), basic_blocks)

    for (i, block) in enumerate(basic_blocks)
        tags = Set{IRTag}(SlotNumber(j) for j = 1:nargs)
        trace_block!(tape, code, sparams, basic_blocks, block.stmts, id, tags, stmts)
        if !isempty(stmts) && (i in sink_blocks)
            branches = Branch[]
            push!(tape.blocks, Block([], [], stmts[:], branches))
            stmts = Statement[]
        end
    end

    return tape
end

function trace_block!(tape, code, sparams, basic_blocks, block_ids, id, tags, stmts)
    ids_map = tape.ids_map
    branched = length(tape.blocks) > 0

    for i in block_ids
        st = code[i]
        if isexpr(st, :call) || (isexpr(st, :(=)) && isexpr(st.args[2], :call))
            isassigment = isexpr(st, :(=))
            rhs = copy(isassigment ? st.args[2] : st)
            sv = resolve_vars!(ids_map, sparams, id, tags, branched, rhs.args)
            ex = Expr(:(=), sv, rhs)
            push!(stmts, Statement(ex, i))
            if isassigment
                ids_map[st.args[1]] = sv
                push!(tags, st.args[1])
            else
                ids_map[SSAValue(i)] = sv
                push!(tags, SSAValue(i))
            end
        elseif isexpr(st, :(=))
            sv, rhs = st.args[1], st.args[2:2]
            resolve_vars!(ids_map, sparams, Ref(0), tags, branched, rhs)
            ids_map[sv] = only(rhs)
            push!(tags, sv)
        elseif st isa SlotNumber
            ids_map[SSAValue(i)] = ids_map[st]
            push!(tags, SSAValue(i))
        elseif st isa GotoNode
            j = st.label
            k = j
            for b in basic_blocks
                isempty(searchsorted(b.stmts, j)) && continue
                k = last(b.stmts)
            end
            trace_block!(tape, code, sparams, basic_blocks, j:k, id, tags, stmts)
        elseif st isa ReturnNode
            val = length(block_ids) == 1 ? Variable(id[] += 1) : ids_map[st.val]
            push!(stmts, Statement(ReturnNode(val), i))
        end
    end
end

function resolve_vars!(ids_map, sparams, id, tags, branched, sv_fargs)
    for (i, sv) in enumerate(sv_fargs)
        if sv isa SlotNumber || sv isa SSAValue
            if id[] != 0 && branched && sv ∉ tags
                id[] += 1
                ids_map[sv] = Variable(id[])
            end
            sv_fargs[i] = ids_map[sv]
        elseif isexpr(sv, :static_parameter)
            sv_fargs[i] = sparams[sv.args[1]]
        else
            sv_fargs[i] = promote_const_value(sv)
        end
    end
    return Variable(id[] += 1)
end

function get_static_params(f, argtypes)
    mi = Base.method_instances(f, argtypes)[1]
    return mi.sparam_vals
end

promote_const_value(x::QuoteNode) = x.value
promote_const_value(x::GlobalRef) = getproperty(x.mod, x.name)
promote_const_value(x) = x


Base.show(io::IO, v::Variable) = print(io, SSAValue(v.id))
Base.show(io::IO, c::Const) = print(io, "Const(", c.value, ")")

function Base.show(io::IO, stmt::Statement)
    print(io, stmt.expr)
end

function Base.show(io::IO, ::MIME"text/plain", tape::Tape)
    print(io, "Arguments: (")
    ins = [string(SSAValue(v), " :: ", T) for (v, T) in zip(tape.inputs.ids, tape.inputs.types)]
    print(io, join(ins, ", "), ")\n")
    for (i, block) in enumerate(tape.blocks)
        print(io, i, ":\n")
        for (j, st) in enumerate(block.stmts)
            print(io, "  ", st, j == length(block.stmts) ? "" : "\n")
        end
        print(io, i == length(tape.blocks) ? "" : "\n")
    end
end


end  # module Recorder
