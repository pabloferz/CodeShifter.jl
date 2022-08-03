module JaxTransforms


# Dependencies
# ============

using ChainRulesCore
using ChainRules
using Umlaut

# import ChainRules: frule
import Umlaut: isprimitive, record_primitive!


# Type definitions
# ================

struct JVPCtx end
struct ChainRulesCtx end

struct Dual{P, T}
    primal::P
    tangent::T
end

Dual(t::Tuple{Any, Any}) = Dual(t...)
Dual(primal) = Dual(primal, tangent(primal))

# Constanst and aliases
# =====================

const BASE_CTX = Umlaut.BaseCtx()
const CR_CTX = ChainRulesCtx()


# Methods
# =======

# ## Umlaut interface

isprimitive(::JVPCtx, f, args...) = isprimitive(BASE_CTX, f, args...)

function isprimitive(::ChainRulesCtx, f, args...)
    F = Core.Typeof(f)
    Args = Core.Typeof.(primal.(args))
    T = Core.Compiler.return_type(frule, Tuple{Any, F, Args...})
    return T !== Nothing && !(T <: Tuple{Any, NoTangent})
end

function record_primitive!(tape::Tape{JVPCtx}, v_fargs...)
    f, args... = [v isa Variable ? tape[v].val : v for v in v_fargs]
    v_primal_args = []
    v_tangent_args = []
    for arg in v_fargs
        if arg isa Variable
            v_primal = push!(tape, mkcall(primal, arg))
            v_tangent = push!(tape, mkcall(tangent, arg))
            push!(v_primal_args, v_primal)
            push!(v_tangent_args, v_tangent)
        else
            v_tangent = push!(tape, Constant(ir_tangent(arg)))
            push!(v_primal_args, arg)
            push!(v_tangent_args, v_tangent)
        end
    end
    if isprimitive(CR_CTX, f, args...)
        v_input_primals = push!(tape, mkcall(tuple, v_primal_args...))
        v_input_tangents = push!(tape, mkcall(tuple, v_tangent_args...))
        v_input_tangents_tuple = push!(tape, mkcall(tuple, v_input_tangents))
        v_inputs = push!(tape, mkcall(tuple, v_input_tangents_tuple, v_input_primals))
        v_fr = push!(tape, mkcall(Core._apply_iterate, iterate, frule, v_input_tangents_tuple, v_input_primals))
        v_outs = push!(tape, mkcall(Dual, v_fr))
        return v_outs
    else
        return push!(tape, mkcall(v_primal_args...))
    end
end

v_getfield(obj, fld) = getfield(obj, fld)

# ## JVPs transformations

function trace_jvp(f, primals, tangents)
    ctx = JVPCtx()
    if isprimitive(ctx, f, primals...)
        t = Umlaut.Tracer(Tape(ctx))
        meth = which(f, map(typeof, primals))
        args = meth.isva ? (primals[1:meth.nargs - 2]..., primals[meth.nargs - 1:end]) : primals
        t.tape.meta[:isva] = meth.isva
        v_fargs = inputs!(t.tape, f, args...)
        rv = record_primitive!(t.tape, v_fargs...)
        t.tape.result = rv
        return t.tape[t.tape.result].val, t.tape
    end
    trace(f, primals...; ctx = JVPCtx())
end

primal(x) = x
primal(x::Dual) = x.primal

tangent(x) = nothing
tangent(x::Number) = one(x)
tangent(x::Dual) = x.tangent

ir_tangent(x) = nothing
ir_tangent(x::Number) = zero(x)


end  # module JaxTransforms
