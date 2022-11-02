module ForwardRules


using ChainRules
using ChainRulesCore
using LinearAlgebra

include("CodeShifter.jl")

import Base.Meta: isexpr
import Core:
    GlobalRef, GotoIfNot, GotoNode, NewvarNode, QuoteNode, ReturnNode,
    SlotNumber, SSAValue, Typeof
import .CodeShifter: @ns, kCallOnlyPrimitives, IRTag


export kInactivePrimitives
export partial_eval, primals_from_pvals, tangents_from_pvals, tdot


const kInactivePrimitives = union!(
    Set([
        Base.CoreLogging.logmsg_code,
        Base.CoreLogging.shouldlog,
        Base.Threads.threadid,
        Base.Threads.nthreads,
        Base.eps,
        Base.methods,
        Base.nextfloat,
        Base.prevfloat,
        Base.print_to_string,
        Base.string,
        Base.to_tuple_type,
        Core.kwfunc,
    ]),
    kCallOnlyPrimitives
)

# This is not the right way store the rules.
# In principle, we should store the rules per method and not per function.
const kJVPRules = Dict{Symbol, Any}()


tdot(::NoTangent, t_in) = NoTangent()
tdot(t, ::NoTangent) = NoTangent()
tdot(t, t_in) = sum(a ⋅ b for (a, b) in zip(t, t_in))
tdot(t::Tuple{NoTangent, Any}, t_in) = tdot(last(t), last(t_in))
function tdot(t::Tuple{NoTangent, Any, Any}, t_in)
    return tdot(Base.rest(t, 2), Base.rest(t_in, 2))
end


getfield_expr(t, i) = Expr(:call, getfield, t, i)

macro jvprule(primal_call, tangent_call, setup...)
    f = primal_call.args[1]
    inputs = @view primal_call.args[2:end]
    nargs = length(inputs)

    primals_in = nargs == 1 ? only(inputs) : :primals_in
    unpack_primals = nargs == 1 ?  () :
        Tuple(Expr(:(=), arg, getfield_expr(:primals_in, i)) for (i, arg) in enumerate(inputs))

    replace_primal = (setup !== ()
        && Meta.isexpr(setup[1], :(=))
        && setup[1].args[1] === :replace_primal
        && setup[1].args[2]
    )

    if replace_primal
        primal_call = ()
    else
        primal_call = Expr(:(=), :Ω, primal_call)
        tangent_call = Expr(:(=), :Ω̇, tangent_call)
    end

    ex = quote
        function rule($primals_in, Δx)
            $(unpack_primals...)
            $primal_call
            $tangent_call
            ΔΩ = tdot(Ω̇, Δx)
            return Ω, ΔΩ
        end
    end

    rule = eval(ex)
    ci = only(code_lowered(rule, (Nothing, Nothing)))

    rn = Ref(1)
    rule_code = Any[]
    tags_map = Dict{IRTag, IRTag}()

    for (id, stmt) in enumerate(@view ci.code[1:end-2])
        new_stmt = transform_rule_stmt(stmt, id, rn, tags_map)
        isa(new_stmt, SlotNumber) && continue
        push!(rule_code, new_stmt)
    end

    kJVPRules[f] = rule_code

    return :nothing
end


function transform_rule_stmt(@ns(stmt), id, rn, tags_map; inner = false)
    transform(stmt) = transform_rule_stmt(stmt, id, rn, tags_map; inner = true)
    if isexpr(stmt, :(=))
        key = stmt.args[1]
        if length(stmt.args) == 2 && isa(stmt.args[2], IRTag)
            return tags_map[key] = tags_map[stmt.args[2]]
        end
        args = Tuple(transform(s) for s in @view stmt.args[2:end])
        sn = tags_map[key] = tags_map[SSAValue(id)] = SlotNumber(rn[] + 3)
        rn[] += 1
        return Expr(:(=), sn, args...)
    elseif !inner && isexpr(stmt, :call)
        ex = Expr(:call, map(transform, stmt.args)...)
        if ex.args[1] in kCallOnlyPrimitives
            return ex
        end
        sn = tags_map[SSAValue(id)] = SlotNumber(rn[] + 3)
        rn[] += 1
        return Expr(:(=), sn, ex)
    elseif isa(stmt, Expr)
        return Expr(stmt.head, map(transform, stmt.args)...)
    elseif isa(stmt, GlobalRef)
        return getproperty(stmt.mod, stmt.name)
    elseif isa(stmt, SlotNumber)
        stmt.id in (2, 3) && return stmt
        if inner
            return tags_map[stmt]
        else
            return tags_map[SSAValue(id)] = tags_map[stmt]
        end
    elseif isa(stmt, SSAValue)
        return tags_map[stmt]
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


@jvprule  one(x)   zero(x)
@jvprule  zero(x)  (Ω = Ω̇ = zero(x))  replace_primal=true

@jvprule  acosh(x)        inv(sqrt(x ^ 2 - 1))
@jvprule  acoth(x)        inv(1 - x ^ 2)
@jvprule  acsch(x)        -(inv(x ^ 2 * sqrt(1 + x ^ -2)))
@jvprule  acsch(x::Real)  -(inv(abs(x) * sqrt(1 + x ^ 2)))
@jvprule  asech(x)        -(inv(x * sqrt(1 - x ^ 2)))
@jvprule  asinh(x)        inv(sqrt(x ^ 2 + 1))
@jvprule  atanh(x)        inv(1 - x ^ 2)

@jvprule  acosd(x)        -inv(deg2rad(sqrt(1 - x ^ 2)))
@jvprule  acotd(x)        -inv(deg2rad(1 + x ^ 2))
@jvprule  acscd(x)        -inv(deg2rad(x^2 * sqrt(1 - x ^ -2)))
@jvprule  acscd(x::Real)  -inv(deg2rad(abs(x) * sqrt(x ^ 2 - 1)))
@jvprule  asecd(x)        inv(deg2rad(x ^ 2 * sqrt(1 - x ^ -2)))
@jvprule  asecd(x::Real)  inv(deg2rad(abs(x) * sqrt(x ^ 2 - 1)))
@jvprule  asind(x)        inv(deg2rad(sqrt(1 - x ^ 2)))
@jvprule  atand(x)        inv(deg2rad(1 + x ^ 2))

@jvprule  cot(x)   -((1 + Ω ^ 2))
@jvprule  cotd(x)  -deg2rad(1 + Ω ^ 2)
@jvprule  coth(x)  -(csch(x) ^ 2)
@jvprule  csc(x)   -Ω * cot(x)
@jvprule  cscd(x)  -deg2rad(Ω * cotd(x))
@jvprule  csch(x)  -(coth(x)) * Ω
@jvprule  sec(x)   Ω * tan(x)
@jvprule  secd(x)  deg2rad(Ω * tand(x))
@jvprule  sech(x)  -(tanh(x)) * Ω

@jvprule  acot(x)        -(inv(1 + x ^ 2))
@jvprule  acsc(x)        -(inv(x ^ 2 * sqrt(1 - x ^ -2)))
@jvprule  acsc(x::Real)  -(inv(abs(x) * sqrt(x ^ 2 - 1)))
@jvprule  asec(x)        inv(x ^ 2 * sqrt(1 - x ^ -2))
@jvprule  asec(x::Real)  inv(abs(x) * sqrt(x ^ 2 - 1))

@jvprule  cosd(x)   -deg2rad(sind(x))
@jvprule  cospi(x)  -π * sinpi(x)
@jvprule  sind(x)   deg2rad(cosd(x))
@jvprule  sinpi(x)  π * cospi(x)
@jvprule  tand(x)   deg2rad(1 + Ω ^ 2)

@jvprule  sin(x)  (sc = sincos(x); Ω = sc[1]; Ω̇ = sc[2])   replace_primal=true
@jvprule  cos(x)  (sc = sincos(x); Ω = sc[2]; Ω̇ = -sc[1])  replace_primal=true
@jvprule  tan(x)  1 + Ω ^ 2

# Trig-Hyperbolic
@jvprule  cosh(x)  sinh(x)
@jvprule  sinh(x)  cosh(x)
@jvprule  tanh(x)  1 - Ω ^ 2

# Trig- Inverses
@jvprule  acos(x)  -(inv(sqrt(1 - x ^ 2)))
@jvprule  asin(x)  inv(sqrt(1 - x ^ 2))
@jvprule  atan(x)  inv(1 + x ^ 2)

# Trig-Multivariate
@jvprule  atan(y, x)  (u = x ^ 2 + y ^ 2; (x / u, -y / u))
@jvprule  sincos(x)   (Ω[2], -Ω[1])

# exponents
@jvprule  cbrt(x)   inv(3 * Ω ^ 2)
@jvprule  inv(x)    -(Ω ^ 2)
@jvprule  sqrt(x)   inv(2Ω)
@jvprule  exp(x)    Ω
@jvprule  exp10(x)  Ω * log(oftype(x, 10))
@jvprule  exp2(x)   Ω * log(oftype(x, 2))
@jvprule  expm1(x)  Ω + 1
@jvprule  log(x)    inv(x)
@jvprule  log10(x)  inv(x) / log(oftype(x, 10))
@jvprule  log1p(x)  inv(x + 1)
@jvprule  log2(x)   inv(x) / log(oftype(x, 2))


end  # module ForwardRules
