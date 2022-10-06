module ForwardRules


using ChainRules
using ChainRulesCore
using LinearAlgebra

include("CodeShifter.jl")

import Core: GlobalRef, Typeof
import .CodeShifter: kCallOnlyPrimitives


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
const kJVPRules = Dict{Function, Function}()


function partial_eval end
function tangents_from_pvals end
function primals_from_pvals end


tdot(::NoTangent, t_in) = NoTangent()
tdot(t, ::NoTangent) = NoTangent()
tdot(t, t_in) = sum(a ⋅ b for (a, b) in zip(t, t_in))
tdot(t::Tuple{NoTangent, Any}, t_in) = tdot(last(t), last(t_in))
function tdot(t::Tuple{NoTangent, Any, Any}, t_in)
    return tdot(Base.rest(t, 2), Base.rest(t_in, 2))
end


macro jvprule(call, tangent_call, setup...)
    f = esc(call.args[1])
    inputs = esc.(call.args[2:end])

    if setup !== () && Meta.isexpr(setup[1], :(=)) && setup[1].args[1] === :peval && setup[1].args[2]
        peval_body = Expr(:tuple, Expr(:call, f, inputs...), inputs...)
        unpacking_stmt = Expr(:(=), Expr(:tuple, esc(:Ω), inputs...), esc(:args))
        primals_result = esc(:Ω)
    else
        peval_body = length(inputs) == 1 ? only(inputs) : Expr(:tuple, inputs...)
        unpacking_stmt = Expr(:(=), peval_body, esc(:args))
        primals_result = Expr(:call, f, inputs...)
    end

    ins = call.args[2:end]
    primals_in = length(ins) == 1 ? only(ins) : :primals_in
    primal_peval = length(ins) == 1 ? () : Expr(:(=), Expr(:tuple, ins...), :primals_in)

    tmp_ex = quote
        function tmp_rule($primals_in, tangents_in)
            $primal_peval
            primals_out = $call
            ptangents = $tangent_call
            tangents_out = ptangents * tangents_in
            return primals_out, tangents_out
        end
    end

    return nothing
end


@jvprule one(x)   zero(x)
@jvprule zero(x)  zero(x)

@jvprule acosh(x)        inv(sqrt(x ^ 2 - 1))
@jvprule acoth(x)        inv(1 - x ^ 2)
@jvprule acsch(x)        -(inv(x ^ 2 * sqrt(1 + x ^ -2)))
@jvprule acsch(x::Real)  -(inv(abs(x) * sqrt(1 + x ^ 2)))
@jvprule asech(x)        -(inv(x * sqrt(1 - x ^ 2)))
@jvprule asinh(x)        inv(sqrt(x ^ 2 + 1))
@jvprule atanh(x)        inv(1 - x ^ 2)

@jvprule acosd(x)        -inv(deg2rad(sqrt(1 - x ^ 2)))
@jvprule acotd(x)        -inv(deg2rad(1 + x ^ 2))
@jvprule acscd(x)        -inv(deg2rad(x^2 * sqrt(1 - x ^ -2)))
@jvprule acscd(x::Real)  -inv(deg2rad(abs(x) * sqrt(x ^ 2 - 1)))
@jvprule asecd(x)        inv(deg2rad(x ^ 2 * sqrt(1 - x ^ -2)))
@jvprule asecd(x::Real)  inv(deg2rad(abs(x) * sqrt(x ^ 2 - 1)))
@jvprule asind(x)        inv(deg2rad(sqrt(1 - x ^ 2)))
@jvprule atand(x)        inv(deg2rad(1 + x ^ 2))

@jvprule cot(x)   -((1 + Ω ^ 2))         peval=true
@jvprule cotd(x)  -deg2rad(1 + Ω ^ 2)    peval=true
@jvprule coth(x)  -(csch(x) ^ 2)
@jvprule csc(x)   -Ω * cot(x)            peval=true
@jvprule cscd(x)  -deg2rad(Ω * cotd(x))  peval=true
@jvprule csch(x)  -(coth(x)) * Ω         peval=true
@jvprule sec(x)   Ω * tan(x)             peval=true
@jvprule secd(x)  deg2rad(Ω * tand(x))   peval=true
@jvprule sech(x)  -(tanh(x)) * Ω         peval=true

@jvprule acot(x)        -(inv(1 + x ^ 2))
@jvprule acsc(x)        -(inv(x ^ 2 * sqrt(1 - x ^ -2)))
@jvprule acsc(x::Real)  -(inv(abs(x) * sqrt(x ^ 2 - 1)))
@jvprule asec(x)        inv(x ^ 2 * sqrt(1 - x ^ -2))
@jvprule asec(x::Real)  inv(abs(x) * sqrt(x ^ 2 - 1))

@jvprule cosd(x)   -deg2rad(sind(x))
@jvprule cospi(x)  -π * sinpi(x)
@jvprule sind(x)   deg2rad(cosd(x))
@jvprule sinpi(x)  π * cospi(x)
@jvprule tand(x)   deg2rad(1 + Ω ^ 2)  peval=true

# @jvprule sin(x)  cos(x)
# @jvprule cos(x)  -sin(x)
# @jvprule tan(x)  1 + Ω ^ 2  peval=true

# Trig-Hyperbolic
@jvprule cosh(x)  sinh(x)
@jvprule sinh(x)  cosh(x)
@jvprule tanh(x)  1 - Ω ^ 2  peval=true

# Trig- Inverses
@jvprule acos(x)  -(inv(sqrt(1 - x ^ 2)))
@jvprule asin(x)  inv(sqrt(1 - x ^ 2))
@jvprule atan(x)  inv(1 + x ^ 2)

# Trig-Multivariate
# @jvprule atan(y, x) @setup(u = x ^ 2 + y ^ 2) (x / u, -y / u)
# @jvprule sincos(x) @setup((sinx, cosx) = Ω) cosx -sinx

# exponents
@jvprule cbrt(x)   inv(3 * Ω ^ 2)          peval=true
@jvprule inv(x)    -(Ω ^ 2)                peval=true
@jvprule sqrt(x)   inv(2Ω)                 peval=true
@jvprule exp(x)    Ω                       peval=true
@jvprule exp10(x)  Ω * log(oftype(x, 10))  peval=true
@jvprule exp2(x)   Ω * log(oftype(x, 2))   peval=true
@jvprule expm1(x)  exp(x)
@jvprule log(x)    inv(x)
@jvprule log10(x)  inv(x) / log(oftype(x, 10))
@jvprule log1p(x)  inv(x + 1)
@jvprule log2(x)   inv(x) / log(oftype(x, 2))


end  # module ForwardRules
