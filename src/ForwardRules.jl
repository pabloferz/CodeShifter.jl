module ForwardRules


using ChainRules
using ChainRulesCore
using LinearAlgebra

import Core: GlobalRef, Typeof


export c_InactivePrimitives
export partial_eval, primals_from_pvals, tangents_from_pvals, tdot


const c_InactivePrimitives = Set([
    Base.CoreLogging.logmsg_code,
    Base.CoreLogging.shouldlog,
    Base.to_tuple_type,
    Base.methods,
    Base.println,
    Base.print,
    Base.show,
    Base.flush,
    Base.string,
    Base.print_to_string,
    Base.Threads.threadid,
    Base.Threads.nthreads,
    Base.eps,
    Base.nextfloat,
    Base.prevfloat,
    Core.kwfunc,
])

const c_Primitives = Set{Function}()


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


macro frule(call, tangent_call, setup...)
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

    peval_expr = quote
        ForwardRules.partial_eval(::Typeof($f), $(inputs...)) = $peval_body
    end

    tangents_expr = quote
        function ForwardRules.tangents_from_pvals(::Typeof($f), $(esc(:args)))
            $unpacking_stmt
            return (NoTangent(), ($(esc(tangent_call))))
        end
    end

    primals_expr = quote
        function ForwardRules.primals_from_pvals(::Typeof($f), $(esc(:args)))
            $unpacking_stmt
            return $primals_result
        end
    end

    return quote
        $(peval_expr)
        $(tangents_expr)
        $(primals_expr)
    end
end


@frule one(x)   zero(x)
@frule zero(x)  zero(x)

@frule acosh(x)        inv(sqrt(x - 1) * sqrt(x + 1))
@frule acoth(x)        inv(1 - x ^ 2)
@frule acsch(x)        -(inv(x ^ 2 * sqrt(1 + x ^ -2)))
@frule acsch(x::Real)  -(inv(abs(x) * sqrt(1 + x ^ 2)))
@frule asech(x)        -(inv(x * sqrt(1 - x ^ 2)))
@frule asinh(x)        inv(sqrt(x ^ 2 + 1))
@frule atanh(x)        inv(1 - x ^ 2)

@frule acosd(x)        -inv(deg2rad(sqrt(1 - x ^ 2)))
@frule acotd(x)        -inv(deg2rad(1 + x ^ 2))
@frule acscd(x)        -inv(deg2rad(x^2 * sqrt(1 - x ^ -2)))
@frule acscd(x::Real)  -inv(deg2rad(abs(x) * sqrt(x ^ 2 - 1)))
@frule asecd(x)        inv(deg2rad(x ^ 2 * sqrt(1 - x ^ -2)))
@frule asecd(x::Real)  inv(deg2rad(abs(x) * sqrt(x ^ 2 - 1)))
@frule asind(x)        inv(deg2rad(sqrt(1 - x ^ 2)))
@frule atand(x)        inv(deg2rad(1 + x ^ 2))

@frule cot(x)   -((1 + Ω ^ 2))         peval=true
@frule cotd(x)  -deg2rad(1 + Ω ^ 2)    peval=true
@frule coth(x)  -(csch(x) ^ 2)
@frule csc(x)   -Ω * cot(x)            peval=true
@frule cscd(x)  -deg2rad(Ω * cotd(x))  peval=true
@frule csch(x)  -(coth(x)) * Ω         peval=true
@frule sec(x)   Ω * tan(x)             peval=true
@frule secd(x)  deg2rad(Ω * tand(x))   peval=true
@frule sech(x)  -(tanh(x)) * Ω         peval=true

@frule acot(x)        -(inv(1 + x ^ 2))
@frule acsc(x)        -(inv(x ^ 2 * sqrt(1 - x ^ -2)))
@frule acsc(x::Real)  -(inv(abs(x) * sqrt(x ^ 2 - 1)))
@frule asec(x)        inv(x ^ 2 * sqrt(1 - x ^ -2))
@frule asec(x::Real)  inv(abs(x) * sqrt(x ^ 2 - 1))

@frule cosd(x)   -deg2rad(sind(x))
@frule cospi(x)  -π * sinpi(x)
@frule sind(x)   deg2rad(cosd(x))
@frule sinpi(x)  π * cospi(x)
@frule tand(x)   deg2rad(1 + Ω ^ 2)  peval=true


@frule sin(x)  cos(x)
@frule cos(x)  -sin(x)
@frule tan(x)  1 + Ω ^ 2  peval=true

# Trig-Hyperbolic
@frule cosh(x)  sinh(x)
@frule sinh(x)  cosh(x)
@frule tanh(x)  1 - Ω ^ 2  peval=true

# Trig- Inverses
@frule acos(x)  -(inv(sqrt(1 - x ^ 2)))
@frule asin(x)  inv(sqrt(1 - x ^ 2))
@frule atan(x)  inv(1 + x ^ 2)

# Trig-Multivariate
# @frule atan(y, x) @setup(u = x ^ 2 + y ^ 2) (x / u, -y / u)
# @frule sincos(x) @setup((sinx, cosx) = Ω) cosx -sinx

# exponents
@frule cbrt(x)   inv(3 * Ω ^ 2)          peval=true
@frule inv(x)    -(Ω ^ 2)                peval=true
@frule sqrt(x)   inv(2Ω)                 peval=true
@frule exp(x)    Ω                       peval=true
@frule exp10(x)  Ω * log(oftype(x, 10))  peval=true
@frule exp2(x)   Ω * log(oftype(x, 2))   peval=true
@frule expm1(x)  exp(x)
@frule log(x)    inv(x)
@frule log10(x)  inv(x) / log(oftype(x, 10))
@frule log1p(x)  inv(x + 1)
@frule log2(x)   inv(x) / log(oftype(x, 2))


end  # module ForwardRules
