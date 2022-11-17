using Test
using CodeShifter
using CodeShifter.JVPTransforms

# Examples taken from
# https://jax.readthedocs.io/en/latest/autodidax.html

deriv(f::F) where {F} = x -> last(jvp(f)((x,), (one(x),)))

function f(x)
    y = 2 * sin(x)
    z = -y + x
    return z
end

g(x) = x > 0 ? 2x : x

code_shifted(f)(3.0)

@testset "BaseContext" begin
    @test FunTransform(f)(3.0) == f(3.0)
end

@testset "JVPs" begin
    @test jvp(f)((3.0,), (1.0,)) == (2.7177599838802657, 2.979984993200891)
    @test jvp(sin)((3.0,), (1.0,)) == sincos(3.0)

    @test deriv(sin)(3.0) == cos(3.0)
    @test deriv(g)(3.0) == 2.0
    @test deriv(g)(-3.0) == 1.0
end
