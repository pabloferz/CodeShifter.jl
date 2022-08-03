Jax = include("src/JaxTransforms.jl")
CS = Jax.CodeShifter

function f(x)
    y = 2 * sin(x)
    z = -y + x
    return z
end

CS.transform_code_info(CS.FunTransform(f), Float64)
CS.FunTransform(f)(3.0), f(3.0)
