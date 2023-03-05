using LinearAlgebra, ForwardDiff, Plots
import ForwardDiff: jacobian

include("../../src/Julia/deflationoperator.jl")
include("../../src/Julia/nonlinearsolver.jl")

"""
Finding two solutions of the Bratu equation
    u''(x) + λeᵘ(x) = 0
    u(0) = u(1) = 0.
This problem has one solution if λ = 0 or λ = λ*, 
two solutions if 0 < λ < λ*, and no solutions if λ > λ* where λ* ≈ 3.513830719. 
"""

n = 100 # number of intervals
h = 1/n # mesh size
Δ = SymTridiagonal(fill(-2/h^2, n), fill(1/h^2,n-1)) # finite difference stencil
x = range(0,1;length=n)

λ = 2.5 # Bratu parameter

# residual + bcs
function F(u)
    r = Δ*u + λ*exp.(u)
    r[1] = 0; r[end] = 0;
    return r
end

# Jacobian + bcs
function J(x)
    A = jacobian(u->F(u), u)
    A[:,1] .= 0; A[1,:] .= 0;
    A[1,1] = 1; A[end,end] = 1;
    return A
end

# Initial guess
u₀ = (1 .- x) .* x

u₁ = newtonls(u₀, F, J)         # first solution
u₂ = newtonls(u₀, F, J, [u₁])   # deflate and find second solution

# Plot solutions
plot(x, u₁)
plot!(x, u₂)