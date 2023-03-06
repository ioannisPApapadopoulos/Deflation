using LinearAlgebra, ForwardDiff
using Plots
import ForwardDiff: jacobian, gradient

"""
Self-contained script for deflation!

Finding two solutions of the Bratu equation
    u''(x) + λeᵘ(x) = 0
    u(0) = u(1) = 0.
This problem has one solution if λ = 0 or λ = λ*, 
two solutions if 0 < λ < λ*, and no solutions if λ > λ* where λ* ≈ 3.513830719. 
"""

# Setup of finite difference stencil
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
function J(u)
    A = jacobian(u->F(u), u)
    A[:,1] .= 0; A[1,:] .= 0;
    A[1,1] = 1; A[end,end] = 1;
    return A
end

# Initial guess
u₀ = (1 .- x) .* x

# Run first Newton loop to find first solution
u₁ = u₀
for _=1:100
    u₁ = u₁ - J(u₁) \ F(u₁)
end 
norm(F(u₁))


# Using ForwarDiff.jl to implement the deflation of the first found solution in 2 lines
μ = u -> 1/dot(u-u₁, u-u₁) + 1
τ = (u, du) -> 1 + (1/μ(u) * dot(gradient(μ, u), du)) ./ (1 .- 1/μ(u) * dot(gradient(μ, u), du))

# Run second Newton loop from the SAME initial guess!
u₂ = u₀
for _=1:100
    # Solve origin undeflated Newton system
    du = -J(u₂)\ F(u₂)
    # Multiply Newton update with the SCALAR τ to 
    # find Newton step of deflated Newton system.
    u₂ = u₂ + τ(u₂, du)*du
end
norm(F(u₂))

# Plot the two different solutions
plot(x, [u₁ u₂])