using LinearAlgebra, ForwardDiff
import ForwardDiff: jacobian
include("../deflationoperator.jl")
include("../nonlinearsolver.jl")

"""
Example using Newton's method & deflation


This is a nonlinear system representing the intersection between a straight
line and a parabola. It consists of two unknowns and has the solutions 
(1,2) and (0,1). 

x - y = -1
y = x^2 + 1
"""

# Residual of nonlinear system
function F(xvec)
    x = xvec[1]
    y = xvec[2]
    [x-y+1.0; y-x^2-1.0]
end

J(x) = jacobian(x->F(x), x)

# Intialise starting value
x = [3.0; 3.0]

# Use Newton's method to find the first solution
x₁ = newtonls(x, F, J)

# Update found solution list in deflation operator and solve again from
# the same initial guess and using the same solver to find a new solution
x₂ = newtonls(x, F, J, [x₁])