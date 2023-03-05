using LinearAlgebra, ForwardDiff
import ForwardDiff: jacobian
include("../deflationoperator.jl")
include("../nonlinearsolver.jl")

"""
Example using a semismooth Newton method & deflation


The following description is taken from Deflation for semismooth
equations by Farrell, Croci and Surowiec (2019) doi: 10.1080/10556788.2019.1613655 

This is a nonconvex quadratic programming problem with linear constraints 
suggestedby N. I. M. Gould in an invited lecture to the 19th biennial 
conference on NA. It is a quadratic minimization problem with an indefinite 
Hessian of the form min_x f(x) = −2(x1−1/4)^2 + 2(x2−1/2)^2
s.t. x1+x2<=1, 6*x1+2*x2<=3, x1,x2>=0.

The first order Karush–Kuhn–Tucker optimality conditions yield an NCP.
The nonconvexity of the function makes this problem difficult; 
it attains two minima with similar functional values and has a saddle point 
at x = [1/4,1/2]ᵀ. The central path to be followed by an interior point method 
is pathological, with different paths converging to the different minima.
"""

function F(xvec)
    # function handle for the KKT conditions
    x1 = xvec[1]; x2 = xvec[2]; l1 = xvec[3]; l2 = xvec[4]
    [-4*(x1-1/4)+3*l1+l2;4*(x2-1/2)+l1+l2;3-6*x1-2*x2;1-x1-x2]
end

J(x) = jacobian(x->F(x), x)

# Intial guess
x = [0.2;0.2;0;0]

# Lower and upper bounds
lb = [0.0;0.0;0.0;0.0]
ub = [1e100;1e100;1e100;1e100]

# Mass matrix, identity matrix since finite-dimensional problem
M = Diagonal(ones(length(x)))

# Use Benson and Munson SSLS solver to find the first solution
x₁ = ssls(x, lb, ub, F, J)

# Deflate found solution and find second
x₂ = ssls(x, lb, ub, F, J, [x₁])

# Repeat again to find the final solution
x₃ = ssls(x, lb, ub, F, J, [x₁, x₂])