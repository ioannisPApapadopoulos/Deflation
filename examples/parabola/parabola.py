import numpy as np
import sys
sys.path.insert(0, "../../")
from src.Python.deflationoperator import DeflationOperator

"""
Example using Newton's method & deflation


This is a nonlinear system representing the intersection between a straight
line and a parabola. It consists of two unknowns and has the solutions 
(1,2) and (0,1). 

x - y = -1
y = x^2 + 1
"""

# Residual of nonlinear system
def F(xvec):
    x = xvec[0]
    y = xvec[1]
    return np.array([x-y+1.0, y-x**2-1.0])

# Jacobian of nonlinear system
def jacobian(xvec):
    x = xvec[0]
    return np.matrix([[1.0, -1.0], [-2*x, 1.0]])

# Intialise starting value
x = np.array([3.0, 3.0])

D = DeflationOperator([], np.eye(2), 2, 1)

# Use Newton's method to find the first solution
x1 = D.newtonls(x, F, jacobian)

# # Update found solution list in deflation operator
D.update_known_roots([x1])
 
# Solve again from the same initial guess and using the same solver to find a new solution
x2 = D.newtonls(x, F, jacobian)