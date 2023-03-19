import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

"""
Self-contained Python script for deflation!

Finding two solutions of the Bratu equation
    u''(x) + λeᵘ(x) = 0
    u(0) = u(1) = 0.
This problem has one solution if λ = 0 or λ = λ*, 
two solutions if 0 < λ < λ*, and no solutions if λ > λ* where λ* ≈ 3.513830719. 

Script generated with the aid of ChatGPT from the Julia script "bratu_self_contained.jl".
"""

# Setup of finite difference stencil
n = 100 # number of intervals
h = 1/n # mesh size
diag_main = np.full(n, -2/h**2)
diag_sub = np.full(n-1, 1/h**2)
Lap = diags([diag_sub, diag_main, diag_sub], [-1, 0, 1]).toarray() # finite difference stencil
x = np.linspace(0, 1, num=n)

lmbda = 2.5 # Bratu parameter

# residual + bcs
def F(u):
    r = Lap.dot(u) + lmbda*np.exp(u)
    r[0] = 0
    r[-1] = 0
    return r

# Jacobian + bcs
def J(u):
    A = Lap + lmbda * diags(np.exp(u)) 
    A[:,0] = 0; A[0,:] = 0
    A[-1,:] = 0; A[:,-1] = 0
    A[0,0] = 1; A[-1,-1] = 1
    return A

# Initial guess
u0 = (1 - x) * x

# Run first Newton loop to find first solution
u1 = u0
for _ in range(100):
    u1 = u1 - np.linalg.solve(J(u1), F(u1))  

print(np.linalg.norm(F(u1)))

# Hard-coded deflation operator
def mu(u, u1):
    return 1/(np.dot(u-u1, u-u1)) + 1

def dmu(u, u1):
    return -2*(u-u1)/(np.dot(u-u1, u-u1))**2

def tau(u, du, u1):
    return 1 + (1/mu(u, u1) * np.dot(dmu(u, u1), du)) / (1 - 1/mu(u, u1) * np.dot(dmu(u, u1), du))

# Run second Newton loop from the SAME initial guess!
u2 = u0

for _ in range(100):
    # Solve origin undeflated Newton system
    du = - np.linalg.solve(J(u2), F(u2))  
    # Multiply Newton update with the SCALAR tau to 
    # find Newton step of deflated Newton system.
    u2 = u2 + tau(u2, du, u1)*du

print(np.linalg.norm(F(u2)))

# Plot the two different solutions
plt.plot(x, np.column_stack((u1, u2)))
plt.show()
