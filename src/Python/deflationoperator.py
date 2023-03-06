import numpy as np

class DeflationOperator(object):    
    def __init__(self, known_roots, M, p, a):
        self.known_roots = known_roots
        self.M = M
        self.p = p
        self.a = a

    def update_known_roots(self, known_roots):
        self.known_roots = known_roots

    def tau(self, x, update):
        dMy = self.getdMy(x, update)
        minv = 1.0 / self.deflation_evaluate(x)
        return 1.0 / (1.0 - minv*dMy)

    def getdMy(self, x, update):
        # defcon has a minus sign here, but that's because PETSc
        # calculates the update so that x = x - update rather
        # than x = x + update
        deriv = self.deflation_derivative(x)
        return deriv.dot(self.M.dot(update))

    def norm_squared(self, y, root):
        return (y-root).dot(self.M.dot(y-root))

    def derivative_norm_squared(self, y, root):
        return 2 * (y-root)

    def deflation_evaluate(self, y):
        m = 1.0
        for i in range(len(self.known_roots)):
            normsq = self.norm_squared(y, self.known_roots[i])
            factor = normsq**(-self.p/2) + self.a
            m = m * factor
        return m

    def deflation_derivative(self, x):
        N = len(self.known_roots)
        normsqs = [self.norm_squared(x, self.known_roots[i]) for i in range(N)]
        dnormsqs =  [self.derivative_norm_squared(x, self.known_roots[i]) for i in range(N)]

        factors = [normsqs[i]**(-self.p/2) + self.a for i in range(N)]
        dfactors = [(-self.p/2) * normsqs[i]**((-self.p/2) - 1.0) for i in range(N)]

        eta = np.prod(factors)
        deta = np.sum([eta/factors[i] * dfactors[i] * dnormsqs[i] for i in range(N)], axis=0)
        return deta        

    def newtonls(self, x, residual, jacobian, tol=1e-9, max_iter=1000):
        M = self.M
        known_roots = self.known_roots
        p = self.p
        a = self.a

        r = residual(x)
        J = jacobian(x)

        norm_residual = np.linalg.norm(r)
        print("Iteration 0, residual norm = %s\n" %norm_residual)

        i = 0
        while norm_residual > tol and i < max_iter:
            update = -np.linalg.solve(J, r)   
            if known_roots:
                tau = self.tau(x, update)
                update = tau * update
            # update = nls.LineSearch.adjust(x, update, nls.damping)
            x = x + update
            r = residual(x)
            J = jacobian(x)
            norm_residual = np.linalg.norm(r)
            i += 1
            print("Iteration %s, residual norm = %s\n" %(i, norm_residual))

        if iter == max_iter:
            print("Iteration max reached")

        return x