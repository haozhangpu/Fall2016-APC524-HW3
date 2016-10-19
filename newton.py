# newton - Newton-Raphson solver
#
# For APC 524 Homework 3
# Hao Zhang, Oct 18, 2016

import numpy as N
import functions as F

class Newton(object):
    def __init__(self, f, Df=None, tol=1.e-6, r=1, maxiter=20, dx=1.e-6):
        """Return a new object to find roots of f(x) = 0 using Newton's method.
        f:       function f(x) = 0
        Df:      Analytical Jacobian of f(x)
        tol:     tolerance for iteration (iterate until |f(x)| < tol)
        maxiter: maximum number of iterations to perform
        dx:      step size for computing approximate Jacobian"""
        self._f = f
        self._Df = Df
        self._tol = tol
        self._r = r
        self._maxiter = maxiter
        self._dx = dx

    def solve(self, x0):
        """Return a root of f(x) = 0, using Newton's method, starting from
        initial guess x0"""
        x = x0
        for i in xrange(self._maxiter):
            fx = self._f(x)
            if N.linalg.norm(fx) < self._tol:
                return x
            if N.linalg.norm(x-x0) > self._r:
                raise Exception('xk outside of radius r of the initial guess x0')
            x = self.step(x, fx)
        if N.linalg.norm(self._f(x)) > self._tol and i >= self._maxiter - 1:
            raise Exception('Newton method fails to converge')
        return x

    def step(self, x, fx=None):
        """Take a single step of a Newton method, starting from x
        If the argument fx is provided, assumes fx = f(x)"""
        if fx is None:
            fx = self._f(x)
        if self._Df is None:
            Df_x = F.ApproximateJacobian(self._f, x, self._dx)
        else:
            Df_x = self._Df(x)
        h = N.linalg.solve(N.matrix(Df_x), N.matrix(fx))
        return x - h