# newton - Newton-Raphson solver
#
# For APC 524 Homework 3
# Hao Zhang, Oct 18, 2016

import numpy as N

def ApproximateJacobian(f, x, dx=1e-6):
    """Return an approximation of the Jacobian Df(x) as a numpy matrix"""
    try:
        n = len(x)
    except TypeError:
        n = 1
    fx = f(x)
    Df_x = N.matrix(N.zeros((n,n)))
    for i in range(n):
        v = N.matrix(N.zeros((n,1)))
        v[i,0] = dx
        Df_x[:,i] = (f(x + v) - fx)/dx
    return Df_x

class Linear(object):
    """Callable linear object.
    
    Example usage: to construct the linear function f(x) = A*x + b, and evaluate
    f(x0):
    
    f = Linear(A,b)
    f(x0)"""
    def __init__(self, A, b):
        self._A = A
        self._b = b
        
    def f(self,x):
        return N.dot(self._A,x)+self._b
    
    def Df(self,x):
        return self._A
    
    def __call__(self, x):
        return self.f(x)
    
class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 2x + 3,
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""

    def __init__(self, coeffs):
        self._coeffs = coeffs
        self._dercoeffs = N.polyder(coeffs)

    def __repr__(self):
        return "Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))

    def f(self,x):
        ans = self._coeffs[0]
        for c in self._coeffs[1:]:
            ans = x*ans + c
        return ans
    
    def Df(self,x):
        ans = self._dercoeffs[0]
        for c in self._dercoeffs[1:]:
            ans = x*ans + c
        return ans

    def __call__(self, x):
        return self.f(x)


class Polynomial2D(object):
    """Callable polynomial object.

    Example usage: to construct the 2D polynomial vector p(x) = 
    [x1^2 + 2x1x2 + 3x2^2 + 4x1 + 5x2 + 6, 2x1^2 + 3x1x2 + 4x2^2 + 5x1 + 6x2 + 7]^T
    and evaluate p([1, 2]^T):

    p = Polynomial2D([1,2,3,4,5,6],[2,3,4,5,6,7])
    p([[1],[2]])"""

    def __init__(self, coeffs1, coeffs2):
        self._coeffs1, self._coeffs2 = coeffs1, coeffs2

    def f(self,x):
        fx = N.matrix(N.zeros((2,1)))
        fx[0] = (self._coeffs1[0]*x[0]**2 + self._coeffs1[1]*x[0]*x[1] +
                self._coeffs1[2]*x[1]**2 + self._coeffs1[3]*x[0] +
                self._coeffs1[4]*x[1] + self._coeffs1[5])
        fx[1] = (self._coeffs2[0]*x[0]**2 + self._coeffs2[1]*x[0]*x[1] +
                self._coeffs2[2]*x[1]**2 + self._coeffs2[3]*x[0] +
                self._coeffs2[4]*x[1] + self._coeffs2[5])
        return fx
    
    def Df(self,x):
        Dfx = N.matrix(N.zeros((2,2)))
        Dfx[0,0] = 2*self._coeffs1[0]*x[0] + self._coeffs1[1]*x[1] + self._coeffs1[3]
        Dfx[0,1] = self._coeffs1[1]*x[0] + 2*self._coeffs1[2]*x[1] + self._coeffs1[4]
        Dfx[1,0] = 2*self._coeffs2[0]*x[0] + self._coeffs2[1]*x[1] + self._coeffs2[3]
        Dfx[1,1] = self._coeffs2[1]*x[0] + 2*self._coeffs2[2]*x[1] + self._coeffs2[4]
        return Dfx

    def __call__(self, x):
        return self.f(x)