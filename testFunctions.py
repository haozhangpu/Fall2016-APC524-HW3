#!/usr/bin/env python

import functions as F
import numpy as N
import unittest

class TestFunctions(unittest.TestCase):
    def testApproxJacobian1(self):
        """Test approximate Jacobian with f(x)=3*x+5, Df(x)=3"""
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)

    def testApproxJacobian2(self):
        """Test approximate Jacobian with f(x)=Ax, Df(x)=A"""
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)
        
    def testApproxJacobianLorenz(self):
        """Test approximate Jacobian with lorenz system, with known f(x), Df(x)"""
        sigma, beta, rho = 10, 8./3, 28
        def f(x):
            fx = N.matrix(N.zeros((3,1)))
            fx[0] = sigma*(x[1]-x[0])
            fx[1] = x[0]*(rho-x[2]) - x[1]
            fx[2] = x[0]*x[1] - beta*x[2]
            return fx
        sigma, beta, rho = 10, 8./3, 28
        x0 = N.matrix("1; 1; 1")
        dx = 1e-6
        Dfx = N.matrix([[-sigma, sigma, 0],[rho-x0[2], -1, -x0[0]],
              [x0[1],x0[0],-beta]])
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (3,3))
        N.testing.assert_array_almost_equal(Df_x, Dfx)

    def testPolynomial(self):
        """Test Polynomial class with f(x)=x^2 + 2x + 3"""
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)
            
    def testLinearAnalyticalJacobian(self):
        """Test Linear class f(x)=Ax+b, and the analytical Jacobian Df(x)=A
        with approximate Jacobian with small dx = 1e-6"""
        A = N.matrix("1. 2.; 3. 4.")
        b = N.matrix("5.; 6")
        linear = F.Linear(A,b)
        x0 = N.matrix("5; 6")
        dx = 1e-6
        Dfx_anal = linear.Df(x0)
        Dfx_appro = F.ApproximateJacobian(linear.f, x0, dx)
        N.testing.assert_array_almost_equal(Dfx_anal, Dfx_appro)
        
    def testPolynomialAnalyticalJacobian(self):
        """Test Polynomial class f(x), and the analytical Jacobian Df(x)
        with approximate Jacobian with small dx = 1e-6"""
        coeffs = [1,2,3]
        p = F.Polynomial(coeffs)
        x0 = 1.
        dx = 1e-6
        Dfx_anal = p.Df(x0)
        Dfx_appro = F.ApproximateJacobian(p.f, x0, dx)
        N.testing.assert_array_almost_equal(Dfx_anal, Dfx_appro)
        
    def testPolynomial2DAnalyticalJacobian(self):
        """Test Polynomial2D class f(x) (x \in R2, f \in R2), 
        and the analytical Jacobian Df(x)
        with approximate Jacobian with small dx = 1e-6"""
        coeffs1 = [1,2,3,4,5,6]
        coeffs2 = [2,3,4,5,6,7]
        poly2D = F.Polynomial2D(coeffs1, coeffs2)
        x0 = N.matrix("1;1")
        dx = 1e-6
        Dfx_anal = poly2D.Df(x0)
        Dfx_appro = F.ApproximateJacobian(poly2D.f, x0, dx)
        N.testing.assert_array_almost_equal(Dfx_anal, Dfx_appro, decimal=4)

if __name__ == '__main__':
    unittest.main()


