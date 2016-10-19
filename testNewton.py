#!/usr/bin/env python

import newton
import unittest
import numpy as N

class TestNewton(unittest.TestCase):
    def testLinear(self):
        """Test Newton root finder with linear function f(x)=3x+6"""
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(-1.8)
        self.assertEqual(x, -2.0)
        
    def testLinear2D(self):
        """Test Newton root finder with 2D linear function f(x)=Ax+b"""
        A = N.matrix("1. 2.; 3. 4.")
        b = N.matrix("-3;-7")
        def f(x):
            return A * x + b
        x0 = N.matrix("1.2; 1.2")
        xsolu = N.linalg.solve(N.matrix(A), N.matrix(-b))
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, xsolu)
        
    def testPolynomial1D(self):
        """Test Newton root finder with polynomial function"""
        def f(x):
            return x**2 - 3*x - 4
        x0, xsolu = 3.5, 4
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(x0)
        self.assertAlmostEqual(x, xsolu)
        
    def testPolynomial2D(self):
        """Test Newton root finder with 2D polynomial function"""
        def f(x):
            fx = N.matrix(N.zeros((2,1)))
            fx[0] = x[0]**2 - x[1]
            fx[1] = x[1] - x[0]
            return fx
        x0, xsolu = N.matrix("0.2;0.2"), N.matrix("0;0")
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, xsolu)
        
    def testStepLinear2D(self):
        """Test single step of Newton root finder with 2D linear function"""
        A = N.matrix("1. 2.; 3. 4.")
        b = N.matrix("-3;-7")
        def f(x):
            return A * x + b
        x0 = N.matrix("1.2; 1.2")
        xsolu = N.linalg.solve(N.matrix(A), N.matrix(-b))
        solver = newton.Newton(f)
        x = solver.step(x0)
        N.testing.assert_array_almost_equal(x, xsolu)
        
    def testConvergeException(self):
        """Test Newton root finder raises an exception if fails to converge"""
        def f(x):
            fx = N.matrix(N.zeros((2,1)))
            fx[0] = x[0]**2 - x[1]
            fx[1] = x[1] - x[0]
            return fx
        x0, xsolu = N.matrix("0.2;0.2"), N.matrix("0;0")
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        with self.assertRaises(Exception) as ExceptionMessage:
            solver.solve(x0)
        self.assertTrue('Newton method fails to converge' in ExceptionMessage.exception)
        
    def testNewtonAnalyticJacobian(self):
        """Test Newton root finder is actually using the analytical Jacobian"""
        def f(x):
            return x**2 + 2*x + 3
        def Df(x):
            return 2*x + 2
        x0 = 1.
        solver = newton.Newton(f,Df,dx=1e-1)
        x1_Newton = solver.step(x0)
        x1_AnalJocabian = x0 - N.linalg.solve(N.matrix(Df(x0)), N.matrix(f(x0)))
        self.assertEqual(x1_Newton, x1_AnalJocabian)
        
    def testRadiusExceptionPolynomial(self):
        """Test Newton root finder raises an exception if xk is outside of 
        radius r of the initial guess, use 1D polynomial as example"""
        def f(x):
            return x**2 - 1
        x0 = 3.
        solver = newton.Newton(f, tol=1.e-15, r=1)
        with self.assertRaises(Exception) as ExceptionMessage:
            solver.solve(x0)
        self.assertTrue('xk outside of radius r of the initial guess x0' in ExceptionMessage.exception)
        
    def testRadiusExceptionPolynomial2D(self):
        """Test Newton root finder raises an exception if xk is outside of 
        radius r of the initial guess, use 2D polynomial as example"""
        def f(x):
            fx = N.matrix(N.zeros((2,1)))
            fx[0] = x[0]**2 - x[1]
            fx[1] = x[1] - x[0]
            return fx
        x0 = N.matrix("3;3")
        solver = newton.Newton(f, tol=1.e-15, r=1)
        with self.assertRaises(Exception) as ExceptionMessage:
            solver.solve(x0)
        self.assertTrue('xk outside of radius r of the initial guess x0' in ExceptionMessage.exception)
        
if __name__ == "__main__":
    unittest.main()
