#!/usr/bin/env python

import newton
import unittest
import numpy as N

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)
        
    def testLinear2D(self):
        A = N.matrix("1. 2.; 3. 4.")
        b = N.matrix("-3;-7")
        def f(x):
            return A * x + b
        x0 = N.matrix("2; 2")
        xsolu = N.linalg.solve(N.matrix(A), N.matrix(-b))
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, xsolu)
        
    def testPolynomial1D(self):
        def f(x):
            return x**2 - 3*x - 4
        x0, xsolu = 3, 4
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(x0)
        self.assertAlmostEqual(x, xsolu)
        
    def testPolynomial2D(self):
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
        A = N.matrix("1. 2.; 3. 4.")
        b = N.matrix("-3;-7")
        def f(x):
            return A * x + b
        x0 = N.matrix("2; 2")
        xsolu = N.linalg.solve(N.matrix(A), N.matrix(-b))
        solver = newton.Newton(f)
        x = solver.step(x0)
        N.testing.assert_array_almost_equal(x, xsolu)
        
    def testConvergeException(self):
        def f(x):
            fx = N.matrix(N.zeros((2,1)))
            fx[0] = x[0]**2 - x[1]
            fx[1] = x[1] - x[0]
            return fx
        x0, xsolu = N.matrix("0.1;0.1"), N.matrix("0;0")
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        with self.assertRaises(Exception) as ExceptionMessage:
            solver.solve(x0)
        self.assertTrue('Newton method fails to converge' in ExceptionMessage.exception)

if __name__ == "__main__":
    unittest.main()
