"""Tests for Lorenz ODE: dfds function."""
import unittest
from collections import OrderedDict

import numpy as np
from scipy.integrate import odeint

from cdriver.datagen.lorenz import dfds


class TestLorenzDfds(unittest.TestCase):
    def setUp(self):
        self.param_dict = OrderedDict(
            sigma1=10.0, rho1=27.0, beta1=8.0 / 3,
            sigma2=12.0, rho2=29.0, beta2=5.0 / 3,
            sigma3=8.0, rho3=25.0, beta3=10.0 / 3,
            kappa=np.zeros((3, 3)),
        )
        self.u0 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])

    def test_output_shape(self):
        deriv = dfds(self.u0, 0.0, self.param_dict)
        self.assertEqual(len(deriv), 9)
        for val in deriv:
            self.assertFalse(np.isnan(val), f"NaN in derivative: {deriv}")
            self.assertFalse(np.isinf(val), f"Inf in derivative: {deriv}")

    def test_steady_state_zero_derivative(self):
        """At the origin (0,0,0...), derivative should be zero for all components."""
        u_zero = np.zeros(9)
        deriv = dfds(u_zero, 0.0, self.param_dict)
        for val in deriv:
            self.assertAlmostEqual(val, 0.0, places=8)

    def test_integration_produces_finite_trajectory(self):
        """Integrate a short trajectory and verify no NaN/Inf."""
        t = np.linspace(0, 10, 100)
        v = odeint(dfds, self.u0, t, args=(self.param_dict,))
        self.assertEqual(v.shape, (100, 9))
        self.assertFalse(np.any(np.isnan(v)))
        self.assertFalse(np.any(np.isinf(v)))

    def test_chaotic_trajectory_diverges_from_small_perturbation(self):
        """A small initial perturbation should produce different trajectories (chaos)."""
        t = np.linspace(0, 10, 100)
        v1 = odeint(dfds, self.u0, t, args=(self.param_dict,))
        u0_perturbed = self.u0 + 1e-6 * np.random.randn(9)
        v2 = odeint(dfds, u0_perturbed, t, args=(self.param_dict,))
        # After some time, trajectories should differ
        diff = np.abs(v1[-1] - v2[-1]).max()
        self.assertGreater(diff, 1e-4)
