from __future__ import annotations

import numpy as np

from gravdyn.constants import (
    GRAVITATIONAL_CONSTANT,
    EPSILON,
    EPSILON_JAX,
    EPSILON_POLYHEDRAL,
)


class TestConstants:
    def test_gravitational_constant_is_positive(self):
        assert GRAVITATIONAL_CONSTANT > 0

    def test_gravitational_constant_magnitude(self):
        assert 1e-20 < GRAVITATIONAL_CONSTANT < 1e-19

    def test_epsilon_is_positive(self):
        assert EPSILON > 0

    def test_epsilon_is_small(self):
        assert EPSILON < 1e-30

    def test_epsilon_jax_is_positive(self):
        assert EPSILON_JAX > 0

    def test_epsilon_jax_is_small(self):
        assert EPSILON_JAX < 1e-25

    def test_epsilon_polyhedral_is_positive(self):
        assert EPSILON_POLYHEDRAL > 0

    def test_epsilon_polyhedral_is_small(self):
        assert EPSILON_POLYHEDRAL < 1e-10

    def test_all_are_float(self):
        assert isinstance(GRAVITATIONAL_CONSTANT, float)
        assert isinstance(EPSILON, float)
        assert isinstance(EPSILON_JAX, float)
        assert isinstance(EPSILON_POLYHEDRAL, float)

    def test_epsilons_are_finite(self):
        assert np.isfinite(EPSILON)
        assert np.isfinite(EPSILON_JAX)
        assert np.isfinite(EPSILON_POLYHEDRAL)
