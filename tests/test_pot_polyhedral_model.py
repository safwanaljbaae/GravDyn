# pytest -v
# tests/test_pot_polyhedral_model.py
from __future__ import annotations
import numpy as np
import pytest
from gravdyn import pot_polyhedral_model


def test_pot_polyhedral_model_raises_for_invalid_single_point_shape():
    mu = 1.0
    stat = [1.0, 2.0]
    mock_data = {
        "faces": np.array([[0, 1, 2]]),
        "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        "edges": np.array([[0, 1, 1, 2]]),
    }

    with pytest.raises((ValueError, KeyError)):
        pot_polyhedral_model(gm_body=mu, stat=stat, polyhedral_data=mock_data)


def test_pot_polyhedral_model_raises_for_invalid_batch_shape():
    mu = 1.0
    stat = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    mock_data = {
        "faces": np.array([[0, 1, 2]]),
        "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        "edges": np.array([[0, 1, 1, 2]]),
    }

    with pytest.raises((ValueError, KeyError)):
        pot_polyhedral_model(gm_body=mu, stat=stat, polyhedral_data=mock_data)


def test_pot_polyhedral_model_zero_mu():
    pytest.skip("Requires valid polyhedral data with proper mesh geometry")
