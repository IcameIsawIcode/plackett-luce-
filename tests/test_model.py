"""Tests for Plackett-Luce model."""

import pytest
import numpy as np
from plackett_luce import PlackettLuce


def test_initialization():
    """Test model initialization."""
    model = PlackettLuce(n_items=5)
    assert model.n_items == 5
    assert len(model.params) == 5
    assert not model.is_fitted


def test_fit_mm():
    """Test fitting with MM algorithm."""
    rankings = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2]
    ]
    
    model = PlackettLuce(n_items=3, method='mm')
    model.fit(rankings)
    
    assert model.is_fitted
    assert model.params[0] > model.params[2]  # Item 0 ranked higher


def test_fit_mle():
    """Test fitting with MLE."""
    rankings = [
        [0, 1, 2],
        [0, 2, 1],
    ]
    
    model = PlackettLuce(n_items=3, method='mle')
    model.fit(rankings)
    
    assert model.is_fitted


def test_probability():
    """Test probability computation."""
    rankings = [[0, 1, 2], [0, 1, 2]]
    
    model = PlackettLuce(n_items=3)
    model.fit(rankings)
    
    prob = model.probability([0, 1, 2])
    assert 0 <= prob <= 1


def test_rank_items():
    """Test item ranking."""
    rankings = [[0, 1, 2]] * 5 + [[1, 0, 2]] * 2
    
    model = PlackettLuce(n_items=3)
    model.fit(rankings)
    
    ranked = model.rank_items()
    assert ranked[0] == 0  # Item 0 should be ranked first


def test_top_k():
    """Test top-k retrieval."""
    rankings = [[0, 1, 2, 3]] * 3
    
    model = PlackettLuce(n_items=4)
    model.fit(rankings)
    
    top_2 = model.top_k(2)
    assert len(top_2) == 2


def test_unfitted_error():
    """Test error when using unfitted model."""
    model = PlackettLuce(n_items=3)
    
    with pytest.raises(ValueError):
        model.probability([0, 1, 2])
    
    with pytest.raises(ValueError):
        model.rank_items()


def test_log_likelihood():
    """Test log-likelihood computation."""
    rankings = [[0, 1, 2]]
    
    model = PlackettLuce(n_items=3)
    model.fit(rankings)
    
    ll = model.log_likelihood(rankings)
    assert isinstance(ll, float)
