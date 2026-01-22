"""Core Plackett-Luce model implementation."""

import numpy as np
from scipy.optimize import minimize


class PlackettLuce:
    """Plackett-Luce model for ranking data.
    
    Parameters
    ----------
    n_items : int
        Number of items to rank
    method : str, default='mm'
        Fitting method: 'mm' (Minorization-Maximization) or 'mle'
    """
    
    def __init__(self, n_items, method='mm'):
        self.n_items = n_items
        self.method = method
        self.params = np.ones(n_items)
        self.is_fitted = False
    
    def probability(self, ranking):
        """Calculate probability of a given ranking.
        
        Parameters
        ----------
        ranking : array-like
            List of item indices in ranked order
            
        Returns
        -------
        float
            Probability of the ranking
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing probabilities")
        
        prob = 1.0
        remaining_items = set(ranking)
        
        for item in ranking:
            denominator = sum(self.params[i] for i in remaining_items)
            prob *= self.params[item] / denominator
            remaining_items.remove(item)
        
        return prob
    
    def log_likelihood(self, rankings):
        """Compute log-likelihood for multiple rankings.
        
        Parameters
        ----------
        rankings : list of array-like
            List of rankings
            
        Returns
        -------
        float
            Log-likelihood value
        """
        ll = 0
        for ranking in rankings:
            remaining = set(ranking)
            for item in ranking:
                ll += np.log(self.params[item]) - np.log(
                    sum(self.params[i] for i in remaining)
                )
                remaining.remove(item)
        return ll
    
    def fit(self, rankings, max_iter=100, tol=1e-6):
        """Fit model to ranking data.
        
        Parameters
        ----------
        rankings : list of array-like
            List of rankings, where each ranking is a list of item indices
        max_iter : int, default=100
            Maximum number of iterations
        tol : float, default=1e-6
            Convergence tolerance
            
        Returns
        -------
        self
            Fitted model
        """
        if self.method == 'mm':
            self._fit_mm(rankings, max_iter, tol)
        elif self.method == 'mle':
            self._fit_mle(rankings)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def _fit_mm(self, rankings, max_iter, tol):
        """Fit using MM algorithm."""
        for iteration in range(max_iter):
            old_params = self.params.copy()
            
            wins = np.zeros(self.n_items)
            gamma = np.zeros(self.n_items)
            
            for ranking in rankings:
                remaining = list(ranking)
                for pos, item in enumerate(ranking[:-1]):
                    wins[item] += 1
                    denom = sum(self.params[i] for i in remaining[pos:])
                    for i in remaining[pos:]:
                        gamma[i] += 1 / denom
            
            self.params = wins / (gamma + 1e-10)
            self.params = self.params / np.sum(self.params) * self.n_items
            
            if np.max(np.abs(self.params - old_params)) < tol:
                break
    
    def _fit_mle(self, rankings):
        """Fit using MLE with optimization."""
        def neg_log_likelihood(params):
            self.params = params
            return -self.log_likelihood(rankings)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - self.n_items}
        bounds = [(1e-6, None)] * self.n_items
        
        result = minimize(
            neg_log_likelihood,
            self.params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.params = result.x
    
    def rank_items(self):
        """Get items ranked by estimated strength.
        
        Returns
        -------
        ndarray
            Item indices in descending order of strength
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before ranking items")
        return np.argsort(self.params)[::-1]
    
    def top_k(self, k):
        """Get top k items by strength.
        
        Parameters
        ----------
        k : int
            Number of top items to return
            
        Returns
        -------
        ndarray
            Top k item indices
        """
        return self.rank_items()[:k]
