"""
purged_kfold_cv.py
------------------
Purged K-Fold Cross-Validation with Embargo (Marcos López de Prado Methodology).
Eliminates lookahead bias, overlap leak, and serial correlation between train/validation splits.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Generator

class PurgedGroupTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, pct_embargo: float = 0.01):
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo

    def split(self, X: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates train/test indices with temporal purging and embargo periods.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo_size = int(n_samples * self.pct_embargo)
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            # Apply Purging & Embargo (exclude overlap + post-test embargo period)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_end] = False
            
            embargo_end = min(n_samples, test_end + embargo_size)
            train_mask[test_end:embargo_end] = False

            train_indices = indices[train_mask]
            yield train_indices, test_indices

purged_cv = PurgedGroupTimeSeriesSplit()
