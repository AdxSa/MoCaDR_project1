import numpy as np


def weighted_imputation(Z):
    """
    Vectorized weighted imputation:
      - user_weights[i]  = count of nonzeros in row i
      - movie_weights[j] = count of nonzeros in col j
      - user_medians[i]  = median of nonzeros in row i (0 if empty)
      - movie_medians[j] = median of nonzeros in col j (0 if empty)
      - for missing Z[i,j]==0:
           Z[i,j] = (uw[i]*um[i] + mw[j]*mm[j])/(uw[i]+mw[j])  if denom>0
                    global_mean                              otherwise
    Finally rounds to the nearest 0.5.
    """
    Z = Z.copy().astype(np.float32)    # work on a copy
    mask = (Z == 0)               # missing entries

    # 1) weights
    user_count = np.count_nonzero(~mask, axis=1)  # shape (n_users,)
    movie_count = np.count_nonzero(~mask, axis=0)  # shape (n_movies,)

    ##new - logarithmic weights
    user_weights = np.log1p(user_count)
    movie_weights = np.log1p(movie_count)

    # 2) medians (fall back to 0 if no nonzeros)
    user_medians = np.array([
        np.median(Z[i, ~mask[i]]) if user_weights[i] > 0 else 0
        for i in range(Z.shape[0])
    ])
    movie_medians = np.array([
        np.median(Z[~mask[:, j], j]) if movie_weights[j] > 0 else 0
        for j in range(Z.shape[1])
    ])

    # 3) broadcast into full matrices
    #    numerator[i,j]   = uw[i]*um[i] + mw[j]*mm[j]
    #    denominator[i,j] = uw[i] + mw[j]
    uw = user_weights[:, None]    # (n_users, 1)
    um = user_medians[:, None]    # (n_users, 1)
    mw = movie_weights[None, :]   # (1, n_movies)
    mm = movie_medians[None, :]   # (1, n_movies)

    numerator = uw * um + mw * mm
    denominator = uw + mw

    # 4) global fallback mean if both weights are zero
    nonzero_vals = Z[~mask]
    global_mean = nonzero_vals.mean() if nonzero_vals.size > 0 else 3.0

    # 5) compute the fill matrix
    fill = np.where(denominator > 0,
                    numerator / denominator,
                    global_mean)

    # 6) apply only to missing entries
    Z[mask] = fill[mask]

    return Z


def impute_global_mean(Z):
    """Fill missing entries with the global mean of observed ratings."""
    mask = (Z == 0)
    vals = Z[~mask]
    m = vals.mean() if vals.size else 0
    Z[mask] = m
    return Z

import numpy as np

def impute_user_mean(Z):
    """
    Fill missing entries (zeros) in each row of Z with that row’s mean of non-zero entries.
    If a row has all zeros, its mean is defined to be 0.
    """
    counts = (Z != 0).sum(axis=1)
    sums = Z.sum(axis=1)
    user_means = np.where(counts > 0, sums / counts, 0)
    return np.where(Z == 0, user_means[:, None], Z)

def impute_movie_mean(Z):
    """
    Fill missing entries (zeros) in each column of Z with that column’s mean of non-zero entries.
    If a column has all zeros, its mean is defined to be 0.
    """
    counts = (Z != 0).sum(axis=0)
    sums = Z.sum(axis=0)
    user_means = np.where(counts > 0, sums / counts, 0)
    return np.where(Z == 0, user_means[None, :], Z)