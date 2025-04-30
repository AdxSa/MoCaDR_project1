import numpy as np

# -----------------------------------------------------------------------------
# weighted_imputation:
#   - Computes per-user and per-movie weights based on counts of observed
#     ratings (log-scaled counts).
#   - Determines each user’s median rating and each movie’s median rating.
#   - For each missing entry, imputes a weighted average of the user median
#     and movie median, falling back to the global mean if no observations.
#   - Rounds results to the nearest 0.5 (optional rounding step can be added).
# -----------------------------------------------------------------------------
def weighted_imputation(Z):
    """
    Vectorized weighted imputation:
      - user_weights[i]  = log(1 + count of nonzero ratings by user i)
      - movie_weights[j] = log(1 + count of nonzero ratings for movie j)
      - user_medians[i]  = median of nonzero ratings by user i (0 if none)
      - movie_medians[j] = median of nonzero ratings for movie j (0 if none)
      - for each missing Z[i,j] == 0, impute:
           (uw[i] * um[i] + mw[j] * mm[j]) / (uw[i] + mw[j])  if denom>0
           global_mean                                       otherwise
    Finally replaces zeros in-place and returns the completed matrix.
    """
    mask = (Z == 0)

    # Count non-missing ratings
    user_count = np.count_nonzero(~mask, axis=1)
    movie_count = np.count_nonzero(~mask, axis=0)

    # Compute log-scaled weights
    user_weights = np.log1p(user_count)
    movie_weights = np.log1p(movie_count)

    # Compute medians of observed ratings
    user_medians = np.array([
        np.median(Z[i, ~mask[i]]) if user_weights[i] > 0 else 0
        for i in range(Z.shape[0])
    ])
    movie_medians = np.array([
        np.median(Z[~mask[:, j], j]) if movie_weights[j] > 0 else 0
        for j in range(Z.shape[1])
    ])

    # Broadcast weights and medians for vectorized arithmetic
    uw = user_weights[:, None]
    um = user_medians[:, None]
    mw = movie_weights[None, :]
    mm = movie_medians[None, :]

    # Prepare numerator and denominator
    numerator = uw * um + mw * mm
    denominator = uw + mw

    # Compute global mean of observed ratings
    nonzero_vals = Z[~mask]
    global_mean = nonzero_vals.mean() if nonzero_vals.size > 0 else 0

    # Compute imputed values
    fill = np.where(denominator > 0, numerator / denominator, global_mean)

    # Replace missing entries
    Z[mask] = fill[mask]

    return Z

# -----------------------------------------------------------------------------
# impute_global_mean:
#   - Fills every missing entry (zeros) with the overall mean of observed ratings.
# -----------------------------------------------------------------------------
def impute_global_mean(Z):
    """
    Fill missing entries with the global mean of all non-zero entries in Z.
    """
    mask = (Z == 0)
    vals = Z[~mask]
    global_mean = vals.mean() if vals.size else 0
    Z[mask] = global_mean
    return Z

# -----------------------------------------------------------------------------
# impute_user_mean:
#   - For each user (row), computes the mean of their observed ratings.
#   - Fills that user’s missing entries with their personal mean.
#   - Users with no ratings get a mean of 0.
# -----------------------------------------------------------------------------
def impute_user_mean(Z):
    """
    Replace zeros in each row by that row’s mean of non-zero entries.
    If a user has no ratings, uses 0.
    """
    counts = (Z != 0).sum(axis=1)
    sums = Z.sum(axis=1)
    user_means = np.where(counts > 0, sums / counts, 0)

    # Broadcast user_means across columns for imputation
    return np.where(Z == 0, user_means[:, None], Z)

# -----------------------------------------------------------------------------
# impute_movie_mean:
#   - For each movie (column), computes the mean of its observed ratings.
#   - Fills that movie’s missing entries with its mean rating.
#   - Movies with no ratings get a mean of 0.
# -----------------------------------------------------------------------------
def impute_movie_mean(Z):
    """
    Replace zeros in each column by that column’s mean of non-zero entries.
    If a movie has no ratings, uses 0.
    """
    counts = (Z != 0).sum(axis=0)
    sums = Z.sum(axis=0)
    movie_means = np.where(counts > 0, sums / counts, 0)

    # Broadcast movie_means across rows for imputation
    return np.where(Z == 0, movie_means[None, :], Z)
