import pandas as pd
from sklearn.model_selection import KFold
from .impute_functions import *


def build_rating_matrix(train_file, user_map=None, movie_map=None, impute=False):
    """
    Builds the user–movie rating matrix Z from a CSV or DataFrame of ratings.

    - Input can be a CSV file path or a DataFrame with columns: userId, movieId, rating.
    - Constructs mappings from user IDs and movie IDs to matrix indices.
    - Fills missing ratings with 0s unless an imputation function is specified.

    Parameters:
        train_file (str or pd.DataFrame): Input ratings data.
        user_map (dict): Optional pre-defined mapping from userId to row index.
        movie_map (dict): Optional pre-defined mapping from movieId to column index.
        impute (callable or bool): If a function is passed, it is used to impute missing entries.

    Returns:
        Z (ndarray): Rating matrix of shape (n_users, n_movies).
        user_map (dict): Mapping from userId to row index.
        movie_map (dict): Mapping from movieId to column index.
    """
    if isinstance(train_file, str):
        df = pd.read_csv(train_file)
    else:
        df = train_file

    if user_map is None:
        unique_users = df["userId"].unique()
        user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}

    if movie_map is None:
        unique_movies = df["movieId"].unique()
        movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    n_users = len(user_map)
    n_movies = len(movie_map)
    Z = np.zeros((n_users, n_movies), dtype=np.float32)

    for row in df.itertuples():
        i = user_map[row.userId]
        j = movie_map[row.movieId]
        Z[i, j] = row.rating

    if impute:
        Z = impute(Z)

    return Z, user_map, movie_map


def split_data(file, n_splits=5):
    """
    Splits a ratings CSV file into k folds for cross-validation.

    Parameters:
        file (str): Path to ratings CSV file.
        n_splits (int): Number of folds for K-Fold cross-validation.

    Returns:
        train_dfs (list of DataFrames): Training subsets.
        test_dfs (list of DataFrames): Testing subsets.
    """
    df = pd.read_csv(file)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_dfs = []
    test_dfs = []

    for train_idx, test_idx in kf.split(df):
        train_dfs.append(df.iloc[train_idx])
        test_dfs.append(df.iloc[test_idx])

    return train_dfs, test_dfs


def optimal_r_finder(train_file, method, n_splits=10, sr=(1, 20), ekw={}):
    """
    Finds the best matrix factorization rank r using cross-validated RMSE.

    Parameters:
        train_file (str): Path to ratings CSV file.
        method (callable): Matrix factorization method (returns W, H, etc.).
        n_splits (int): Number of folds in cross-validation.
        sr (tuple): Range of r values to search (inclusive).
        ekw (dict): Extra keyword arguments for the method.

    Returns:
        r_best (int): Rank with lowest cross-validated RMSE.
        RMSE_matrix (ndarray): RMSE values for each (fold, r) pair.
    """
    df = pd.read_csv(train_file)
    user_map = {uid: i for i, uid in enumerate(sorted(df["userId"].unique()))}
    movie_map = {mid: j for j, mid in enumerate(sorted(df["movieId"].unique()))}

    train_dfs, test_dfs = split_data(train_file, n_splits=n_splits)
    RMSE_matrix = np.zeros((n_splits, sr[1] - sr[0] + 1), dtype=np.float32)

    for fold, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
        Z_test, _, _ = build_rating_matrix(test_df, user_map, movie_map, impute=False)
        test_i, test_j = np.nonzero(Z_test)
        true_val = Z_test[test_i, test_j]

        for r in range(sr[0], sr[1] + 1):
            W, H, _, _ = method(train_df, user_map, movie_map, r, **ekw)
            Z_approx = np.dot(W, H)
            approxed_val = Z_approx[test_i, test_j]
            rmse = np.sqrt(np.mean((true_val - approxed_val) ** 2))
            RMSE_matrix[fold, r - sr[0]] = rmse
            print(f"r={r}, fold={fold}, rmse={rmse:.4f}")

    RMSE_mean = RMSE_matrix.mean(axis=0)
    r_best = np.argmin(RMSE_mean) + sr[0]
    print(f"Best r = {r_best} (CV RMSE = {np.min(RMSE_mean):.4f})")
    return r_best, RMSE_matrix


def optimal_sgd_hyperparams(train_file, method, n_splits=10, sr=(10, 30), ekw={}):
    """
    Grid-search over matrix rank r and regularization parameter lambda for SGD-based MF.

    Parameters:
        train_file (str): Path to ratings CSV file.
        method (callable): SGD matrix factorization method.
        n_splits (int): Number of CV folds.
        sr (tuple): Range of rank values to try (inclusive).
        ekw (dict): Additional kwargs to pass to `method`.

    Returns:
        (best_r, best_lam): Tuple of best rank and lambda.
        RMSE_mean (ndarray): RMSE values across all (r, lambda) pairs.
    """
    r_list = np.arange(sr[0], sr[1] + 1)
    lam_list = [0]  # Can be adjusted to a wider lambda grid if needed

    df = pd.read_csv(train_file)
    user_map = {uid: i for i, uid in enumerate(sorted(df["userId"].unique()))}
    movie_map = {mid: j for j, mid in enumerate(sorted(df["movieId"].unique()))}

    train_dfs, test_dfs = split_data(train_file, n_splits=n_splits)
    cv_errors = {(r, lam): [] for r in r_list for lam in lam_list}

    for fold, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
        Z_test, _, _ = build_rating_matrix(test_df, user_map, movie_map, impute=False)
        test_i, test_j = np.nonzero(Z_test)
        true_val = Z_test[test_i, test_j]

        for r in r_list:
            for lam in lam_list:
                W, H, _, _ = method(train_df, user_map, movie_map, r, lam=lam, **ekw)
                Z_approx = np.dot(W, H)
                approxed_val = Z_approx[test_i, test_j]
                rmse = np.sqrt(np.mean((true_val - approxed_val) ** 2))
                cv_errors[(r, lam)].append(rmse)
                print(f"Fold {fold + 1}/{n_splits} | r={r}, lam={lam:.3f} → RMSE={rmse:.4f}")

    mean_errors = {k: np.mean(v) for k, v in cv_errors.items()}
    best_r, best_lam = min(mean_errors, key=mean_errors.get)
    columns = list(cv_errors.values())
    RMSE_mean = np.column_stack(columns)

    print(f"\nBest pair: r = {best_r}, lam = {best_lam:.3f}  (CV RMSE = {mean_errors[(best_r, best_lam)]:.4f})")
    return (best_r, best_lam), RMSE_mean
