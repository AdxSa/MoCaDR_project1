import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
# import torch


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
    user_weights = np.count_nonzero(~mask, axis=1)  # shape (n_users,)
    movie_weights = np.count_nonzero(~mask, axis=0)  # shape (n_movies,)

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

    # 7) final rounding to nearest 0.5
    return np.round(Z * 2) / 2


def build_rating_matrix(train_file, user_map=None, movie_map=None, impute=False):
    """
    Reads a ratings CSV file with columns: userId, movieId, rating.
    Builds and returns the user–movie matrix Z (missing entries set to 0),
    along with mappings from userId to row index and movieId to column index.

    Parameters:
      - train_file (str): Path to the training CSV file.

    Returns:
      - Z (ndarray): Rating matrix of shape (n_users, n_movies).
      - user_map (dict): Mapping from userId to row index.
      - movie_map (dict): Mapping from movieId to column index.
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

    # Build matrix Z with zeros for missing ratings
    Z = np.zeros((n_users, n_movies), dtype=np.float32)
    for row in df.itertuples():
        i = user_map[row.userId]
        j = movie_map[row.movieId]
        Z[i, j] = row.rating

    # print('build')
    if impute:
        Z = weighted_imputation(Z)
    return Z, user_map, movie_map


def split_data(file, n_splits=5):
    df = pd.read_csv(file)
    # X = df.drop(columns = ['userId', 'movieId'])
    # y = df['rating']
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_dfs = []
    test_dfs = []

    for train_idx, test_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_dfs.append(train_df)
        test_dfs.append(test_df)

    return train_dfs, test_dfs


def optimal_r_finder(file, method, n_splits=10):
    df = pd.read_csv(file)
    unique_users = df["userId"].unique()
    user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    unique_movies = df["movieId"].unique()
    movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    train_dfs, test_dfs = split_data(file, n_splits=n_splits)
    # print("data splited")

    RMSE_matrix = np.zeros((n_splits, 20), dtype=np.float32)

    for fold, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
        RMSE_list = []

        Z_test, _, _ = build_rating_matrix(test_df, user_map, movie_map, impute=False)

        # for r in range(1, min(Z.shape[0], Z.shape[1]) + 1):
        for r in range(1, 21):
            Z_approx, _, _ = method(train_df, user_map, movie_map, r)
            # print("Z approximated")

            test_i, test_j = np.where(Z_test != 0)
            true_val = Z_test[test_i, test_j]
            approxed_val = Z_approx[test_i, test_j]

            rmse = np.sqrt(np.mean((true_val - approxed_val) ** 2))

            RMSE_list.append(rmse)
            print(f"r={r}, fold = {fold}, rmse = {rmse}")
        RMSE_matrix[fold,] = RMSE_list
    RMSE_mean = RMSE_matrix.mean(axis=0)
    r_best = np.argmin(RMSE_mean) + 1
    print(r_best)
    return r_best, RMSE_matrix

from sklearn.model_selection import   RandomizedSearchCV
from scipy.stats import uniform



# def optimal_lam_finder(file, r, n_splits=10):
#     df = pd.read_csv(file)
#     unique_users = df["userId"].unique()
#     user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
#     unique_movies = df["movieId"].unique()
#     movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

#     train_dfs, test_dfs = split_data(file, n_splits=n_splits)
#     # print("data splited")
#     param_dist = {'lam': uniform(0, 5)}

#     # Create a RandomizedSearchCV object with 100 iterations
#     random_search = RandomizedSearchCV(train_sgd_model(), param_dist, cv=n_splits, scoring='neg_root_mean_squared_error',
#                                     n_iter=100, random_state=42)
#     random_search.fit(x_train, y_train)

#     print("RandomizedSearchCV best lambda:", random_search.best_params_['lam'])
#     print("RandomizedSearchCV best CV MSE:", -random_search.best_score_)

#     best_param_random = random_search.best_params_['lam']

def optimal_lam_finder(file, method, r=14, n_splits=10):
    df = pd.read_csv(file)
    unique_users = df["userId"].unique()
    user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    unique_movies = df["movieId"].unique()
    movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}
    lam_list = np.arange(0, 5, 100)

    train_dfs, test_dfs = split_data(file, n_splits=n_splits)
    # print("data splited")

    RMSE_matrix = np.zeros((n_splits, 20), dtype=np.float32)

    for fold, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
        RMSE_list = []

        Z_test, _, _ = build_rating_matrix(test_df, user_map, movie_map, impute=False)
        Z_test = torch.tensor(Z_test)

        # for r in range(1, min(Z.shape[0], Z.shape[1]) + 1):
        for lam in lam_list:
            Z_approx, _, _ = method(train_df, user_map, movie_map, lam=lam)
            # print("Z approximated")

            test_i, test_j = np.where(Z_test != 0)
            true_val = Z_test[test_i, test_j]
            approxed_val = Z_approx[test_i, test_j]

            rmse = torch.sqrt(torch.mean((true_val - approxed_val) ** 2))

            RMSE_list.append(rmse)
            print(f"lambda={lam}, fold = {fold}, rmse = {rmse}")
        RMSE_matrix[fold,] = RMSE_list
    RMSE_mean = RMSE_matrix.mean(axis=0)
    lam_best = np.argmin(RMSE_mean) 
    print(lam_best)
    return lam_best, RMSE_matrix


if __name__ == "__main__":

    data = {
        "userId": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "movieId": [101, 102, 101, 103, 102, 103, 101, 104, 103, 104],
        "rating": [5.0, 0.0, 4.0, 2.0, 3.5, 4.0, 0.0, 1.0, 3.0, 4.5]
    }

    pd.DataFrame(data).to_csv("test.csv", index=False)

    Z, user_map, movie_map = build_rating_matrix("test.csv", impute=True)
    print(Z)
