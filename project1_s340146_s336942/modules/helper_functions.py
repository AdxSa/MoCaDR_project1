import pandas as pd
from sklearn.model_selection import KFold
from .impute_functions import *



def build_rating_matrix(train_file, user_map=None, movie_map=None, impute = False):

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
        Z = impute(Z)

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


def optimal_r_finder(train_file, method, n_splits=10, sr = (1,20), ekw = {}):



    df = pd.read_csv(train_file)
    unique_users = df["userId"].unique()
    user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    unique_movies = df["movieId"].unique()
    movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    train_dfs, test_dfs = split_data(train_file, n_splits=n_splits)
    # print("data splited")

    RMSE_matrix = np.zeros((n_splits, sr[1]-sr[0]+1), dtype=np.float32)

    for fold, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
        RMSE_list = []

        Z_test, _, _ = build_rating_matrix(test_df, user_map, movie_map, impute=False)
        test_i, test_j = np.nonzero(Z_test)
        true_val = Z_test[test_i, test_j]
        # for r in range(1, min(Z.shape[0], Z.shape[1]) + 1):
        for r in range(sr[0], sr[1]+1):
            W, H, _, _ = method(train_df, user_map, movie_map, r, **ekw)
            Z_approx = np.dot(W, H)


            approxed_val = Z_approx[test_i, test_j]

            rmse = np.sqrt(np.mean((true_val - approxed_val) ** 2))

            RMSE_list.append(rmse)
            print(f"r={r}, fold = {fold}, rmse = {rmse}")
        RMSE_matrix[fold,] = RMSE_list
    RMSE_mean = RMSE_matrix.mean(axis=0)
    r_best = np.argmin(RMSE_mean) + 1
    print(f"Best r = {r_best} (CV RMSE = {np.min(RMSE_mean):.4f})")
    return r_best, RMSE_matrix


def optimal_sgd_hyperparams(train_file,
                            method,
                            n_splits=10,
                            sr=(10,30),
                            ekw={}):
    """
    Grid‐search over ranks and regularizations for SGD-based MF.

    Returns:
      best_r:    rank with lowest CV RMSE
      best_lam:  lambda  with lowest CV RMSE
      cv_errors: dict mapping (r,lam) -> mean RMSE over folds
    """
    r_list = np.arange(sr[0], sr[1]+1, 1)
    # lam_list = np.array([0, 0.01, 0.1, 1])
    lam_list = [0]
    # lam_list = np.array([0,1,10,100,1000])
    df = pd.read_csv(train_file)
    unique_users = df["userId"].unique()
    user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    unique_movies = df["movieId"].unique()
    movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    train_dfs, test_dfs = split_data(train_file, n_splits=n_splits)

    cv_errors = {(r, lam): [] for r in r_list for lam in lam_list}

    # For each fold: train on train_idx, test on test_idx
    for fold, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):

        # Build the test‐matrix once per fold
        Z_test, _, _ = build_rating_matrix(test_df, user_map, movie_map, impute=False)
        test_i, test_j = np.nonzero(Z_test)
        true_val = Z_test[test_i, test_j]

        for r in r_list:
            for lam in lam_list:
                # Train on the fold’s training split
                W, H, _, _ = method(train_df, user_map, movie_map, r, lam=lam, **ekw)
                Z_approx = np.dot(W, H)

                approxed_val = Z_approx[test_i, test_j]

                rmse = np.sqrt(np.mean((true_val - approxed_val) ** 2))
                cv_errors[(r, lam)].append(rmse)

                print(f"Fold {fold + 1}/{n_splits} | r={r} lam={lam:.3f} → RMSE={rmse:.4f}")

    # Compute mean RMSE for each (r, lam)
    mean_errors = {k: np.mean(v) for k, v in cv_errors.items()}
    best_r, best_lam = min(mean_errors, key=mean_errors.get)

    columns = list(cv_errors.values())
    RMSE_mean = np.column_stack(columns)

    print(f"\nBest pair: r = {best_r}, lam = {best_lam:.3f}  (CV RMSE = {mean_errors[(best_r, best_lam)]:.4f})")
    print(RMSE_mean)
    return (best_r, best_lam), RMSE_mean


# def optimal_lam_finder(file, method, r=14, n_splits=10):
#     df = pd.read_csv(file)
#     unique_users = df["userId"].unique()
#     user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
#     unique_movies = df["movieId"].unique()
#     movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}
#     lam_list = np.linspace(0, 5, 50)
#
#     train_dfs, test_dfs = split_data(file, n_splits=n_splits)
#     # print("data splited")
#
#     RMSE_matrix = np.zeros((n_splits, len(lam_list)), dtype=np.float32)
#
#     for fold, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
#         RMSE_list = []
#
#         Z_test, _, _ = build_rating_matrix(test_df, user_map, movie_map, impute=False)
#         # Z_test = torch.tensor(Z_test)
#
#         # for r in range(1, min(Z.shape[0], Z.shape[1]) + 1):
#         for lam in lam_list:
#             Z_approx, _, _ = method(train_df, user_map, movie_map, lam=lam)
#             # print("Z approximated")
#             Z_approx = np.array(Z_approx)
#
#             test_i, test_j = np.where(Z_test != 0)
#             true_val = Z_test[test_i, test_j]
#             approxed_val = Z_approx[test_i, test_j]
#
#             rmse = np.sqrt(np.mean((true_val - approxed_val) ** 2))
#
#             RMSE_list.append(rmse)
#             print(f"lambda={lam}, fold = {fold + 1}, rmse = {rmse}")
#         RMSE_matrix[fold,] = RMSE_list
#     RMSE_mean = RMSE_matrix.mean(axis=0)
#     lam_best = np.argmin(RMSE_mean)
#     print(lam_best)
#     return lam_best, RMSE_matrix


if __name__ == "__main__":

    data = {
        "userId": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "movieId": [101, 102, 101, 103, 102, 103, 101, 104, 103, 104],
        "rating": [5.0, 0.0, 4.0, 2.0, 3.5, 4.0, 0.0, 1.0, 3.0, 4.5]
    }

    pd.DataFrame(data).to_csv("test.csv", index=False)

    Z, user_map, movie_map = build_rating_matrix("test.csv", impute=weighted_imputation)
    print(Z)
