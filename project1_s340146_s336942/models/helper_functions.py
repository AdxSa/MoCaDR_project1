import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
###########################################################################################
# def impute_missing_values(Z):
#     user_medians = np.zeros(Z.shape[0])
#     for i in range(Z.shape[0]):
#         non_zero = Z[i, :][Z[i, :] != 0]
#         user_medians[i] = np.median(non_zero) if non_zero.size > 0 else 0.0

#     movie_medians = np.zeros(Z.shape[1])
#     for j in range(Z.shape[1]):
#         non_zero = Z[:, j][Z[:, j] != 0]
#         movie_medians[j] = np.median(non_zero) if non_zero.size > 0 else 0.0

#     imputed_values = np.round((user_medians[:, np.newaxis] + movie_medians)) / 2
#     Z[Z == 0] = imputed_values[Z == 0]
    
#     return Z
def weighted_imputation(Z):
    user_weights = np.sum(Z != 0, axis=1)
    movie_weights = np.sum(Z != 0, axis=0)
    
    user_medians = np.array([np.median(Z[i][Z[i] != 0]) if np.any(Z[i] != 0) else 0 for i in range(Z.shape[0])])
    movie_medians = np.array([np.median(Z[:, j][Z[:, j] != 0]) if np.any(Z[:, j] != 0) else 0 for j in range(Z.shape[1])])
    
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if Z[i, j] == 0:
                total_weight = user_weights[i] + movie_weights[j]
                if total_weight > 0:
                    Z[i, j] = (user_medians[i] * user_weights[i] + movie_medians[j] * movie_weights[j]) / total_weight
                else:
                    Z[i, j] = np.mean(Z[Z != 0]) if np.any(Z != 0) else 3.0
    return np.round(Z * 2) / 2.0
###########################################################################################
# # def build_rating_matrix(train_file, user_map = None, movie_map = None, impute = False):
#     """
#     Reads a ratings CSV file with columns: userId, movieId, rating.
#     Builds and returns the user–movie matrix Z (missing entries set to 0),
#     along with mappings from userId to row index and movieId to column index.

#     Parameters:
#       - train_file (str): Path to the training CSV file.

#     Returns:
#       - Z (ndarray): Rating matrix of shape (n_users, n_movies).
#       - user_map (dict): Mapping from userId to row index.
#       - movie_map (dict): Mapping from movieId to column index.
#     """

#     df = pd.read_csv(train_file)

#     if user_map is None:
#         unique_users = df["userId"].unique()
#         user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    
#     if movie_map is None:
#         unique_movies = df["movieId"].unique()
#         movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}


#     # Extract unique users and movies
#     unique_users = df["userId"].unique()
#     unique_movies = df["movieId"].unique()

#     # Create mappings: userId -> row index, movieId -> column index
#     user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
#     movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

#     n_users = len(user_map)
#     n_movies = len(movie_map)

#     # Build matrix Z with zeros for missing ratings
#     Z = np.zeros((n_users, n_movies), dtype=np.float32)
#     for row in df.itertuples():
#         u = row.userId
#         m = row.movieId
#         rating = row.rating
#         i = user_map[u]
#         j = movie_map[m]
#         Z[i, j] = rating
#     # Z = impute_missing_values(Z)
#     print('build')
#     if impute:
#         Z = weighted_imputation(Z)

#     return Z, user_map, movie_map


def build_rating_matrix(df, user_map = None, movie_map = None, impute = False):
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

    # df = pd.read_csv(train_file)

    if user_map is None:
        unique_users = df["userId"].unique()
        user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    
    if movie_map is None:
        unique_movies = df["movieId"].unique()
        movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    # # Extract unique users and movies
    # unique_users = df["userId"].unique()
    # unique_movies = df["movieId"].unique()

    # # Create mappings: userId -> row index, movieId -> column index
    # user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    # movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    n_users = len(user_map)
    n_movies = len(movie_map)

    # Build matrix Z with zeros for missing ratings
    Z = np.zeros((n_users, n_movies), dtype=np.float32)
    for row in df.itertuples():
        i = user_map[row.userId]
        j = movie_map[row.movieId]
        Z[i, j] = row.rating

    print('build')
    if impute:
        Z = weighted_imputation(Z)

    return Z#, user_map, movie_map

def split_data(file):
    df = pd.read_csv(file)
    X = df.drop(columns = ['userId', 'movieId'])
    # y = df['rating']
    kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
    train_dfs = []
    test_dfs = []
    
    for train_idx, test_idx in kf.split(X):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    
    return train_dfs, test_dfs


def optimal_r_finder(file, method):
    train_dfs, test_dfs = split_data(file)
    RMSE_fold = []
    train_df = train_dfs[0]
    test_df = test_dfs[0]

    # train_df["userId"]  -=1
    # train_df["movieId"] -=1
    # test_df["userId"]  -=1
    # test_df["movieId"] -=1

    users = np.sort(pd.concat([train_df["userId"], test_df["userId"]], axis=0).unique())

    movies = np.sort(pd.concat([train_df["movieId"], test_df["movieId"]], axis=0).unique())

    user_map = {uid: i for i, uid in enumerate(sorted(users))}
    movie_map = {mid: j for j, mid in enumerate(sorted(movies))}

    for (train_df, test_df) in zip(train_dfs, test_dfs):
        RMSE_list = []       

        Z = build_rating_matrix(train_df, user_map, movie_map, impute=True)
        Z_test = build_rating_matrix(test_df, user_map, movie_map, impute=False)
        # for r in range(1, min(Z.shape[0], Z.shape[1]) + 1):
        for r in range(1, 30):
            Z_approx = method(Z, r)
            test_i, test_j = np.where(Z_test != 0)  
            pred_ratings = Z_approx[test_i, test_j]  
            test_ratings = Z_test[test_i, test_j]
            rmse = np.sqrt(np.mean((pred_ratings - test_ratings) ** 2))
            RMSE_list.append(rmse)
        RMSE_fold.append(RMSE_list)
        # r_best = np.argmin(RMSE_list)
    return RMSE_fold


# TO DAŁ CZAT:

# def build_rating_matrix(df, user_map=None, movie_map=None, impute=False):
#     if user_map is None:
#         unique_users = df["userId"].unique()
#         user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    
#     if movie_map is None:
#         unique_movies = df["movieId"].unique()
#         movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

#     filtered_df = df[
#         df["userId"].isin(user_map.keys()) & 
#         df["movieId"].isin(movie_map.keys())
#     ]

#     # Build matrix using final mappings
#     n_users = len(user_map)
#     n_movies = len(movie_map)
#     Z = np.zeros((n_users, n_movies), dtype=np.float32)
    
#     for row in filtered_df.itertuples():
#         i = user_map[row.userId]
#         j = movie_map[row.movieId]
#         Z[i, j] = row.rating

#     if impute:
#         Z = weighted_imputation(Z)
        
#     return Z, user_map, movie_map

# def optimal_r_finder(file, method):
#     train_dfs, test_dfs = split_data(file)
#     RMSE_fold = []
    
#     for train_df, test_df in zip(train_dfs, test_dfs):
#         # Build training matrix with its mappings
#         Z_train, train_user_map, train_movie_map = build_rating_matrix(train_df, impute=True)
        
#         # Build test matrix using TRAINING mappings
#         Z_test, _, _ = build_rating_matrix(test_df, 
#                                          user_map=train_user_map,
#                                          movie_map=train_movie_map,
#                                          impute=False)
        
#         # Get test ratings that exist in both sets
#         test_i, test_j = np.where(Z_test != 0)
        
#         # Skip fold if no valid test ratings
#         if len(test_i) == 0:
#             continue
            
#         min_rank = min(Z_train.shape)
#         RMSE_list = []
        
#         for r in range(1, min(min_rank, 20) + 1):  # Cap at 20 for practicality
#             # Factorize the TRAINING matrix
#             Z_approx = method(Z_train, r)
            
#             # Calculate predictions only for valid test indices
#             valid_mask = (test_i < Z_approx.shape[0]) & (test_j < Z_approx.shape[1])
#             valid_i = test_i[valid_mask]
#             valid_j = test_j[valid_mask]
            
#             if len(valid_i) == 0:
#                 rmse = np.nan
#             else:
#                 pred_ratings = Z_approx[valid_i, valid_j]
#                 test_ratings = Z_test[valid_i, valid_j]
#                 rmse = np.sqrt(np.mean((pred_ratings - test_ratings) ** 2))
            
#             RMSE_list.append(rmse)
        
#         RMSE_fold.append(RMSE_list)
    
#     return RMSE_fold

############################################################################################################

if __name__ == "__main__":

    data = {
        "userId": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "movieId": [101, 102, 101, 103, 102, 103, 101, 104, 103, 104],
        "rating": [5.0, 3.0, 4.0, 2.0, 3.5, 4.0, 2.5, 1.0, 3.0, 4.5]
    }

    pd.DataFrame(data).to_csv("test.csv", index=False)


    Z, user_map, movie_map = build_rating_matrix("test.csv", impute= True)
    print(Z)






