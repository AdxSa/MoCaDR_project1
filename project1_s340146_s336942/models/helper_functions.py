
import pandas as pd
import numpy as np
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
def build_rating_matrix(train_file, user_map = None, movie_map = None, impute = False):
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

    df = pd.read_csv(train_file)

    if user_map is None:
        unique_users = df["userId"].unique()
        user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    
    if movie_map is None:
        unique_movies = df["movieId"].unique()
        movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}


    # Extract unique users and movies
    unique_users = df["userId"].unique()
    unique_movies = df["movieId"].unique()

    # Create mappings: userId -> row index, movieId -> column index
    user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    n_users = len(user_map)
    n_movies = len(movie_map)

    # Build matrix Z with zeros for missing ratings
    Z = np.zeros((n_users, n_movies), dtype=np.float32)
    for row in df.itertuples():
        u = row.userId
        m = row.movieId
        rating = row.rating
        i = user_map[u]
        j = movie_map[m]
        Z[i, j] = rating
    # Z = impute_missing_values(Z)
    print('build')
    if impute:
        Z = weighted_imputation(Z)

    return Z, user_map, movie_map

if __name__ == "__main__":

    data = {
        "userId": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "movieId": [101, 102, 101, 103, 102, 103, 101, 104, 103, 104],
        "rating": [5.0, 3.0, 4.0, 2.0, 3.5, 4.0, 2.5, 1.0, 3.0, 4.5]
    }

    pd.DataFrame(data).to_csv("test.csv", index=False)


    Z, user_map, movie_map = build_rating_matrix("test.csv", impute= True)
    print(Z)






