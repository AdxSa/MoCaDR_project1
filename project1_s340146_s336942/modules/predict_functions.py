import pandas as pd


def predict_ratings(test_file, model_data, default_rating=0.0):
    """
    Generate rating predictions for user–movie pairs from a trained factorization model.

    - Reads a CSV or DataFrame with columns: userId, movieId.
    - Uses W and H from model_data to reconstruct Z_approx = W @ H.
    - Looks up each (userId, movieId) in the reconstructed matrix; if unseen, uses default_rating.
    - Rounds each predicted rating to the nearest 0.5.

    Parameters:
      - test_file (str or pd.DataFrame): Path to CSV or DataFrame containing the test pairs.
      - model_data (dict): Contains keys:
          • "W": user-factor matrix of shape (n_users, r)
          • "H": factor-movie matrix of shape (r, n_movies)
          • "user_map": mapping userId → row index in W
          • "movie_map": mapping movieId → column index in H
      - default_rating (float): Rating to assign when userId or movieId is unseen.

    Returns:
      - predictions (list of dict): Each dict has keys "userId", "movieId", and "rating".
    """
    # Load test pairs
    if isinstance(test_file, str):
        df = pd.read_csv(test_file)
    else:
        df = test_file

    # Reconstruct the full rating matrix approximation
    W = model_data["W"]
    H = model_data["H"]
    Z_approx = W @ H

    # Retrieve the user/movie index mappings
    user_map = model_data["user_map"]
    movie_map = model_data["movie_map"]

    predictions = []
    for row in df.itertuples(index=False):
        uid, mid = row.userId, row.movieId

        # If both user and movie were seen during training, look up the approximation
        if uid in user_map and mid in movie_map:
            i = user_map[uid]
            j = movie_map[mid]
            raw_pred = Z_approx[i, j]
        else:
            # Unseen user or movie: fall back to default_rating
            raw_pred = default_rating

        # Round to nearest 0.5
        rating = round(raw_pred * 2) / 2

        predictions.append({
            "userId": uid,
            "movieId": mid,
            "rating": rating
        })

    return predictions
