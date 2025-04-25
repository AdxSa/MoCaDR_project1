## Methods of classification and dimensionality reduction
## University of Wrocław
## author: Paweł Lorek

import argparse
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from project1_s340146_s336942.models.train_functions import train_nmf_model
from project1_s340146_s336942.models.predict_functions import predict_nmf
from project1_s340146_s336942.models.helper_functions import build_rating_matrix




def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple NMF-based Recommender")
    parser.add_argument("--train", type=str, default="no",
                        help="Train mode: 'yes' to train NMF model, 'no' otherwise.")
    parser.add_argument("--predict", type=str, default="no",
                        help="Predict mode: 'yes' to predict ratings, 'no' otherwise.")
    parser.add_argument("--train_file", type=str, default="data/ratings.csv",
                        help="CSV file with training data (userId,movieId,rating).")
    parser.add_argument("--test_file", type=str, default="data/test_file.csv",
                        help="CSV file with (userId,movieId) for predictions.")
    parser.add_argument("--model_path", type=str, default="models_trained/nmf_model.pkl",
                        help="Path to save/load the trained NMF model.")
    parser.add_argument("--output_file", type=str, default="predictions/preds.csv",
                        help="Where to save predictions.")
    parser.add_argument("--alg", type=str, default="NMF",
                        help="Algorithm to use (only 'NMF' is implemented).")
    return parser.parse_args()

###########################################################################################
class RecommenderSystem:

    def __init__(self):
        self.n_movies = 0
        self.n_users = 0
        self.train_file = []
        self.test_file = []

    def load_data(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file

        df_train = pd.read_csv(
            self.train_file,
            dtype={
                "userId": "int64",
                "movieId": "int64",
                "rating": "float64",
                "timestamp": "int64",
            }
        )
        df_test = pd.read_csv(
            self.test_file,
            dtype={
                "userId": "int64",
                "movieId": "int64",
                "rating": "float64",
                "timestamp": "int64",
            }
        )

        df_train.drop(columns=['timestamp'])

        df_train["userId"] -= 1
        df_train["movieId"] -= 1
        df_test["userId"] -= 1
        df_test["movieId"] -= 1

        self.users = np.sort((df_train["userId"] + df_test["userId"]).unique())
        self.movies = np.sort((df_train["movieId"] + df_test["movieId"]).unique())

        user_map = {uid: i for i, uid in enumerate(sorted(self.users))}
        movie_map = {mid: j for j, mid in enumerate(sorted(self.movies))}

        self.n_users = len(user_map)
        self.n_movies = len(movie_map)

        # funkcja build_rating_matrix do przebudowy!!!

        # self.train_matrix, _, _ = build_rating_matrix(train_file)
        # self.test_matrix, _, _ = build_rating_matrix(test_file)

    def NMF(self):
        Z_test, user_map, movie_map = build_rating_matrix(self.test_file, impute=False)
        Z, user_map, movie_map = build_rating_matrix(self.train_file, impute=True)
        # RMSE_list = []
        # print(self.n_movies)
        # print(self.n_users)
        # for r in range(1, min(self.n_users, self.n_movies) + 1):
        #     Z_approx = train_nmf_model(Z, r)
        #     test_i, test_j = np.where(Z_test != 0)
        #     pred_ratings = Z_approx[test_i, test_j]
        #     test_ratings = Z_test[test_i, test_j]
        #     rmse = np.sqrt(np.mean((pred_ratings - test_ratings) ** 2))
        #     RMSE_list.append(rmse)
        # RMSE_list.append(root_mean_squared_error(Z_approx, Z_test))
        r_best, best_error = optimal_r_finder(Z, Z_test, train_nmf_model)
        Z_approx = train_nmf_model(Z, r_best)
        print(r_best)
        print(best_error)

        return Z_approx, best_error

    def SVD1(self):
        pass

    def SVD2(self):
        pass

    def SGD(self):
        pass

    def predict(self):
        pass


###########################################################################################
def main():
    args = parse_arguments()
    train_mode = (args.train.lower() == "yes")
    predict_mode = (args.predict.lower() == "yes")

    if args.alg.upper() != "NMF":
        print("Only --alg NMF is implemented in this demo.")
        return

    if train_mode:
        print("Training mode activated (NMF).")
###########################################################################################

        Z, _, _ = build_rating_matrix(args.train_file)
        train_nmf_mse_list = []
        for r in range():
            Z_approx, user_map, movie_map = train_nmf_model(args.train_file, r)
            train_nmf_mse_list.append()
###########################################################################################
        # Save the model
        model_data = {
            "Z_approx": Z_approx,
            "user_map": user_map,
            "movie_map": movie_map
        }
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        with open(args.model_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {args.model_path}")

    if predict_mode:
        print("Prediction mode activated (NMF).")
        if not os.path.exists(args.model_path):
            print("Model file does not exist. Please run training first.")
            return
        with open(args.model_path, "rb") as f:
            model_data = pickle.load(f)
        predictions = predict_nmf(args.test_file, model_data)

        # Save predictions
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            f.write("userId,movieId,rating\n")
            for row in predictions:
                f.write(f"{row['userId']},{row['movieId']},{row['rating']}\n")
        print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    pass
