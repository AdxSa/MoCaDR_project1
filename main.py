## Methods of classification and dimensionality reduction
## University of Wrocław
## author: Paweł Lorek

import argparse
import os
import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error
from project1_s340146_s336942.models.train_functions import train_nmf_model, train_svd1_model
from project1_s340146_s336942.models.predict_functions import predict_nmf
from project1_s340146_s336942.models.helper_functions import build_rating_matrix, split_data, optimal_r_finder
from sklearn.decomposition import TruncatedSVD


# Tu zadeklarujemy zmienne globalne


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
        self.r = 14
        self.movie_map = {}
        self.user_map = {}

    def load_data(self, train_file, test_file):

        self.df_train = pd.read_csv(
            train_file,
            dtype={
                "userId":   "int64",
                "movieId":  "int64",
                "rating":  "float64",
                "timestamp": "int64",
            }
        )
        self.df_test  = pd.read_csv(
            test_file,
            dtype={
                "userId": "int64",
                "movieId": "int64",
                "rating": "float64",
                "timestamp": "int64",
            }
        )

        self.df_train.drop(columns=['timestamp'])

        self.df_train["userId"] -= 1
        self.df_train["movieId"] -= 1
        self.df_test["userId"] -= 1
        self.df_test["movieId"] -= 1

        self.users = np.sort(pd.concat([self.df_train["userId"], self.df_test["userId"]], axis=0).unique())

        self.movies = np.sort(pd.concat([self.df_train["movieId"], self.df_test["movieId"]], axis=0).unique())

        self.user_map = {uid: i for i, uid in enumerate(sorted(self.users))}
        self.movie_map = {mid: j for j, mid in enumerate(sorted(self.movies))}

        self.n_users = len(self.user_map)
        self.n_movies = len(self.movie_map)


        self.Z = build_rating_matrix(train_file, impute=True)
        # funkcja build_rating_matrix do przebudowy!!!

        # self.train_matrix, _, _ = build_rating_matrix(train_file)
        # self.test_matrix, _, _ = build_rating_matrix(test_file)

    def NMF(self):
        Z_test = build_rating_matrix(self.df_test, user_map=self.user_map, movie_map=self.movie_map, impute=False)
        Z = build_rating_matrix(self.df_train, user_map=self.user_map, movie_map=self.movie_map, impute=True)
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
        # r_best = np.argmin(RMSE_list)
        Z_approx = train_nmf_model(Z, 2)
        test_i, test_j = np.where(Z_test != 0)
        pred_ratings = Z_approx[test_i, test_j]
        test_ratings = Z_test[test_i, test_j]
        rmse = np.sqrt(np.mean((pred_ratings - test_ratings) ** 2))

        print("2")
        print(rmse)

        Z_approx = train_nmf_model(Z, 14)
        test_i, test_j = np.where(Z_test != 0)
        pred_ratings = Z_approx[test_i, test_j]
        test_ratings = Z_test[test_i, test_j]
        rmse = np.sqrt(np.mean((pred_ratings - test_ratings) ** 2))
        print("4")
        print(rmse)

        Z_approx = train_nmf_model(Z, 14)
        test_i, test_j = np.where(Z_test != 0)
        pred_ratings = Z_approx[test_i, test_j]
        test_ratings = Z_test[test_i, test_j]
        rmse = np.sqrt(np.mean((pred_ratings - test_ratings) ** 2))
        print("8")
        print(rmse)

        Z_approx = train_nmf_model(Z, 14)
        test_i, test_j = np.where(Z_test != 0)
        pred_ratings = Z_approx[test_i, test_j]
        test_ratings = Z_test[test_i, test_j]
        rmse = np.sqrt(np.mean((pred_ratings - test_ratings) ** 2))
        print("14")
        print(rmse)

        return Z_approx, rmse

    def train_SVD1(self):
        r = optimal_r_finder(self.train_file, method=TruncatedSVD)
        svd = TruncatedSVD(n_components=self.r, random_state=42)
        svd.fit(self.Z)
        Sigma2 = np.diag(svd.singular_values_)
        VT = svd.components_

        W = svd.transform(self.Z) / svd.singular_values_
        H = np.dot(Sigma2, VT)
        print(self.Z)
        print(W @ H)
        return W @ H
    def SVD1_predict(self, n_components=14):
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

    available_models = {"NMF", "SVD1"}

    if args.alg.upper() not in available_models:
        print(f"--alg {args.alg.upper()} is not implemented in project.")
        return

    if train_mode:
        print("Training mode activated.")
###########################################################################################
        if args.alg.upper() == "NMF":
            Z_approx, user_map, movie_map = train_nmf_model(args.train_file)
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

        if args.alg.upper() == "SVD1":

            best_r, _ = optimal_r_finder(args.train_file, train_svd1_model)

            Z_approx, user_map, movie_map = train_svd1_model(args.train_file, best_r)
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
        print("Prediction mode activated.")
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
    main()
    # # kf = split_data("project1_s340146_s336942/data/ratings.csv")
    #
    # a = RecommenderSystem()
    # a.load_data("project1_s340146_s336942/data/ratings.csv", "sample_test_with_ratings.csv")
    # a.train_SVD1()
    # # print(optimal_r_finder("project1_s340146_s336942/data/ratings.csv", train_nmf_model))
