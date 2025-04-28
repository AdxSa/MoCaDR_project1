## Methods of classification and dimensionality reduction
## University of Wrocław
## author: Paweł Lorek

import argparse
import os
import pickle
import pandas as pd
import numpy as np
# import torch
from sklearn.metrics import root_mean_squared_error
from project1_s340146_s336942.modules.train_functions import train_nmf_model, train_svd1_model, train_svd2_model, train_sgd_model
from project1_s340146_s336942.modules.predict_functions import predict_ratings
from project1_s340146_s336942.modules.helper_functions import optimal_r_finder, optimal_lam_finder
from sklearn.decomposition import TruncatedSVD
from project1_s340146_s336942.modules.plot_makers import plot_rmse


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


def main():
    args = parse_arguments()
    train_mode = (args.train.lower() == "yes")
    predict_mode = (args.predict.lower() == "yes")

    ALGORITHMS = {
        "NMF": {
            "trainer": train_nmf_model,
            "n_splits": 10,
            "train_kwargs": {"init": "nndsvda", "max_iter": 3000}
        },
        "SVD1": {
            "trainer": train_svd1_model,
            "n_splits": 10,
            "train_kwargs": {}
        },
        "SVD2": {
            "trainer": train_svd2_model,
            "n_splits": 10,
            "train_kwargs": {"n_iter": 10}
        },
        "SGD": {
            "trainer": train_sgd_model,
            "n_splits": 10,
            "train_kwargs": {"lam": 0.1, "lr": 0.01, "n_epochs": 5000, "optimizer_name": "adam"}
        },
    }

    if args.alg.upper() not in ALGORITHMS:
        print(f"--alg {args.alg.upper()} is not implemented in project.")
        return

    if train_mode:
        alg = args.alg.upper()
        entry = ALGORITHMS[alg]
        trainer = entry["trainer"]
        n_splits = entry["n_splits"]
        extra_kw = entry.get("train_kwargs", {})

        print(f"Training mode activated.  Algorithm = {alg}")

        # 2) find the best hyper‐param r
        best_r, rmse_matrix = optimal_r_finder(args.train_file, trainer, n_splits)
        print(f" -> Best r = {best_r}")
        print(min(rmse_matrix.mean(axis=0)))
        plot_rmse(rmse_matrix, alg)


        Z_approx, user_map, movie_map = trainer(
            args.train_file,
            r=best_r,
            **extra_kw
        )
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
        predictions = predict_ratings(args.test_file, model_data)

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
    # a.SGD()
    # print(optimal_lam_finder("project1_s340146_s336942/data/ratings.csv", train_sgd_model))
