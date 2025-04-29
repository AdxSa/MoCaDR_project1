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
from project1_s340146_s336942.modules.helper_functions import optimal_r_finder, optimal_sgd_hyperparams
from sklearn.decomposition import TruncatedSVD
from project1_s340146_s336942.modules.plot_functions import plot_rmse,plot_impute_diff
from project1_s340146_s336942.modules.impute_functions import *

# Tu zadeklarujemy zmienne globalne


def parse_arguments():
    parser = argparse.ArgumentParser(description="simple recommender system")
    parser.add_argument("--train", type=str, default="no",
                        help="Train mode: 'yes' to train NMF model, 'no' otherwise.")
    parser.add_argument("--predict", type=str, default="no",
                        help="Predict mode: 'yes' to predict ratings, 'no' otherwise.")
    parser.add_argument("--train_file", type=str, default="project1_s340146_s336942/data/ratings.csv",
                        help="CSV file with training data (userId,movieId,rating).")
    parser.add_argument("--input_file", type=str, default="pred.csv",
                        help="CSV file with (userId,movieId) for predictions.")
    parser.add_argument("--model_path", type=str, default="project1_s340146_s336942/models_trained/all_models.pkl",
                        help="Path to save/load the trained NMF model.")
    parser.add_argument("--output_file", type=str, default="project1_s340146_s336942/results/preds.csv",
                        help="Where to save predictions.")
    parser.add_argument("--alg", type=str, default="ALL",
                        help="Algorithm to use.")

    parser.add_argument("--r", type=int, default=0,
                        help="r to use for training (default -> searching for best r)")
    parser.add_argument("--print_rmse_plots", type=str, default="no",
                        help="plot printing: 'yes' to save algorithm rmse plot")
    parser.add_argument("--print_impute_plot", type=str, default="no",
                        help="plot printing: 'yes' to save algorithm impute plot")
    return parser.parse_args()


def main():
    args = parse_arguments()
    train_mode = (args.train.lower() == "yes")
    predict_mode = (args.predict.lower() == "yes")

    ALGORITHMS = {
        "ALL": {
            "NMF":"NMF",
            "SVD1":"SVD1",
            "SVD2":"SVD2",
            "SGD":"SGD"
        },
        "NMF": {
            "hp_finder": optimal_r_finder,
            "method": train_nmf_model,
            "n_splits": 10,
            "sr": 20,
            "train_kwargs": {"init": "nndsvda", "max_iter": 1000, "impute": weighted_imputation}
        },
        "SVD1": {
            "hp_finder": optimal_r_finder,
            "method": train_svd1_model,
            "n_splits": 10,
            "sr": 20,
            "train_kwargs": {"impute": weighted_imputation}
        },
        "SVD2": {
            "hp_finder": optimal_r_finder,
            "method": train_svd2_model,
            "n_splits": 10,
            "sr": 20,
            "train_kwargs": {"impute": weighted_imputation, "n_iter": 5}
        },
        "SGD": {
            "hp_finder": optimal_sgd_hyperparams,
            "method": train_sgd_model,
            "n_splits": 10,
            "sr": 20,
            "train_kwargs": { "lr": 0.01, "n_epochs": 50, "optimizer_name": "adam"}
        },
    }

    if args.alg.upper() not in ALGORITHMS:
        print(f"--alg {args.alg.upper()} is not implemented in project.")
        return



    if train_mode:
        def training_time(alg):
            entry = ALGORITHMS[alg]
            method = entry["method"]
            n_splits = entry["n_splits"]
            sr = entry["sr"]
            hp_finder = entry["hp_finder"]
            extra_kw = entry.get("train_kwargs", {})

            print(f"Training mode activated.  Algorithm = {alg}")

            if not args.r:
                if args.print_impute_plot.lower() == "yes":
                    best_hyper, rmse_matrix = plot_impute_diff(args.train_file, alg, method, extra_kw,
                                                               n_splits=n_splits, sr=sr)

                else:
                    print(f"Searching for optimal r parameter")
                    # r, rmse_matrix = optimal_r_finder(args.train_file, method, n_splits, sr, ekw=extra_kw)
                    best_hyper, rmse_matrix = hp_finder(args.train_file, method, n_splits, sr, ekw=extra_kw)
                print(f" -> Best hyperparameters = {best_hyper}")
                print(min(rmse_matrix.mean(axis=0)))
                if args.print_rmse_plots.lower() == "yes":
                    plot_rmse(rmse_matrix, alg)
            else:
                r = args.r

            if alg == "SGD":
                best_hyper = {"r": best_hyper[0], "lam": best_hyper[1]}
            else: best_hyper = {"r": best_hyper}

            Z_approx, user_map, movie_map = method(
                args.train_file,
                **best_hyper,
                **extra_kw
            )
            model_data = {
                "Z_approx": Z_approx,
                "user_map": user_map,
                "movie_map": movie_map
            }

            return model_data

        if args.alg.upper() =="ALL":
            all_model_data = {}
            for alg in ALGORITHMS["ALL"]:
                all_model_data[alg] = training_time(alg)
        else: all_model_data = training_time(args.alg.upper())

        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        with open(args.model_path, "wb") as f:
            pickle.dump(all_model_data, f)

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
