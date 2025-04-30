import argparse
import pickle
import os
from modules.predict_functions import *
from modules.helper_functions import *
from modules.plot_functions import *
from modules.train_functions import *

def parse_arguments():
    """
    Parse command-line arguments for the recommender system.

    Returns:
        argparse.Namespace: Contains the following attributes:
            - train (str): "yes"/"no" flag to trigger training.
            - predict (str): "yes"/"no" flag to trigger prediction.
            - train_file (str): Path to CSV with training data (userId,movieId,rating).
            - input_file (str): Path to CSV with (userId,movieId) pairs for prediction.
            - model_path (str): Path to save or load the trained model pickle.
            - output_file (str): Path to write the prediction results CSV.
            - alg (str): Algorithm choice ("ALL", "NMF", "SVD1", "SVD2", "SGD").
            - r (int): Fixed rank; if zero, performs hyperparameter search.
            - print_rmse_plots (str): "yes"/"no" to save RMSE boxplots.
            - print_impute_plot (str): "yes"/"no" to save imputation comparison plots.
    """
    parser = argparse.ArgumentParser(description="simple recommender system")
    parser.add_argument("--train", type=str, default="no",
                        help="Train mode: 'yes' to train models, 'no' otherwise.")
    parser.add_argument("--predict", type=str, default="no",
                        help="Predict mode: 'yes' to predict ratings, 'no' otherwise.")
    parser.add_argument("--train_file", type=str, default="project1_s340146_s336942/data/ratings.csv",
                        help="CSV file with training data (userId,movieId,rating).")
    parser.add_argument("--input_file", type=str, default="pred.csv",
                        help="CSV file with (userId,movieId) for predictions.")
    parser.add_argument("--model_path", type=str, default="project1_s340146_s336942/models_trained/all_models.pkl",
                        help="Path to save/load the trained model(s) pickle.")
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
        "ALL": ["NMF", "SVD1", "SVD2", "SGD"],
        "NMF": {
            "hp_finder": optimal_r_finder,
            "method": train_nmf_model,
            "n_splits": 10,
            "sr": (1,20),
            "train_kwargs": {"init": "nndsvda", "max_iter": 1000, "impute": weighted_imputation}
        },
        "SVD1": {
            "hp_finder": optimal_r_finder,
            "method": train_svd1_model,
            "n_splits": 10,
            "sr": (1,20),
            "train_kwargs": {"impute": weighted_imputation}
        },
        "SVD2": {
            "hp_finder": optimal_r_finder,
            "method": train_svd2_model,
            "n_splits": 10,
            "sr": (1,20),
            "train_kwargs": {"impute": weighted_imputation, "n_iter": 5}
        },
        "SGD": {
            "hp_finder": optimal_sgd_hyperparams,
            "method": train_sgd_model,
            "n_splits": 10,
            "sr": (10,30),
            "train_kwargs": {"lr": 0.01, "n_epochs": 50, "optimizer_name": "adam"}
        },
    }

    if args.alg.upper() not in ALGORITHMS:
        print(f"--alg {args.alg.upper()} is not implemented in project.")
        return

    if train_mode:
        def training_time(alg):
            entry = ALGORITHMS[alg]
            method = entry["method"]
            n_splits = entry.get("n_splits")
            sr = entry.get("sr")
            hp_finder = entry.get("hp_finder")
            extra_kw = entry.get("train_kwargs", {})

            print(f"Training mode activated. Algorithm = {alg}")


            if not args.r and hp_finder is not None:
                if args.print_impute_plot.lower() == "yes":
                    best_hyper, rmse_matrix = plot_impute_diff(
                        args.train_file, alg, method, extra_kw, n_splits=n_splits, sr=sr)
                else:
                    print("Searching for optimal r parameter")
                    best_hyper, rmse_matrix = hp_finder(
                        args.train_file, method, n_splits, sr, ekw=extra_kw)
                if args.print_rmse_plots.lower() == "yes":
                    plot_rmse(rmse_matrix, alg, sr)
            else:
                best_hyper = args.r if alg != "SGD" else (args.r, None)

            if alg == "SGD":
                hyperparams = {"r": best_hyper[0], "lam": best_hyper[1]}
            else:
                hyperparams = {"r": best_hyper}


            W_approx, H_approx, user_map, movie_map = method(
                args.train_file,
                **hyperparams,
                **extra_kw
            )
            return {alg: {"W": W_approx, "H": H_approx, "user_map": user_map, "movie_map": movie_map}}


        if args.alg.upper() == "ALL":
            all_model_data = {}
            for alg in ALGORITHMS["ALL"]:
                model_data = training_time(alg)
                all_model_data.update(model_data)

            base, ext = os.path.splitext(args.model_path)
            for alg, data in all_model_data.items():
                model_file = f"{base}_{alg}{ext}"
                os.makedirs(os.path.dirname(model_file), exist_ok=True)
                with open(model_file, "wb") as f:
                    pickle.dump({alg: data}, f)
                print(f"Model {alg} saved to {model_file}")
        else:
            model_data = training_time(args.alg.upper())
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            with open(args.model_path, "wb") as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {args.model_path}")

    if predict_mode:
        print("Prediction mode activated.")
        if not os.path.exists(args.model_path):
            print("Model file does not exist. Please run training first.")
            return


        model_data = {}
        if args.alg.upper() == "ALL":
            base, ext = os.path.splitext(args.model_path)
            for alg in ALGORITHMS["ALL"]:
                model_file = f"{base}_{alg}{ext}"
                with open(model_file, "rb") as f:
                    model_data.update(pickle.load(f))
        else:
            with open(args.model_path, "rb") as f:
                model_data = pickle.load(f)


        def pred_time(alg, out_file):
            predictions = predict_ratings(args.input_file, model_data[alg])
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, "w") as f:
                f.write("userId,movieId,rating\n")
                for row in predictions:
                    f.write(f"{row['userId']},{row['movieId']},{row['rating']}\n")
            print(f"Predictions with {alg} saved to {out_file}")


        if args.alg.upper() == "ALL":
            for alg in ALGORITHMS["ALL"]:
                out_file = args.output_file.replace(ext, f"_{alg}{ext}")
                pred_time(alg, out_file)
        else:
            pred_time(args.alg.upper(), args.output_file)

if __name__ == "__main__":
    main()
