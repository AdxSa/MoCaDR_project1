import argparse
import pickle
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
                        help="Train mode: 'yes' to train NMF model, 'no' otherwise.")
    parser.add_argument("--predict", type=str, default="no",
                        help="Predict mode: 'yes' to predict ratings, 'no' otherwise.")
    parser.add_argument("--train_file", type=str, default="data/ratings.csv",
                        help="CSV file with training data (userId,movieId,rating).")
    parser.add_argument("--input_file", type=str, default="pred.csv",
                        help="CSV file with (userId,movieId) for predictions.")
    parser.add_argument("--model_path", type=str, default="models_trained/all_models.pkl",
                        help="Path to save/load the trained NMF model.")
    parser.add_argument("--output_file", type=str, default="results/preds.csv",
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
    """
    Entry point of the recommender system script.

    Depending on command-line flags, it can:
      - Train one or more algorithms (NMF, SVD1, SVD2, SGD), optionally searching for the best rank.
      - Save the trained model data (W, H, mappings) to a pickle file.
      - Load the trained model and generate rating predictions for a test set.
      - Save prediction outputs to CSV.

    Workflow:
      1. Parse arguments.
      2. If train mode:
           a. For each selected algorithm, search or fix hyperparameters.
           b. Train final model with chosen hyperparameters.
           c. Serialize all trained models to `--model_path`.
      3. If predict mode:
           a. Load model(s) from `--model_path`.
           b. For each selected algorithm, call `predict_ratings` on `--input_file`.
           c. Write predictions to `--output_file` (one CSV per algorithm if ALL).
    """
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
                    best_hyper, rmse_matrix = hp_finder(args.train_file, method, n_splits, sr, ekw=extra_kw)
                if args.print_rmse_plots.lower() == "yes":
                    plot_rmse(rmse_matrix, alg, sr)
            else:
                r = args.r

            if alg == "SGD":
                best_hyper = {"r": best_hyper[0], "lam": best_hyper[1]}
            else:
                best_hyper = {"r": best_hyper}

            W_approx, H_approx, user_map, movie_map = method(
                args.train_file,
                **best_hyper,
                **extra_kw
            )
            model_data = {alg: {
                "W": W_approx,
                "H": H_approx,
                "user_map": user_map,
                "movie_map": movie_map
            }}

            return model_data

        if args.alg.upper() == "ALL":
            all_model_data = dict()
            for alg in ALGORITHMS["ALL"]:
                all_model_data = all_model_data | training_time(alg)
        else:
            all_model_data = training_time(args.alg.upper())

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

        def pred_time(alg, output_file):
            predictions = predict_ratings(args.input_file, model_data[alg])

            # Save predictions
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, "w") as f:
                f.write("userId,movieId,rating\n")
                for row in predictions:
                    f.write(f"{row['userId']},{row['movieId']},{row['rating']}\n")
            print(f"Predictions with {alg} saved to {output_file}")

        if args.alg.upper() == "ALL" and len(model_data) != 1:
            for alg in ALGORITHMS["ALL"]:
                pred_time(alg, args.output_file + f"/{alg}")
        else:
            pred_time(args.alg.upper(), args.output_file)

if __name__ == "__main__":
    main()