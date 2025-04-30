import os
import matplotlib.pyplot as plt
from .impute_functions import *
from .helper_functions import optimal_r_finder


def plot_rmse(rmse_matrix, alg_name, r_range, output_dir="plots"):
    """
    Create and save a boxplot of RMSE distributions across CV folds for each rank r.

    - Computes per-rank mean RMSE to highlight the best r.
    - Draws a vertical line at the best-rank position.
    - Saves figure as PNG under `output_dir/boxplot_{alg_name}.png`.

    Parameters:
      - rmse_matrix (ndarray): shape (n_folds, n_ranks), RMSE values.
      - alg_name (str): Name of the algorithm (used in title and filename).
      - r_range (tuple): (min_r, max_r) inclusive range of ranks corresponding to columns of rmse_matrix.
      - output_dir (str): Directory in which to save the plot (created if needed).
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Compute mean RMSE across folds for each rank
    rmse_means = rmse_matrix.mean(axis=0)
    best_idx = int(np.argmin(rmse_means))

    # Build list of rank labels
    min_r, max_r = r_range
    rank_labels = [str(r) for r in range(min_r, max_r + 1)]

    plt.figure(figsize=(12, 6))
    plt.boxplot(
        rmse_matrix,
        patch_artist=True,
        labels=rank_labels,
        showmeans=True,
        meanline=True
    )
    plt.title(f"RMSE per Rank for {alg_name}", fontsize=14, pad=15)
    plt.xlabel("Rank (r)", labelpad=10)
    plt.ylabel("RMSE", labelpad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Mark the best rank
    plt.axvline(
        best_idx + 1,
        linestyle="--",
        linewidth=1.5,
        label=f"Best r = {min_r + best_idx}"
    )
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Save to file and close
    save_path = os.path.join(output_dir, f"boxplot_{alg_name}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[plot_rmse] Saved boxplot to: {save_path}")


def plot_impute_diff(
    train_file,
    alg_name,
    method,
    method_kwargs,
    output_dir="plots",
    n_splits=10,
    max_r=20
):
    """
    Compare RMSE curves of a matrix-factorization method using different imputation strategies.

    - Iterates over four imputation functions, plugging each into `method_kwargs['impute']`.
    - Uses `optimal_r_finder` to get CV-RMSE matrix for each imputer.
    - Overlays their mean-RMSE-vs-rank curves on a single plot.
    - Returns the best-rank and RMSE matrix corresponding to the original imputer.

    Parameters:
      - train_file (str): Path to the ratings CSV.
      - alg_name (str): Label for plot titles and filenames.
      - method (callable): Matrix-factorization training function.
      - method_kwargs (dict): Keyword args passed to `method`, must include key `"impute"`.
      - output_dir (str): Directory for saving plot.
      - n_splits (int): Number of CV folds for RMSE evaluation.
      - max_r (int): Maximum rank to evaluate (assumes ranks 1..max_r).

    Returns:
      - best_r_original (int): Best rank found for the original imputer.
      - rmse_matrix_original (ndarray): RMSE matrix for the original imputer.
    """
    # Available imputation options
    imputers = {
        "User mean": impute_user_mean,
        "Movie mean": impute_movie_mean,
        "Global mean": impute_global_mean,
        "Weighted mean": weighted_imputation,
    }

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))

    # Remember original imputer to identify its results later
    original_imputer = method_kwargs.get("impute")

    best_r_original = None
    rmse_matrix_original = None

    # Evaluate RMSE for each imputer
    for label, imputer in imputers.items():
        method_kwargs["impute"] = imputer
        best_r, rmse_matrix = optimal_r_finder(
            train_file,
            method,
            n_splits=n_splits,
            sr=(1, max_r),
            ekw=method_kwargs
        )

        # Plot mean RMSE vs. rank
        mean_curve = rmse_matrix.mean(axis=0)
        plt.plot(
            range(1, max_r + 1),
            mean_curve,
            label=label
        )

        # Capture the results for the original imputer
        if imputer is original_imputer:
            best_r_original = best_r
            rmse_matrix_original = rmse_matrix

    plt.title(f"RMSE vs. Rank for {alg_name} with Different Imputers")
    plt.xlabel("Rank (r)")
    plt.ylabel("Mean CV RMSE")
    plt.xticks(np.arange(1, max_r + 1, step=1))
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Save the comparison plot
    save_path = os.path.join(output_dir, f"impute_comparison_{alg_name}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[plot_impute_diff] Saved comparison plot to: {save_path}")
    return best_r_original, rmse_matrix_original
