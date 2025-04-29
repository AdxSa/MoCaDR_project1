import matplotlib.pyplot as plt
import os
import numpy as np
from .impute_functions import *
from .train_functions import *
from .helper_functions import optimal_r_finder

#   NMF

# rmse_list = [[0.9768225, 0.959451, 0.9676893, 0.96581763, 0.96715623, 0.966544, 0.9680487, 0.9680791, 0.96805257, 0.9687701, 0.9687976, 0.9689091, 0.96903896, 0.9692207, 0.9693714, 0.9694193, 0.96962637, 0.9695921, 0.96977204, 0.9698381, 0.9701705, 0.97016495, 0.97023195, 0.97043604, 0.9703521, 0.97032005, 0.9704065, 0.97050756, 0.9704634], [0.98548555, 0.96959287, 0.9754305, 0.97356427, 0.97458464, 0.97581154, 0.97529614, 0.9754349, 0.9748774, 0.97624844, 0.97638017, 0.9763991, 0.97664225, 0.97675806, 0.97703016, 0.97711456, 0.9771203, 0.9773303, 0.9773557, 0.977368, 0.97731084, 0.97755575, 0.9775593, 0.97755253, 0.97761005, 0.9774812, 0.97732836, 0.9775037, 0.9775944], [0.9863121, 0.9695507, 0.9741726, 0.9734286, 0.97438216, 0.97469693, 0.9752909, 0.9756198, 0.97586447, 0.9758479, 0.9759609, 0.97588307, 0.9759302, 0.9761522, 0.97647846, 0.97650814, 0.9763809, 0.97662634, 0.97657186, 0.9768618, 0.97688484, 0.97698367, 0.9769933, 0.977019, 0.97718006, 0.9772232, 0.9770166, 0.9771941, 0.97720176], [0.985166, 0.9686618, 0.97664326, 0.97457933, 0.9755513, 0.97596216, 0.9757139, 0.97658443, 0.97680366, 0.97714, 0.97734267, 0.97746885, 0.97753155, 0.9778338, 0.9780062, 0.97809035, 0.9782728, 0.97832876, 0.97843003, 0.97845364, 0.9785093, 0.97865355, 0.9787155, 0.978718, 0.97871023, 0.9787138, 0.978845, 0.97878236, 0.97884375], [0.98114944, 0.9731195, 0.97228926, 0.96989995, 0.9715343, 0.97201705, 0.9732934, 0.97375304, 0.9739094, 0.9740551, 0.9742323, 0.97422576, 0.9744778, 0.9745389, 0.9745502, 0.97454333, 0.97474813, 0.97506446, 0.97486603, 0.97488457, 0.9751145, 0.97506607, 0.97528505, 0.97522336, 0.97537774, 0.9755208, 0.9753535, 0.97549945, 0.9755569], [0.9840963, 0.9645391, 0.97193897, 0.9703169, 0.9724045, 0.9718258, 0.9724262, 0.9729272, 0.9728506, 0.9734411, 0.97350556, 0.97360784, 0.9735018, 0.97377414, 0.9736001, 0.97379446, 0.9738985, 0.97395295, 0.97410166, 0.97412413, 0.97397846, 0.97397095, 0.9740793, 0.9742032, 0.97438765, 0.9742603, 0.9745534, 0.97450775, 0.9747168], [0.9858841, 0.96711, 0.97425187, 0.9723733, 0.9742126, 0.9735099, 0.97410285, 0.9755915, 0.97516227, 0.9752409, 0.97591215, 0.9763251, 0.9762025, 0.9764152, 0.9765619, 0.97680366, 0.97661537, 0.97680694, 0.97683835, 0.9770205, 0.976876, 0.97689223, 0.97692245, 0.9769365, 0.9770995, 0.9771312, 0.9772065, 0.9772561, 0.9774596], [0.97985977, 0.9722543, 0.97082835, 0.9697631, 0.9713123, 0.9716112, 0.97224647, 0.97339827, 0.973303, 0.9733555, 0.97350913, 0.97370845, 0.97378767, 0.9738415, 0.9739111, 0.97394365, 0.97395897, 0.9741792, 0.97435534, 0.9742932, 0.9743583, 0.974447, 0.97463924, 0.9746502, 0.9746957, 0.9747004, 0.9748413, 0.9749138, 0.97521937], [0.98027146, 0.97271127, 0.9712631, 0.96995455, 0.97173965, 0.97141546, 0.9718251, 0.97303075, 0.9733366, 0.9733525, 0.9733606, 0.9734611, 0.9735842, 0.9738116, 0.9740491, 0.9739823, 0.9739674, 0.9740669, 0.97416437, 0.97423637, 0.9743217, 0.9742283, 0.974534, 0.97456425, 0.9743853, 0.9744755, 0.97451323, 0.9746928, 0.9745714], [0.9849817, 0.96735847, 0.97508883, 0.9735883, 0.97484255, 0.9758831, 0.97642905, 0.9759785, 0.9766763, 0.97662055, 0.9769396, 0.977055, 0.9771981, 0.9770602,
# 0.9775487, 0.977359, 0.9773063, 0.977555, 0.97751814, 0.977642, 0.97774607, 0.97774386, 0.97762436, 0.97770333, 0.97793525, 0.9778389, 0.9779816, 0.9781167, 0.97807825]]
#
# rmse_per_r = list(map(list, zip(*rmse_list)))
def plot_rmse(rmse_matrix, alg, folder_path = "plots"):

    rmse_means = rmse_matrix.mean(axis=0)
    best_idx = np.argmin(rmse_means)
    """
        Save boxplot of RMSE values across folds for each r.

        Parameters:
          - rmse_matrix: np.array shape (n_splits, len(rs))
          - alg_name: name of the algorithm (e.g., "SVD2")
          - folder_path: directory where to save the plot
        """
    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.boxplot(
        rmse_matrix,
        patch_artist=True,
        labels=[str(r) for r in range(1, rmse_matrix.shape[1] + 1)],
        showmeans=True,
        meanline=True
    )

    plt.title(f"Boxplot of RMSE per r for {alg}", fontsize=14, pad=15)
    plt.xlabel("Number of components (r)", labelpad=10)
    plt.ylabel("RMSE", labelpad=10)
    # plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.axvline(
        best_idx + 1,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'Best r = {best_idx+1}'
    )

    plt.legend(loc='upper right')
    save_path = os.path.join(folder_path, f"boxplot_{alg}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Boxplot saved to {save_path}")


def plot_impute_diff(train_file, alg_name, alg, alg_atr ,folder_path = "plots", n_splits = 10, sr=20):
    imputes = {"User mean": impute_user_mean, "Movie mean": impute_movie_mean,
               "Global mean": impute_global_mean, "Weighted mean": weighted_imputation}
    os.makedirs(folder_path, exist_ok=True)
    plt.figure(figsize=(12, 6))
    impute = alg_atr["impute"]
    for im in imputes:
        alg_atr["impute"] = imputes[im]
        r, rmse_matrix = optimal_r_finder(train_file, alg, n_splits, sr, ekw=alg_atr)
        plt.plot(
            range(sr),
            rmse_matrix.mean(axis=0),
            label=f"{im}"
        )
        if impute == alg_atr["impute"]:
            r_p, rmse_matrix_p = r, rmse_matrix
    plt.title(f"RMSE comparison for {alg_name} with different imputation methods")
    plt.legend(loc='upper right')
    save_path = os.path.join(folder_path, f"impute_{alg_name}.png")
    plt.savefig(save_path)
    plt.close()
    return r_p, rmse_matrix_p
