import numpy as np
import torch
from sklearn.decomposition import NMF, TruncatedSVD

from .helper_functions import build_rating_matrix, weighted_imputation


def train_nmf_model(
        train_file,
        user_map=None,
        movie_map=None,
        r=5,
        impute=weighted_imputation,
        init="nndsvda",
        max_iter=3000
):
    """
    Train an NMF-based recommender.

    - Reads ratings via build_rating_matrix (with imputation).
    - Applies sklearn.decomposition.NMF to factor Z ≈ W @ H.

    Parameters:
      - train_file (str or DataFrame): CSV or DataFrame of (userId, movieId, rating).
      - user_map, movie_map (dict): Optional pre-built ID→index maps.
      - r (int): Number of latent components (n_components).
      - impute (callable): Function to fill missing entries before factoring.
      - init (str): Initialization method passed to NMF.
      - max_iter (int): Maximum NMF iterations.

    Returns:
      - W (ndarray): User-factor matrix of shape (n_users, r).
      - H (ndarray): Component matrix of shape (r, n_movies).
      - user_map (dict), movie_map (dict): Index mappings.
    """
    Z, user_map, movie_map = build_rating_matrix(
        train_file, user_map, movie_map, impute=impute
    )
    nmf = NMF(n_components=r, init=init, max_iter=max_iter, random_state=42)
    W = nmf.fit_transform(Z)
    H = nmf.components_
    return W, H, user_map, movie_map


def train_svd1_model(
        train_file,
        user_map=None,
        movie_map=None,
        r=14,
        impute=weighted_imputation
):
    """
    Train a one-shot truncated‐SVD recommender.

    - Builds and imputes Z via build_rating_matrix.
    - Uses sklearn TruncatedSVD to compute Z ≈ W @ H in one pass.

    Parameters:
      - train_file (str or DataFrame): Ratings data source.
      - user_map, movie_map (dict): Optional ID→index mappings.
      - r (int): Number of SVD components.
      - impute (callable): Pre‐SVD imputation function.

    Returns:
      - W (ndarray): User-factor matrix of shape (n_users, r).
      - H (ndarray): Component matrix of shape (r, n_movies).
      - user_map (dict), movie_map (dict): Index mappings.
    """
    Z, user_map, movie_map = build_rating_matrix(
        train_file, user_map, movie_map, impute=impute
    )
    svd = TruncatedSVD(n_components=r, random_state=42)
    svd.fit(Z)

    # Σ = diag(singular_values), VT = components_
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_

    # W = Z V Σ^{-1}, H = Σ VT
    W = svd.transform(Z) / svd.singular_values_
    H = Sigma2 @ VT
    return W, H, user_map, movie_map


def train_svd2_model(
        train_file,
        user_map=None,
        movie_map=None,
        r=14,
        impute=weighted_imputation,
        n_iter=3
):
    """
    Train iterative SVD (SVD2) recommender.

    - Initializes Z_approx by imputing missing entries.
    - Repeatedly:
        1) Compute truncated SVD on Z_approx.
        2) Reconstruct Z_approx = W @ H.
        3) Re‐inject original known ratings into Z_approx.

    Parameters:
      - train_file (str or DataFrame): Ratings source.
      - user_map, movie_map (dict): Optional mappings.
      - r (int): Number of SVD components.
      - impute (callable): Initial imputation function.
      - n_iter (int): Number of SVD–reconstruct iterations.

    Returns:
      - W (ndarray): Final user-factor matrix.
      - H (ndarray): Final component matrix.
      - user_map (dict), movie_map (dict): Index mappings.
    """
    Z_orig, user_map, movie_map = build_rating_matrix(
        train_file, user_map, movie_map
    )
    Z_approx, _, _ = build_rating_matrix(
        train_file, user_map, movie_map, impute=impute
    )
    known_mask = Z_orig != 0

    for _ in range(n_iter):
        svd = TruncatedSVD(n_components=r, random_state=42)
        svd.fit(Z_approx)
        Sigma2 = np.diag(svd.singular_values_)
        VT = svd.components_

        W = svd.transform(Z_approx) / svd.singular_values_
        H = Sigma2 @ VT

        Z_approx = W @ H
        Z_approx[known_mask] = Z_orig[known_mask]

    return W, H, user_map, movie_map


def train_sgd_model(
        train_file,
        user_map=None,
        movie_map=None,
        r=1,
        lam=0.0,
        lr=0.01,
        n_epochs=1000,
        optimizer_name="sgd"
):
    """
    Train an SGD-based matrix factorization recommender in PyTorch.

    - Reads Z (no imputation) via build_rating_matrix.
    - Initializes W, H randomly scaled by data mean.
    - Optimizes MSE+λ·(||W||²+||H||²) over observed entries.
    - Prints loss every 10 epochs.

    Parameters:
      - train_file (str or DataFrame): Ratings data.
      - user_map, movie_map (dict): Optional mappings.
      - r (int): Latent dimension.
      - lam (float): L2 regularization weight.
      - lr (float): Learning rate.
      - n_epochs (int): Number of SGD iterations.
      - optimizer_name (str): "sgd" or "adam".

    Returns:
      - W (ndarray): Learned user-factor matrix.
      - H (ndarray): Learned factor–movie matrix.
      - user_map (dict), movie_map (dict): Index mappings.
    """
    # Load data and build Z
    Z_np, user_map, movie_map = build_rating_matrix(train_file, user_map, movie_map)
    n_users, n_items = Z_np.shape

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize W, H based on data scale
    scale = 4 * Z_np[Z_np > 0].mean() / r
    W = torch.rand(n_users, r) * scale
    H = torch.rand(r, n_items) * scale
    W.requires_grad_(True)
    H.requires_grad_(True)

    # Observed indices and values
    us, is_ = np.nonzero(Z_np)
    ratings = torch.tensor(Z_np[us, is_], dtype=torch.float32)
    us = torch.tensor(us, dtype=torch.long)
    is_ = torch.tensor(is_, dtype=torch.long)

    # Select optimizer
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam([W, H], lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD([W, H], lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # SGD loop
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        Wu = W[us]  # (n_obs, r)
        Hi = H[:, is_].T  # (n_obs, r)
        preds = (Wu * Hi).sum(dim=1)
        mse = ((preds - ratings) ** 2).sum()
        reg = lam * (W.norm() ** 2 + H.norm() ** 2)
        loss = mse + reg

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d}  loss={loss.item():.4f}")

        loss.backward()
        optimizer.step()

    return W.detach().numpy(), H.detach().numpy(), user_map, movie_map
