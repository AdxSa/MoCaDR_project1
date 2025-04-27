import numpy as np
# import torch
from sklearn.decomposition import NMF, TruncatedSVD
from .helper_functions import build_rating_matrix


def train_nmf_model(train_file, user_map=None, movie_map=None, r=5, init='nndsvda', max_iter=3000):
    """
    Reads the ratings CSV file, builds the rating matrix using build_rating_matrix,
    performs NMF, and returns the approximated rating matrix along with mappings.

    Parameters:
      - train_file (str): Path to the training CSV file.
      - n_components (int): Rank for the NMF decomposition.

    Returns:
      - Z_approx (ndarray): Approximated rating matrix from NMF.
      - user_map (dict): Mapping from userId to row index.
      - movie_map (dict): Mapping from movieId to column index.
    """
    Z, user_map, movie_map = build_rating_matrix(train_file, user_map, movie_map, impute=True)

    model = NMF(n_components=r, init=init, max_iter=max_iter, random_state=42)

    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.dot(W, H)
    print('train')

    return Z_approx, user_map, movie_map


def train_svd1_model(train_file, user_map=None, movie_map=None, r=14):

    Z, user_map, movie_map = build_rating_matrix(train_file, user_map, movie_map, impute=True)
    svd = TruncatedSVD(n_components=r, random_state=42)
    svd.fit(Z)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_

    W = svd.transform(Z) / svd.singular_values_
    H = np.dot(Sigma2, VT)
    Z_approx = np.dot(W, H)
    # print(Z_approx)
    # print('train')
    return Z_approx, user_map, movie_map

def train_sgd_model(train_file, user_map=None, movie_map=None, r=14, lam=0, lr=0.0001, n_epochs=1000, optimizer_name="sgd"):
    Z, user_map, movie_map = build_rating_matrix(train_file, user_map, movie_map)
    n, d = Z.shape
    W = torch.randn(n, r, requires_grad=True)
    H = torch.randn(r, d, requires_grad=True)
    Z = torch.tensor(Z)
    mask = (Z > 0).nonzero(as_tuple=True)

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam([W, H], lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD([W, H], lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        # print(W)
        Z_hat = W @ H
        diff = (Z - Z_hat)[mask]
        # print(diff)
        loss = torch.sum(diff ** 2) + lam * (torch.norm(W, 'fro') ** 2 + torch.norm(H, 'fro') ** 2)

        loss.backward()
        optimizer.step()

    W = W.detach()
    H = H.detach()

    Z_approx = W @ H

    return Z_approx, user_map, movie_map


def train_svd2_model(train_file, user_map=None, movie_map=None, r=14, n_iter=3):
    """
    SVD2: iterative scheme
      Z_with_zeros = original matrix (0 where missing)
      Z_approx     = weighted‐imputed start
      repeat n_iter times:
        1) low‐rank SVD of Z_approx
        2) reconstruct Z_approx = W @ H
        3) re‐inject original (non‐zero) ratings into Z_approx
    """
    Z_with_zeros, user_map, movie_map = build_rating_matrix(train_file, user_map, movie_map)
    Z_approx, _, _ = build_rating_matrix(train_file, user_map, movie_map, impute=True)

    not_missing = (Z_with_zeros != 0)

    for t in range(n_iter):

        svd = TruncatedSVD(n_components=r, random_state=42)

        svd.fit(Z_approx)
        Sigma2 = np.diag(svd.singular_values_)
        VT = svd.components_

        W = svd.transform(Z_approx) / svd.singular_values_
        H = np.dot(Sigma2, VT)
        Z_approx = np.dot(W, H)

        Z_approx[not_missing] = Z_with_zeros[not_missing]

        # optional debug
        # rmse_known = np.sqrt(np.mean((Z_with_zeros[not_missing] -
        #                              Z_approx[not_missing])**2))
        # print(f"[SVD2] iter {t+1}/{n_iter}: RMSE on known = {rmse_known:.4f}")

    return Z_approx, user_map, movie_map
