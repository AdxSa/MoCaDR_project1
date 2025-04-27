import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import NMF, TruncatedSVD
from .helper_functions import build_rating_matrix


# Wersja oryginalna:

# def train_nmf_model(train_file, n_components=5):
#     """
#     Reads the ratings CSV file, builds the rating matrix using build_rating_matrix,
#     performs NMF, and returns the approximated rating matrix along with mappings.

#     Parameters:
#       - train_file (str): Path to the training CSV file.
#       - n_components (int): Rank for the NMF decomposition.

#     Returns:
#       - Z_approx (ndarray): Approximated rating matrix from NMF.
#       - user_map (dict): Mapping from userId to row index.
#       - movie_map (dict): Mapping from movieId to column index.
#     """
#     Z, user_map, movie_map = build_rating_matrix(train_file, impute=True)

#     model = NMF(n_components=n_components, init='random', random_state=0)
#     W = model.fit_transform(Z)
#     H = model.components_
#     Z_approx = np.dot(W, H)

#     return Z_approx, user_map, movie_map


# Moja wersja:


def train_nmf_model(Z, n_components=5):
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

    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=1000)
    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.dot(W, H)
    print('train')

    return Z_approx


def train_svd1_model(train_file, r):
    Z, user_map, movie_map = build_rating_matrix(train_file, impute=True)
    svd = TruncatedSVD(n_components=r, random_state=42)
    svd.fit(Z)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_

    W = svd.transform(Z) / svd.singular_values_
    H = np.dot(Sigma2, VT)
    Z_approx = np.dot(W, H)
    # print(Z_approx)
    print('train')
    return Z_approx, user_map, movie_map

def sgd(Z, r, lr=0.01, n_epochs=1000, optimizer_name="SGD"):
    n, d = Z.shape
    torch.manual_seed(42)
    A = torch.randn(n, r, requires_grad=True)
    H = torch.randn(r, d, requires_grad=True)

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam([A, H], lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD([A, H], lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        Z_hat = A @ H
        loss = torch.norm(Z - Z_hat, p=4)
        loss.backward()

        optimizer.step()

    return A.detach(), H.detach()
