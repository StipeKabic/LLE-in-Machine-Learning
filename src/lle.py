import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import math
import pandas as pd


def lle(r, k, X, n_components):
    # input: D x N matrix X (D-dimensionality, N - number of vectors)
    # output d x N matrix (d-reduced dimension)
    # k - number of nearest neighbors

    X = X.to_numpy()
    N = len(X[0])
    D = len(X)

    # find nn
    index = []
    for i in range(N):
        l = sum(np.power(X - X[:, i][:, np.newaxis], 2))
        index.append(sorted(range(len(l)), key=lambda k: l[k]))
    index = np.transpose(index)[1:(k + 1), :]

    # step2-find weights
    if (k > D):
        tol = r
    else:
        tol = 0

    W = np.zeros((k, N))
    z_ = np.zeros((k, N))
    for i in range(N):
        z = X[:, index[:, i]] - X[:, i][:, np.newaxis]
        C = np.dot(z.transpose(), z)
        C = C + r * np.dot(np.identity(k), C.trace())
        W[:, i] = np.linalg.solve(C, np.ones(k))
        W[:, i] = W[:, i] / sum(W[:, i])

    # step 3 - compute embedding
    M = sp.csr_matrix(np.identity(N))
    for i in range(N):
        w = W[:, i]
        j = index[:, i]
        M[i, j] = M[i, j] - w
        M[j, i] = M[j, i] - w[:, np.newaxis]
        M[j[:, np.newaxis], j] = M[j[:, np.newaxis], j] + np.dot(w[:, np.newaxis], w[np.newaxis])
    print(M)
    vals, vecs = eigsh(M, k=n_components, which='SM')
    y = vecs[:, [i for i in range(1, n_components)]].transpose() * math.sqrt(N)
    df = pd.DataFrame(np.transpose(y))
    return df