import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import math
import pandas as pd


class LLEClass:
    def __init__(self, r, k, X, n_components):
        self.r = r
        self.k = k
        self.X = X.to_numpy().transpose()
        self.N = len(self.X[0])
        self.D = len(self.X)
        self.index = self.init_index()
        self.eigenvectors = self.init_eigenvectors()

    def init_index(self):
        print("index started")
        index = []
        for i in range(self.N):
            l = sum(np.power(self.X - self.X[:, i][:, np.newaxis], 2))
            index.append(sorted(range(len(l)), key=lambda k: l[k]))
        index = np.transpose(index)[1:(self.k + 1), :]
        print("index done")
        return index

    def init_eigenvectors(self):
        if (self.k > self.D):
            tol = self.r
        else:
            tol = 0
        W = np.zeros((self.k, self.N))
        for i in range(self.N):
            z = self.X[:, self.index[:, i]] - self.X[:, i][:, np.newaxis]
            C = np.dot(z.transpose(), z)
            C = C + tol * np.dot(np.identity(self.k), C.trace())
            W[:, i] = np.linalg.solve(C, np.ones(self.k))
            W[:, i] = W[:, i] / sum(W[:, i])

        # step 3 - compute embedding
        M = sp.csr_matrix(np.identity(self.N))
        # M = np.identity(N)
        for i in range(self.N):
            w = W[:, i]
            j = self.index[:, i]
            M[i, j] = M[i, j] - w
            M[j, i] = M[j, i] - w[:, np.newaxis]
            M[j[:, np.newaxis], j] = M[j[:, np.newaxis], j] + np.dot(w[:, np.newaxis], w[np.newaxis])
        vals, vecs = eigsh(M, k=self.N - 1, which='SM')
        print("eigenvectors done")
        return vecs

    def return_dataframe(self, n_components):
        y = self.eigenvectors[:, [i for i in range(1, n_components + 1)]].transpose() * math.sqrt(self.N)
        df = pd.DataFrame(np.transpose(y))
        return df
