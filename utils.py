import numpy as np
from sklearn.preprocessing import OneHotEncoder

def onehot_trans(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x).reshape(-1, 1)
    try:
        onehot_coder = OneHotEncoder(sparse=False, dtype=np.float32)
    except TypeError:
        onehot_coder = OneHotEncoder(dtype=np.float32, sparse_output=False)
    x = onehot_coder.fit_transform(x.reshape(-1, 1))
    return x


def MSE_global(V):
    mse = np.sum(V) / 0.5
    return mse


def MSE_ind(V, W):
    I1 = np.sum(np.diag(V))
    I2 = np.sum(V * W)
    I3 = 0
    R = V.shape[0]
    for i in range(R):
        for j in range(R):
            numerator = np.sum(W[i, :] * W[j, :])
            I3 += numerator * V[i, j]
    mse = (I1 + 2 * I2 + I3) / 0.5
    return mse

def label2dict(cluster):
    if isinstance(cluster, dict):
        pass
    elif isinstance(cluster, np.ndarray) or isinstance(cluster, list):
        cluster = dict([(k, np.where(cluster == k)[0]) for k in np.unique(cluster)])
    return cluster

def cov2cor(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation