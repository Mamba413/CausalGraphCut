import numpy as np

def neighbor_n_cluster(W, cluster):
    R = W.shape[0]
    W_new = W + np.eye(R)
    num_cluster = np.zeros((R, 1))
    for i in range(R):
        nnz_idx = np.nonzero(W_new[:, i])[0]
        old_nnz = nnz_idx.shape[0]
        for _, value in cluster.items():
            nnz_idx = np.setdiff1d(nnz_idx, value)
            curr_nnz = nnz_idx.shape[0]
            if old_nnz > curr_nnz:
                num_cluster[i] += 1
                old_nnz = curr_nnz
            else:
                pass
            if curr_nnz == 0:
                break
    return num_cluster