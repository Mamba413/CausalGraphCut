import numpy as np


def MSE_global(V):
    mse = np.sum(V) / 0.25
    return mse


def MSE_ind(V, W):
    R = W.shape[0]
    N_mat = (W + np.eye(R)) @ (W + np.eye(R))
    weight_V = 2.0 * np.power(2.0, N_mat) * V * (N_mat > 0).astype(np.int8)
    mse = np.sum(weight_V)
    return mse


def bi_objective(W, V):
    def objective_W_V(x):
        R = W.shape[0]
        u_1 = (
            ((np.eye(R) @ (W + np.eye(R)) @ ((x + 1) / 2)) > 0)
            .astype(np.int8)
            .reshape(-1, 1)
        )
        u_m1 = (
            ((np.eye(R) @ (W + np.eye(R)) @ ((1 - x) / 2)) > 0)
            .astype(np.int8)
            .reshape(-1, 1)
        )
        M_mat = u_1 @ u_1.transpose() + u_m1 @ u_m1.transpose()
        weight_V = 2.0 * np.power(2.0, M_mat) * V * (M_mat > 0).astype(np.int8)

        obj_value = np.sum(weight_V)
        return obj_value

    return objective_W_V


def bi_relax_objective(W, V, eta):
    def relax_objective_W_V(x):
        R = W.shape[0]
        weight_mat = (W + np.eye(R)) @ V @ (W + np.eye(R))
        obj_value = x.transpose() @ (weight_mat - eta * W) @ x
        obj_value = np.reshape(obj_value, ())
        return obj_value

    return relax_objective_W_V


def multi_objective(W, V):
    """
    @description The result of this function matches to `bi_objective`. see
    from utils import onehot_trans
    print("cluster MSE (Oracle):", bi_objective(W, V)(x))
    print("cluster MSE (Oracle):", multi_objective(W, V)(onehot_trans(x)))
    """

    def objective_W_V(X):
        R = W.shape[0]
        join_mat = (((W + np.eye(R)) @ X) > 0).astype(np.int8)
        M_mat = join_mat @ join_mat.transpose()
        weight_mat = 2.0 * np.power(2.0, M_mat) * V * (M_mat > 0).astype(np.int8)
        mse = np.sum(weight_mat)
        return mse

    return objective_W_V


def multi_objective_new(W, V):
    """
    @description Another implementation for `multi_objective`. It is designed to validate `multi_objective`.
    """

    def objective_W_V(X):
        R = W.shape[0]
        join_mat = (((W + np.eye(R)) @ X) > 0).astype(np.int8)
        M_mat = join_mat @ join_mat.transpose()
        weight_mat = 2.0 * (np.power(2.0, M_mat) + (M_mat > 0).astype(np.int8) - 1) * V
        mse = np.sum(weight_mat)
        return mse

    return objective_W_V


if __name__ == "__main__":
    from utils import onehot_trans
    from data import EnvSimulator

    env = EnvSimulator(
        pattern="hexagon",
        rho=0.6,
        cor_type="example1",
        grid_size=12,
    )
    W = env.get_adj_matrix()
    V = env.get_cov_matrix()
    x = np.round(np.random.uniform(low=1, high=7, size=W.shape[0]), decimals=0)
    print("cluster MSE (Oracle):", multi_objective(W, V)(onehot_trans(x)))
    print("cluster MSE (Oracle):", multi_objective_new(W, V)(onehot_trans(x)))
