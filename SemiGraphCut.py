import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from utils_semi import *
import os
from utils import onehot_trans, label2dict
from utils_semi import MSE_global
from utils_cluster import SpectralClustering
from semi_sp_design import IndividualDesign, ClusterDesign

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def compute_similarity(V, W, m=None):
    R = W.shape[0]
    sim_mat = (R / m) * np.maximum(V, 0) * (W + np.eye(R)) - V
    sim_mat = np.diag(np.sum(sim_mat, axis=1)) - sim_mat
    return sim_mat


def graph_cut_algo(
    W,
    V,
    m=2,
    eta=0.0,
    seed=1,
    verbose=True,
):
    """
    @param W: a R-by-R adjacent matrix
    @param V: correlation matrix among residual
    @note. The usage of spectral clustering has some tricks and the current usage is recommended in most of cases.
    """

    np.random.seed(seed=seed)
    sim_mat = compute_similarity(V, W, m)

    if eta > 0.0:
        sim_mat = sim_mat - eta * W  ## total variation
        # sim_mat = sim_mat + eta * np.eye(V.shape[0])   ## ridge penalty

    cut_algo = SpectralClustering(
        n_clusters=m,
        affinity="precomputed",
        random_state=0,
        # assign_labels="cluster_qr",
        assign_labels="discretize",  # for the new loss function, this option is also nice
        # assign_labels="kmeans",
    )
    if m == 2:
        curr_x = 2 * cut_algo.fit_predict(sim_mat) - 1
    else:
        curr_x = cut_algo.fit_predict(sim_mat)
    curr_x = curr_x.reshape(-1, 1)

    return curr_x


def multi_graph_cut(
    W,
    V,
    m_max=None,
    eta=0.0,
    seed=1,
    verbose=True,
):
    if m_max is None:
        m_max = np.round(np.power(W.shape[0], 2 / 3)).astype(np.int8)
    x_m = []
    obj_value = np.zeros(m_max)
    for i, m in enumerate(range(1, m_max + 1)):
        if m == 1:
            curr_x = np.zeros(shape=(V.shape[0], 1))
            obj_value[i] = MSE_global(V)
        elif m == 2:
            curr_x = graph_cut_algo(W, V, m=m, eta=eta, seed=seed, verbose=verbose)
            obj_value[i] = bi_objective(W, V)(curr_x)
        else:
            curr_x = graph_cut_algo(W, V, m=m, eta=eta, seed=seed, verbose=verbose)
            obj_value[i] = multi_objective(W, V)(onehot_trans(curr_x))
        x_m.append(curr_x)

    x_m = [x_m[i] for i in np.argsort(-obj_value)]
    return x_m, np.sort(obj_value)[::-1]

def online_graph_cut(env, semi_est, sample_num, batch_size=5, prob=0.5, init_design=None, seed=1, m_max=None):
    batch_size = 5
    num_sample_iter = int(sample_num / batch_size)
    tau_value_list = np.zeros(num_sample_iter)
    for i in range(num_sample_iter):
        if i == 0:
            if init_design is None:
                init_design = IndividualDesign(p=prob, W=W)
            semi_est.update_design(init_design)
            tau_value, prev_data, hat_V = semi_est.estimate(
                env,
                N=batch_size,
                seed=seed,
                random=True,
                regression_type='pool', 
                return_cov=True,
            )
        else:
            gc_cluster, _ = multi_graph_cut(
                W=W,
                V=hat_V,
                m_max=m_max,
                verbose=False,
            )
            gc_cluster = gc_cluster[-1]
            c_design = ClusterDesign(prob, W, label2dict(gc_cluster))
            semi_est.update_design(c_design)
            tau_value, prev_data, hat_V = semi_est.estimate(
                env,
                N=batch_size,
                seed=seed+i,
                random=True,
                regression_type='pool', 
                prev_data=prev_data,
                return_cov=True,
            )
        tau_value_list[i] = tau_value
    hat_tau = np.mean(tau_value_list)
    return hat_tau, hat_V, 

if __name__ == "__main__":
    from data import *
    from plot_region import plot_hexagon

    env = EnvSimulator(
        pattern="hexagon", grid_size=6,
        rho=0.9,
        exposure=1,
        cor_type="example9",
    )
    W = env.get_adj_matrix()
    V = env.get_cov_matrix()
    R = W.shape[0]
    bi_cluster = graph_cut_algo(W=W, V=V)
    plot_hexagon(env.grid, bi_cluster)
    multi_cluster, _ = multi_graph_cut(W=W, V=V)
    plot_hexagon(env.grid, multi_cluster[-1])
    for cluster_m in multi_cluster:
        plot_hexagon(env.grid, cluster_m)

    print("individual MSE:", MSE_ind(V, W))
    print("global MSE:", MSE_global(V))
    print("GC MSE (Bi-partition):", bi_objective(W, V)(bi_cluster))
    print("GC MSE (Multi-partition):", multi_objective(W, V)(onehot_trans(multi_cluster[-1])))
