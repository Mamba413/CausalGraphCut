import numpy as np
import networkx as nx
from scipy import spatial
from sklearn.cluster import SpectralClustering
import cvxpy as cp
from sklearn.cluster import KMeans
from utils import onehot_trans


def cluster_rand(positions, num_clusters, p, seed):
    """Generates spatial clusters and treatments via cluster randomization.

    Parameters
    ----------
    positions : numpy array
        n x d array of d-dimensional positions, one for each of the n units.
    num_clusters : int
        number of clusters.
    p : float
        probability of assignment to treatment.
    seed : int
        set seed for k-means clustering initialization.

    Returns
    -------
    D : numpy array
        n x 1 array of indicators, one for each of the n units.
    clusters : numpy array
        n x 1 array of cluster assignments, one for each of the n units. Clusters are labeled 0 to num_clusters-1.
    """
    clustering = SpectralClustering(n_clusters=num_clusters, random_state=seed).fit(
        positions
    )
    clusters = clustering.labels_
    cluster_rand = np.random.binomial(1, p, num_clusters)
    D = np.array([cluster_rand[clusters[i]] for i in range(positions.shape[0])])
    return D, clusters


def AOS(positions, m_nCoef=2 / 3):
    n = positions.shape[0]
    m_n = int(np.round(n ** (m_nCoef)))
    _, clusters = cluster_rand(positions, m_n, p=0.5, seed=1)
    return clusters


def KDD(adj_matrix):
    """
    @ description Implement based on R-package "interference" in Github, which implements 3-net clustering algorithm
    @ example
    >>> adj_matrix = np.loadtxt('adj.txt')
    >>> labels = three_net_cluster(adj_matrix)
    >>> print(labels)
    """
    kdd_adj_matrix = np.copy(adj_matrix)
    R = kdd_adj_matrix.shape[0]
    zero_idx = np.where(np.sum(kdd_adj_matrix, axis=1) == 0)[0]
    if zero_idx.shape[0] > 0:
        for idx in zero_idx:
            manual_idx = np.random.choice(np.setdiff1d(range(R), idx), size=3, replace=False)
            for manual_idx_i in manual_idx:
                kdd_adj_matrix[idx, manual_idx_i] = 1.0

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(kdd_adj_matrix)

    # Find all nodes within 2 hops of each node
    B_2 = {
        node: nx.single_source_shortest_path_length(G, node, cutoff=2)
        for node in G.nodes
    }

    # Initialize cluster centers and mark arrays
    cluster_centers = np.full(G.number_of_nodes(), np.nan)
    marked = np.zeros(G.number_of_nodes(), dtype=bool)
    j = 0

    # Assign cluster centers
    while not all(marked):
        unmarked_nodes = np.where(~marked)[0]
        v_j = np.random.choice(unmarked_nodes)
        cluster_centers[v_j] = j
        marked[v_j] = True
        # Mark all nodes within 2 hops
        for neighbor in B_2[v_j]:
            if B_2[v_j][neighbor] <= 2:
                marked[neighbor] = True
        j += 1

    # Assign each node to the nearest cluster center
    cluster_center_indices = np.where(~np.isnan(cluster_centers))[0]
    distances = dict(nx.all_pairs_shortest_path_length(G))
    cluster_assignment = np.zeros(G.number_of_nodes(), dtype=int)

    for i in G.nodes:
        # Find the nearest cluster center
        distances_to_centers = {
            center: distances[i][center] for center in cluster_center_indices
        }
        nearest_center = min(distances_to_centers, key=distances_to_centers.get)
        cluster_assignment[i] = cluster_centers[nearest_center]

    return cluster_assignment


def CausalClustering_obj(X, W, xi):
    n = W.shape[0]
    est_var = xi * np.sum(np.square(np.sum(X, axis=0))) / np.square(n)
    n_diff_neighbor = np.sum((1 - X @ X.transpose()) * W, axis=0)
    W_rowsum = np.sum(W, axis=0)
    W_rowsum += (W_rowsum == 0).astype(np.float16) * 1e-6
    est_bias = np.mean(n_diff_neighbor / W_rowsum)
    obj = est_bias**2 + xi * est_var
    return obj


def CausalClustering(W, xi):
    """
    xi: a tuning parameter in "Causal clustering: design of cluster  experiments under network interference"
    """
    n = W.shape[0]
    W_rowsum = np.sum(W, axis=0)
    W_rowsum += (W_rowsum == 0).astype(np.float16) * 1e-6
    V_inv = np.diag(1 / W_rowsum)
    L = V_inv @ W
    L_xi = n * L - xi * np.ones((n, n))

    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.diag(X) == np.ones(n)]
    prob = cp.Problem(cp.Minimize(cp.trace(L_xi @ X)), constraints)
    prob.solve()
    X_opt = X.value

    _, e_vector = np.linalg.eigh(X_opt)
    e_vector = e_vector[:, ::-1]

    K_min = max([int(0.09 * n), 1])
    K_max = max([int(0.42 * n), K_min])
    obj_list = []
    label_list = []
    for K in range(K_min, K_max):
        kmeans = KMeans(n_clusters=K)
        label = kmeans.fit_predict(e_vector[:, 1:K])
        obj = CausalClustering_obj(onehot_trans(label), W, xi)
        label_list.append(label)
        obj_list.append(obj)
    obj_list = np.array(obj_list)
    label = label_list[np.argmin(obj_list)]
    return label


def cluster_IPW(Y, D, design, W):
    W_new = W + np.eye(W.shape[0])
    exposure_1 = W_new @ D
    num_neighbor_plus1 = np.sum(W_new, axis=1, keepdims=True)
    exposure_1 = (exposure_1 == num_neighbor_plus1).astype(np.float64)
    exposure_0 = W_new @ (1 - D)
    exposure_0 = (exposure_0 == num_neighbor_plus1).astype(np.float64)
    ipw_1 = exposure_1 / design.propensity_score(1)
    ipw_0 = exposure_0 / design.propensity_score(0)
    est = np.mean(np.sum(((ipw_1 - ipw_0) * Y), axis=0))
    return est


class IPWEstimator:
    def __init__(self, design=None) -> None:
        self.design = design
        pass

    def compute_tau(self, data, W):
        A_mat = []
        Y_mat = []
        for _, value in data.items():
            A_mat.append(value["tre"])
            Y_mat.append(value["outcome"])
        A_mat = np.hstack(A_mat).transpose()
        Y_mat = np.hstack(Y_mat).transpose()
        hat_tau = cluster_IPW(
            Y_mat,
            A_mat,
            self.design,
            W,
        )
        return hat_tau

    def estimate(self, env, N, seed, random=True):
        data = env.sample_data(
            interior=np.array([False] * env.R),
            policy=self.design.policy(),
            N=N,
            seed=seed,
            random=random,
        )
        W = env.get_adj_matrix()
        tau = self.compute_tau(data, W)
        return tau

    def estimate_from_data(self, data, adj_mat):
        W = np.copy(adj_mat)
        tau = self.compute_tau(data, W)
        return tau

def naive_IPW(Y, D, p):
    est = np.sum(((D / p) - ((1 - D) / (1 - p))) * Y, axis=0)
    est = np.mean(est)
    return est


class NaiveIPWEstimator:
    def __init__(self, design=None) -> None:
        self.design = design
        pass

    def compute_tau(self, data):
        A_mat = []
        Y_mat = []
        for _, value in data.items():
            A_mat.append(value["tre"])
            Y_mat.append(value["outcome"])
        A_mat = np.hstack(A_mat).transpose()
        Y_mat = np.hstack(Y_mat).transpose()
        hat_tau = naive_IPW(Y_mat, A_mat, self.design.p)
        return hat_tau

    def estimate(self, env, N, seed, random=True):
        data = env.sample_data(
            interior=np.array([False] * env.R),
            policy=self.design.policy(),
            N=N,
            seed=seed,
            random=random,
        )
        tau = self.compute_tau(data)
        return tau

    def estimate_from_data(self, data):
        tau = self.compute_tau(data)
        return tau


if __name__ == "__main__":
    from data import EnvSimulator
    from utils import label2dict
    from semi_sp_design import ClusterDesign
    
    env = EnvSimulator(rho=0.1, model_type="semi-static", grid_size=8)
    print("True tau:", env.tau)
    W = env.get_adj_matrix()
    SAMPLE_NUM = 20000
    PROB = 0.5
    
    kdd_cluster = KDD(W)
    c_design = ClusterDesign(PROB, W, label2dict(kdd_cluster))
    ipw_est = IPWEstimator(c_design)
    hat_tau = ipw_est.estimate(env, N=SAMPLE_NUM, seed=0, random=True)
    print("KDD:", hat_tau)
    
    aos_cluster = AOS(env.grid)
    c_design = ClusterDesign(PROB, W, label2dict(aos_cluster))
    ipw_est = IPWEstimator(c_design)
    hat_tau = ipw_est.estimate(env, N=SAMPLE_NUM, seed=0, random=True)
    print("AOS:", hat_tau)
    
    cc_cluster = CausalClustering(W, xi=8)
    c_design = ClusterDesign(PROB, W, label2dict(cc_cluster))
    naive_ipw_est = NaiveIPWEstimator(c_design)
    hat_tau = naive_ipw_est.estimate(env, N=SAMPLE_NUM, seed=0, random=True)
    print("CC:", hat_tau)
