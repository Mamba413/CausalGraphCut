from data import *
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy
from utils_estimator import neighbor_n_cluster

from sklearn.model_selection import KFold
import numpy as np


def extract_2d_array(arr, axis, index):
    slicer = [slice(None), slice(None)]
    slicer[axis] = index
    return arr[tuple(slicer)]


def adjmat_to_neigh_indices(adj_mat):
    adj_indices = []
    for i in range(adj_mat.shape[0]):
        adj_indices.append(np.nonzero(adj_mat[:, i])[0])
    return adj_indices

def compute_Nc_group(adj_mat, cluster):
    adj_indices = adjmat_to_neigh_indices(adj_mat)
    Nc_group = {}

    for i in range(len(adj_indices)):
        neighbor_indices = adj_indices[i] + [i]
        unique_clusters = set()  

        for region in neighbor_indices:
            for cluster_id, regions in cluster.items():
                if region in regions:  
                    unique_clusters.add(cluster_id)
        
        Nc_group[i] = len(unique_clusters)
        
    return Nc_group

class CrossFitting:
    def __init__(self, n_splits, shuffle, random_state=42) -> None:
        self.n_splits = n_splits
        if not shuffle:
            random_state=None
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        pass

    def split(self, num):
        self.idx = tuple(self.kfold.split(np.arange(num)))
        self.train_size = [len(t[0]) for t in self.idx]
        self.test_size = [len(t[1]) for t in self.idx]
        self.idx = dict(enumerate([{"train": t[0], "test": t[1]} for t in self.idx]))
        pass


class CrossFittingIID(CrossFitting):
    def __init__(self, n_splits, shuffle, random_state=42) -> None:
        super(CrossFittingIID, self).__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        pass

    def split(self, x, y, x_axis=0, y_axis=0):
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.x_axis = x_axis
        self.y_axis = y_axis
        num = x.shape[x_axis]
        super().split(num)
        pass

    def get_ifold_data(self, i):
        train_idx = self.idx[i]["train"]
        test_idx = self.idx[i]["test"]
        x_train = extract_2d_array(self.x, self.x_axis, train_idx)
        x_test = extract_2d_array(self.x, self.x_axis, test_idx)
        y_train = extract_2d_array(self.y, self.y_axis, train_idx)
        y_test = extract_2d_array(self.y, self.y_axis, test_idx)
        return x_train, x_test, y_train, y_test, train_idx, test_idx
    

class IndividualDesign:
    def __init__(self, p, W) -> None:
        self.p = p
        self.W = W
        self.num_neighbor = np.sum(self.W, axis=0) + 1
        pass

    def propensity_score(self, treatment, idx=None):
        tmp = treatment * self.p + (1 - self.p) * (1 - treatment)
        if idx is not None:
            score = np.power(tmp, self.num_neighbor[idx])
        else:
            score = np.power(tmp, self.num_neighbor).reshape(-1, 1)
        return score
    
    def propensity_score_smooth(self, treatment, idx=None):
        tmp = treatment * self.p + (1 - self.p) * (1 - treatment)
        if idx is not None:
            score = np.power(tmp, self.num_neighbor[idx])
        else:
            score = np.max(np.power(tmp, self.num_neighbor).reshape(-1, 1), 0.125)
        return score

    def policy(self):
        def individual_policy(obs):
            """
            obs (np.array): R-by-d matrix
            p (float): a scalar
            """
            R = obs.shape[0]
            A = np.random.binomial(n=1, p=self.p, size=(R, 1))
            return A

        return individual_policy


class GlobalDesign:
    def __init__(self, p, W) -> None:
        self.p = p
        self.W = W
        pass

    def propensity_score(self, treatment, idx=None):
        tmp = treatment * self.p + (1 - self.p) * (1 - treatment)
        if idx is None or isinstance(idx, int):
            score = tmp * np.ones((1, 1))
        else:
            score = tmp * np.ones((len(idx), 1))
        return score
    
    def propensity_score_smooth(self, treatment, idx=None):
        tmp = treatment * self.p + (1 - self.p) * (1 - treatment)
        if idx is None or isinstance(idx, int):
            score = tmp * np.ones((1, 1))
        else:
            score = np.maximum(tmp * np.ones((len(idx), 1)), 0.125)
        return score

    def policy(self):
        def global_policy(obs):
            R = obs.shape[0]
            A = np.random.binomial(n=1, p=self.p, size=1) * np.ones((R, 1))
            return A

        return global_policy


class ClusterDesign:
    def __init__(self, p, W, cluster) -> None:
        self.p = p
        self.W = W
        self.cluster = deepcopy(cluster)
        self.num_cluster = neighbor_n_cluster(W, cluster)

    def propensity_score(self, treatment, idx=None):
        tmp = treatment * self.p + (1 - self.p) * (1 - treatment)
        if idx is None:
            score = np.power(tmp, self.num_cluster)
        else:
            score = np.power(tmp, self.num_cluster[idx])
            if len(score.shape) == 3:
                score = score[:, :, 0]
            elif len(score.shape) == 1:
                score = score.reshape(-1, 1)
        return score
    
    def propensity_score_smooth(self, treatment, idx=None):
        tmp = treatment * self.p + (1 - self.p) * (1 - treatment)
        if idx is None:
            score = np.power(tmp, self.num_cluster)
        else:
            score = np.power(tmp, self.num_cluster[idx])
            score = np.maximum(np.power(tmp, self.num_cluster[idx]),0.125)
            if len(score.shape) == 3:
                score = score[:, :, 0]
            elif len(score.shape) == 1:
                score = score.reshape(-1, 1)
        return score

    def policy(self):
        def fixed_cluster_policy(obs):
            R = obs.shape[0]
            A = np.zeros((R, 1))
            for _, value in self.cluster.items():
                A[value] = np.random.binomial(n=1, p=self.p, size=1) * np.ones(
                    (len(value), 1)
                )
            return A

        return fixed_cluster_policy


class SemiEstimator:
    def __init__(self, n_splits, model, design=None) -> None:
        self.n_splits = n_splits
        self.model = model
        self.design = design
        pass

    def compute_tau(
        self,
        data,
        adj_indices,
        grid,
        correct_model,
        regression_type='pool',
        estimator_type='DR',
        prev_data = None,
        return_error = False,
    ):
        eval_indices = dict()
        if  prev_data is not None:
            for key, value in prev_data.items():
                for sub_key, sub_value in value.items():
                    if sub_key == 'outcome':
                        un_eval_num = sub_value.shape[0]
                        total_num = data[key][sub_key].shape[0] + un_eval_num
                        eval_indices_arr = np.ones((total_num, 1), dtype=np.int8)
                        eval_indices_arr[range(un_eval_num), :] = 0
                        eval_indices[key] = eval_indices_arr
                    if sub_key != 'interior':
                        data[key][sub_key] = np.vstack([sub_value, data[key][sub_key]])
        else:
            for key, value in data.items():
                total_num = data[key]['outcome'].shape[0]
                eval_indices[key] = np.ones((total_num, 1), dtype=np.int8)

        cf = CrossFittingIID(n_splits=self.n_splits, shuffle=True)

        A_mat = []
        O_mat = []
        for key, value in data.items():
            A_mat.append(value["tre"])
            O_mat.append(value["obs"])
        A_mat = np.hstack(A_mat)
        O_mat = np.stack(O_mat, axis=2)
        O_dim = O_mat.shape[1]
        TIME_REP = A_mat.shape[0]
        SPATIAL_NUM = len(data)

        tau = 0.0
        ## data preparation
        x_mat = []
        y_mat = []
        loc_mat = []
        A_dim_list = []
        A_dim_mat = []
        eval_mat = []
        max_adj_num = np.max([len(value) for value in adj_indices])
        for key, value in data.items():
            neighbor_indices = adj_indices[key]
            if regression_type == 'pool' and correct_model:
                current_A_dim = 2
                neighbor_tre = np.mean(A_mat[:, neighbor_indices], axis=1)
                neighbor_tre = neighbor_tre.reshape(-1, 1)
                neighbor_obs = np.mean(O_mat[:, :, neighbor_indices], axis=2)
            else:
                adj_num = len(neighbor_indices)
                current_A_dim = adj_num + 1
                neighbor_tre = A_mat[:, neighbor_indices]
                if regression_type == 'pool' and max_adj_num > adj_num:
                    neighbor_tre = np.hstack([neighbor_tre, np.zeros((TIME_REP, max_adj_num - adj_num))])
                neighbor_obs = []
                for k in range(O_dim):
                    tmp_O_mat = O_mat[:, k, neighbor_indices]
                    if regression_type == 'pool' and max_adj_num > adj_num:
                        tmp_O_mat = np.hstack([tmp_O_mat, np.zeros((TIME_REP, max_adj_num - adj_num))])
                    neighbor_obs.append(tmp_O_mat)
                neighbor_obs = np.hstack(neighbor_obs)
            A_dim_list.append(current_A_dim)

            x_key = np.hstack(
                [
                    value["tre"],
                    neighbor_tre,
                    value["obs"],
                    neighbor_obs,
                ]
            )
            if regression_type == "pool":
                x_key = np.hstack(
                    [
                        x_key,
                        np.repeat(grid[key, :].reshape(1, -1), TIME_REP, axis=0), 
                    ]
                )
            x_mat.append(x_key)
            y_mat.append(value["outcome"])
            loc_mat.append(np.array([key] * A_mat.shape[0]).reshape(-1, 1))
            A_dim_mat.append(np.array([current_A_dim] * A_mat.shape[0]).reshape(-1, 1))
            eval_mat.append(eval_indices[key])
        if return_error:
            if regression_type == "pool":
                error = np.zeros((y_mat[0].shape[0] * SPATIAL_NUM, 1))
            elif regression_type == "local":
                error = dict()
                for key, value in data.items():
                    error[key] = np.zeros(value["tre"].shape)
                error_key = [x[0] for x in data.items()]

        if regression_type == 'pool':
            tau = 0.0
            x = np.vstack(x_mat)
            y = np.vstack(y_mat)
            loc = np.vstack(loc_mat)
            eval_mat = np.vstack(eval_mat)
            A_dim = np.vstack(A_dim_mat)
            cf.split(x, y)
            for k in range(self.n_splits):
                x_train, x_test, y_train, y_test, _, idx_test = cf.get_ifold_data(k)
                self.model.fit(x_train, y_train.ravel())
                x_1 = np.copy(x_test)
                x_0 = np.copy(x_test)
                indicator_1 = np.zeros((x_test.shape[0], 1))
                indicator_0 = np.zeros((x_test.shape[0], 1))
                for i, test_i in enumerate(idx_test):
                    x_1[i, :A_dim[test_i, 0]] = 1
                    x_0[i, :A_dim[test_i, 0]] = 0
                    indicator_1[i, 0] = np.prod((x_test[i, :A_dim[test_i, 0]] == 1).astype(np.int8))
                    indicator_0[i, 0] = np.prod((x_test[i, :A_dim[test_i, 0]] == 0).astype(np.int8))
                y_1_pred = self.model.predict(x_1).reshape(-1, 1)
                y_0_pred = self.model.predict(x_0).reshape(-1, 1)
                # score_1 = self.design.propensity_score(1, loc[idx_test])
                # score_0 = self.design.propensity_score(0, loc[idx_test])
                score_1 = self.design.propensity_score_smooth(1, loc[idx_test])
                score_0 = self.design.propensity_score_smooth(0, loc[idx_test])

                eval_indicator = (eval_mat[idx_test, :] == 1).flatten()
                tau += self.ope_estimation(
                # tau += self.ope_estimation_smoothing(
                    y_test[eval_indicator, :],
                    y_0_pred[eval_indicator, :],
                    y_1_pred[eval_indicator, :],
                    indicator_0[eval_indicator, :],
                    indicator_1[eval_indicator, :],
                    score_0[eval_indicator, :],
                    score_1[eval_indicator, :],
                    estimator_type,
                )
                if return_error:
                    y_pred = self.model.predict(x_test).reshape(-1, 1)
                    error[idx_test, :] = y_test - y_pred
            tau = (SPATIAL_NUM *  tau) / self.n_splits
        elif regression_type == "local":  
            for i in range(len(y_mat)):
                x = x_mat[i]
                y = y_mat[i]
                A_dim = A_dim_list[i]
                loc = loc_mat[i]
                eval_i = eval_mat[i]
                cf.split(x, y)
                tau_key = 0.0

                for k in range(self.n_splits):
                    x_train, x_test, y_train, y_test, _, idx_test = cf.get_ifold_data(k)
                    self.model.fit(x_train, y_train.ravel())
                    x_1 = np.copy(x_test)
                    x_1[:, :A_dim] = 1
                    x_0 = np.copy(x_test)
                    x_0[:, :A_dim] = 0
                    y_1_pred = self.model.predict(x_1).reshape(-1, 1)
                    y_0_pred = self.model.predict(x_0).reshape(-1, 1)
                    # score_1 = self.design.propensity_score(1, loc[idx_test])
                    # score_0 = self.design.propensity_score(0, loc[idx_test])
                    score_1 = self.design.propensity_score_smooth(1, loc[idx_test])
                    score_0 = self.design.propensity_score_smooth(0, loc[idx_test])
                    indicator_1 = np.prod(
                        (x_test[:, :A_dim] == 1).astype(np.int8),
                        axis=1,
                        keepdims=True,
                    )
                    indicator_0 = np.prod(
                        (x_test[:, :A_dim] == 0).astype(np.int8),
                        axis=1,
                        keepdims=True,
                    )

                    eval_indicator = (eval_i[idx_test, :] == 1).flatten()
                    tau_key += self.ope_estimation(
                    # tau_key += self.ope_estimation_smoothing(
                        y_test[eval_indicator, :],
                        y_0_pred[eval_indicator, :],
                        y_1_pred[eval_indicator, :],
                        indicator_0[eval_indicator, :],
                        indicator_1[eval_indicator, :],
                        score_0[eval_indicator, :],
                        score_1[eval_indicator, :],
                        estimator_type,
                    )
                    if return_error:
                        y_pred = self.model.predict(x_test).reshape(-1, 1)
                        error[error_key[i]][idx_test, :] = y_test - y_pred
                tau_key = tau_key / self.n_splits
                tau += tau_key

        if return_error and regression_type == 'pool':
            error_arr = np.copy(error)
            NUM_TIME = int(error_arr.shape[0] / SPATIAL_NUM)
            error = dict()
            for key, value in data.items():
                error[key] = (error_arr[range(key * NUM_TIME, (key+1)*NUM_TIME), 0]).reshape(-1, 1)

        if return_error:
            return tau, error
        else:
            return tau

    def ope_estimation(
        self,
        y_test,
        y_0_pred,
        y_1_pred,
        indicator_0,
        indicator_1,
        score_0,
        score_1,
        estimator_type,
    ):
        if estimator_type == "DR":
            tau_tmp = np.mean(
                ((y_test - y_1_pred) * (indicator_1 / score_1) + y_1_pred)
                - ((y_test - y_0_pred) * (indicator_0 / score_0) + y_0_pred)
            )
        elif estimator_type == "Direct":
            tau_tmp = np.mean(y_1_pred - y_0_pred)
        elif estimator_type == "IS":
            tau_tmp = np.mean(
                ((indicator_1 / score_1) - (indicator_0 / score_0)) * y_test
            )
        return tau_tmp
    
    # def ope_estimation_smoothing(
    #     self,
    #     y_test,
    #     y_0_pred,
    #     y_1_pred,
    #     indicator_0,
    #     indicator_1,
    #     score_0,
    #     score_1,
    #     estimator_type,
    # ):
    #     if estimator_type == "DR":
    #         adj_mat = env.get_adj_matrix()
    #         Nc_group = compute_Nc_group(adj_mat, cluster)
    #         # print("y_test.shape, len(y_test), R:",y_test.shape, len(y_test), R)
    #         ww = np.array([0.5 / max(0.5 ** Nc_group[k], 0.025) for k in range(R)])
    #         # print(ww.shape)
    #         # print(ww)
    #         # ww = np.repeat(ww, (len(y_test)/R)).reshape(-1,1)
    #         # print(ww.shape)
    #         # aa = (y_test - y_1_pred) * (indicator_1 / score_1)
    #         # print(aa.shape)
    #         # bb = ww * aa
    #         # print(bb.shape)
    #         # ww = np.repeat(ww, (len(y_test)/R))
    #         # print(y_1_pred.shape)
    #         tau_tmp1 = np.mean(
    #             (ww * (y_test - y_1_pred) * (indicator_1 / score_1) + y_1_pred)
    #             - (ww * (y_test - y_0_pred) * (indicator_0 / score_0) + y_0_pred)
    #         )
    #         print(tau_tmp1)
    #         tau_tmp = np.mean(
    #             (((y_test - y_1_pred) * (indicator_1 / score_1) + y_1_pred)
    #             - ((y_test - y_0_pred) * (indicator_0 / score_0) + y_0_pred))
    #         )
    #         print(tau_tmp)
    #     elif estimator_type == "Direct":
    #         tau_tmp = np.mean(y_1_pred - y_0_pred)
    #     elif estimator_type == "IS":
    #         tau_tmp = np.mean(
    #             ((indicator_1 / score_1) - (indicator_0 / score_0)) * y_test
    #         )
    #     return tau_tmp

    def estimate(
        self,
        env,
        N,
        seed,
        correct_model=False,
        regression_type='local',
        estimator_type="DR",
        random=True,
        prev_data=None,
        return_cov=False,
        prev_error=None,
        return_error=False,
    ):
        data = env.sample_data(
            interior=np.array([False] * env.R),
            policy=self.design.policy(),
            N=N,
            seed=seed,
            random=random,
        )

        adj_indices = env.adj_indices
        if return_cov:
            tau, error = self.compute_tau(
                data,
                adj_indices,
                env.grid,
                correct_model,
                regression_type,
                estimator_type,
                prev_data=prev_data,
                return_error=True,
            )
            error_mat = []
            for _, value in error.items():
                error_mat.append(value)
            error_mat = np.hstack(error_mat)
            if prev_error is not None:
                error_mat = np.vstack([prev_error, error_mat])
            est_cov = np.cov(error_mat, rowvar=False)
            if not return_error:
                return tau, data, est_cov
            else:
                return tau, data, est_cov, error_mat
        else:
            tau = self.compute_tau(
                data,
                adj_indices,
                env.grid,
                correct_model,
                regression_type,
                estimator_type,
                prev_data=prev_data,
            )
            return tau, data

    def estimate_from_data(
        self,
        data,
        adj_mat, 
        grid = None, 
        correct_model=False,
        regression_type='local',
        estimator_type="DR",
        prev_data=None,
        return_cov=False,
        prev_error=None,
        return_error=False,
    ):
        adj_indices = adjmat_to_neigh_indices(adj_mat)
        if return_cov:
            tau, error = self.compute_tau(
                data,
                adj_indices,
                grid,
                correct_model,
                regression_type,
                estimator_type,
                prev_data=prev_data,
                return_error=True,
            )
            error_mat = []
            for _, value in error.items():
                error_mat.append(value)
            error_mat = np.hstack(error_mat)
            if prev_error is not None:
                error_mat = np.vstack([prev_error, error_mat])
            est_cov = np.cov(error_mat, rowvar=False)
            # est_cov = np.corrcoef(np.hstack(error_mat), rowvar=False)
            if not return_error:
                return tau, est_cov
            else:
                return tau, est_cov, error_mat
        else:
            tau = self.compute_tau(
                data,
                adj_indices,
                grid,
                correct_model,
                regression_type,
                estimator_type,
                prev_data=prev_data,
            )
            return tau

    def update_design(self, design):
        self.design = design

def cluster_yang(grid, block_size):
    R = grid.shape[0]
    grid_size = int(np.sqrt(R))
    num_blocks_per_dim = grid_size // block_size
    num_blocks_per_dim = int(num_blocks_per_dim)
    cluster_index = 0
    cluster = {}
    for i in range(num_blocks_per_dim):
        for j in range(num_blocks_per_dim):
            cluster_indices = []
            for bi in range(block_size):
                for bj in range(block_size):
                    idx = (i * block_size + bi) * grid_size + (j * block_size + bj)
                    if idx < grid_size**2:  # Ensure index is within bounds
                        cluster_indices.append(idx)
            cluster[cluster_index] = cluster_indices
            cluster_index += 1
    return cluster


if __name__ == "__main__":
    block_size = 4
    grid_size=12
    ## Both settings 'semi-static' and 'homo-semi-static' are available
    env = EnvSimulator(rho=0.1, model_type="semi-static", grid_size=12)
    R = grid_size ** 2
    print("True tau:", env.tau)
    model = RandomForestRegressor(random_state=0, n_estimators=10)
    semi_est = SemiEstimator(n_splits=2, model=model)
    NUM = 100
    SEED = 1
    REGRESS_TYPE = 'pool'

    cluster = cluster_yang(env.grid, block_size)
    c_design = ClusterDesign(p=0.5, W=env.get_adj_matrix(), cluster=cluster)
    semi_est.update_design(c_design)
    hat_tau_C, _ = semi_est.estimate(env, N=NUM, seed=SEED, 
                                     regression_type=REGRESS_TYPE, random=False)
    print("tau_C:", hat_tau_C)
    data = env.sample_data(
        interior=np.array([False] * env.R),
        policy=c_design.policy(),
        N=NUM,
        seed=SEED,
        random=False,
    )
    hat_tau_C = semi_est.estimate_from_data(data, adj_mat=env.get_adj_matrix(), grid=env.grid, 
                                            regression_type=REGRESS_TYPE)
    print('tau_C: {} (Data interface)'.format(hat_tau_C))

    i_design = IndividualDesign(p=0.5, W=env.get_adj_matrix())
    semi_est.update_design(i_design)
    hat_tau_I, _ = semi_est.estimate(env, N=NUM, seed=SEED, random=False, regression_type=REGRESS_TYPE)
    print("tau_I:", hat_tau_I)

    g_design = GlobalDesign(p=0.5, W=env.get_adj_matrix())
    semi_est.update_design(g_design)
    hat_tau_G, _ = semi_est.estimate(env, N=NUM, seed=SEED, random=False, regression_type=REGRESS_TYPE)
    print("tau_G:", hat_tau_G)

    ################### smoothing ###################
    # semi_est_smooth = SemiEstimatorSmooth(n_splits=2, model=model)

    # cluster = cluster_yang(env.grid, block_size)
    # c_design = ClusterDesign(p=0.5, W=env.get_adj_matrix(), cluster=cluster)

    # semi_est_smooth.update_design(c_design)
    # hat_tau_C_smooth, _ = semi_est_smooth.estimate(env, N=NUM, seed=SEED, 
    #                                  regression_type=REGRESS_TYPE, random=False)
    # print("tau_C_smooth:", hat_tau_C_smooth)
    # data = env.sample_data(
    #     interior=np.array([False] * env.R),
    #     policy=c_design.policy(),
    #     N=NUM,
    #     seed=SEED,
    #     random=False,
    # )
    # hat_tau_C_smooth = semi_est_smooth.estimate_from_data(data, adj_mat=env.get_adj_matrix(), grid=env.grid, 
    #                                         regression_type=REGRESS_TYPE)
    # print('tau_C_smooth: {} (Data interface)'.format(hat_tau_C_smooth))


    ################### smoothing end ###################

    # cluster = cluster_yang(env.grid, block_size)
    # c_design = ClusterDesign(p=0.5, W=env.get_adj_matrix(), cluster=cluster)
    # semi_est.update_design(c_design)
    # hat_tau_C = semi_est.estimate(env, N=NUM, seed=SEED, random=True)
    # print("tau_C (w random):", hat_tau_C)

    # i_design = IndividualDesign(p=0.5, W=env.get_adj_matrix())
    # semi_est.update_design(i_design)
    # hat_tau_I = semi_est.estimate(env, N=NUM, seed=SEED, random=True)
    # print("tau_I (w random):", hat_tau_I)

    # g_design = GlobalDesign(p=0.5, W=env.get_adj_matrix())
    # semi_est.update_design(g_design)
    # hat_tau_G = semi_est.estimate(env, N=NUM, seed=SEED, random=True)
    # print("tau_G (w random):", hat_tau_G)

    # print(">>> Test covariance estimation")
    # est_cov, _ = semi_est.estimate_cov(env, N=30, seed=SEED, random=True)
    # cov_error_F = np.mean(np.square(est_cov - env.cov_mat))
    # print("F-norm of \|V - \hat V\|: ", cov_error_F)

    ################### test in parametric model setting ###################
    # env = EnvSimulator(rho=0.3, model_type='static')
    # print("True tau:", env.tau)
    # model = LinearRegression()
    # semi_est = SemiEstimator(n_splits=2, model=model)
    # cluster = cluster_yang(env.grid, block_size)
    # c_design = ClusterDesign(p=0.5, W=env.get_adj_matrix(), cluster=cluster)
    # semi_est.update_design(c_design)
    # hat_tau_C = semi_est.estimate(env, N=30, seed=SEED, random=False)
    # print("tau_C:", hat_tau_C)
    ########################################################################
