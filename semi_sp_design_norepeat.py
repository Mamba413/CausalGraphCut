from data import *
from sklearn.ensemble import RandomForestRegressor
from semi_sp_design import cluster_yang, ClusterDesign, GlobalDesign, IndividualDesign
import numpy as np

def adjmat_to_neigh_indices(adj_mat):
    adj_indices = []
    for i in range(adj_mat.shape[0]):
        adj_indices.append(np.nonzero(adj_mat[:, i])[0])
    return adj_indices

class SemiEstimatorNoRepeat:
    def __init__(self, model, design=None) -> None:
        self.model = model
        self.design = design
        pass

    def compute_tau(
        self,
        data,
        adj_indices,
        grid,
        correct_model,
        estimator_type='DR',
        prev_data = None,
    ):
        regression_type='pool'

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

        if regression_type == 'pool':
            tau = 0.0
            x = np.vstack(x_mat)
            y = np.vstack(y_mat)
            loc = np.vstack(loc_mat)
            eval_mat = np.vstack(eval_mat)
            A_dim = np.vstack(A_dim_mat)

            self.model.fit(x, y.ravel())
            x_1 = np.copy(x)
            x_0 = np.copy(x)
            indicator_1 = np.zeros((x.shape[0], 1))
            indicator_0 = np.zeros((x.shape[0], 1))

            for i in range(x_1.shape[0]):
                x_1[i, :A_dim[i, 0]] = 1
                x_0[i, :A_dim[i, 0]] = 0
                indicator_1[i, 0] = np.prod((x[i, :A_dim[i, 0]] == 1).astype(np.int8))
                indicator_0[i, 0] = np.prod((x[i, :A_dim[i, 0]] == 0).astype(np.int8))
            y_1_pred = self.model.predict(x_1).reshape(-1, 1)
            y_0_pred = self.model.predict(x_0).reshape(-1, 1)
            # score_1 = self.design.propensity_score(1, loc[idx_test])
            # score_0 = self.design.propensity_score(0, loc[idx_test])
            score_1 = self.design.propensity_score_smooth(1, loc)
            score_0 = self.design.propensity_score_smooth(0, loc)

            tau = self.ope_estimation(
            # tau += self.ope_estimation_smoothing(
                y,
                y_0_pred,
                y_1_pred,
                indicator_0,
                indicator_1,
                score_0,
                score_1,
                estimator_type,
            )
            tau = SPATIAL_NUM *  tau

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
    ):
        data = env.sample_data(
            interior=np.array([False] * env.R),
            policy=self.design.policy(),
            N=N,
            seed=seed,
            random=random,
        )

        adj_indices = env.adj_indices
        tau = self.compute_tau(
            data,
            adj_indices,
            env.grid,
            correct_model,
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
            est_cov = 4 * np.cov(error_mat, rowvar=False) 
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


if __name__ == "__main__":
    block_size = 4
    grid_size=12
    ## Both settings 'semi-static' and 'homo-semi-static' are available
    env = EnvSimulator(rho=0.1, model_type="semi-static", grid_size=80)
    R = grid_size ** 2
    print("True tau:", env.tau)
    model = RandomForestRegressor(random_state=0, n_estimators=10)   # no annoying warning messages
    semi_est = SemiEstimatorNoRepeat(model=model)
    NUM = 1
    SEED = 1
    REGRESS_TYPE = 'pool'

    cluster = cluster_yang(env.grid, block_size)
    c_design = ClusterDesign(p=0.5, W=env.get_adj_matrix(), cluster=cluster)
    semi_est.update_design(c_design)
    hat_tau_C, _ = semi_est.estimate(env, N=NUM, seed=SEED, random=False)
    print("tau_C:", hat_tau_C)
    # data = env.sample_data(
    #     interior=np.array([False] * env.R),
    #     policy=c_design.policy(),
    #     N=NUM,
    #     seed=SEED,
    #     random=False,
    # )
    # hat_tau_C = semi_est.estimate_from_data(data, adj_mat=env.get_adj_matrix(), grid=env.grid, 
    #                                         regression_type=REGRESS_TYPE)
    # print('tau_C: {} (Data interface)'.format(hat_tau_C))

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
