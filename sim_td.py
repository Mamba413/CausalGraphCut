from SemiGraphCut import multi_graph_cut
from data import *
import argparse
from semi_sp_design import (
    SemiEstimator,
    ClusterDesign,
    IndividualDesign,
    GlobalDesign,
)
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from utils import label2dict, onehot_trans
from utils_semi import multi_objective

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Simulation on semi-parametric models on data leverage (single or complete)")
parser.add_argument("--pattern", type=str, default="hexagon")
parser.add_argument("--exposure", type=int, default=1)
parser.add_argument("--model", type=str, default="semi-static")
parser.add_argument("--rho", type=float, default=0.5)
parser.add_argument("--cor-type", type=str, default="example1")
parser.add_argument("--grid-size", type=int, default=12)
parser.add_argument("--loc-std", type=float, default=0.0)
parser.add_argument("--nrep", type=int, default=50)
parser.add_argument("--m-max", type=int, default=None)
parser.add_argument("--sample-num", type=int, default=100)

SAVE = True
CONSISTENCY = True
DIR_NAME = "result-td"   ## result for different train datasets
PROB = 0.5

if __name__ == "__main__":
    args = parser.parse_args()

    PATTERN = args.pattern
    EXPOSURE = args.exposure
    MODEL = args.model
    RHO = args.rho
    COR_TYPE = args.cor_type
    GRID_SIZE = args.grid_size
    LOC_NOISE = args.loc_std
    NREP = args.nrep
    M_MAX = args.m_max
    SAMPLE_NUM = args.sample_num

    METHOD_LIST = [
        "GC (Oracle)",
        # "GC",
        "OGC-ST",  # use data at single iteration for training machine learning model
        "OGC",     # use all data at all iteration for training machine learning model
    ]

    env = EnvSimulator(
        pattern=PATTERN,
        model_type=MODEL,
        exposure=EXPOSURE,
        grid_noise=("uniform", LOC_NOISE),
        rho=RHO,
        cor_type=COR_TYPE,
        grid_size=GRID_SIZE,
    )
    W = env.get_adj_matrix()
    V = env.get_cov_matrix()
    R = W.shape[0]
    true_tau = env.tau

    ## Oracle V
    error_gc_oracle = -9999.99 * np.ones(NREP)
    error_gc = -9999.99 * np.ones(NREP)
    error_ogc = -9999.99 * np.ones(NREP)
    error_ogc_st = -9999.99 * np.ones(NREP)

    for r in tqdm(range(NREP)):
        model = RandomForestRegressor(random_state=r, n_estimators=10, min_samples_leaf=1)
        semi_est = SemiEstimator(n_splits=2, model=model)
        if "GC (Oracle)" in METHOD_LIST:
            gc_oracle, gc_oracle_value = multi_graph_cut(
                W=W,
                V=V,
                m_max=M_MAX,
                verbose=False,
            )
            gc_oracle = gc_oracle[-1]
            c_design = ClusterDesign(PROB, W, label2dict(gc_oracle))
            semi_est.update_design(c_design)
            hat_tau, _ = semi_est.estimate(
                env, N=SAMPLE_NUM, seed=r, random=True
            )
            error_gc_oracle[r] = hat_tau - true_tau
        if "GC" in METHOD_LIST:
            i_design = IndividualDesign(p=PROB, W=env.get_adj_matrix())
            semi_est.update_design(i_design)
            _, _, hat_V = semi_est.estimate(
                env, N=SAMPLE_NUM, seed=r, random=True, regression_type='pool', return_cov=True, 
            )
            gc_cluster, gc_oracle_objective = multi_graph_cut(
                W=W,
                V=hat_V,
                m_max=M_MAX,
                verbose=False,
            )
            gc_cluster = gc_cluster[-1]
            c_design = ClusterDesign(PROB, W, label2dict(gc_cluster))
            semi_est.update_design(c_design)
            hat_tau, _ = semi_est.estimate(
                env, N=SAMPLE_NUM, seed=r, random=True
            )
            error_gc[r] = hat_tau - true_tau
        if "OGC" in METHOD_LIST:
            BATCH_SAMPLE_NUM = 10
            num_sample_iter = int(SAMPLE_NUM / BATCH_SAMPLE_NUM)
            tau_value_list = np.zeros(num_sample_iter)
            for i in range(num_sample_iter):
                if i == 0:
                    # init_design = IndividualDesign(p=PROB, W=env.get_adj_matrix())
                    init_design = GlobalDesign(p=PROB, W=W)
                    semi_est.update_design(init_design)
                    tau_value, prev_data, hat_V = semi_est.estimate(
                        env,
                        N=BATCH_SAMPLE_NUM,
                        seed=r,
                        random=True,
                        regression_type='pool', 
                        return_cov=True,
                    )
                else:
                    gc_cluster, gc_oracle_objective = multi_graph_cut(
                        W=W,
                        V=hat_V,
                        m_max=M_MAX,
                        verbose=False,
                    )
                    gc_cluster = gc_cluster[-1]
                    c_design = ClusterDesign(PROB, W, label2dict(gc_cluster))
                    semi_est.update_design(c_design)
                    tau_value, prev_data, hat_V = semi_est.estimate(
                        env,
                        N=BATCH_SAMPLE_NUM,
                        seed=r + i * 2025,
                        random=True,
                        regression_type='pool', 
                        prev_data=prev_data,
                        return_cov=True,
                    )
                tau_value_list[i] = tau_value
            hat_tau = np.mean(tau_value_list)
            error_ogc[r] = hat_tau - true_tau
        if "OGC-ST" in METHOD_LIST:
            BATCH_SAMPLE_NUM = 10
            num_sample_iter = int(SAMPLE_NUM / BATCH_SAMPLE_NUM)
            tau_value_list = np.zeros(num_sample_iter)
            hat_MSE_list = np.zeros(num_sample_iter)
            oracle_MSE_list = np.zeros(num_sample_iter)
            for i in range(num_sample_iter):
                if i == 0:
                    # init_design = IndividualDesign(p=PROB, W=env.get_adj_matrix())
                    init_design = GlobalDesign(p=PROB, W=W)
                    semi_est.update_design(init_design)
                    tau_value, _, hat_V, prev_error = semi_est.estimate(
                        env,
                        N=BATCH_SAMPLE_NUM,
                        seed=r,
                        random=True,
                        regression_type='pool', 
                        return_cov=True,
                        return_error=True,
                    )
                    hat_MSE = multi_objective(W, hat_V)(np.ones((W.shape[0], 1)))
                    oracle_MSE = multi_objective(W, V)(np.ones((W.shape[0], 1)))
                else:
                    gc_cluster, gc_oracle_objective = multi_graph_cut(
                        W=W,
                        V=hat_V,
                        m_max=M_MAX,
                        verbose=False,
                    )
                    gc_cluster = gc_cluster[-1]
                    c_design = ClusterDesign(PROB, W, label2dict(gc_cluster))
                    semi_est.update_design(c_design)
                    tau_value, _, hat_V, prev_error = semi_est.estimate(
                        env,
                        N=BATCH_SAMPLE_NUM,
                        seed=r + i * 2025,
                        random=True,
                        regression_type='pool', 
                        return_cov=True,
                        prev_error=prev_error,
                        return_error=True,
                    )
                    hat_MSE = multi_objective(W, hat_V)(onehot_trans(gc_cluster))
                    oracle_MSE = multi_objective(W, V)(onehot_trans(gc_cluster))
                tau_value_list[i] = tau_value
                hat_MSE_list[i] = hat_MSE
                oracle_MSE_list[i] = oracle_MSE
            hat_tau = np.mean(tau_value_list)
            inv_hat_MSE_list = 1 / np.sqrt(hat_MSE_list)
            inv_oracle_MSE_list = 1 / np.sqrt(oracle_MSE_list)
            hat_tau_3 = np.dot(inv_hat_MSE_list / np.sum(inv_hat_MSE_list), tau_value_list)
            hat_tau_4 = np.dot(inv_oracle_MSE_list / np.sum(inv_oracle_MSE_list), tau_value_list)
            error_ogc_st[r] = hat_tau - true_tau

    if SAVE:
        import pandas as pd

        df = pd.DataFrame(
            {
                "true": true_tau * np.ones(NREP),
                "gc-oracle": error_gc_oracle,
                "gc": error_gc,
                "ogc": error_ogc, 
                "ogc-st": error_ogc_st, 
            }
        )
        if CONSISTENCY:
            filename = "{}/td_num{}_grid{}_{}_exposure{}_{}_rho{}_lnoise{}.csv".format(
                DIR_NAME,
                SAMPLE_NUM,
                GRID_SIZE,
                PATTERN,
                EXPOSURE,
                COR_TYPE,
                RHO,
                LOC_NOISE,
            )
        df.to_csv(filename)
