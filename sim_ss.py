from SemiGraphCut import multi_graph_cut
from data import *
import argparse
from semi_sp_design import (
    SemiEstimator,
    ClusterDesign,
    GlobalDesign,
)
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from utils import label2dict
from utils_semi import multi_objective

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Simulation on semi-parametric models on training style (local or pool)")
parser.add_argument("--pattern", type=str, default="hexagon")
parser.add_argument("--exposure", type=int, default=1)
parser.add_argument("--model", type=str, default="semi-static")
parser.add_argument("--rho", type=float, default=0.5)
parser.add_argument("--cor-type", type=str, default="example1")
parser.add_argument("--grid-size", type=int, default=12)
parser.add_argument("--loc-std", type=float, default=0.0)
parser.add_argument("--nrep", type=int, default=20)
parser.add_argument("--m-max", type=int, default=None)
parser.add_argument("--sample-num", type=int, default=20)

SAVE = True
CONSISTENCY = True
DIR_NAME = "result-ss"
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
        "GC-Oracle-pool",
        "GC-Oracle-local",
        "OGC-pool",
        "OGC-local",
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
    error_oracle_pool = -9999.99 * np.ones(NREP)
    error_oracle_local = -9999.99 * np.ones(NREP)
    error_ogc_pool = -9999.99 * np.ones(NREP)
    error_ogc_local = -9999.99 * np.ones(NREP)

    for r in tqdm(range(NREP)):
        model = RandomForestRegressor(random_state=r, n_estimators=10, min_samples_leaf=1)
        semi_est = SemiEstimator(n_splits=2, model=model)
        if "GC-Oracle-pool" in METHOD_LIST:
            gc_oracle, gc_oracle_value = multi_graph_cut(
                W=W,
                V=V,
                m_max=M_MAX,
                verbose=False,
            )
            gc_oracle = gc_oracle[-1]
            c_design = ClusterDesign(PROB, W, label2dict(gc_oracle))
            semi_est.update_design(c_design)
            hat_tau = semi_est.estimate(
                env, N=SAMPLE_NUM, seed=r, random=True, regression_type='pool'
            )
            error_oracle_pool[r] = hat_tau - true_tau
        if "GC-Oracle-local" in METHOD_LIST:
            gc_oracle, gc_oracle_value = multi_graph_cut(
                W=W,
                V=V,
                m_max=M_MAX,
                verbose=False,
            )
            gc_oracle = gc_oracle[-1]
            c_design = ClusterDesign(PROB, W, label2dict(gc_oracle))
            semi_est.update_design(c_design)
            hat_tau = semi_est.estimate(
                env, N=SAMPLE_NUM, seed=r, random=True, regression_type='local'
            )
            error_oracle_local[r] = hat_tau - true_tau
        if "OGC-pool" in METHOD_LIST:
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
                    hat_V, prev_error, tau_value = semi_est.estimate_cov(
                        env, N=BATCH_SAMPLE_NUM, seed=r, random=True, regression_type='pool',
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
                    hat_V, prev_error, tau_value = semi_est.estimate_cov(
                        env, N=BATCH_SAMPLE_NUM, seed=r + i * 2025, random=True, regression_type='pool', prev_data=prev_error,
                    )
                tau_value_list[i] = tau_value
            hat_tau = np.mean(tau_value_list)
            error_ogc_pool[r] = hat_tau - true_tau
        if "OGC-local" in METHOD_LIST:
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
                    hat_V, prev_error, tau_value = semi_est.estimate_cov(
                        env, N=BATCH_SAMPLE_NUM, seed=r, random=True, regression_type='local',
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
                    hat_V, prev_error, tau_value = semi_est.estimate_cov(
                        env, N=BATCH_SAMPLE_NUM, seed=r + i * 2025, random=True, regression_type='local', prev_data=prev_error,
                    )
                tau_value_list[i] = tau_value
            hat_tau = np.mean(tau_value_list)
            error_ogc_local[r] = hat_tau - true_tau
    if SAVE:
        import pandas as pd

        df = pd.DataFrame(
            {
                "true": true_tau * np.ones(NREP),
                "gc-oracle-pool": error_oracle_pool,
                "gc-oracle-local": error_oracle_local,
                "ogc-pool": error_ogc_pool, 
                "ogc-local": error_ogc_local, 
            }
        )
        if CONSISTENCY:
            filename = "{}/num{}_grid{}_{}_{}_rho{}.csv".format(
                DIR_NAME,
                SAMPLE_NUM,
                GRID_SIZE,
                PATTERN,
                COR_TYPE,
                RHO,
            )
        df.to_csv(filename)

