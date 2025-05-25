from RSemiGraphCut import recursive_graph_cut
from SemiGraphCut import multi_graph_cut
from data import *
import argparse
# from semi_sp_design import (
from semi_sp_design_norepeat import (
    # SemiEstimator,
    SemiEstimatorNoRepeat,
    ClusterDesign,
    IndividualDesign,
    GlobalDesign,
    cluster_yang, 
)
from sklearn.ensemble import RandomForestRegressor
from benchmark import AOS, KDD, CausalClustering, IPWEstimator, NaiveIPWEstimator
from tqdm import tqdm
from utils import label2dict, onehot_trans
from utils_semi import multi_objective
from plot_region import plot_hexagon
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Simulation on semi-parametric models")
parser.add_argument("--pattern", type=str, default="hexagon")
parser.add_argument("--exposure", type=int, default=1)
parser.add_argument("--model", type=str, default="semi-static")
parser.add_argument("--rho", type=float, default=0.5)
parser.add_argument("--cor-type", type=str, default="example3")
parser.add_argument("--grid-size", type=int, default=12)
parser.add_argument("--loc-std", type=float, default=0.0)
parser.add_argument("--nrep", type=int, default=100)
parser.add_argument("--m-max", type=int, default=None)
parser.add_argument("--sample-num", type=int, default=50)

SAVE = True
CONSISTENCY = True
DIR_NAME = "result-error-ex12-norepeat-100"
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
    # SAMPLE_NUM = args.sample_num
    SAMPLE_NUM = 1

    METHOD_LIST = [
        "Global",
        "Individual",
        "GC (Oracle)",
        "GC (0.01)",
        "GC (0.1)",
        "GC (1.0)",
        # "GC",
        # "OGC",     # use all data at all iteration for training machine learning model
        # "OGC-ST",
        "AOS",
        "KDD",
        "CC",
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
    R = W.shape[0]
    true_tau = env.tau

    ## Oracle V
    error_global = -9999.99 * np.ones(NREP)
    error_individual = -9999.99 * np.ones(NREP)
    error_gc_oracle = -9999.99 * np.ones(NREP)
    error_gc001 = -9999.99 * np.ones(NREP)
    error_gc01 = -9999.99 * np.ones(NREP)
    error_gc1 = -9999.99 * np.ones(NREP)
    error_gc = -9999.99 * np.ones(NREP)
    error_ogc = -9999.99 * np.ones(NREP)
    error_ogc_st = -9999.99 * np.ones(NREP)
    error_aos = -9999.99 * np.ones(NREP)
    error_kdd = -9999.99 * np.ones(NREP)
    error_cc = -9999.99 * np.ones(NREP)

    if "CC" in METHOD_LIST or 'CC-DR' in METHOD_LIST:
        cc_cluster = CausalClustering(W, xi=8)
    if "KDD" in METHOD_LIST or 'KDD-DR' in METHOD_LIST:
        kdd_cluster = KDD(W)
    if "AOS" in METHOD_LIST or "AOS-DR" in METHOD_LIST:
        aos_cluster = AOS(env.grid) 

    for r in tqdm(range(NREP)):
        model = RandomForestRegressor(random_state=r, n_estimators=10, min_samples_leaf=1)
        # semi_est = SemiEstimator(n_splits=2, model=model)
        semi_est = SemiEstimatorNoRepeat(model=model)
        if "Global" in METHOD_LIST:
            g_design = GlobalDesign(PROB, W)
            semi_est.update_design(g_design)
            hat_tau, _ = semi_est.estimate(
                env, N=SAMPLE_NUM, seed=2*r+200, random=True
            )
            error_global[r] = hat_tau - true_tau
        if "Individual" in METHOD_LIST:
            i_design = IndividualDesign(PROB, W)
            semi_est.update_design(i_design)
            hat_tau, _ = semi_est.estimate(
                env, N=SAMPLE_NUM, seed=2*r+200, random=True
            )
            error_individual[r] = hat_tau - true_tau
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
                # env, N=SAMPLE_NUM, seed=r, random=True
                # env, N=SAMPLE_NUM, seed=2*r+200, random=True #changeseed 1
                env, N=SAMPLE_NUM, seed=2*r+200, random=True #changeseed 2
                # env, N=SAMPLE_NUM, seed=3*r+500, random=True #changeseed 3
                # env, N=SAMPLE_NUM, seed=5*r+200, random=True #changeseed 4
            )
            # print(hat_tau)
            error_gc_oracle[r] = hat_tau - true_tau
        if "GC (0.01)" in METHOD_LIST:
            noise = np.random.normal(size=(R, R))
            V_hat = V + 0.01 * ((noise + noise.T)/2)
            gc_oracle, gc_oracle_value = multi_graph_cut(
                W=W,
                V=V_hat,
                m_max=M_MAX,
                verbose=False,
            )
            gc_oracle = gc_oracle[-1]
            c_design = ClusterDesign(PROB, W, label2dict(gc_oracle))
            semi_est.update_design(c_design)
            hat_tau, _ = semi_est.estimate(
                env, N=SAMPLE_NUM, seed=2*r+200, random=True
            )
            error_gc001[r] = hat_tau - true_tau
        if "GC (0.1)" in METHOD_LIST:
            noise = np.random.normal(size=(R, R))
            V_hat = V + 0.1 * ((noise + noise.T)/2)
            gc_oracle, gc_oracle_value = multi_graph_cut(
                W=W,
                V=V_hat,
                m_max=M_MAX,
                verbose=False,
            )
            gc_oracle = gc_oracle[-1]
            c_design = ClusterDesign(PROB, W, label2dict(gc_oracle))
            semi_est.update_design(c_design)
            hat_tau, _ = semi_est.estimate(
                env, N=SAMPLE_NUM, seed=2*r+200, random=True
            )
            error_gc01[r] = hat_tau - true_tau
        if "GC (1.0)" in METHOD_LIST:
            noise = np.random.normal(size=(R, R))
            V_hat = V + 1.0 * ((noise + noise.T)/2)
            gc_oracle, gc_oracle_value = multi_graph_cut(
                W=W,
                V=V_hat,
                m_max=M_MAX,
                verbose=False,
            )
            gc_oracle = gc_oracle[-1]
            c_design = ClusterDesign(PROB, W, label2dict(gc_oracle))
            semi_est.update_design(c_design)
            hat_tau, _ = semi_est.estimate(
                env, N=SAMPLE_NUM, seed=2*r+200, random=True
            )
            error_gc1[r] = hat_tau - true_tau
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
            BATCH_SAMPLE_NUM = 5
            num_sample_iter = int(SAMPLE_NUM / BATCH_SAMPLE_NUM)
            # num_sample_iter = 10
            tau_value_list = np.zeros(num_sample_iter)
            for i in range(num_sample_iter):
                if i == 0:
                    # init_design = IndividualDesign(p=PROB, W=env.get_adj_matrix())
                    init_design = GlobalDesign(p=PROB, W=W)
                    semi_est.update_design(init_design)
                    tau_value, prev_data, hat_V = semi_est.estimate(
                        env,
                        N=BATCH_SAMPLE_NUM,
                        # seed=r,
                        # seed=r+200, # changeseed 1
                        seed=2*r+200, # changeseed 2
                        # seed=3*r+500, # changeseed 3
                        # seed=5*r+200, # changeseed 4
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
                        # seed=r + i * 2025,
                        # seed=r+200 +i * 2025, # changeseed 1
                        seed=2*r+200 +i * 2025, # changeseed 2
                        # seed=3*r+500 +i * 2025, # changeseed 3
                        # seed=5*r+200 + i * 2025, # changeseed 4
                        random=True,
                        regression_type='pool', 
                        prev_data=prev_data,
                        return_cov=True,
                    )
                tau_value_list[i] = tau_value
            # hat_tau = np.mean(tau_value_list)
            burn_in = int(num_sample_iter//2)
            hat_tau = np.mean(tau_value_list[burn_in:])
            error_ogc[r] = hat_tau - true_tau
            print("error:", error_ogc[r])
            # plot_hexagon(env.grid, gc_cluster)
        if "OGC-ST" in METHOD_LIST:
            BATCH_SAMPLE_NUM = 50
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
            error_ogc_st[r] = hat_tau - true_tau
        if "KDD" in METHOD_LIST:
            c_design = ClusterDesign(PROB, W, label2dict(kdd_cluster))
            ipw_est = IPWEstimator(c_design)
            hat_tau = ipw_est.estimate(env, N=SAMPLE_NUM, seed=r, random=True)
            error_kdd[r] = hat_tau - true_tau
        if "AOS" in METHOD_LIST:
            c_design = ClusterDesign(PROB, W, label2dict(aos_cluster))
            ipw_est = IPWEstimator(c_design)
            hat_tau = ipw_est.estimate(env, N=SAMPLE_NUM, seed=r, random=True)
            error_aos[r] = hat_tau - true_tau
        if "CC" in METHOD_LIST:
            c_design = ClusterDesign(PROB, W, label2dict(cc_cluster))
            naive_ipw_est = NaiveIPWEstimator(c_design)
            hat_tau = naive_ipw_est.estimate(env, N=SAMPLE_NUM, seed=r, random=True)
            error_cc[r] = hat_tau - true_tau

    if SAVE:
        import pandas as pd

        df = pd.DataFrame(
            {
                "true": true_tau * np.ones(NREP),
                "global": error_global,
                "individual": error_individual,
                "gc-oracle": error_gc_oracle,
                "gc": error_gc,
                "gc(001)": error_gc001,
                "gc(01)": error_gc01,
                "gc(1)": error_gc1,
                "ogc": error_ogc, 
                "ogc-st": error_ogc_st, 
                "aos": error_aos,
                "kdd": error_kdd,
                "cc": error_cc,
            }
        )
        if CONSISTENCY:
            filename = "{}/num{}_grid{}_{}_exposure{}_{}_rho{}_lnoise{}.csv".format(
                DIR_NAME,
                SAMPLE_NUM,
                GRID_SIZE,
                PATTERN,
                EXPOSURE,
                COR_TYPE,
                RHO,
                LOC_NOISE,
            )
        else:
            filename = "{}/MSE{}_{}_exposure{}_{}_rho{}_lnoise{}.csv".format(
                DIR_NAME,
                GRID_SIZE,
                PATTERN,
                EXPOSURE,
                COR_TYPE,
                RHO,
                LOC_NOISE,
            )
        df.to_csv(filename)
    else:
        if "Global" in METHOD_LIST:
            print("Global Mean: {} (Std: {})".format(np.mean(error_global), np.std(error_global)))
        if "Individual" in METHOD_LIST:
            print("Individual Mean: {} (Std: {})".format(np.mean(error_individual), np.std(error_individual)))
        if "GC (Oracle)" in METHOD_LIST:
            print("OracleGC Mean: {} (Std: {})".format(np.mean(error_gc_oracle), np.std(error_gc_oracle)))
            plot_hexagon(env.grid, gc_oracle)
        if "GC" in METHOD_LIST:
            print("GC Mean: {} (Std: {})".format(np.mean(error_gc), np.std(error_gc)))
        if "OGC" in METHOD_LIST:
            print("OGC Mean: {} (Std: {})".format(np.mean(error_ogc), np.std(error_ogc)))
            plot_hexagon(env.grid, gc_cluster)
        if "OGC-ST" in METHOD_LIST:
            print("OGC-ST Mean: {} (Std: {})".format(np.mean(error_ogc_st), np.std(error_ogc_st)))
        if "AOS" in METHOD_LIST:
            print("AOS Mean: {} (Std: {})".format(np.mean(error_aos), np.std(error_aos)))
        if "KDD" in METHOD_LIST:
            print("KDD Mean: {} (Std: {})".format(np.mean(error_kdd), np.std(error_kdd)))
        if "CC" in METHOD_LIST:
            print("CC Mean: {} (Std: {})".format(np.mean(error_cc), np.std(error_cc)))
