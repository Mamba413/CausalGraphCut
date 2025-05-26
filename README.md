Balancing Interference and Correlation in Spatial Experimental Designs: A Causal Graph Cut Approach
---------------

This repository contains the implementation for the paper "[Balancing Interference and Correlation in Spatial Experimental Designs: A Causal Graph Cut Approach](https://icml.cc/virtual/2025/poster/43725)" (ICML 2025) in Python.

### Summary of the paper

This paper focuses on the design of spatial experiments to optimize the amount of information derived from the experimental data and enhance the accuracy of the resulting causal effect estimator. We propose a surrogate function for the mean squared error of the estimator, which facilitates the use of classical graph cut algorithms to learn the optimal design. Our proposal offers three key advances: (1) it accommodates moderate to large spatial interference effects; (2) it adapts to different spatial covariance functions; (3) it is computationally efficient.

![](figure/m-cut.png)

### Reproduction guidance

- Change your working directory to this main folder, run `setup.sh` to configure the environment and install all requirements.
- `./figure3b.sh` --> reproduce Figure 3(b)
- `./figure6.sh` --> reproduce Figure 6
- `./figure7.sh` --> reproduce Figure 7
- `./figure8&9.sh` --> reproduce Figure 8 and Figure 9

### Using the method

**Warm-up**. If you can assess the _spatial covariance_, then you can employ _oracel_ causal graph cut by following these steps:

```python
### 1. configure the double robust estimator
from sklearn.ensemble import RandomForestRegressor
from semi_sp_design import SemiEstimator

model = RandomForestRegressor(random_state=0, n_estimators=10)
semi_est = SemiEstimator(n_splits=2, model=model)

### 2. get spatial clusters by (oracle) causal graph cut
from SemiGraphCut import multi_graph_cut

W = your_env.get_adj_matrix()
V = your_env.get_cov_matrix()
spat_cluster, _ = multi_graph_cut(W=W, V=V)

### 3. get the ATE estimation based on the cluster design
c_design = ClusterDesign(p=0.5, W=W, cluster=spat_cluster)
semi_est.update_design(c_design)
hat_tau_C, _ = semi_est.estimate(your_env, N=100)
print("Estimator:", hat_tau_C)
```

**More realistic cases**. Iteratively estimate spatial covariance via the **causal graph cut** by following these steps:

```python
### configure the double robust estimator as previous
from sklearn.ensemble import RandomForestRegressor
from semi_sp_design import SemiEstimator
model = RandomForestRegressor(random_state=0, n_estimators=10)
semi_est = SemiEstimator(n_splits=2, model=model)

### perform the causal graph cut algorithm
from SemiGraphCut import online_graph_cut
online_graph_cut(your_env, semi_est)
hat_tau_C, _, _ = semi_est.estimate(your_env, N=100)
print("Estimator:", hat_tau_C)
```

### Citation

Please cite our paper [Balancing Interference and Correlation in Spatial Experimental Designs: A Causal Graph Cut Approach (ICML 2025)](https://icml.cc/virtual/2025/poster/43725)

```
@inproceedings{zhu2025balancing,
  title={Balancing Interference and Correlation in Spatial Experimental Designs: A Causal Graph Cut Approach},
  author={Zhu, Jin and Li, Jingyi and Zhou, Hongyi and Lin, Yinan and Lin, Zhenhua and Shi, Chengchun},
  booktitle={International Conference on Machine Learning},
  year={2025},
  organization={PMLR}
}
```
