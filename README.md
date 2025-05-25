Balancing Interference and Correlation in Spatial Experimental Designs: A Causal Graph Cut Approach
---------------

This repository contains the implementation for the paper "Balancing Interference and Correlation in Spatial Experimental Designs: A Causal Graph Cut Approach" (ICML 2025) in Python. 

### Summary of the paper

This paper focuses on the design of spatial experiments to optimize the amount of information derived from the experimental data and enhance the accuracy of the resulting causal effect estimator. We propose a surrogate function for the mean squared error of the estimator, which facilitates the use of classical graph cut algorithms to learn the optimal design. Our proposal offers three key advances: (1) it accommodates moderate to large spatial interference effects; (2) it adapts to different spatial covariance functions; (3) it is computationally efficient.

### Reproduction guidance

- Change your working directory to this main folder, run `setup.sh` to install all requirements.
- `./figure3b.sh` --> reproduce Figure 3(b)
- `./figure6.sh` --> reproduce Figure 6
- `./figure7.sh` --> reproduce Figure 7

### Citation

Please cite our paper [Balancing Interference and Correlation in Spatial Experimental Designs: A Causal Graph Cut Approach (ICML 2025)]()

```
@inproceedings{zhu2025balancing,
  title={Balancing Interference and Correlation in Spatial Experimental Designs: A Causal Graph Cut Approach},
  author={Zhu, Jin and Li, Jingyi and Zhou, Hongyi and Lin, Yinan and Lin, Zhenhua and Shi, Chengchun},
  booktitle={International Conference on Machine Learning},
  year={2025},
  organization={PMLR}
}
```
