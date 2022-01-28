# DS-UI
DS-UI: Dual-Supervised Mixture of Gaussian Mixture Models for Uncertainty Inference in Image Recognition (IEEE TIP 2021) [IEEE Xplore](https://ieeexplore.ieee.org/document/9605222 "IEEE Xplore") or [ArXiv](https://arxiv.org/abs/2011.08595 "ArXiv")

## Code List
+ main.py
	+ Main file for running
+ model_resnet.py
	+ Implementation for ResNet
+ gmm_layer.py
	+ Implementation for MoGMM-FC layer
+ uncertainty_measurements.py
	+ Implementation for uncertainty measurements

## Backbone
### ResNet

## Dataset
### CIFAR-10

## Requirements
- python >= 3.6
- PyTorch >= 1.1.0
- torchvision >= 0.3.0
- sklearn >= 0.19.1
- GPU memory >= 5000MiB (GTX 1080Ti)

## Training
- Download datasets
- Train and evaluate: `python main.py` or use nohup `nohup python main.py >1.out 2>&1 &`

## Args in main.py
- savepath: Save path of checkpoint and results
- repeattimes: Times of independent repeated tests
- card: Index of the used GPU
- n_component: Number of components of each GMM in MoGMM

## Citation
If you find this paper useful in your research, please consider citing:
```
@ARTICLE{9605222,
  author={Xie, Jiyang and Ma, Zhanyu and Xue, Jing-Hao and Zhang, Guoqiang and Sun, Jian and Zheng, Yinhe and Guo, Jun},
  journal={IEEE Transactions on Image Processing}, 
  title={{DS-UI}: {D}ual-Supervised Mixture of {G}aussian Mixture Models for Uncertainty Inference in Image Recognition}, 
  year={2021},
  volume={30},
  number={},
  pages={9208-9219},
  doi={10.1109/TIP.2021.3123555}}
```
