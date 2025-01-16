# Pytorch implementation of Diffusion models in SE(3) for grasp and motion generation

This library provides the tools for training and sampling diffusion models in SE(3),
implemented in PyTorch. 
We apply them to learn 6D grasp distributions. We use the learned distribution as cost function
for grasp and motion optimization problems.
See reference [1] for additional details.

[[Website]](https://sites.google.com/view/se3dif/home)      [[Preprint]](https://arxiv.org/pdf/2209.03855.pdf)

<img src="assets/grasp_dif.gif" alt="diffusion" style="width:800px;"/>

## Installation

Create conda environment

```bash
conda env create -f environment.yaml
conda activate se3diff
python -m pip install -r requirements.txt
python -m pip install -e .
```

Prepare data

```bash
ln -s /mnt/kostas-graid/datasets/boshu/data data
```

## Run

```bash
python scripts/train/train_pointcloud_6d_grasp_diffusion.py
```
