# Towards Black-Box Membership Inference Attack for Diffusion Models
Official repo for [Towards Black-Box Membership Inference Attack for Diffusion Models](https://arxiv.org/pdf/2405.20771).

This repository contains the code of the paper [Towards Black-Box Membership Inference Attack for Diffusion Models](https://arxiv.org/pdf/2405.20771).




## DDIM

### Requirements



### Train DDIM model

We provide dataset splits such as `DDIM/CIFAR10_train_ratio0.5.npz` (placed under the `DDIM/` folder).
Put datasets (`cifar10`, `cifar100`, `stl10`, `svhn`, `tiny-imagenet`) under:

```
DDIM/data/datasets/pytorch
```

You can change paths in:

* `train.py` → `get_dataset`
* `dataset_utils.load_member_data`

Change log dir / dataset in `train.py` by editing:

```python
FLAGS.logdir   # e.g., './logs/DDPM_SVHN_EPS'
FLAGS.dataset  # 'CIFAR10' | 'CIFAR100' | 'STL10' | 'SVHN' | 'TINY-IN'
```

**Command:**

```bash
cd DDIM
python train.py
```

---

### Attack DDIM model

Runs Membership Inference Attacks on a trained checkpoint (expects files like `logs/DDPM_<DATASET>_EPS/ckpt-step<STEP>.pt`).

**Command:**

```bash
cd DDIM
python attack.py \
  --checkpoint 800000 \
  --dataset CIFAR10 \
  --attacker_name ReDiffuse \
  --attack_num 1 \
  --interval 200 \
  --norm 1 \
  --k 100
```

**Parameters:**

* `--checkpoint` — training step of the saved ckpt (e.g., `800000` → `ckpt-step800000.pt`)
* `--dataset` — `CIFAR10` | `CIFAR100` | `STL10` | `SVHN` | `TINY-IN`
* `--attacker_name` (case-sensitive) —
  `Naive` (Loss) | `SecMI` | `PIA` | `PIAN` | `ReDiffuse` (ours)
* `--attack_num`, `--interval`, `--norm`, `--k` — attack hyperparameters

**Outputs:**

* Training: checkpoints under `logs/DDPM_*_EPS/`
* Attack: prints `AUC`, `ASR`, `TPR@1%FPR` and appends to `result.csv`

## DiT

### train DiT model

We provide the split of the dataset. They are `DDPM/CIFAR10_train_ratio0.5.npz` and `DDPM/TINY-IN_train_ratio0.5.npz`. To train the DDPM, you need put the `cifar10`, `cifar100`, `stl10`, `tiny-imagenet`dataset into `DDPM/data/pytorch`. You can also change the directory by modifying the path in `main.get_dataset` function and `dataset_utils.load_member_data`.  You can change the log directory by modifying `FLAGS.logdir` in `main.py`. You can change the `FLAGS.dataset` to select the dataset.

Then, to train the DDPM, just run command below.
```bash
cd DDPM
python main.py
```

### attack DiT model
Just run command below.
```bash
cd DDPM
python micro_attack.py 
```

The meaning of those parameters:

`--checkpoint` The checkpoint you saved.

`--dataset` The dataset to attack. It can be `cifar10`,`cifar100`,`stl10` or `TINY-IN`.

`--attacker_name` The attack method. `naive` for Loss attack. `SecMI` for SecMI attack. `PIA` for PIA and `PIAN` for PIAN, `Denoise` for our algorithm rediffuse attack.

## Stable Diffusion

We conduct experiments on the original Stable Diffusion model, i.e., stable-diffusion-v1-5 provided by Huggingface, without further fine-tuning or modifications.

### attack Stable Diffusion

for our algorithm rediffuse attack, just run command below.
```bash
cd stable_diffusion
python stable_attack.py 
```

for our algorithm rediffuse+ attack, just run command below.
```bash
cd stable_diffusion
python two_attack.py 
```


**Acknowledgements**: this repository uses codes and resources from [An Efficient Membership Inference Attack for the Diffusion Model by Proximal Initialization](https://github.com/kong13661/PIA).


## Citation

```
@inproceedings{li2025towards,
  title={Towards black-box membership inference attack for diffusion models},
  author={Li, Jingwei and Dong, Jing and He, Tianxing and Zhang, Jingzhao},
  booktitle={International Conference on Machine Learning},
  <!-- pages={8717--8730}, -->
  year={2025},
  organization={PMLR}
}

```
