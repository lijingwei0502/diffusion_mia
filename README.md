# Towards Black-Box Membership Inference Attack for Diffusion Models
This repository contains the code of the paper [Towards Black-Box Membership Inference Attack for Diffusion Models](https://arxiv.org/pdf/2405.20771).

## DDIM

### Requirements

Install dependencies from the provided conda environment file:

```bash
conda env create -f DDIM/environment.yml
```

---

### Train DDIM Model

We provide pre-defined dataset splits such as `DDIM/CIFAR10_train_ratio0.5.npz` (stored under the `DDIM/` directory).
Place the raw datasets (`cifar10`, `cifar100`, `stl10`) under:

```
DDIM/data/datasets/pytorch
```

To train the model:

```bash
cd DDIM
python train.py
```

---

### Attack DDIM Model

Run membership inference attacks on a trained checkpoint.
Checkpoints are expected in the format `logs/DDPM_<DATASET>_EPS/ckpt-step<STEP>.pt`.

**Example command:**

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

**Key parameters:**

* `--checkpoint` — Training step of the saved checkpoint (e.g., `800000` → `ckpt-step800000.pt`)
* `--dataset` — One of: `CIFAR10` | `CIFAR100` | `STL10` | `SVHN` | `TINY-IN`
* `--attacker_name` — Attack method (case-sensitive):
  `Naive` (Loss) | `SecMI` | `PIA` | `PIAN` | `ReDiffuse` (ours)
* `--attack_num`, `--interval`, `--norm`, `--k` — Attack hyperparameters

**Outputs:**

* **Training:** Checkpoints stored under `logs/DDPM_*_EPS/`
* **Attack:**

  * Prints `AUC`, `ASR`, `TPR@1%FPR` to console
  * Appends results to `result.csv` in the current directory

---

## DiT

### Requirements

Install dependencies from the provided conda environment file:

```bash
conda env create -f DiT/environment.yml
```

---

### Train DiT Model

We use the DiT-XL/2 architecture by default. The script supports multi-GPU training via PyTorch DDP.


#### Example command

```bash
torchrun --nproc_per_node=<NUM_GPUS> train.py \
  --data-path /path/to/dataset \
  --results-dir ./results \
  --model DiT-XL/2 \
  --image-size 256 \
  --num-classes 1000 \
  --epochs 1400 \
  --global-batch-size 256 \
  --vae ema
```

#### Key arguments

* `--data-path` — Path to dataset root (ImageFolder format)
* `--results-dir` — Output directory for logs and checkpoints
* `--model` — Model architecture (`DiT-XL/2`, etc.)
* `--image-size` — Image resolution (`128`, `256`, `512`)
* `--num-classes` — Number of classes in the dataset
* `--epochs` — Training epochs
* `--global-batch-size` — Total batch size across all GPUs
* `--vae` — Pretrained VAE variant (`ema` or `mse`)

#### Outputs

* Training logs in `<results-dir>/<EXP_ID>-<MODEL>/log.txt`
* Checkpoints in `<results-dir>/<EXP_ID>-<MODEL>/checkpoints/`

---

### Attack DiT Model

Runs membership inference attacks (MIA) on a trained DiT checkpoint.
Supports multiple attack types: `Naive`, `SecMI`, `PIA`, `PIAN`, `Denoise`.

#### Example command

```bash
python attack.py \
  --model DiT-XL/2 \
  --vae mse \
  --image-size 256 \
  --ckpt /path/to/checkpoint.pt \
  --member-data-path /path/to/member/images \
  --nonmember-data-path /path/to/nonmember/images \
  --mia-type denoise \
  --t-step 1 \
  --k 10 \
  --experiments 10
```

#### Key arguments

* `--model` — Model architecture (`DiT-XL/2`, etc.)
* `--vae` — Pretrained VAE variant (`ema` or `mse`)
* `--image-size` — Image resolution (`128`, `256`, `512`)
* `--ckpt` — Path to trained checkpoint (`.pt`)
* `--member-data-path` — Path to member dataset
* `--nonmember-data-path` — Path to non-member dataset
* `--mia-type` — Attack type:
  `naive` | `secmi` | `pia` | `pian` | `denoise`
* `--t-step` — Number of timesteps for adding noise
* `--k` — Step size for DDIM sampling
* `--experiments` — Number of repeated runs for averaging

#### Outputs

* Console: `AUC`, `ASR`, `TPR@1%FPR`
* Appends results to `results.csv`

---

## Stable Diffusion

### Requirements

Install dependencies from the provided conda environment file:

```bash
conda env create -f Stable_Diffusion/environment.yml
```

---

### Use Pretrained Checkpoint

This pipeline uses a pretrained Stable Diffusion v1.4 checkpoint and **does not** train any model.
By default it loads from: `CompVis/stable-diffusion-v1-4`.

---

### Attack Stable Diffusion

Runs membership inference attacks (MIA) against Stable Diffusion using your provided script.
Supported attackers: `SecMI`, `PIA`, `Naive`, `PIAN`, `ReDiffuse` (default).

#### Example command

```bash
python attack_sd.py \
  --attacker_name ReDiffuse \
  --dataset laion5 \
  --checkpoint CompVis/stable-diffusion-v1-4 \
  --attack_num 1 \
  --interval 50 \
  --k 50 \
  --average 1 \
  --seed 0
```

> The script auto-selects a GPU with free memory via NVML; no need to pass `--device`.

#### Key parameters

* `--attacker_name` — Attack method (case-sensitive):
  `SecMI` | `PIA` | `Naive` | `PIAN` | `ReDiffuse`
* `--dataset` — Dataset name passed to `load_member_data` (e.g., `laion5`).
* `--checkpoint` — pretrained model of stable diffusion (default: `CompVis/stable-diffusion-v1-4`)
* `--attack_num` — Number of attack rounds per sample
* `--interval` — DDIM interval used inside the attacker
* `--k` — DDIM step size
* `--average` — Number of repeated runs for averaging (some attackers)
* `--seed` — Random seed


---

### Outputs

* Console metrics: `AUC`, `ASR`, `TPR@1%FPR`
* Appends a CSV row to `result.csv` with:

  ```
  dataset,attacker_name,update,attack_num,interval,k,average,auc,asr,tpr@1%fpr
  ```

---

**Acknowledgements**: this repository uses codes and resources from [An Efficient Membership Inference Attack for the Diffusion Model by Proximal Initialization](https://github.com/kong13661/PIA), [Scalable Diffusion Models with Transformers (DiT)](https://github.com/facebookresearch/DiT).

## Citation

```
@inproceedings{li2025towards,
  title={Towards black-box membership inference attack for diffusion models},
  author={Li, Jingwei and Dong, Jing and He, Tianxing and Zhang, Jingzhao},
  booktitle={International Conference on Machine Learning},
  year={2025},
  organization={PMLR}
}

```

