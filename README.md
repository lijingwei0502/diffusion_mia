# Towards Black-Box Membership Inference Attack for Diffusion Models

## DDPM

### train DDPM model

We provide the split of the dataset. They are `DDPM/CIFAR10_train_ratio0.5.npz` and `DDPM/TINY-IN_train_ratio0.5.npz`. To train the DDPM, you need put the `cifar10`, `cifar100`, `stl10`, `tiny-imagenet`dataset into `DDPM/data/pytorch`. You can also change the directory by modifying the path in `main.get_dataset` function and `dataset_utils.load_member_data`.  You can change the log directory by modifying `FLAGS.logdir` in `main.py`. You can change the `FLAGS.dataset` to select the dataset.

Then, to train the DDPM, just run command below.
```bash
cd DDPM
python main.py
```

### attack DDPM model
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
