import subprocess
import time
processes = []
image_sizes = [128,256] 
kk = [50]
t_steps = [50,100,150,200,250,300]
experiments = [10]
mia_type = ['denoise','secmi','pia','pian','naive']

for image_size in image_sizes:
    for t_step in t_steps:
        for m in mia_type:
            for e in experiments:
                for k in kk:
                    if mia_type == 'denoise':
                        e = 10
                    else:
                        e = 1
                    if image_size == 128:
                        cmd = f'HF_ENDPOINT=https://hf-mirror.com python attack.py --image-size {image_size} --seed 1 --ckpt results/XL-imagenet128/checkpoints/0200000.pt --model DiT-XL/2 --t-step {t_step} --mia-type {m} --experiments {e} --k {k}'
                    else:
                        cmd = f'HF_ENDPOINT=https://hf-mirror.com python attack.py --image-size {image_size} --seed 1 --ckpt results/XL-imagenet256/checkpoints/0300000.pt --model DiT-XL/2 --t-step {t_step} --mia-type {m} --experiments {e} --k {k}'
                    p = subprocess.Popen(cmd, shell=True)
                    processes.append(p)
                    time.sleep(30)

for p in processes:
    p.wait()
