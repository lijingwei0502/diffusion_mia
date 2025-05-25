import subprocess
import time

#HF_ENDPOINT=https://hf-mirror.com python mia.py --image-size 256 --seed 1 --ckpt /data/ljw/results/000-DiT-B-2/checkpoints/0125000.pt --model DiT-B/2 

processes = []
image_sizes = [256] 
kk = [10,50,100]
t_steps = [10,100,150,200,250,300]
experiments = [1,5,10]
mia_type = ['denoise','secmi','pia','pian','naive']

for image_size in image_sizes:
    for t_step in t_steps:
        for m in mia_type:
            for e in experiments:
                for k in kk:
                    if mia_type == 'denoise':
                        e = 10
                    if image_size == 128:
                        cmd = f'HF_ENDPOINT=https://hf-mirror.com python mia.py --image-size {image_size} --seed 1 --ckpt pretrained_models/DiT-XL-2-256x256.pt --model DiT-XL/2 --t-step {t_step} --mia-type {m} --experiments {e} --k {k}'
                    else:
                        cmd = f'HF_ENDPOINT=https://hf-mirror.com python mia.py --image-size {image_size} --seed 1 --ckpt pretrained_models/DiT-XL-2-256x256.pt --model DiT-XL/2 --t-step {t_step} --mia-type {m} --experiments {e} --k {k}'
                    p = subprocess.Popen(cmd, shell=True)
                    processes.append(p)
                    time.sleep(200)

                    

# for image_size in image_sizes:
#     for t_step in t_steps:
#         cmd = f'HF_ENDPOINT=https://hf-mirror.com python mia.py --image-size {image_size} --seed 1 --ckpt /data/ljw/results/000-DiT-L-2/checkpoints/0060000.pt --model DiT-L/2 --t-step {t_step}'
#         subprocess.Popen(cmd, shell=True)
#         time.sleep(30)

# for image_size in image_sizes:
#     for t_step in t_steps:
#         cmd = f'HF_ENDPOINT=https://hf-mirror.com python mia.py --image-size {image_size} --seed 1 --ckpt /data/ljw/results/000-DiT-L-2/checkpoints/0150000.pt --model DiT-L/2 --t-step {t_step}'
#         subprocess.Popen(cmd, shell=True)
#         time.sleep(30)

for p in processes:
    p.wait()
