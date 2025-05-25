import subprocess
import time

#HF_ENDPOINT=https://hf-mirror.com python mia.py --image-size 256 --seed 1 --ckpt /data/ljw/results/000-DiT-B-2/checkpoints/0125000.pt --model DiT-B/2 


processes = []

cmd = f'HF_ENDPOINT=https://hf-mirror.com torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py --ckpt results/003-DiT-S-8/checkpoints/0400000.pt'
p = subprocess.Popen(cmd, shell=True)
processes.append(p)

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
