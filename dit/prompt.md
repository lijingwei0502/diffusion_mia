HF_ENDPOINT=https://hf-mirror.com python sample.py --image-size 256 --seed 1 --ckpt results/010-DiT-S-2/checkpoints/0400000.pt --model DiT-S/2

HF_ENDPOINT=https://hf-mirror.com python sample.py --image-size 256 --seed 1 --ckpt results/005-DiT-S-2/checkpoints/0050000.pt --model DiT-S/2

HF_ENDPOINT=https://hf-mirror.com torchrun --nnodes=1 --nproc_per_node=8 train_all.py --model DiT-S/2 --data-path /data_server3/ljw/imagenet/train

HF_ENDPOINT=https://hf-mirror.com torchrun --nnodes=1 --nproc_per_node=4 sample_ddp.py --ckpt results/003-DiT-S-2/checkpoints/0200000.pt --model DiT-S/2


HF_ENDPOINT=https://hf-mirror.com torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py --ckpt results/005-DiT-S-2/checkpoints/0200000.pt --model DiT-S/2

HF_ENDPOINT=https://hf-mirror.com torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py --ckpt results/003-DiT-S-8/checkpoints/0400000.pt --model DiT-S/8


python evaluator.py 100000.npz S-2-5000.npz 
