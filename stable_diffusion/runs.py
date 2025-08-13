import subprocess
import time

dataset = ['laion5','laion5_blip']
attacker_name = ['ReDiffuse','SecMI','PIA','naive','PIAN']
interval = ['100','200']
kk = ['50','100']
processes = []

for ds in dataset:
    for an in attacker_name:
        for inr in interval:
            for k in kk:
                cmd = f'HF_ENDPOINT=https://hf-mirror.com python attack.py --dataset {ds} --attacker_name {an} --interval {inr} --k {k}'
                processes.append(subprocess.Popen(cmd, shell=True))
                time.sleep(10)
            

for p in processes:
    p.wait()
