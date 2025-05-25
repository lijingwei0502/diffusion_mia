import subprocess
import time

dataset = ['laion5','laion5_blip','laion5_dalle']
attacker_name = ['SecMI','PIA','naive','PIAN']
interval = ['800','900','950','1000']
kk = ['10','20','50','100','200']
processes = []

for ds in dataset:
    for an in attacker_name:
        for inr in interval:
            for k in kk:
                cmd = f'HF_ENDPOINT=https://hf-mirror.com python stable_attack.py --dataset {ds} --attacker_name {an} --interval {inr} --k {k}'
                processes.append(subprocess.Popen(cmd, shell=True))
                time.sleep(500)
            


# dataset = ['laion5_dalle']
# attacker_name = ['PIA','SecMI']
# interval = ['50','100','150']
# processes = []
# for ds in dataset:
#     for an in attacker_name:
#         for inr in interval:
#             cmd = f'HF_ENDPOINT=https://hf-mirror.com python pia_attack.py --dataset {ds} --attacker_name {an} --interval {inr} '
#             processes.append(subprocess.Popen(cmd, shell=True))
#             time.sleep(30)  

# for p in processes:
#     p.wait()
