import os
import time
import random
import subprocess
    
random.seed(0)

ckpt = ['800000']
dataset = ['CIFAR10','CIFAR100','STL10']
attacker_name = ['ReDiffuse',"SecMI","PIA","Naive","PIAN"]
attack_num = ['1']
interval = ['100']
kk = ['10']
seed = ['1']
averages = ['1']
processes = []

for se in seed:
    for ck in ckpt:
        for ds in dataset:
            for an in attacker_name:
                for anum in attack_num:
                    for inr in interval:
                        for k in kk:
                            for av in averages:
                                cmd = f'python attack.py --checkpoint {ck} --dataset {ds} --attacker_name {an} --attack_num {anum} --interval {inr} --k {k} --seed {se} --average {av}'
                                processes.append(subprocess.Popen(cmd, shell=True))
                                time.sleep(20)

for p in processes:
    p.wait()
