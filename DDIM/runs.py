import os
import time
import random
import subprocess
    
random.seed(0)

ckpt = ['487000']
dataset = ['SVHN']
#dataset = ['CIFAR100','STL10','TINY-IN']
attacker_name = ['ReDiffuse']
attack_num = ['1']
interval = ['200']
kk = ['100']
seed = ['1']
averages = ['10']
processes = []

for se in seed:
    for ck in ckpt:
        for ds in dataset:
            for an in attacker_name:
                for anum in attack_num:
                    for inr in interval:
                        for k in kk:
                            for av in averages:
                                cmd = f'python nn_attack.py --checkpoint {ck} --dataset {ds} --attacker_name {an} --attack_num {anum} --interval {inr} --k {k} --seed {se} --average {av}'
                                processes.append(subprocess.Popen(cmd, shell=True))
                                time.sleep(20)

for p in processes:
    p.wait()
