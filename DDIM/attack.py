import numpy as np
import torch
from rich.progress import track
import fire
import logging
from rich.logging import RichHandler
from pytorch_lightning import seed_everything
import components
from typing import Type, Dict
from itertools import chain
from model_unet import UNet
from dataset_utils import load_member_data
from torchmetrics.classification import BinaryAUROC, BinaryROC
import tqdm
from sklearn import metrics
import resnet
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from sklearn.metrics import roc_curve, auc

import pynvml
import copy

def found_device():
    default_device=0
    default_memory_threshold=500
    pynvml.nvmlInit()
    while True:
        handle=pynvml.nvmlDeviceGetHandleByIndex(default_device)
        meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
        used=meminfo.used/1024**2
        if used<default_memory_threshold:
            break
        else:
            default_device+=1
        if default_device>=8:
            default_device=0
            default_memory_threshold+=1000
    pynvml.nvmlShutdown()
    return str(default_device)


device_str = 'cuda:' + found_device() if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(device_str)


def get_FLAGS():

    def FLAGS(x): return x
    FLAGS.T = 1000
    FLAGS.ch = 128
    FLAGS.ch_mult = [1, 2, 2, 2]
    FLAGS.attn = [1]
    FLAGS.num_res_blocks = 2
    FLAGS.dropout = 0.1
    FLAGS.beta_1 = 0.0001
    FLAGS.beta_T = 0.02

    return FLAGS


def get_model(ckpt, WA=True):
    FLAGS = get_FLAGS()
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # load model and evaluate
    ckpt = torch.load(ckpt)

    if WA:
        weights = ckpt['ema_model']
    else:
        weights = ckpt['net_model']

    new_state_dict = {}
    for key, val in weights.items():
        if key.startswith('module.'):
            new_state_dict.update({key[7:]: val})
        else:
            new_state_dict.update({key: val})

    model.load_state_dict(new_state_dict)

    model.eval()

    return model

class EpsGetter(components.EpsGetter):
    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, noise_level=None, t: int = None) -> torch.Tensor:
        t = torch.ones([xt.shape[0]], device=xt.device).long() * t
        return self.model(xt, t=t)


attackers: Dict[str, Type[components.DDIMAttacker]] = {
    "SecMI": components.SecMIAttacker,
    "PIA": components.PIA,
    "Naive": components.NaiveAttacker,
    "PIAN": components.PIAN,
    "ReDiffuse": components.ReDiffuseAttacker,
}

def split_nn_datasets(member_diffusion, member_sample, nonmember_diffusion, nonmember_sample, norm, train_portion=0.2, batch_size=128):
    # split training and testing
    # train num
    member_concat = (member_diffusion - member_sample).abs() ** norm
    nonmember_concat = (nonmember_diffusion - nonmember_sample).abs() ** norm
    
    # train num
    num_train = int(member_concat.size(0) * train_portion)
    # split
    train_member_concat = member_concat[:num_train]
    train_member_label = torch.ones(train_member_concat.size(0))
    train_nonmember_concat = nonmember_concat[:num_train]
    train_nonmember_label = torch.zeros(train_nonmember_concat.size(0))
    test_member_concat = member_concat[num_train:]
    test_member_label = torch.ones(test_member_concat.size(0))
    test_nonmember_concat = nonmember_concat[num_train:]
    test_nonmember_label = torch.zeros(test_nonmember_concat.size(0))

    # datasets
    if num_train == 0:
        train_dataset = None
        train_loader = None
    else:
        train_dataset = MIDataset(train_member_concat, train_nonmember_concat, train_member_label,
                                  train_nonmember_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MIDataset(test_member_concat, test_nonmember_concat, test_member_label, test_nonmember_label)
    # dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def nns_attack(device, members_diffusion, members_sample, nonmembers_diffusion, nonmembers_sample, norm, train_portion=0.2):
    n_epoch = 15
    lr = 0.001
    batch_size = 128
    # model training
    train_loader, test_loader = split_nn_datasets(members_diffusion, members_sample, nonmembers_diffusion, nonmembers_sample, norm, train_portion=train_portion,
                                                                batch_size=batch_size)
    # initialize NNs
    model = resnet.ResNet18(num_channels=3 * 1, num_classes=1).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    test_acc_best_ckpt = None
    test_acc_best = 0
    for epoch in range(n_epoch):
        train_loss, train_acc = nn_train(device, epoch, model, optim, train_loader)
        test_loss, test_acc = nn_eval(device, model, test_loader)
        if test_acc > test_acc_best:
            test_acc_best_ckpt = copy.deepcopy(model.state_dict())

    
    # resume best ckpt
    model.load_state_dict(test_acc_best_ckpt)
    
    model.eval()
    member_scores = []
    nonmember_scores = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            logits = model(data.to(device))
            logits_cpu = logits.detach().cpu()
            member_scores.append(logits_cpu[label == 1])
            nonmember_scores.append(logits_cpu[label == 0])

    member_scores = torch.cat(member_scores).reshape(-1)
    nonmember_scores = torch.cat(nonmember_scores).reshape(-1)
    return member_scores, nonmember_scores


def nn_train(device, epoch, model, optimizer, data_loader):
    model.train()

    mean_loss = 0
    total = 0
    acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        label = label.to(device).reshape(-1, 1)
        logit = model(data)
        loss = ((logit - label) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        total += data.size(0)

        logit[logit >= 0.5] = 1
        logit[logit < 0.5] = 0
        acc += (logit == label).sum()

    mean_loss /= len(data_loader)
    print(f'Epoch: {epoch} \t Loss: {mean_loss:.4f} \t Acc: {acc / total:.4f} \t')
    return mean_loss, acc / total

def nn_eval(device, model, data_loader):
    model.eval()

    mean_loss = 0
    total = 0
    acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device).reshape(-1, 1)
        logit = model(data)
        loss = ((logit - label) ** 2).mean()

        mean_loss += loss.item()
        total += data.size(0)

        logit[logit >= 0.5] = 1
        logit[logit < 0.5] = 0

        acc += (logit == label).sum()

    mean_loss /= len(data_loader)
    print(f'Test: \t Loss: {mean_loss:.4f} \t Acc: {acc / total:.4f} \t')
    return mean_loss, acc / total

class MIDataset():

    def __init__(self, member_data, nonmember_data, member_label, nonmember_label):
        self.data = torch.concat([member_data, nonmember_data])
        self.label = torch.concat([member_label, nonmember_label]).reshape(-1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, item):
        data = self.data[item]
        return data, self.label[item]

def roc(member_scores, nonmember_scores, n_points=1000):
    max_asr = 0
    max_threshold = 0

    min_conf = min(member_scores.min(), nonmember_scores.min()).item()
    max_conf = max(member_scores.max(), nonmember_scores.max()).item()

    FPR_list = []
    TPR_list = []

    for threshold in torch.arange(min_conf, max_conf, (max_conf - min_conf) / n_points):
        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ASR = (TP + TN) / (TP + TN + FP + FN)

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        if ASR > max_asr:
            max_asr = ASR
            max_threshold = threshold

    FPR_list = np.asarray(FPR_list)
    TPR_list = np.asarray(TPR_list)
    auc = metrics.auc(FPR_list, TPR_list)
    return auc, max_asr, torch.from_numpy(FPR_list), torch.from_numpy(TPR_list), max_threshold

def calculate_distance(x0, x1):
    return ((x0 - x1).abs() ** 2).flatten(2).sum(dim=-1)

def main(checkpoint=800000,
         dataset='CIFAR10',
         attacker_name="ReDiffuse",
         attack_num=1, interval=200,
         seed=0,average=10, norm = 1, k = 100):
    
    seed_everything(seed)

    FLAGS = get_FLAGS()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler())

    logger.info("loading model...")
    ckpt = 'logs/DDPM_' + str(dataset) + '_EPS/ckpt-step' + str(checkpoint) + '.pt'
    model_unet = get_model(ckpt, WA = True).to(DEVICE)
    model_unet.eval()

    logger.info("loading dataset...")
    if dataset == 'CIFAR10':
        _, _, train_loader, test_loader = load_member_data(dataset_name='CIFAR10', batch_size=64,
                                                           shuffle=False, randaugment=False)
    elif dataset == 'CIFAR100':
        _, _, train_loader, test_loader = load_member_data(dataset_name='CIFAR100', batch_size=64,
                                                           shuffle=False, randaugment=False)
    elif dataset == 'STL10':
        _, _, train_loader, test_loader = load_member_data(dataset_name='STL10', batch_size=64,
                                                           shuffle=False, randaugment=False)
    elif dataset == 'TINY-IN':
        _, _, train_loader, test_loader = load_member_data(dataset_name='TINY-IN', batch_size=64,
                                                           shuffle=False, randaugment=False)

    print(FLAGS.beta_1, FLAGS.beta_T)
    attacker = attackers[attacker_name](
        torch.from_numpy(np.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T)).to(DEVICE), interval, attack_num, k, EpsGetter(model_unet), average, lambda x: x * 2 - 1)

    logger.info("attack start...")

    members_diffusion, members_sample, nonmembers_diffusion, nonmembers_sample = [], [], [], []

    with torch.no_grad():
        for member, nonmember in track(zip(train_loader, chain(*([test_loader]))), total=len(test_loader)):
            member, nonmember = member[0].to(DEVICE), nonmember[0].to(DEVICE)

            intermediate, intermediate_denoise = attacker(member)
            members_diffusion.append(intermediate[0])
            members_sample.append(intermediate_denoise[0])
            intermediate, intermediate_denoise = attacker(nonmember)
            nonmembers_diffusion.append(intermediate[0])
            nonmembers_sample.append(intermediate_denoise[0])
        
    members_diffusion = torch.concat(members_diffusion)
    members_sample = torch.concat(members_sample)
    nonmembers_diffusion = torch.concat(nonmembers_diffusion)
    nonmembers_sample = torch.concat(nonmembers_sample)
    members_distance = calculate_distance(members_diffusion, members_sample)
    nonmembers_distance = calculate_distance(nonmembers_diffusion, nonmembers_sample)
   
    auc, asr, fpr_list, tpr_list, threshold = roc(members_distance, nonmembers_distance, n_points=2000)
    # 保存fpr和tpr
    # fpr_list = fpr_list.numpy()
    # tpr_list = tpr_list.numpy()
    # f = open('roc_curve/fpr_tpr' + str(attacker_name) + '.csv', 'w')
    # f.write('fpr,tpr\n')
    # for i in range(len(fpr_list)):
    #     f.write(str(fpr_list[i]) + ',' + str(tpr_list[i]) + '\n')
    
    # TPR @ 1% FPR
    asr = asr.item()
    tpr_1_fpr = tpr_list[(fpr_list - 0.01).abs().argmin(dim=0)]
    tpr_1_fpr = tpr_1_fpr.item()
  
    print('AUC:', auc)
    print('ASR:', asr)
    print('TPR @ 1% FPR:', tpr_1_fpr)

    result_dir = 'result.csv'
    f = open(result_dir, 'a')
    f.write(str(checkpoint) + ',' + dataset + ',' + attacker_name + ',' + str(attack_num) + ',' + str(interval) + ',' + str(norm) + ',' + str(k))
    f.write(',' + str(auc) + ',' + str(asr) + ',' + str(tpr_1_fpr) + '\n')

if __name__ == '__main__':
    fire.Fire(main)
