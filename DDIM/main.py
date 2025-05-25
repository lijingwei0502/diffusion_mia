import copy
import json
import os
import sys

import numpy as np
import warnings
from absl import app, flags
import torch
from tensorboardX import SummaryWriter
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, CelebA, SVHN
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
from dataset_utils import MIACIFAR10, MIACIFAR100, MIASTL10, MIAImageFolder, MIASVHN
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model_unet import UNet


def FLAGS(x): return x

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
device = torch.device(device_str)

FLAGS.ch = 128
FLAGS.ch_mult = [1, 2, 2, 2]
FLAGS.attn = [1]
FLAGS.num_res_blocks = 2
FLAGS.dropout = 0.1
FLAGS.beta_1 = 1e-4
FLAGS.beta_T = 0.02
FLAGS.T = 1000
FLAGS.mean_type = "epsilon"
FLAGS.var_type = "fixedlarge"
FLAGS.lr = 2e-4
FLAGS.grad_clip = 1.
FLAGS.total_steps = 800001
FLAGS.img_size = 32
FLAGS.warmup = 5000
FLAGS.batch_size = 128
FLAGS.num_workers = 4
FLAGS.ema_decay = 0.9999
FLAGS.parallel = False
FLAGS.dataset = "SVHN"
FLAGS.logdir = './logs/DDPM_SVHN_EPS'
FLAGS.sample_size = 64
FLAGS.sample_step = 1000
FLAGS.save_step = 1000
FLAGS.num_images = 25000


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def cutout(mask_size, p, cutout_inside, mask_color=-1):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = image.clone()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout


def get_dataset(FLAGS, only_member=False):
    splits = np.load(f'./{FLAGS.dataset.upper()}_train_ratio0.5.npz')
    member_idxs = splits['mia_train_idxs']

    if FLAGS.dataset.upper() == 'CIFAR10':

        transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                      (0.5, 0.5, 0.5))])
        dataset = MIACIFAR10(member_idxs, root='data/datasets/pytorch', train=True,
                             transform=transforms, download=True)
    elif FLAGS.dataset.upper() == 'CIFAR100':
        transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                      (0.5, 0.5, 0.5))])
        dataset = MIACIFAR100(member_idxs, root='data/datasets/pytorch', train=True,
                             transform=transforms, download=True)
    elif FLAGS.dataset.upper() == 'STL10':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
        dataset = MIASTL10(member_idxs, root='data/datasets/pytorch', split='unlabeled',
                               download=True, transform=transforms)
    elif FLAGS.dataset.upper() == 'TINY-IN':
        transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                     torchvision.transforms.Resize((FLAGS.img_size, FLAGS.img_size)),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                      (0.5, 0.5, 0.5))])
        dataset = MIAImageFolder(member_idxs, root='data/datasets/pytorch/tiny-imagenet-200/train', 
                                 transform=transforms)
        
    elif FLAGS.dataset.upper() == 'SVHN':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = MIASVHN(member_idxs, root='data/datasets/pytorch', split='train',
                               download=True, transform=transforms)
        
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True,
                                              num_workers=FLAGS.num_workers)

    return data_loader

def load_checkpoint(checkpoint_path, net_model, ema_model, optim, sched, device):
    """
    加载检查点
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net_model.load_state_dict(checkpoint['net_model'])
    ema_model.load_state_dict(checkpoint['ema_model'])
    optim.load_state_dict(checkpoint['optim'])
    sched.load_state_dict(checkpoint['sched'])
    step = checkpoint['step']
    x_T = checkpoint['x_T']
    print(f"Resuming from step {step}...")
    return step, x_T


def train():
    dataloader = get_dataset(FLAGS, only_member=True)
    cutout_op = cutout(mask_size=4, p=0.1, cutout_inside=False)
    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)

    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)

    ema_model = copy.deepcopy(net_model)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    datalooper = infiniteloop(dataloader)

    # checkpoint_path = 'logs/DDPM_STL10_EPS/ckpt-step800000.pt'  # 替换为你的实际路径
    # # 加载检查点
    # step = 0
    # x_T = None
    # if checkpoint_path:
    #     step, x_T = load_checkpoint(checkpoint_path, net_model, ema_model, optim, sched, device)


    # log setup
    if not os.path.exists(os.path.join(FLAGS.logdir, 'sample')):
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            x_0 = cutout_op(x_0)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, f'ckpt-step{step}.pt'))

    writer.close()


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    train()


if __name__ == '__main__':
    app.run(main)
