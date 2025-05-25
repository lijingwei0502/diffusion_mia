import argparse
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from models import DiT_models  # 假设DiT模型定义在这个文件中
from diffusion import create_diffusion
from download import find_model
from collections import defaultdict
from diffusers import AutoencoderKL
from sklearn import metrics
import pynvml
import copy
import resnet
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import scipy.io

# COCO to ImageNet Category Mapping
coco_to_imagenet_mapping = {
    14: 527,  # parking meter
    24: 80,   # zebra
    28: 220,  # umbrella
    38: 397,  # kite
    47: 859,  # cup
    52: 323,  # banana
    55: 319,  # orange
    56: 737,  # broccoli
    86: 874   # vase
}

class FilteredImageFolder(Dataset):
    def __init__(self, root, transform, target_classes, category_id_to_name):
        self.dataset = ImageFolder(root)
        self.target_classes = set(target_classes)
        self.transform = transform
        self.filtered_indices = []
        self.class_indices = defaultdict(list)
        self.category_id_to_name = category_id_to_name
        
        # Filter dataset to include only target classes
        for idx, (path, label) in enumerate(self.dataset.samples):
            if label in self.target_classes:
                self.filtered_indices.append(idx)
                self.class_indices[label].append(idx)
        
        # 打印每个目标类别的样本数量
        for target_class in self.target_classes:
            num_samples = len(self.class_indices[target_class])
            print(f"Target class {target_class} ({self.category_id_to_name.get(target_class, 'Unknown')}) has {num_samples} samples")

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        if index >= len(self.filtered_indices):
            raise IndexError(f"Index {index} is out of range for filtered_indices with length {len(self.filtered_indices)}")
        sample_idx = self.filtered_indices[index]
        path, target = self.dataset.samples[sample_idx]
        sample = self.dataset.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def get_indices_by_class(self, class_id):
        return self.class_indices[class_id]

class COCODataset(Dataset):
    def __init__(self, coco_data, transform=None, category_mapping=None, num_samples_per_class=10, nonmember_data_path=''):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.nonmember_data_path = nonmember_data_path

        for coco_id, imagenet_id in category_mapping.items():
            images = self.get_coco_images_by_category(coco_data, coco_id)
            num_available_samples = len(images)
            num_samples_to_select = min(num_samples_per_class, num_available_samples)
            if num_samples_to_select > 0:
                selected_images = np.random.choice(images, num_samples_to_select, replace=False)
                self.image_paths.extend(selected_images)
                self.labels.extend([imagenet_id] * num_samples_to_select)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_coco_images_by_category(self, coco_data, category_id):
        images = []
        for annotation in coco_data['annotations']:
            if annotation['category_id'] == category_id:
                image_id = annotation['image_id']
                image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
                if image_info:
                    images.append(f"{self.nonmember_data_path}/{image_info['file_name']}")
        return images
        
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

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

def process_and_compare_batch(loader, model, vae, diffusion, device, cfg_scale, t_step, distance_type, k, num_experiments):
    ssim_distances = []
    l2_distances = []
    latent_l2_distances = []
    noise_losses = []
    nn_original = []
    nn_final = []

    t_step = torch.tensor([t_step], device=device)

    for image_batch, labels in loader:
        image_batch = image_batch.to(device)
        labels = labels.to(device)

        accumulated_samples = None

        with torch.no_grad():
            for _ in range(num_experiments):
                latent_batch = vae.encode(image_batch).latent_dist.sample().mul_(0.18215)
                noise = torch.randn_like(latent_batch)
                noisy_latent_batch = diffusion.q_sample(latent_batch, t_step, noise)

                y = torch.cat([labels, torch.tensor([1000] * labels.size(0), device=device)], 0)
                noisy_latent_batch = torch.cat([noisy_latent_batch, noisy_latent_batch], 0)

                model_kwargs = dict(y=y, cfg_scale=cfg_scale)

                samples = []
                for sample_dict in diffusion.ddim_sample_progressive(
                    model.forward_with_cfg,
                    noisy_latent_batch.shape,
                    noise=noisy_latent_batch,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    device=device,
                    start_step=t_step,
                    eta=0.0,
                    progress=False,
                    k=k,
                ):
                    samples.append(sample_dict["sample"])

                final_samples = samples[-1]
                final_samples, _ = final_samples.chunk(2, dim=0)

                if accumulated_samples is None:
                    accumulated_samples = final_samples
                else:
                    accumulated_samples += final_samples

            average_samples = accumulated_samples / num_experiments
            denoised_image_batch = vae.decode(average_samples / 0.18215).sample

        for i in range(image_batch.size(0)):
            denoised_image = denoised_image_batch[i].cpu().numpy().transpose(1, 2, 0)
            original_image = image_batch[i].cpu().numpy().transpose(1, 2, 0)
            denoised_image_normalized = (denoised_image * 0.5) + 0.5
            original_image_normalized = (original_image * 0.5) + 0.5

            if distance_type == 'ssim':
                ssim_distance = ssim(original_image_normalized, denoised_image_normalized, win_size=5, channel_axis=2, data_range=original_image_normalized.max() - original_image_normalized.min())
                ssim_distances.append(ssim_distance)
                print(f'SSIM distance for image {i}: {ssim_distance}')
                save_image(denoised_image_batch[i], f"denoised_image_{i}.png")
                save_image(image_batch[i], f"original_image_{i}.png")

            elif distance_type == 'l2':
                l2_distance = np.linalg.norm(original_image - denoised_image)
                print(f'L2 distance for image {i}: {l2_distance}')
                l2_distances.append(l2_distance)
                save_image(denoised_image_batch[i], f"denoised_image_{i}.png")
                save_image(image_batch[i], f"original_image_{i}.png")

            elif distance_type == 'latent_l2':
                original_latent = latent_batch[i].cpu().numpy()
                final_latent = average_samples[i].cpu().numpy()
                latent_l2_distance = np.linalg.norm(original_latent - final_latent)
                print(f'Latent L2 distance for image {i}: {latent_l2_distance}')
                latent_l2_distances.append(latent_l2_distance)
                save_image(denoised_image_batch[i], f"denoised_image_{i}.png")
                save_image(image_batch[i], f"original_image_{i}.png")

            elif distance_type == 'nn':
                original_latent = latent_batch[i].cpu().numpy()
                final_latent = average_samples[i].cpu().numpy()
                nn_original.append(original_latent)
                nn_final.append(final_latent)

            elif distance_type == 'loss':
                predicted_noise = model(noisy_latent_batch[i:i+1], t_step, y=labels[i:i+1])
                if predicted_noise.shape[1] == noise.shape[1] * 2:
                    predicted_noise, _ = torch.split(predicted_noise, noise.shape[1], dim=1)
                noise_loss = F.mse_loss(predicted_noise, noise[i:i+1])
                noise_losses.append(noise_loss.item())
                print(f'Noise prediction loss for image: {noise_loss.item()}')
                print(f'Mean noise prediction loss: {np.mean(noise_losses)}')

    if distance_type == 'ssim':
        return ssim_distances
    elif distance_type == 'l2':
        return l2_distances
    elif distance_type == 'latent_l2':
        return latent_l2_distances
    elif distance_type == 'nn':
        return nn_original, nn_final
    elif distance_type == 'loss':
        return noise_losses

def roc(member_scores, nonmember_scores, n_points=1000):
    max_asr = 0
    max_threshold = 0

    min_conf = min(member_scores.min(), nonmember_scores.min())
    max_conf = max(member_scores.max(), nonmember_scores.max())

    FPR_list = []
    TPR_list = []

    for threshold in np.linspace(min_conf, max_conf, n_points):
        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ASR = (TP + TN) / (TP + TN + FP + FN)

        TPR_list.append(TPR)
        FPR_list.append(FPR)

        if ASR > max_asr:
            max_asr = ASR
            max_threshold = threshold

    FPR_list = np.asarray(FPR_list)
    TPR_list = np.asarray(TPR_list)
    auc = metrics.auc(FPR_list, TPR_list)
    return auc, max_asr, torch.from_numpy(FPR_list), torch.from_numpy(TPR_list), max_threshold

def plot_scores_distribution(member_scores, nonmember_scores):
    if torch.is_tensor(member_scores):
        member_scores = member_scores.cpu().numpy()
    if torch.is_tensor(nonmember_scores):
        nonmember_scores = nonmember_scores.cpu().numpy()

    all_scores = np.concatenate((member_scores, nonmember_scores))
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)

    bins = np.linspace(min_score, max_score, 50)

    print('Member Scores: mean: {:.4f}, std: {:.4f}'.format(np.mean(member_scores), np.std(member_scores)))
    print('Non-Member Scores: mean: {:.4f}, std: {:.4f}'.format(np.mean(nonmember_scores), np.std(nonmember_scores)))

    plt.figure(figsize=(10, 8))
    plt.hist(member_scores, bins=bins, alpha=0.5, label='Member Scores')
    plt.hist(nonmember_scores, bins=bins, alpha=0.5, label='Non-Member Scores')
    plt.legend(loc='upper right', fontsize=18)
    plt.title('Distribution of Member vs Non-Member Scores', fontsize=18)
    plt.xlabel('Scores', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig('distribution.png')

def get_random_subset(dataset, num_samples):
    indices = torch.randperm(len(dataset)).tolist()[:num_samples]
    return Subset(dataset, indices)

class MIDataset():

    def __init__(self, member_data, nonmember_data, member_label, nonmember_label):
        self.data = torch.concat([member_data, nonmember_data])
        self.label = torch.concat([member_label, nonmember_label]).reshape(-1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, item):
        data = self.data[item]
        return data, self.label[item]

def split_nn_datasets(member_diffusion, member_sample, nonmember_diffusion, nonmember_sample, norm, train_portion=0.2, batch_size=128):
    member_concat = (member_diffusion - member_sample).abs() ** norm
    nonmember_concat = (nonmember_diffusion - nonmember_sample).abs() ** norm
    
    num_train = int(member_concat.size(0) * train_portion)
    train_member_concat = member_concat[:num_train]
    train_member_label = torch.ones(train_member_concat.size(0))
    train_nonmember_concat = nonmember_concat[:num_train]
    train_nonmember_label = torch.zeros(train_nonmember_concat.size(0))
    test_member_concat = member_concat[num_train:]
    test_member_label = torch.ones(test_member_concat.size(0))
    test_nonmember_concat = nonmember_concat[num_train:]
    test_nonmember_label = torch.zeros(test_nonmember_concat.size(0))

    if num_train == 0:
        train_dataset = None
        train_loader = None
    else:
        train_dataset = MIDataset(train_member_concat, train_nonmember_concat, train_member_label,
                                  train_nonmember_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MIDataset(test_member_concat, test_nonmember_concat, test_member_label, test_nonmember_label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

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

class MappedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        if index >= len(self.indices) or index < 0:
            raise IndexError(f"Index {index} is out of range for indices with length {len(self.indices)}")
        actual_index = self.indices[index]
        if actual_index not in self.dataset.filtered_indices:
            raise IndexError(f"Actual index {actual_index} not found in filtered_indices")
        mapped_index = self.dataset.filtered_indices.index(actual_index)
        return self.dataset[mapped_index]

    def __len__(self):
        return len(self.indices)

    
def main(args):
    torch.manual_seed(args.seed)
    device_str = 'cuda:' + found_device() if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # 加载ImageNet类别标签文件
    meta_mat_file = '/data_server3/ljw/imagenet/ILSVRC2012_devkit_t12/data/meta.mat'
    meta = scipy.io.loadmat(meta_mat_file)
    imagenet_category_id_to_name = {}
    for synset in meta['synsets']:
        synset = synset[0]
        try:
            category_id = int(synset[0][0])
            category_name = synset[1][0]  # 确保正确索引
            imagenet_category_id_to_name[category_id] = category_name
        except IndexError as e:
            print(f"IndexError: {e}, synset: {synset}")
        except Exception as e:
            print(f"Unexpected error: {e}, synset: {synset}")

    # 加载COCO注释文件
    with open(args.coco_annotation_file, 'r') as f:
        coco_data = json.load(f)
    coco_categories = coco_data['categories']
    coco_category_id_to_name = {category['id']: category['name'] for category in coco_categories}
    imagenet_name_to_id = {value: key for key, value in imagenet_category_id_to_name.items()}

    # 获取目标类别
    target_classes = set(coco_to_imagenet_mapping.values())
    
    # 预加载ImageNet数据集并缓存类别索引
    imagenet_dataset = FilteredImageFolder(args.member_data_path, transform=transform, target_classes=target_classes, category_id_to_name=imagenet_category_id_to_name)
    
    # 检查self.filtered_indices的长度
    print(f"Filtered ImageNet samples count: {len(imagenet_dataset.filtered_indices)}")

    selected_imagenet_indices = []
    for coco_id, imagenet_id in coco_to_imagenet_mapping.items():
        imagenet_class_indices = imagenet_dataset.get_indices_by_class(imagenet_id)
        num_samples = len(imagenet_class_indices)
        #print(f"ImageNet class {imagenet_id} ({imagenet_category_id_to_name.get(imagenet_id, 'Unknown')}) has {num_samples} available samples")
        if num_samples > 0:
            selected_indices = np.random.choice(imagenet_class_indices, min(num_samples, args.num_samples_per_class), replace=False)
            selected_imagenet_indices.extend(selected_indices)

    print(f"Total selected ImageNet samples: {len(selected_imagenet_indices)}")
    print(selected_imagenet_indices)
    print(imagenet_dataset.filtered_indices)
    member_dataset = MappedSubset(imagenet_dataset, selected_imagenet_indices)
    nonmember_dataset = COCODataset(coco_data, transform=transform, category_mapping=coco_to_imagenet_mapping, num_samples_per_class=args.num_samples_per_class, nonmember_data_path=args.nonmember_data_path)
    member_loader = DataLoader(
        member_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    nonmember_loader = DataLoader(
        nonmember_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print(f"Member dataset contains {len(member_dataset):,} images ({args.member_data_path})")
    print(f"Non-member dataset contains {len(nonmember_dataset):,} images ({args.nonmember_data_path})")

    if args.distance_type == 'nn':
        member_original, member_final = process_and_compare_batch(member_loader, model, vae, diffusion, device, args.cfg_scale, args.t_step, args.distance_type, args.k, args.experiments)
        nonmember_original, nonmember_final = process_and_compare_batch(nonmember_loader, model, vae, diffusion, device, args.cfg_scale, args.t_step, args.distance_type, args.k, args.experiments)
        n_epoch = 15
        lr = 0.001
        batch_size = 16
        norm = 1
        train_portion=0.5

        member_original = [torch.from_numpy(arr) for arr in member_original]
        member_original = torch.stack(member_original)

        member_final = [torch.from_numpy(arr) for arr in member_final]
        member_final = torch.stack(member_final)

        nonmember_original = [torch.from_numpy(arr) for arr in nonmember_original]
        nonmember_original = torch.stack(nonmember_original)

        nonmember_final = [torch.from_numpy(arr) for arr in nonmember_final]
        nonmember_final = torch.stack(nonmember_final)

        train_loader, test_loader = split_nn_datasets(member_original, member_final, nonmember_original, nonmember_final, norm, train_portion=train_portion, batch_size=batch_size)
        model = resnet.ResNet18(num_channels=4, num_classes=1).to(device)
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        test_acc_best_ckpt = None
        test_acc_best = 0
        for epoch in range(n_epoch):
            train_loss, train_acc = nn_train(device, epoch, model, optim, train_loader)
            test_loss, test_acc = nn_eval(device, model, test_loader)
            if test_acc > test_acc_best:
                test_acc_best_ckpt = copy.deepcopy(model.state_dict())

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

        member_ssim_scores = torch.cat(member_scores).reshape(-1)
        nonmember_ssim_scores = torch.cat(nonmember_scores).reshape(-1)

    else:  
        # Calculate SSIM scores for member and nonmember images
        nonmember_ssim_scores = process_and_compare_batch(nonmember_loader, model, vae, diffusion, device, args.cfg_scale, args.t_step, args.distance_type, args.k, args.experiments)

        member_ssim_scores = process_and_compare_batch(member_loader, model, vae, diffusion, device, args.cfg_scale, args.t_step, args.distance_type, args.k, args.experiments)

                                          
        # Convert SSIM scores to numpy arrays
        member_ssim_scores = np.array(member_ssim_scores)
        nonmember_ssim_scores = np.array(nonmember_ssim_scores)

    if args.distance_type == 'ssim' or args.distance_type == 'nn':
        member_ssim_scores *= -1
        nonmember_ssim_scores *= -1

    # Calculate ROC metrics
    auc, asr, fpr_list, tpr_list, threshold = roc(member_ssim_scores, nonmember_ssim_scores, n_points=2000)

    # TPR @ 1% FPR
    asr = asr.item()
    tpr_1_fpr = tpr_list[(fpr_list - 0.01).abs().argmin(dim=0)]
    tpr_1_fpr = tpr_1_fpr.item()
    print('AUC:', auc)
    print('ASR:', asr)
    print('TPR @ 1% FPR:', tpr_1_fpr)
    
    # Save results to CSV file
    result_dir = 'results.csv'
    with open(result_dir, 'a') as f:
        f.write(f"{args.model},{args.image_size},{args.t_step},{args.k},{args.distance_type},{args.experiments},{auc},{asr},{tpr_1_fpr}\n")
    plot_scores_distribution(member_ssim_scores, nonmember_ssim_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--member-data-path", type=str, default='/data_server3/ljw/imagenet/member', help="Path to the member dataset")
    parser.add_argument("--nonmember-data-path", type=str, default='/nfs/nfs-home/ljw/mia-diffusion/stable_diffusion/coco_data/val2017', help="Path to the nonmember dataset")
    parser.add_argument("--coco-annotation-file", type=str, default='/nfs/nfs-home/ljw/mia-diffusion/stable_diffusion/coco_data/annotations/instances_val2017.json', help="Path to the COCO annotation file")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--t-step", type=int, default=1, help="Number of timesteps for adding noise")
    parser.add_argument("--num-samples", type=int, default=480, help="Number of samples to use from each dataset")
    parser.add_argument("--num-samples-per-class", type=int, default=10, help="Number of samples per class")
    parser.add_argument("--distance-type", type=str, default='loss', help="Type of distance metric to use")
    parser.add_argument("--k", type=int, default=10, help="Step size for DDIM")
    parser.add_argument("--experiments", type=int, default=1, help="Number of experiments to average over")
    args = parser.parse_args()
    main(args)
