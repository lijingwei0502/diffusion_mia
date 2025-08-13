import argparse
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from models import DiT_models  # 假设DiT模型定义在这个文件中
from diffusion import create_diffusion
from download import find_model
from diffusers import AutoencoderKL
from sklearn import metrics
import pynvml
import numpy as np
from PIL import Image

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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


def process_and_compare_batch(loader, model, vae, diffusion, device, cfg_scale, t_step, mia_type, k, num_experiments):
    latent_l2_distances = []
    noise_losses = []

    # Convert t_step to a tensor
    t_step = torch.tensor([t_step], device=device)

    for image_batch, labels in loader:
        image_batch = image_batch.to(device)
        labels = labels.to(device)

        accumulated_samples = None

        with torch.no_grad():
            if mia_type == 'denoise':
                for _ in range(num_experiments):
                    # Map input images to latent space + normalize latents
                    latent_batch = vae.encode(image_batch).latent_dist.sample().mul_(0.18215)
                    # Add noise to latent
                    noise = torch.randn_like(latent_batch)
                    noisy_latent_batch = diffusion.q_sample(latent_batch, t_step, noise)

                    # Class labels for guidance
                    y = torch.cat([labels, torch.tensor([1000] * labels.size(0), device=device)], 0)
                    noisy_latent_batch = torch.cat([noisy_latent_batch, noisy_latent_batch], 0)

                    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

                    # Denoise using the model starting from t_step
                    samples = []

                    for sample_dict in diffusion.ddim_sample_progressive(
                        model.forward_with_cfg,
                        noisy_latent_batch.shape,
                        noise=noisy_latent_batch,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        device=device,
                        start_step=t_step,
                        eta=0.0,  # Set eta to 0.0 for deterministic DDIM
                        progress=False,
                        k=k,  # Add the step size parameter
                    ):
                        samples.append(sample_dict["sample"])

                    final_samples = samples[-1]  # Get the final denoised samples
                    final_samples, _ = final_samples.chunk(2, dim=0)  # Remove null class samples

                    if accumulated_samples is None:
                        accumulated_samples = final_samples
                    else:
                        accumulated_samples += final_samples

                average_samples = accumulated_samples / num_experiments

                # Decode the averaged latent samples to images
                denoised_image_batch = vae.decode(average_samples / 0.18215).sample

                # Calculate distances
                for i in range(image_batch.size(0)):
                    # Calculate L2 distance in latent space
                    original_latent = latent_batch[i].cpu().numpy()
                    final_latent = average_samples[i].cpu().numpy()
                    latent_l2_distance = np.linalg.norm(original_latent - final_latent)
                    latent_l2_distances.append(latent_l2_distance)
                
            elif mia_type == 'secmi':
                # Map input images to latent space + normalize latents
                latent_batch = vae.encode(image_batch).latent_dist.sample().mul_(0.18215)
                latent_batch = torch.cat([latent_batch, latent_batch], 0)
                # Class labels for guidance
                y = torch.cat([labels, torch.tensor([1000] * labels.size(0), device=device)], 0)

                # Denoise using the model starting from t_step
                samples = []
                model_kwargs = dict(y=y, cfg_scale=cfg_scale)
                for sample_dict in diffusion.ddim_reverse_sample_progressive(
                    model.forward_with_cfg,
                    latent_batch.shape,
                    noise=latent_batch,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    device=device,
                    end_step=t_step,
                    eta=0.0,  # Set eta to 0.0 for deterministic DDIM
                    progress=False,
                    k=k,  # Add the step size parameter
                ):
                    samples.append(sample_dict["sample"])
                    
                final_samples = samples[-1]  # Get the final denoised samples
                latent_batch = samples[-1]
                latent_batch, _ = latent_batch.chunk(2, dim=0)  # Remove null class samples
                # Decode the averaged latent samples to images
                denoised_image_batch = vae.decode(latent_batch / 0.18215).sample
                
                t_step_in = torch.tensor([t_step] * final_samples.shape[0], device=device)
                new_sample_dict = diffusion.ddim_k_reverse_sample(
                    model.forward_with_cfg,
                    final_samples,
                    t_step_in,
                    k,  # Add the step size parameter
                    clip_denoised=True,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=model_kwargs,
                    eta=0.0,
                )
                final_samples = new_sample_dict["sample"]
                t_step_in = torch.tensor([t_step+k] * final_samples.shape[0], device=device)
                new_sample_dict = diffusion.ddim_k_sample(
                    model.forward_with_cfg,
                    final_samples,
                    t_step_in,
                    k,  # Add the step size parameter
                    clip_denoised=True,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=model_kwargs,
                    eta=0.0,
                )

                final_samples = new_sample_dict["sample"]
                final_samples, _ = final_samples.chunk(2, dim=0)  # Remove null class samples
                average_samples = final_samples
                # Decode the averaged latent samples to images
                denoised_image_batch = vae.decode(average_samples / 0.18215).sample

                # Calculate distances
                for i in range(image_batch.size(0)):
                    # Calculate L2 distance in latent space
                    original_latent = latent_batch[i].cpu().numpy()
                    final_latent = average_samples[i].cpu().numpy()
                    latent_l2_distance = np.linalg.norm(original_latent - final_latent)
                    latent_l2_distances.append(latent_l2_distance)

            elif mia_type == 'pia' or mia_type == 'pian':
                latent_batch = vae.encode(image_batch).latent_dist.sample().mul_(0.18215)

                latent_batch = torch.cat([latent_batch, latent_batch], 0)

                y = torch.cat([labels, labels], 0)

                for i in range(image_batch.size(0)):
                    prime_noise = model(latent_batch[i:i+1], torch.tensor([0], device=device), y=y[i:i+1])
                    prime_noise = prime_noise[:, :prime_noise.shape[1] // 2]
                    if mia_type == 'pian':
                        prime_noise = prime_noise / prime_noise.abs().mean(list(range(1, prime_noise.ndim)), keepdim=True) * (2 / torch.pi) ** 0.5
                    noisy_latent_batch = diffusion.q_sample(latent_batch[i:i+1], t_step, prime_noise)

                    noisy_latent_batch = torch.cat([noisy_latent_batch, noisy_latent_batch], 0)
                    y_guided = torch.cat([labels, torch.tensor([1000] * labels.size(0), device=device)], 0)

                    predicted_noise = model(noisy_latent_batch[i:i+1], t_step, y=y_guided[i:i+1])
            
                    predicted_noise = predicted_noise[:, :predicted_noise.shape[1] // 2]
                    
                    noise_loss = F.mse_loss(prime_noise, predicted_noise)
                    noise_losses.append(noise_loss.item())

            elif mia_type == 'naive':
                latent_batch = vae.encode(image_batch).latent_dist.sample().mul_(0.18215)
                # Add noise to latent
                noise = torch.randn_like(latent_batch)
                noisy_latent_batch = diffusion.q_sample(latent_batch, t_step, noise)

                # Class labels for guidance
                y = torch.cat([labels, torch.tensor([1000] * labels.size(0), device=device)], 0)
                noisy_latent_batch = torch.cat([noisy_latent_batch, noisy_latent_batch], 0)

                for i in range(image_batch.size(0)):
                    # Predict noise
                    predicted_noise = model(noisy_latent_batch[i:i+1], t_step, y=labels[i:i+1])
                    # Handle the case where model_output has double the channels
                    if predicted_noise.shape[1] == noise.shape[1] * 2:
                        predicted_noise, _ = torch.split(predicted_noise, noise.shape[1], dim=1)

                    noise_loss = F.mse_loss(predicted_noise, noise[i:i+1])
                    noise_losses.append(noise_loss.item())

    if mia_type == 'denoise' or mia_type == 'secmi':
        return latent_l2_distances
    elif mia_type == 'naive':
        return noise_losses
    elif mia_type == 'pia' or mia_type == 'pian':
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

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    device_str = 'cuda:' + found_device() if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    
    model.load_state_dict(state_dict)
    model.eval()  
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup data transformations
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    nonmember_dataset = ImageFolder(args.nonmember_data_path, transform=transform)
    member_dataset = ImageFolder(args.member_data_path, transform=transform)

    if args.num_samples > 0:
        num_samples = min(args.num_samples, len(member_dataset))
        member_dataset = get_random_subset(member_dataset, num_samples)
        num_samples = min(args.num_samples, len(nonmember_dataset))
        nonmember_dataset = get_random_subset(nonmember_dataset, num_samples)

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
    # Calculate SSIM scores for member and nonmember images
    member_ssim_scores = process_and_compare_batch(member_loader, model, vae, diffusion, device, args.cfg_scale, args.t_step, args.mia_type, args.k, args.experiments)
    nonmember_ssim_scores = process_and_compare_batch(nonmember_loader, model, vae, diffusion, device, args.cfg_scale, args.t_step, args.mia_type, args.k, args.experiments)
                    
    # Convert SSIM scores to numpy arrays
    member_ssim_scores = np.array(member_ssim_scores)
    nonmember_ssim_scores = np.array(nonmember_ssim_scores)

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
        f.write(f"{args.model},{args.ckpt},{args.image_size},{args.t_step},{args.k},{args.mia_type},{args.experiments},{auc},{asr},{tpr_1_fpr}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--member-data-path", type=str, default='/data_server3/ljw/imagenet/member_512', help="Path to the member dataset")
    parser.add_argument("--nonmember-data-path", type=str, default='/data_server3/ljw/imagenet/val', help="Path to the nonmember dataset")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--t-step", type=int, default=1, help="Number of timesteps for adding noise")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to use from each dataset")
    parser.add_argument("--mia-type", type=str, default='denoise', help="Type of MIA attack to use")
    parser.add_argument("--k", type=int, default=10, help="Step size for DDIM")
    parser.add_argument("--experiments", type=int, default=10, help="Number of experiments to average over")
    args = parser.parse_args()
    main(args)
