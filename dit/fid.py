import os
from PIL import Image
from torchvision import transforms, models
import torch
import numpy as np
from scipy.stats import entropy
from pytorch_fid import fid_score
from tqdm import tqdm  # 引入tqdm库
import pynvml

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

def load_images_from_folder(folder_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 将图像大小统一调整为299x299
        transforms.ToTensor()           # 将调整大小后的图像转换为张量
    ])
    images = []  # 创建一个空列表来存储处理后的图像张量
    valid_extensions = {'png', 'jpg', 'jpeg'}  # 定义有效的文件扩展名集合

    for root, _, files in os.walk(folder_path):  # 遍历文件夹中的所有文件
        for img_file in files:
            if img_file.split('.')[-1].lower() in valid_extensions:  # 检查文件扩展名是否有效
                img_path = os.path.join(root, img_file)  # 获取完整的文件路径
                try:
                    img = Image.open(img_path).convert('RGB')  # 尝试打开图像并转换为RGB
                    img = transform(img)  # 应用之前定义的转换
                    images.append(img)    # 将转换后的图像张量添加到列表中
                except Exception as e:    # 捕捉并处理任何异常
                    print(f"Error processing image {img_path}: {e}")  # 打印错误信息
            else:
                print(f"Skipped non-image file {img_file}")  # 如果文件不是图像，打印跳过的信息
    return images  # 返回处理后的图像列表

def calculate_inception_score(imgs, inception_model, splits=10):
    N = len(imgs)
    probs = []
    for img in tqdm(imgs, desc="Calculating Inception Scores"):  # 添加进度条
        img = img.unsqueeze(0).to(device)
        pred = inception_model(img)
        probs.append(torch.nn.functional.softmax(pred, dim=1).data.cpu().numpy())
    probs = np.concatenate(probs, 0)
    split_scores = []

    for k in range(splits):
        part = probs[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = np.exp(entropy(part, py).mean())
        split_scores.append(scores)

    return np.mean(split_scores), np.std(split_scores)

def main(folder_path, real_path):
    inception_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inception_model.to(device)
    inception_model.eval()

    # 加载生成图像
    images = load_images_from_folder(folder_path)
    is_mean, is_std = calculate_inception_score(images, inception_model)
    print(f"Inception Score: Mean = {is_mean}, Std = {is_std}")

    # 确保FID的batch size不超过数据集大小
    fid_batch_size = 32
    
    # Calculating FID
    fid = fid_score.calculate_fid_given_paths([real_path, folder_path], batch_size=fid_batch_size, device=device, dims=192)
    print(f"FID score: {fid}")

if __name__ == '__main__':
    #folder_path = 'samples/DiT-XL-2-0245000-size-256-vae-ema-cfg-1.5-seed-0'  # 更新到你的生成图像文件夹路径
    #folder_path = 'samples/DiT-XL-2-DiT-XL-2-256x256-size-256-vae-ema-cfg-1.5-seed-0'
    #folder_path = 'samples/DiT-XL-2-0270000-size-256-vae-ema-cfg-1.5-seed-0'  # 更新到你的生成图像文件夹路径
    #real_path = '/data_server3/ljw/imagenet/val_50000'  # 真实图像集的路径
    #real_path = 'samples/DiT-XL-2-DiT-XL-2-256x256-size-256-vae-ema-cfg-1.5-seed-0'  # 真实图像集的路径
    real_path = 'samples/DiT-XL-2-0270000-size-256-vae-ema-cfg-1.5-seed-0'  # 更新到你的生成图像文件夹路径
    #real_path = 'samples/DiT-XL-2-0245000-size-256-vae-ema-cfg-1.5-seed-0'  # 更新到你的生成图像文件夹路径
    #real_path = 'samples/DiT-XL-2-DiT-XL-2-256x256-size-256-vae-ema-cfg-1.5-seed-0'  # 更新到你的生成图像文件夹路径
    folder_path = '/data_server3/ljw/imagenet/val_50000'  # 真实图像集的路径

    main(folder_path, real_path)
