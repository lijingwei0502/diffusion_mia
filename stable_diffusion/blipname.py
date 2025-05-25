import os
import random
import shutil
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# 文件夹路径和CSV文件路径
image_folder = 'laion_nonmember'
output_folder = 'nonmember'
csv_file = 'nonmember_titles.csv'

# 如果输出文件夹不存在，则创建它
os.makedirs(output_folder, exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# 初始化BLIP模型和处理器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# 用于存储成功的图像及其标题
titles = []

# 循环直到成功处理100张图像
while len(titles) < 500 and image_files:
    # 随机选择一张图片
    image_file = random.choice(image_files)
    image_path = os.path.join(image_folder, image_file)
    
    try:
        # 尝试打开并处理图像
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        # 复制成功的图像到输出文件夹
        shutil.copy(image_path, os.path.join(output_folder, image_file))

        # 存储图像名称和生成的标题
        titles.append((image_file, caption))
        print(f'Processed: {image_file} -> {caption}')
    except Exception as e:
        # 打印错误信息并跳过该图像
        print(f'Error processing {image_file}: {e}')
    finally:
        # 无论是否成功，都从文件列表中移除该图像，以免重复处理
        image_files.remove(image_file)

# 将结果保存到CSV文件
df = pd.DataFrame(titles, columns=['Image Name', 'Title'])
df.to_csv(csv_file, index=False)

print(f'Titles saved to {csv_file} and images copied to {output_folder}')
