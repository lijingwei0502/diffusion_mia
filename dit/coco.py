import json
import scipy.io

# COCO数据集的注释文件路径
coco_annotation_file = '/nfs/nfs-home/ljw/mia-diffusion/stable_diffusion/coco_data/annotations/instances_val2017.json'

# 加载ImageNet类别标签文件
meta_mat_file = f'/data_server3/ljw/imagenet/ILSVRC2012_devkit_t12/data/meta.mat'
meta = scipy.io.loadmat(meta_mat_file)

# 检查meta['synsets']的结构
print("meta['synsets'] example:", meta['synsets'][0])

# 提取ImageNet类别ID和类别名称的映射
imagenet_category_id_to_name = {}
for synset in meta['synsets']:
    # 解包synset元组
    synset = synset[0]
    
    try:
        category_id = int(synset[0][0])
        category_name = synset[2][0]
        imagenet_category_id_to_name[category_id] = category_name
    except IndexError as e:
        print(f"IndexError: {e}, synset: {synset}")
    except Exception as e:
        print(f"Unexpected error: {e}, synset: {synset}")

# 打印前几个映射进行检查
print("ImageNet Categories (sample):", list(imagenet_category_id_to_name.items())[:10])

# 加载COCO注释文件
with open(coco_annotation_file, 'r') as f:
    coco_data = json.load(f)

# 提取COCO类别ID到类别名称的映射
coco_categories = coco_data['categories']
coco_category_id_to_name = {category['id']: category['name'] for category in coco_categories}

# 创建类别名称到ImageNet类别ID的映射
imagenet_name_to_id = {value: key for key, value in imagenet_category_id_to_name.items()}

# 找到COCO和ImageNet相同类别的映射
coco_to_imagenet_mapping = {}
for coco_id, coco_name in coco_category_id_to_name.items():
    if coco_name in imagenet_name_to_id:
        imagenet_id = imagenet_name_to_id[coco_name]
        coco_to_imagenet_mapping[coco_id] = imagenet_id

# 打印映射结果
print("COCO to ImageNet Category Mapping:")
for coco_id, imagenet_id in coco_to_imagenet_mapping.items():
    print(f"COCO Category ID {coco_id} ({coco_category_id_to_name[coco_id]}) -> ImageNet Category ID {imagenet_id} ({imagenet_category_id_to_name[imagenet_id]})")

# 示例：根据类别ID获取COCO和ImageNet的图片文件名（假设已经有函数实现）
def get_coco_images_by_category(coco_data, category_id):
    images = []
    for annotation in coco_data['annotations']:
        if annotation['category_id'] == category_id:
            image_id = annotation['image_id']
            image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
            if image_info:
                images.append(image_info['file_name'])
    return images

def get_imagenet_images_by_category(imagenet_data, category_id):
    # 假设你有加载ImageNet图片文件名的实现，这里提供一个示例
    # return list_of_image_files_corresponding_to_imagenet_category_id
    pass

# 获取示例类别ID的对应图片文件名
example_coco_category_id = 1  # 示例COCO类别ID
if example_coco_category_id in coco_to_imagenet_mapping:
    example_imagenet_category_id = coco_to_imagenet_mapping[example_coco_category_id]
    coco_images = get_coco_images_by_category(coco_data, example_coco_category_id)
    imagenet_images = get_imagenet_images_by_category(imagenet_data, example_imagenet_category_id)

    print(f"COCO images for category {example_coco_category_id}: {coco_images}")
    print(f"ImageNet images for category {example_imagenet_category_id}: {imagenet_images}")
