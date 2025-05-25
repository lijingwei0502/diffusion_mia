from typing import Any
import torch
from torchvision.datasets import CocoDetection as _CocoDetection, VisionDataset
import os
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
import numpy as np
import pandas as pd

coco_dataset_root = 'coco_data/val2017'
coco_dataset_anno = 'coco_data/annotations/captions_val2017.json'
stable_diffusion_data = 'stable_diffusion_data'

class Laion5(VisionDataset):
    def __init__(self, root: str, metadata, transforms = None, transform = None, target_transform = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.metadata = np.load(metadata)

    def __getitem__(self, index: int) -> Any:
        
        img = Image.open(os.path.join(self.root, self.metadata[index][0] + ".jpg")).convert('RGB')
        caption = self.metadata[index][1]
        if self.transforms is not None:
            img, caption = self.transforms(img, caption)
        return img, caption

    def __len__(self) -> int:
        return len(self.metadata)

    @staticmethod
    def collate_fn(examples):
        imgs = []
        captions = []
        for e in examples:
            imgs.append(e[0])
            captions.append(e[1])
        return torch.stack(imgs), captions


class Laion5Generated(Laion5):
    def __init__(self, root: str, metadata, generated, transforms = None, transform = None, target_transform = None) -> None:
        super().__init__(root, metadata, transforms, transform, target_transform)
        self.generated = pd.read_csv(generated)

    def __getitem__(self, index: int) -> Any:
        
        img = Image.open(os.path.join(self.root, self.metadata[index][0] + ".jpg")).convert('RGB')
        caption = self.generated[self.generated.file_name == self.metadata[index][0] + ".jpg"].iloc[0, 0]
        if self.transforms is not None:
            img, caption = self.transforms(img, caption)
        return img, caption

class Laion5GeneratedDALLE(Laion5):
    def __init__(self, root: str, metadata, generated, file_path, transforms=None, transform=None, target_transform=None):
        super().__init__(root, metadata, transforms, transform, target_transform)
        
        self.image_files = {file.split('.')[0]: file for file in os.listdir(file_path) if file.lower().endswith(('png', 'jpg', 'jpeg'))}
        self.name = root
        
        self.generated = pd.read_csv(generated)
        self.generated = self.generated[self.generated['file_name'].str.split('.').str[0].isin(self.image_files.keys())]

    def __getitem__(self, index: int) -> Any:
        return self._get_valid_item(index, set())

    def _get_valid_item(self, index: int, tried_indices: set) -> Any:
        if index >= len(self.generated) or index in tried_indices:
            raise IndexError("No valid image file found in the specified range or all files have been tried.")
        
        tried_indices.add(index)
        file_name = self.generated.iloc[index]['file_name']
        file_path = os.path.join(self.root, self.image_files[file_name.split('.')[0]])
       
        try:
            img = Image.open(file_path).convert('RGB')
            caption = self.generated.iloc[index]['caption']
            if self.transforms is not None:
                img, caption = self.transforms(img, caption)
            
            return img, caption
        except FileNotFoundError:
            return self._get_valid_item(index + 1, tried_indices)

    def __len__(self) -> int:
        return len(self.generated)
    
class Laion5Generatednew(Laion5):
    def __init__(self, root: str, metadata, generated: str, transforms=None, transform=None, target_transform=None) -> None:
        super().__init__(root, metadata, transforms, transform, target_transform)
        self.generated = pd.read_csv(generated)
    
    def __getitem__(self, index: int) -> Any:
        # Get the file name from the generated DataFrame
        file_name = self.generated.iloc[index]['Image Name']
        
        # Load the image
        img_path = os.path.join(self.root, file_name)
        img = Image.open(img_path).convert('RGB')
        
        # Get the caption
        caption = self.generated.iloc[index]['Title']
        
        # Apply transforms if provided
        if self.transforms is not None:
            img, caption = self.transforms(img, caption)
                
        return img, caption

    def __len__(self) -> int:
        # Return the total number of items
        return len(self.generated)

class CocoDetection(_CocoDetection):
    def __init__(self, root: str, annFile: str, transform = None, target_transform = None, transforms = None) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.splited_id = OmegaConf.load(f'coco-2500-random.yaml')

    def __len__(self) -> int:
        return len(self.splited_id)

    def __getitem__(self, index: int):
        return super().__getitem__(self.splited_id[index])

    @staticmethod
    def collate_fn(examples):
        imgs = []
        captions = []
        for e in examples:
            imgs.append(e[0])
            captions.append(e[1][0]['caption'])
        return torch.stack(imgs), captions


class CocoDetectionGenerated(CocoDetection):
    def __init__(self, root: str, annFile: str, generated, transform = None, target_transform = None, transforms = None) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.generated = pd.read_csv(generated)

    def __getitem__(self, index: int):
        img, _ = super().__getitem__(index)
        file_name = self.coco.loadImgs(self.ids[self.splited_id[index]])[0]["file_name"]
        caption = self.generated[self.generated.file_name == file_name].iloc[0, 0]
        return img, caption

    @staticmethod
    def collate_fn(examples):
        imgs = []
        captions = []
        for e in examples:
            imgs.append(e[0])
            captions.append(e[1])
        return torch.stack(imgs), captions

def load_member_data(dataset_name, batch_size=4):
    resolution = 512
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    if dataset_name == 'laion5':
        member_set = Laion5(f"{stable_diffusion_data}/images-random",
                            f"{stable_diffusion_data}/val-list-2500-random.npy",
                            transform=transform)
        nonmember_set = CocoDetection(root=coco_dataset_root,
                                      annFile=coco_dataset_anno,
                                      transform=transform)
    elif dataset_name == 'laion5_blip':
        member_set = Laion5Generated(f"{stable_diffusion_data}/images-random",
                                     f"{stable_diffusion_data}/val-list-2500-random.npy",
                                     f'{stable_diffusion_data}/text_generation/images-random.csv',
                                     transform=transform)
        nonmember_set = CocoDetectionGenerated(root=coco_dataset_root,
                                               annFile=coco_dataset_anno,
                                               generated=f'{stable_diffusion_data}/text_generation/val2017.csv',
                                               transform=transform)
    elif dataset_name == 'laion5_dalle':
        generate_path = 'image_dalle'
        member_set = Laion5GeneratedDALLE(f"{stable_diffusion_data}/images-random",
                                     f"{stable_diffusion_data}/val-list-2500-random.npy", 
                                     f'{stable_diffusion_data}/text_generation/images-random.csv', f"{stable_diffusion_data}/{generate_path}",
                                     transform=transform)
        nonmember_set = Laion5GeneratedDALLE(f"{stable_diffusion_data}/{generate_path}",
                                     f"{stable_diffusion_data}/val-list-2500-random.npy", 
                                     f'{stable_diffusion_data}/text_generation/images-random.csv', f"{stable_diffusion_data}/{generate_path}",
                                     transform=transform)
    elif dataset_name == 'laion5_new':
        member_set = Laion5Generatednew(f"member",
                                        f"{stable_diffusion_data}/val-list-2500-random.npy",
                                     f'member_titles.csv', 
                                     transform=transform)
        nonmember_set = Laion5Generatednew(f"nonmember",
                                           f"{stable_diffusion_data}/val-list-2500-random.npy",
                                     f'nonmember_titles.csv', 
                                     transform=transform)


    member_loader = torch.utils.data.DataLoader(member_set, batch_size=batch_size, collate_fn=member_set.collate_fn)
    nonmember_loader = torch.utils.data.DataLoader(nonmember_set, batch_size=batch_size, collate_fn=nonmember_set.collate_fn)
    return member_set, nonmember_set, member_loader, nonmember_loader


if __name__ == '__main__':
    resolution = 512
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
