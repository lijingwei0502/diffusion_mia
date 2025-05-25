import numpy as np
import torch
import torchvision
from torchvision import transforms
import random
import os

# Custom Dataset Class for SVHN
class MIASVHN(torchvision.datasets.SVHN):
    def __init__(self, idxs, **kwargs):
        super(MIASVHN, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIASVHN, self).__getitem__(item)

def main(dataset_name, output_dir, num_member_samples=1000, num_nonmember_samples=1000, split='train'):
    # Set paths and parameters
    dataset_root = 'data/datasets/pytorch'

    # Define the transformations (SVHN images are 32x32 by default, no need to resize)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # Load the full SVHN dataset (train or extra split)
    if dataset_name.upper() == 'SVHN':
        dataset = torchvision.datasets.SVHN(
            root=os.path.join(dataset_root, 'svhn'),
            split=split,  # 'train', 'test', or 'extra'
            transform=transforms,
            download=True
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported!")

    # Debugging: Check the length of the dataset
    print(f"Loaded SVHN {split} dataset with {len(dataset)} samples.")

    # Randomly select indices for member and nonmember
    total_samples = len(dataset)
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)

    # Select member and nonmember indices
    member_indices = all_indices[:num_member_samples]
    nonmember_indices = all_indices[num_member_samples:num_member_samples + num_nonmember_samples]

    # Debugging: Check the number of member and nonmember samples
    print(f"Selected {len(member_indices)} member samples and {len(nonmember_indices)} non-member samples.")

    # Save to npz file
    output_filename = os.path.join(output_dir, f'SVHN_train_ratio0.5.npz')
    np.savez(
        output_filename,
        mia_train_idxs=np.array(member_indices),
        mia_eval_idxs=np.array(nonmember_indices)
    )

    print(f"Generated npz file at {output_filename}")

    # Now create the MIASVHN dataset with member indices
    member_dataset = MIASVHN(
        idxs=member_indices,
        root=os.path.join(dataset_root, 'svhn'),
        split=split,
        transform=transforms
    )
    nonmember_dataset = MIASVHN(
        idxs=nonmember_indices,
        root=os.path.join(dataset_root, 'svhn'),
        split=split,
        transform=transforms
    )

    print(f"Member dataset length: {len(member_dataset)}")
    print(f"Non-member dataset length: {len(nonmember_dataset)}")

if __name__ == "__main__":
    # Example Usage
    dataset_name = 'SVHN'  # Only SVHN is supported here
    output_dir = './output'  # Directory where npz file will be saved
    os.makedirs(output_dir, exist_ok=True)

    # Generate npz file with 1000 member and 1000 nonmember samples
    main(
        dataset_name,
        output_dir,
        num_member_samples=25000,
        num_nonmember_samples=25000,
        split='train'  # 'train', 'test', or 'extra'
    )