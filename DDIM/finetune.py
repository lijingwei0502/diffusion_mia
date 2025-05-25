import numpy as np
import torch
import torchvision
from torchvision import transforms
import random
import os

# Custom Dataset Classes
class MIACIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, idxs, **kwargs):
        super(MIACIFAR10, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIACIFAR10, self).__getitem__(item)

# Main function to create npz files for member and non-member data
def main(dataset_name, output_dir, num_member_samples=1000, num_nonmember_samples=1000):
    # Set paths and parameters
    dataset_root = 'data/datasets/pytorch'

    # Define the transformations
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # Load the full CIFAR-10 dataset initially to get all indices
    if dataset_name.upper() == 'CIFAR10':
        # Load CIFAR-10 dataset without idxs since we want the full dataset first
        dataset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_root, 'cifar10'), train=True, transform=transforms, download=True)
    elif dataset_name.upper() == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root=os.path.join(dataset_root, 'cifar100'), train=True, transform=transforms, download=True)
    elif dataset_name.upper() == 'STL10':
        transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()])
        dataset = torchvision.datasets.STL10(root=os.path.join(dataset_root, 'stl10'), split='unlabeled', transform=transforms, download=True)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported!")

    # Debugging: Check the length of the dataset
    print(f"Loaded {dataset_name} dataset with {len(dataset)} samples.")

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
    np.savez(os.path.join(output_dir, f'{dataset_name}_ft_ratio0.5.npz'),
             mia_train_idxs=np.array(member_indices),
             mia_eval_idxs=np.array(nonmember_indices))

    print(f"Generated npz file for {dataset_name} at {output_dir}/{dataset_name}_ft_ratio0.5.npz")

    # Now create the MIACIFAR10 dataset with member indices
    if dataset_name.upper() == 'CIFAR10':
        member_dataset = MIACIFAR10(idxs=member_indices, root=os.path.join(dataset_root, 'cifar10'), train=True, transform=transforms)
        nonmember_dataset = MIACIFAR10(idxs=nonmember_indices, root=os.path.join(dataset_root, 'cifar10'), train=True, transform=transforms)

        print(f"Member dataset length: {len(member_dataset)}, Non-member dataset length: {len(nonmember_dataset)}")

if __name__ == "__main__":
    # Example Usage
    dataset_name = 'CIFAR10'  # Choose between 'CIFAR10', 'CIFAR100', 'STL10', etc.
    output_dir = './output'  # Directory where npz file will be saved
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate npz file with 1000 member and 1000 nonmember samples
    main(dataset_name, output_dir, num_member_samples=1000, num_nonmember_samples=1000)
