import os
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms

class TeamMateDataset(Dataset):
    def __init__(self, n_images=50, train=True, transform=None):
        """
        Args:
            n_images (int): Number of images per class (0 and 1).
            train (bool): If True, use training set, else test set.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if train:
            dataset_type = 'train'
        else:
            dataset_type = 'test'

        subject_0 = os.listdir(f'/home/pi/ee347/lab8/data/{dataset_type}/0')
        subject_1 = os.listdir(f'/home/pi/ee347/lab8/data/{dataset_type}/1')

        assert len(subject_0) >= n_images and len(subject_1) >= n_images, f'Number of images in each folder should be {n_images}'

        subject_0 = subject_0[: n_images]
        subject_1 = subject_1[: n_images]

        image_paths = subject_0 + subject_1

        self.dataset = []
        self.labels = []

        # Define the normalization transform for MobileNetV3
        self.transform = transform
        self.mean = [0.485, 0.456, 0.406]  # ImageNet mean
        self.std = [0.229, 0.224, 0.225]   # ImageNet std

        # Load and preprocess images
        for i, image_path in tqdm(enumerate(image_paths), desc="Loading Images", total=n_images * 2, leave=False):
            if i >= n_images:
                subject = 1
            else:
                subject = 0

            image = cv2.imread(f'/home/pi/ee347/lab8/data/{dataset_type}/{subject}/' + image_path)

            # Resize the image to 64x64
            image = cv2.resize(image, (64, 64))

            # Normalize the image
            image = image / 255.0  # Normalize to [0, 1]

            # Convert image from HWC to CHW format (required by PyTorch)
            image = torch.tensor(image).permute(2, 0, 1).float()

            # Apply transformation if provided
            if self.transform:
                image = self.transform(image)

            # Normalize the image using ImageNet mean and std
            image = transforms.Normalize(mean=self.mean, std=self.std)(image)

            self.dataset.append(image)
            self.labels.append(subject)

        # Convert lists to tensors
        self.dataset = torch.stack(self.dataset)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]
