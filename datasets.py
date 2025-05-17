import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class CIFARDataset(Dataset):
    def __init__(self, root="data", train=True):
        data_path = os.path.join(root, "cifar-10-batches-py")
        if train:
            data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(data_path, "test_batch")]

        self.images = []
        self.labels = []
        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                self.images.extend(dict[b'data'])
                self.labels.extend(dict[b'labels'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((3, 32, 32)).astype(np.float32)   # RGB
        label = self.labels[idx]
        return image/255., label


class AnimalDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.image_paths = []
        self.labels = []
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                           "squirrel"]
        self.transform = transform

        data_path = os.path.join(root, "animals")

        if train:
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "test")

        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            for item in os.listdir(data_files):
                path = os.path.join(data_files, item)
                self.image_paths.append(path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        # image = cv2.imread(image_path)
        # image = torch.from_numpy(image)
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    # dataset = AnimalDataset(root="./data", train=True, transform=transform)
    dataset = AnimalDataset(root="/mnt/e/Dataset/animals_v2", train=True, transform= transform)
    # image, label = dataset.__getitem__(5555)
    # print(image.shape)
    # print(label)
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True
    )
    for images, labels in train_dataloader:
        print(images.shape)
        print(labels)