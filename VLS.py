import glob
import os
import random

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from tqdm import tqdm


class VLSDataset(Dataset):
    def __init__(self, root, is_train=True, transform=None):
        if is_train:
            self.root = os.path.join(root, "train")
        else:
            self.root = os.path.join(root, "val")

        self.transform = transform
        self.data = sorted(glob.glob(os.path.join(self.root, "data", "*.npy")))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        mask = os.path.join(self.root, "masks", os.path.basename(data))
        data, mask = np.load(data), np.load(mask)
        if self.transform:
            data, mask = self.transform(data, mask)
        return {"image": data, "mask": mask}


class ToTensor:
    def __call__(self, data, mask):
        # data = torch.as_tensor(data).permute((2, 0, 1))
        data = torch.as_tensor(data).contiguous()
        mask = torch.as_tensor(mask).contiguous()
        return data, mask


class MaskToLabel:
    def __init__(self, boundaries):
        self.boundaries = torch.tensor(boundaries)

    def __call__(self, data, mask):
        mask = torch.bucketize(mask, self.boundaries)
        return data, mask


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def evaluate_acc(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    accuracy = 0
    accuracy_fg = 0
    # precision = 0
    # recall = 0
    # fscore = 0

    # iterate over the validation set
    for batch in tqdm(
        dataloader, total=num_val_batches, desc="Validation round", unit="batch", leave=False
    ):
        image, mask_true = batch["image"], batch["mask"]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long).flatten()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            mask_pred = torch.softmax(mask_pred, dim=1).argmax(dim=1).flatten()

            tp = mask_pred == mask_true
            # print((mask_pred == 0).sum(), tp.sum())
            accuracy += tp.sum() / mask_true.numel()
            fg = torch.bitwise_or(mask_true > 0, mask_pred > 0)
            accuracy_fg += tp[fg].sum() / fg.sum()
            # prec, rec, fs, _ = precision_recall_fscore_support(
            #     mask_true.cpu() > 0, mask_pred.cpu() > 0, average="weighted"
            # )
            # precision += prec
            # recall += rec
            # fscore += fs

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return accuracy
    return accuracy / num_val_batches, accuracy_fg / num_val_batches
    # return precision / num_val_batches, recall / num_val_batches, fscore / num_val_batches
