import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

from config import *


def get_data():
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    # dataset = dset.ImageFolder(root=dataroot,
    #                            transform=transforms.Compose([
    #                                transforms.TenCrop(64),
    #                                transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops]))
    #                            ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    return dataset, dataloader
