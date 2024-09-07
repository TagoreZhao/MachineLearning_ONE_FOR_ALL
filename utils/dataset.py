
import torch
import torchvision
import torchvision.transforms as T
from .augmentation import get_augmentations


def get_CIFAR_10(batch_size=64, num_workers=5, agumentation='basic'):

    transform = get_augmentations(agumentation)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

    return trainloader, testloader
