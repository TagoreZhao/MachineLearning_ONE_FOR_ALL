
import torch
import torchvision
import torchvision.transforms as T
from .augmentation import get_augmentations


def get_CIFAR_10(batch_size=64, num_workers=5, augmentation='basic'):
    """
    Loads the CIFAR-10 dataset with specific augmentations for training and basic transformations for testing.
    Args:
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        augmentation (str): Augmentation type for training data.
    Returns:
        trainloader, testloader: Data loaders for training and testing datasets.
    """
    # Augmentation for training data
    train_transform = get_augmentations(augmentation)

    # Basic transformation for test data (no augmentations)
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load training dataset with augmentations
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    # Load test dataset without augmentations
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return trainloader, testloader
