import torch
import torchvision
import torchvision.transforms as T

def get_augmentations(augmentation = 'basic'):
    """
    Returns the augmentation strategy specified by the user.

    Args:
        augmentation (str): The augmentation strategy to use.

    Returns:
        torchvision.transforms.Compose: A composition of the specified augmentations.
    """
    if augmentation == 'resnet':
        return resnet_augmentations()
    elif augmentation == 'basic':
        return basic_augmentations()
    else:
        raise ValueError(f'Invalid augmentation strategy: {augmentation}')
    
def basic_augmentations():
    """
    Basic augmentations including normalization.
    Suitable for simple training without heavy augmentation.
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

def resnet_augmentations():
    """
    Augmentation strategy based on ResNet paper description:
    - The image is resized with its shorter side randomly sampled between 256 and 480 pixels.
    - A 224x224 crop is randomly sampled from the resized image or its horizontal flip.
    - Standard per-pixel mean subtraction and color augmentation are applied.

    Returns:
        torchvision.transforms.Compose: A composition of the specified augmentations.
    """
    return T.Compose([
        T.RandomResizedCrop(224, scale=(256/480, 1.0)),  # Resizes shorter side randomly sampled in [256, 480]
        T.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Per-pixel mean subtraction
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)  # Standard color augmentation
    ])
