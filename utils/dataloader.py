from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

def get_loaders(batch_size):

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True,
                                     download=True, transform=transform_train)

    test_dataset = datasets.CIFAR10(root="./data", train=False,
                                    download=True, transform=transform_test)

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
