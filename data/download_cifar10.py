import os
import torch
import torchvision
from torchvision import datasets, transforms


def main(root: str = './data'):
    os.makedirs(root, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # CIFAR-10 のダウンロード
    datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    print("CIFAR-10 データセットをダウンロードした。")


if __name__ == '__main__':
    main()
