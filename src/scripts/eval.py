import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.alexnet import AlexNet


def main(config_path, model_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=cfg['training']['batch_size'], shuffle=False)

    model = AlexNet(num_classes=cfg['model']['num_classes']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total
    print(f'Test Accuracy: {acc:.2%}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.model_path)
