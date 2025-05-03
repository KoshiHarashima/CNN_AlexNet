import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from src.utils.early_stopping import EarlyStopping

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        patience: int = 7,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.early_stopping = EarlyStopping(patience=patience)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6, verbose=True
        )
        self.train_losses = []
        self.val_losses = []

    def train(self, train_loader, val_loader, epochs: int):
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f'Train Epoch {epoch}'):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            avg_train = running_loss / len(train_loader)
            self.train_losses.append(avg_train)

            val_loss = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            print(f"Epoch {epoch}: Train Loss={avg_train:.4f}, Val Loss={val_loss:.4f}")

            if self.early_stopping(val_loss):
                print("Early stopping")
                break

    def evaluate(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(loader)

    def accuracy(self, loader) -> float:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total
