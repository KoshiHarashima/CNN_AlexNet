import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from src.models.alexnet import AlexNet
from src.trainers.trainer import Trainer

def make_dummy_loader(num_samples=8, batch_size=4, num_classes=10):
    # ダミーの画像データとラベルを生成
    data = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(data, labels)
    return DataLoader(ds, batch_size=batch_size)

def test_trainer_train_and_evaluate_runs_without_error():
    """
    Trainer.train が例外を出さず動き、
    train_losses, val_losses に正しい長さの履歴が残ることを確認。
    """
    device = torch.device('cpu')
    model = AlexNet(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = Trainer(model, criterion, optimizer, device, patience=1)

    train_loader = make_dummy_loader()
    val_loader   = make_dummy_loader()

    # 1 エポックだけ学習させてみる
    trainer.train(train_loader, val_loader, epochs=1)

    assert len(trainer.train_losses) == 1, "train_losses の長さが 1 ではありません"
    assert len(trainer.val_losses) == 1,   "val_losses の長さが 1 ではありません"

def test_trainer_evaluate_and_accuracy():
    """
    evaluate が float を返し、accuracy が 0～1 の範囲になることを確認。
    """
    device = torch.device('cpu')
    model = AlexNet(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = Trainer(model, criterion, optimizer, device, patience=1)

    loader = make_dummy_loader()

    loss = trainer.evaluate(loader)
    acc  = trainer.accuracy(loader)

    assert isinstance(loss, float), "evaluate は float を返すべき"
    assert 0.0 <= acc <= 1.0,      "accuracy は 0～1 の範囲で返すべき"
