import torch
from src.models.alexnet import AlexNet

def test_alexnet_output_shape():
    """
    AlexNet にランダムなバッチを入力して、
    出力の shape が (batch_size, num_classes) になっているか確認する。
    """
    batch_size = 4
    num_classes = 10
    model = AlexNet(num_classes=num_classes)
    x = torch.randn(batch_size, 3, 224, 224)
    y = model(x)
    assert y.shape == (batch_size, num_classes), \
        f"Expected shape {(batch_size, num_classes)}, but got {y.shape}"
