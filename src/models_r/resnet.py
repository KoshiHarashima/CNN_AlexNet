import torch
import torch.nn as nn
from blocks import Bottleneck

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 use_se=False, dropout_rate=0.0):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
                                       use_se=use_se, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       use_se=use_se, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       use_se=use_se, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       use_se=use_se, dropout_rate=dropout_rate)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride,
                    use_se=False, dropout_rate=0.0):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample,
                        use_se=use_se, dropout_rate=dropout_rate)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels,
                                use_se=use_se, dropout_rate=dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet50(num_classes=1000, pretrained=False, use_se=False, dropout_rate=0.0):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                   use_se=use_se, dropout_rate=dropout_rate)
    if pretrained:
        state_dict = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).state_dict()
        model.load_state_dict(state_dict, strict=False)  # ignore mismatch for custom final layer
    return model
