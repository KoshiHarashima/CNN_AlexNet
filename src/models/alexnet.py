import torch                             # PyTorch本体をインポート
import torch.nn as nn                   # ニューラルネットワークのモジュール群をインポート

class AlexNet(nn.Module):               # nn.Moduleを継承してAlexNetクラスを定義
    def __init__(self, num_classes: int = 10):
        super().__init__()              # 親クラス（nn.Module）の初期化を呼び出し
        # 特徴抽出器（畳み込み層＋プーリング層）の定義
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 入力3チャネル→出力64チャネル、11x11カーネル、ストライド4、パディング2
            nn.ReLU(inplace=True),                                 # 活性化関数ReLU（インプレースでメモリ節約）
            nn.MaxPool2d(kernel_size=3, stride=2),                 # 3x3カーネルの最大プーリング、ストライド2

            nn.Conv2d(64, 192, kernel_size=5, padding=2),          # 64チャネル→192チャネル、5x5カーネル、パディング2
            nn.ReLU(inplace=True),                                 # ReLU活性化
            nn.MaxPool2d(kernel_size=3, stride=2),                 # プーリング

            nn.Conv2d(192, 384, kernel_size=3, padding=1),         # 192チャネル→384チャネル、3x3カーネル、パディング1
            nn.ReLU(inplace=True),                                 # ReLU活性化

            nn.Conv2d(384, 256, kernel_size=3, padding=1),         # 384チャネル→256チャネル、3x3カーネル、パディング1
            nn.ReLU(inplace=True),                                 # ReLU活性化

            nn.Conv2d(256, 256, kernel_size=3, padding=1),         # 256チャネル→256チャネル、3x3カーネル、パディング1
            nn.ReLU(inplace=True),                                 # ReLU活性化
            nn.MaxPool2d(kernel_size=3, stride=2),                 # プーリング
        )
        # 平均プーリングで出力サイズを6×6に固定
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # 分類器（全結合層＋ドロップアウト＋ReLU）の定義
        self.classifier = nn.Sequential(
            nn.Dropout(),                                         # ドロップアウト層（過学習防止）
            nn.Linear(256 * 6 * 6, 4096),                         # 全結合層: フラットにした特徴量→4096次元
            nn.ReLU(inplace=True),                                # ReLU活性化

            nn.Dropout(),                                         # ドロップアウト
            nn.Linear(4096, 4096),                                # 全結合層: 4096→4096
            nn.ReLU(inplace=True),                                # ReLU活性化

            nn.Linear(4096, num_classes),                         # 全結合層: 4096→分類クラス数
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)         # 特徴抽出器に入力を通す
        x = self.avgpool(x)          # 平均プーリングで6×6に変換
        x = torch.flatten(x, 1)      # バッチ次元を残して平坦化（全結合層に入力可能に）
        x = self.classifier(x)       # 分類器に通して最終出力を得る
        return x                     # ログit（未正規化スコア）を返す
