# CNN_AlexNet
CNNのAlexNetをscratchから構築をする練習コード

````markdown
# AlexNet PyTorch 実装リポジトリ

このリポジトリは、PyTorch で実装された AlexNet を使って画像分類を行うためのサンプルプロジェクトです。
初心者でも実験やカスタマイズがしやすいように、構造化されたコードと設定ファイルを提供しています。

---

## 📋 特徴

- **シンプルかつ再利用可能**: モデル定義、学習ループ、ユーティリティ関数を明確に分割
- **設定ファイル管理**: YAML 形式でハイパーパラメータを一元管理
- **データ自動ダウンロード**: CIFAR-10 などのデータセットをスクリプトで自動取得
- **可視化サポート**: 学習曲線や混同行列を簡単に出力
- **テスト付き**: `pytest` を使った単体テストで安心

---

## 🚀 環境構築

```bash
# リポジトリをクローン
git clone https://github.com/KoshiHarashima/CNN_AlexNet.git
cd CNN_AlexNet

# 仮想環境作成（推奨）
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows

# 依存パッケージインストール
pip install -r requirements.txt
````

---

## 📂 ディレクトリ構成

```text
alexnet-pytorch/
├── README.md                # このドキュメント
├── .gitignore               # Git 無視ファイル設定
├── requirements.txt         # 必要パッケージ一覧
├── configs/                 # ハイパーパラメータ設定
│   └── default.yaml
├── data/                    # データ取得・前処理スクリプト
│   └── download_cifar10.py
├── src/                     # 実装コード
│   ├── models/              # モデル定義 (alexnet.py)
│   ├── trainers/            # 学習・評価ロジック (trainer.py)
│   ├── utils/               # ユーティリティ (early_stopping.py, visualize.py)
│   └── scripts/             # 実行用スクリプト (train.py, eval.py)
└── tests/                   # 単体テスト (test_*.py)
```

---

## 🔧 設定ファイル (`configs/default.yaml`)

```yaml
model:
  num_classes: 10
  dropout: 0.5
training:
  batch_size: 64
  epochs: 50
  lr: 0.01
  momentum: 0.9
  patience: 7
```

* **num\_classes**: 分類クラス数
* **dropout**: ドロップアウト率
* **batch\_size**: バッチサイズ
* **epochs**: 最大エポック数
* **lr**: 学習率
* **momentum**: SGD のモーメンタム
* **patience**: 早期停止の待機エポック数

---

## 🏃 実行方法

### データ準備

```bash
python data/download_cifar10.py
```

### 学習

```bash
python src/scripts/train.py \
  --config configs/default.yaml \
  --device cuda  # または cpu
```

### 評価

```bash
python src/scripts/eval.py \
  --config configs/default.yaml \
  --model-path path/to/checkpoint.pth
```

---

##  テスト実行

```bash
pytest tests/
```

---

## 貢献

* Pull Request、大歓迎です。

