from pytorch_lightning.loggers import TensorBoardLogger
import os

def get_tensorboard_logger(save_dir="logs", name="resnet"):
    os.makedirs(save_dir, exist_ok=True)
    return TensorBoardLogger(save_dir=save_dir, name=name)
