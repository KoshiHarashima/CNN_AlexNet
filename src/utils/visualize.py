import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, val_losses, window_size: int = 10):
    # 移動平均
    def moving_avg(x):
        return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

    plt.figure()
    plt.plot(moving_avg(train_losses), label='Train')
    plt.plot(moving_avg(val_losses), label='Val')
    plt.legend()
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
