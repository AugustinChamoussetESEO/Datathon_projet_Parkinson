import matplotlib.pyplot as plt
import numpy as np


class TrainingStatistics:
    """
    Computes statistics per epoch.
    """

    def __init__(self):
        self.batch_losses = []

        self.train_loss = []
        self.test_loss = []

    def on_training_step(self, loss):
        self.batch_losses.append(loss)

    def on_epoch_started(self):
        self.batch_losses = []

    def on_epoch_ended(self):
        self.train_loss.append(np.mean(self.batch_losses))

    def get_progbar_postfix(self):
        return {
            "loss": "%1.4f" % np.mean(self.batch_losses),
        }

    def plot_losses(self, epochs: list):
        plt.plot(epochs, self.train_loss, label='Training Loss')
        plt.plot(epochs, self.test_loss, label='Test Loss')

        plt.xlabel('Epochs')
        plt.xlabel('Loss')

        plt.legend()
        plt.savefig('test.png')