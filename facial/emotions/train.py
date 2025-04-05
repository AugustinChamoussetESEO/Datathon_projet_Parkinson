import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from torch import optim
from torch.optim import Adam
from torchmetrics import ConfusionMatrix
from torchsummary import summary
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import wandb
from models import MyModel
from utils import TrainingStatistics

import torch.nn.functional as F


def train_model():
    model.train()
    stats.on_epoch_started()
    pbar = tqdm(unit="batch", file=sys.stdout, total=len(train_loader))

    n = 0
    correct = 0
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = F.one_hot(labels, num_classes=5).float()

        outputs = model(inputs)
        loss = criterion(outputs.to(device), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats.on_training_step(loss.item())
        train_loss += loss.item()

        correct += (torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).sum()
        n += len(labels)

        pbar.set_postfix(stats.get_progbar_postfix())
        pbar.update(1)

    pbar.close()
    stats.on_epoch_ended()

    return train_loss / len(train_loader), 100.0 * correct / n


def test_model():
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = F.one_hot(labels, num_classes=7).float()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss

            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == torch.argmax(labels, dim=1)).sum().item()

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_labels = torch.tensor(np.concatenate(all_labels))
    all_predictions = torch.tensor(np.concatenate(all_predictions))
    confmat = ConfusionMatrix(task="multiclass", num_classes=len(train_dataset.classes), normalize='true')
    con = confmat(all_labels, all_predictions)

    stats.test_loss.append(test_loss.item())

    acc = 100.0 * correct / total
    eval_loss = test_loss / len(test_loader)

    print('Eval loss: %.4f, Eval Accuracy: %.4f %%' % (eval_loss, acc))

    return eval_loss, acc, con


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset")
    parser.add_argument("--out-dir", type=str, default="out", help="Path where to save log files")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs of training")
    parser.add_argument("--batch-size", type=int, default=32, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam: learning rate")
    parser.add_argument("--image-size", type=int, default=48, help="Image size")
    parser.add_argument("--channels", type=int, default=3, help="Image channels")

    opt = parser.parse_args()
    print(opt)

    wandb.login(key='5fb5063613945dcce0d74c1f1b9f3b7506c4dece')
    wandb.init(project="FER", config={"Epochs": opt.epochs}, name="")

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device used: " + device)

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(20, interpolation=InterpolationMode.BILINEAR),
                                          transforms.Grayscale(),
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Grayscale(),
                                         transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(opt.dataset_path + '/train', train_transform)
    test_dataset = datasets.ImageFolder(opt.dataset_path + '/test', test_transform)

    print("Using %d images for training." % len(train_dataset))
    print("Using %d images for evaluation." % len(test_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    model = MyModel(len(train_dataset.classes)).to(device)
    summary(model, (opt.channels, opt.image_size, opt.image_size))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6, min_lr=1e-6)  # goal: minimize loss

    stats = TrainingStatistics()
    wandb.watch(model, log='all')

    best_loss = float("inf")

    for epoch in range(opt.epochs):
        print("[Epoch: %d/%d]" % (epoch + 1, opt.epochs))

        train_loss, train_accuracy = train_model()
        test_loss, test_accuracy, confusion_matrix = test_model()

        scheduler.step(test_loss)

        wandb.log({'Train Loss': train_loss,
                   'Test Loss': test_loss,
                   'Train Accuracy': train_accuracy,
                   'Test Accuracy': test_accuracy,
                   'Epoch': epoch + 1,
                   'Learning Rate': optimizer.param_groups[0]['lr']})

        df_cm = pd.DataFrame(confusion_matrix, index=[i for i in train_dataset.classes],
                             columns=[i for i in train_dataset.classes])
        plt.figure(figsize=(10, 7))
        ax = sn.heatmap(df_cm, annot=True)

        wandb.log({"Confusion Matrix": wandb.Image(plt)})
        plt.close()

        # Save the best checkpoint
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, opt.out_dir + '/checkpoints/best_model.pkl')

    wandb.finish()
