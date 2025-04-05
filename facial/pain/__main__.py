import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchsummary import summary
from torchvision import datasets, transforms
from tqdm import tqdm

import wandb
from models import get_convnext, Baseline
from vgg_mask import VGGMask

random.seed(2000)
np.random.seed(2000)
torch.manual_seed(2000)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self,
                 model,
                 training_dataloader,
                 validation_dataloader,
                 testing_dataloader,
                 num_classes,
                 output_dir,
                 max_epochs: int = 10000,
                 early_stopping_patience: int = 12,
                 lr: float = 1e-4,
                 amp: bool = False,
                 label_smoothing: float = 0.2,
                 ema_decay: float = 0.99,
                 ema_update_every: int = 16,

                 checkpoint_path: str = None,
                 ):

        self.epochs = max_epochs

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.testing_dataloader = testing_dataloader

        self.num_classes = num_classes

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device used: " + self.device.type)

        self.amp = amp

        self.model = model.to(self.device)
        # summary(self.model, (3, 44, 44))

        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=128)
        self.ema = EMA(model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        self.early_stopping_patience = early_stopping_patience
        self.label_smoothing = label_smoothing

        self.output_directory = Path(output_dir)
        self.output_directory.mkdir(exist_ok=True)

        self.best_val_loss = float('inf')

        if checkpoint_path:
            self.load(checkpoint_path)

        wandb.watch(model, log='all')

    def run(self):
        counter = 0  # Counter for epochs with no validation loss improvement

        for epoch in range(self.epochs):
            print("[Epoch: %d/%d]" % (epoch + 1, self.epochs))

            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.val_epoch()
            self.visualize_stn()

            wandb.log({'Train Loss': train_loss,
                       'Val Loss': val_loss,
                       'Train Accuracy': train_accuracy,
                       'Val Accuracy': val_accuracy,
                       'Epoch': epoch + 1,
                       'Learning Rate': self.optimizer.param_groups[0]['lr']})

            # Early stopping
            if val_loss < self.best_val_loss:
                self.save()
                counter = 0
                self.best_val_loss = val_loss
            else:
                counter += 1
                if counter >= self.early_stopping_patience:
                    print(
                        "Validation loss did not improve for %d epochs. Stopping training." % self.early_stopping_patience)
                    break

        self.test_model()
        wandb.finish()

    def train_epoch(self):
        self.model.train()

        avg_accuracy = []
        avg_loss = []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.training_dataloader))
        for batch_idx, data in enumerate(self.training_dataloader):
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                outputs, att_loss = self.model(inputs)
            loss = F.cross_entropy(outputs, labels, label_smoothing=self.label_smoothing) + att_loss

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update()
            self.scheduler.step()

            batch_accuracy = (torch.argmax(outputs, dim=1) == labels).sum().item() / labels.size(0)

            avg_loss.append(loss.item())
            avg_accuracy.append(batch_accuracy)

            # Update progress bar
            pbar.set_postfix({'loss': np.mean(avg_loss), 'acc': np.mean(avg_accuracy) * 100.0})
            pbar.update(1)

        pbar.close()

        return np.mean(avg_loss), np.mean(avg_accuracy) * 100.0

    def val_epoch(self):
        self.model.eval()

        avg_loss = []
        avg_accuracy = []

        for batch_idx, (inputs, labels) in enumerate(self.validation_dataloader):
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                outputs, att_loss = self.model(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
            loss = F.cross_entropy(outputs_avg, labels) + att_loss

            batch_accuracy = (torch.argmax(outputs_avg, dim=1) == labels).sum().item() / labels.size(0)

            avg_loss.append(loss.item())
            avg_accuracy.append(batch_accuracy)

        print('Eval loss: %.4f, Eval Accuracy: %.4f %%' % (np.mean(avg_loss) * 1.0, np.mean(avg_accuracy) * 100.0))
        return np.mean(avg_loss), np.mean(avg_accuracy) * 100.0

    def test_model(self):
        self.ema.eval()

        avg_accuracy = []

        for batch_idx, (inputs, labels) in enumerate(self.testing_dataloader):
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                outputs, _ = self.ema(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

            batch_accuracy = (torch.argmax(outputs_avg, dim=1) == labels).sum().item() / labels.size(0)
            avg_accuracy.append(batch_accuracy)

        print('Test Accuracy: %.4f %%' % (np.mean(avg_accuracy) * 100.0))

    def visualize_stn(self):
        self.model.eval()

        batch = next(iter(self.training_dataloader))[0].to(self.device)

        grid = torchvision.utils.make_grid(batch, nrow=8, padding=4)
        transforms.ToPILImage()(grid).save(str(self.output_directory / 'grid_images.jpg'))

        with torch.autocast(self.device.type, enabled=self.amp):
            stn_batch = self.model.stn(batch)

        grid = torchvision.utils.make_grid(stn_batch, nrow=8, padding=4)
        transforms.ToPILImage()(grid).save(str(self.output_directory / 'stn_images.jpg'))

    def save(self):
        data = {
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_val_loss,
        }

        torch.save(data, str(self.output_directory / 'model.pt'))

    def load(self, path):
        data = torch.load(path, map_location=self.device)

        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.scaler.load_state_dict(data['scaler'])
        self.scheduler.load_state_dict(data['scheduler'])
        self.best_val_loss = data['best_loss']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-path", type=str, help="Path to the dataset")
    parser.add_argument("--output-dir", type=str, default="out", help="Path where to save log files and best model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs of training")
    parser.add_argument("--batch-size", type=int, default=32, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam: learning rate")
    parser.add_argument("--image-size", type=int, default=48, help="Image size")
    parser.add_argument("--crop-size", type=int, default=44, help="Crop size")
    parser.add_argument("--channels", type=int, default=1, help="Image channels")
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument("--num-workers", type=int, default=0,
                        help="The number of subprocesses to use for data loading."
                             "0 means that the data will be loaded in the main process.")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--model-size', choices=['tiny', 'small', 'base', 'large'], default='tiny',
                        help='Choose size: tiny, base, or large')

    opt = parser.parse_args()
    print(opt)

    wandb.login(key='334a10b3731793af77ee9f3b73ad7b350a52089e')
    wandb.init(project="EmoNext")

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(degrees=20),
                                          transforms.Grayscale(),
                                          transforms.Resize(236),
                                          transforms.RandomCrop(size=opt.crop_size),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
    test_transform = transforms.Compose([transforms.Grayscale(),
                                         transforms.Resize(236),
                                         transforms.TenCrop(opt.crop_size),
                                         transforms.Lambda(lambda crops: torch.stack(
                                             [transforms.ToTensor()(crop) for crop in crops])),
                                         transforms.Lambda(lambda crops: torch.stack(
                                             [transforms.Lambda(lambda x: x.repeat(3, 1, 1))(crop) for crop in crops]))])

    train_dataset = datasets.ImageFolder(opt.dataset_path + '/train', train_transform)
    val_dataset = datasets.ImageFolder(opt.dataset_path + '/val', test_transform)
    test_dataset = datasets.ImageFolder(opt.dataset_path + '/test', test_transform)

    print("Using %d images for training." % len(train_dataset))
    print("Using %d images for evaluation." % len(val_dataset))
    print("Using %d images for testing." % len(test_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    net = get_convnext(
        len(train_dataset.classes),
        opt.model_size
    )

    Trainer(
        model=net,
        training_dataloader=train_loader,
        validation_dataloader=val_loader,
        testing_dataloader=test_loader,
        num_classes=len(train_dataset.classes),
        lr=opt.lr,
        output_dir=opt.output_dir,
        checkpoint_path=opt.checkpoint,
        max_epochs=opt.epochs,
        amp=opt.amp
    ).run()

#python __main__.py --dataset-path=C:\Users\abohi\Desktop\10_DATA\FER2013 --batch-size=64 --lr=0.0001 --epochs=10000 --amp --crop-size=224 --model-size=tiny --num-workers=0