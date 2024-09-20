import math
from datetime import datetime

import torch
import os
import json
import torch.optim as optim
import tensorflow_datasets as tfds
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models import model_select
from src.data import naca_dataset, dataset
from src import loss

from src import utils
from src import config



class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.model_config, self.model = model_select.get_model(train_config.model_name)
        self.output_dir = config.output_dir
        self.train_dataset = dataset.Airfoil_Dataset(self.config, mode='train')
        self.validation_dataset = dataset.Airfoil_Dataset(self.config, mode='validation')
        #self.train_dataset = naca_dataset.get_data_from_tfds(self.config, mode='train')
        #self.validation_dataset = naca_dataset.get_data_from_tfds(self.config, mode='validation')
        self.train_dataloader = DataLoader(self.train_dataset, config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True)
        self.val_dataloader = DataLoader(self.validation_dataset, config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True)
        self.loss_func = loss.get_loss_function(config.loss_function)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs, eta_min=0)
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters())
        print("Num parameters: {}".format(self.num_model_parameters))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        os.mkdir(self.output_dir)
        for dir in [os.path.join(self.output_dir, "checkpoints"),
                    os.path.join(self.output_dir, "logs"),
                    os.path.join(self.output_dir, "config"),
                    os.path.join(self.output_dir, "images"),
                    os.path.join(self.output_dir, "images/predictions"),
                    os.path.join(self.output_dir, "images/targets")]:
            os.mkdir(dir)

    def train_model(self):
        with open("{}/configs/config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.config.to_dict(), json_file, indent=4)

        with open("{}/configs/model_config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.model_config.to_dict(), json_file, indent=4)

        with open("{}/configs/model_size.txt".format(self.output_dir), '+w') as file:
            file.write("Number of model parameters: {}".format(self.num_model_parameters))

        train_curve = []
        val_curve = []

        for epoch in range(self.config.num_epochs):
            print("Epoch:{}, Started at:{}".format(epoch, datetime.now()))
            self.model.train()
            train_loss = 0.0

            for inputs, targets, label in self.train_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, targets)
                if math.isinf(loss) | math.isnan(loss):
                    print("{}, {}".format(label, loss))
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            train_loss = train_loss / len(self.train_dataset)
            train_curve.append(train_loss)

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets, label in self.val_dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_func(outputs, targets)
                    val_loss += loss.item()
            val_loss = val_loss / len(self.validation_dataset)
            val_curve.append(val_loss)

            with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
                file.write("{},{}\n".format(str(train_loss), str(val_loss)))

            if epoch % self.config.checkpoint_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, "{}/checkpoints/{}.pth".format(self.output_dir, epoch))

                utils.save_images(outputs, self.output_dir, "predictions", epoch)
                utils.save_images(targets, self.output_dir, "targets", epoch)

                loss_plot = utils.plot_losses(train_curve, val_curve)
                loss_plot.savefig("{}/logs/loss_curves.png".format(self.output_dir))
                loss_plot.close()

        return val_curve[-1]


if __name__ == '__main__':
    train_config = config.get_config()
    trainer = Trainer(train_config)
    trainer.train_model()