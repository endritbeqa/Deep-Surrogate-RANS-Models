import math

from datetime import datetime
import numpy
import torch
import os
import json
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset

from src.models import U_net_SwinV2, Config_UNet_Swin
from src.dataloader import dataset
from src.losses import loss
from torch.utils.data import DataLoader
from src import utils
from src import config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.model_config = Config_UNet_Swin.get_config()
        self.model = U_net_SwinV2.U_NET_Swin(self.model_config)
        self.output_dir = config.output_dir

        self.train_dataset = dataset.Airfoil_Dataset(config, mode='train')
        self.val_dataset = dataset.Airfoil_Dataset(config, mode='validation')
        self.train_dataloader = DataLoader(self.train_dataset, config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2)
        self.val_dataloader = DataLoader(self.val_dataset, config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2)
        self.loss_func = loss.KLD #loss.get_loss_function(config.loss_function)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs, eta_min=0)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print("Num parameters: {}".format(count_parameters(self.model)))
        os.mkdir(self.output_dir)
        for dir in [os.path.join(self.output_dir, "checkpoints"),
                    os.path.join(self.output_dir, "logs"),
                    os.path.join(self.output_dir, "config"),
                    os.path.join(self.output_dir, "images"),
                    os.path.join(self.output_dir, "images/predictions"),
                    os.path.join(self.output_dir, "images/targets")]:
            os.mkdir(dir)

        all_train_inputs = []
        all_train_targets = []

        for inputs, targets, labels in self.train_dataloader:
            all_train_inputs.append(inputs)
            all_train_targets.append(targets)

        all_train_inputs = torch.from_numpy(numpy.concatenate(all_train_inputs, axis=0)).to(self.device)
        all_train_targets = torch.from_numpy(numpy.concatenate(all_train_targets, axis=0)).to(self.device)

        all_validation_inputs = []
        all_validation_targets = []

        for inputs, targets, labels in self.train_dataloader:
            all_validation_inputs.append(inputs)
            all_validation_targets.append(targets)

        all_validation_inputs = torch.from_numpy(numpy.concatenate(all_validation_inputs, axis=0)).to(self.device)
        all_validation_targets = torch.from_numpy(numpy.concatenate(all_validation_targets, axis=0)).to(self.device)

        self.GPU_train_dataset = TensorDataset(all_train_inputs, all_train_targets)
        self.GPU_validation_dataset = TensorDataset(all_validation_inputs, all_validation_targets)
        self.GPU_train_dataloader = DataLoader(self.GPU_train_dataset, batch_size=config.batch_size, shuffle=True)
        self.GPU_validation_dataloader = DataLoader(self.GPU_validation_dataset, batch_size=config.batch_size, shuffle=True)

    def train_model(self):

        train_curve = []
        val_curve = []

        for epoch in range(self.config.num_epochs):
            print("Epoch:{}".format(epoch))
            self.model.train()
            train_loss = 0.0

            for inputs, targets in self.GPU_train_dataloader:
                self.optimizer.zero_grad()
                outputs, mu, logvar = self.model(inputs, targets)
                loss = self.loss_func(outputs, targets , mu, logvar)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            train_loss = train_loss / len(self.GPU_train_dataset)
            train_curve.append(train_loss)

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in self.GPU_validation_dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs, mu, logvar = self.model(inputs, targets)
                    loss = self.loss_func(outputs, targets , mu, logvar)
                    val_loss += loss.item()
            val_loss = val_loss / len(self.GPU_validation_dataset)
            val_curve.append(val_loss)

            with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
                file.write("{},{}\n".format(str(train_loss), str(val_loss)))

            current_time = datetime.now().strftime("%H:%M:%S")

            # Print the current time in the console
            print(f"Current Time: {current_time}")

            if epoch % self.config.checkpoint_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, "{}/checkpoints/{}.pth".format(self.output_dir, epoch))

                utils.save_images(outputs, self.output_dir, "predictions", epoch)
                utils.save_images(targets, self.output_dir, "targets", epoch)

        loss_plot = utils.plot_losses(train_curve, val_curve)
        loss_plot.savefig("{}/logs/loss_curves.png".format(self.output_dir))
        loss_plot.close()

        with open("{}/config/config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.config.to_dict(), json_file, indent=4)

        with open("{}/config/model_config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.model_config.to_dict(), json_file, indent=4)

        with open("{}/config/model_size.txt".format(self.output_dir), '+w') as file:
            file.write("Number of model parameters: {}".format(count_parameters(self.model)))

        return val_curve[-1]


if __name__ == '__main__':
    train_config = config.get_config()
    trainer = Trainer(train_config)
    trainer.train_model()
