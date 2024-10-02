import os
import json
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.models import model_select
from src.data import dataset
from src import utils
from src import loss
from src import config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class Trainer(object):
    def __init__(self, train_config):
        self.config = train_config
        self.model_config, self.model = model_select.get_model(train_config.model_name)
        self.output_dir = train_config.output_dir
        self.train_dataset = dataset.Airfoil_Dataset(train_config, mode='train')
        self.val_dataset = dataset.Airfoil_Dataset(train_config, mode='validation')
        self.train_dataloader = DataLoader(self.train_dataset, train_config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, train_config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True)
        self.loss_func = loss.get_loss_function(self.config.loss_function)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_config.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=40)
        self.device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters())
        self.beta = train_config.KLD_beta
        print("Num parameters: {}".format(self.num_model_parameters))
        os.makedirs(self.output_dir, exist_ok=True)
        for dir in [os.path.join(self.output_dir, "checkpoints"),
                    os.path.join(self.output_dir, "logs"),
                    os.path.join(self.output_dir, "configs"),
                    os.path.join(self.output_dir, "images"),
                    os.path.join(self.output_dir, "images/predictions"),
                    os.path.join(self.output_dir, "images/targets")]:
            os.makedirs(dir, exist_ok=True)

    def train_model(self):
        torch.cuda.empty_cache()
        with open("{}/configs/config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.config.to_dict(), json_file, indent=4)

        with open("{}/configs/model_config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.model_config.to_dict(), json_file, indent=4)

        with open("{}/configs/model_size.txt".format(self.output_dir), '+w') as file:
            file.write("Number of model parameters: {}".format(self.num_model_parameters))

        with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
            file.write("train_loss, val_loss\n")

        with open("{}/logs/recon_vs_KLD_curves.txt".format(self.output_dir), "+a") as file:
            file.write("train_reconstruction, train_KLD, val_reconstruction, val_KLD\n")


        train_curve = []
        val_curve = []

        train_reconstruction_curve = []
        train_KLD_curve = []
        val_reconstruction_curve = []
        val_KLD_curve = []

        for epoch in range(self.config.num_epochs):
            print("Epoch:{}, Started at:{}".format(epoch, datetime.now()))
            self.model.train()
            train_loss = 0.0
            train_reconstruction_loss = 0.0
            train_KLD_loss = 0.0

            for inputs, targets, label in self.train_dataloader:

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs, mu, logvar = self.model(inputs, targets)

                loss = self.loss_func(outputs, targets, mu, logvar, self.beta)
                reconstruction_loss = F.l1_loss(outputs, targets)
                kld = loss.item() - reconstruction_loss.item()

                train_loss += loss.item()
                train_reconstruction_loss += reconstruction_loss.item()
                train_KLD_loss += kld

                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            train_loss = train_loss / len(self.train_dataloader.dataset)
            train_reconstruction_loss = train_reconstruction_loss / len(self.train_dataloader.dataset)
            train_KLD_loss = train_KLD_loss / len(self.train_dataloader.dataset)
            train_curve.append(train_loss)
            train_reconstruction_curve.append(train_reconstruction_loss)
            train_KLD_curve.append(train_KLD_loss)

            self.model.eval()
            val_loss = 0.0
            val_reconstruction_loss = 0.0
            val_KLD_loss = 0.0

            with torch.no_grad():
                for inputs, targets, label in self.val_dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs, mu, logvar = self.model(inputs, targets)
                    loss = self.loss_func(outputs, targets, mu, logvar, self.beta)
                    mae = F.l1_loss(outputs, targets)
                    kld = loss.item() - mae.item()
                    val_loss += loss.item()
                    val_reconstruction_loss += mae.item()
                    val_KLD_loss += kld

            val_loss = val_loss / len(self.val_dataloader.dataset)
            val_reconstruction_loss = val_reconstruction_loss / len(self.val_dataloader.dataset)
            val_KLD_loss = val_KLD_loss / len(self.val_dataloader.dataset)

            val_curve.append(val_loss)
            val_reconstruction_curve.append(val_reconstruction_loss)
            val_KLD_curve.append(val_KLD_loss)

            with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
                file.write("{},{}\n".format(str(train_loss), str(val_loss)))

            with open("{}/logs/recon_vs_KLD_curves.txt".format(self.output_dir), "+a") as file:
                file.write("{},{},{},{}\n".format(str(train_reconstruction_loss), str(train_KLD_loss),str(val_reconstruction_loss), str(val_KLD_loss)))

            if epoch % self.config.checkpoint_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'train_config': self.config,
                    'model_config': self.model_config,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, "{}/checkpoints/{}.pth".format(self.output_dir, epoch))

                utils.save_images(outputs, self.output_dir, "predictions", epoch)
                utils.save_images(targets, self.output_dir, "targets", epoch)

                loss_plot = utils.plot_losses(train_curve, val_curve)
                loss_plot.savefig("{}/logs/loss_curves.png".format(self.output_dir))
                loss_plot.close()

                loss_plot = utils.plot_recon_vs_KLD(train_reconstruction_curve, train_KLD_curve,val_reconstruction_curve, val_KLD_curve)
                loss_plot.savefig("{}/logs/recon_vs_KLD.png".format(self.output_dir))
                loss_plot.close()


        return val_curve[-1]


if __name__ == '__main__':
    train_config = config.get_config()
    trainer = Trainer(train_config)
    trainer.train_model()
