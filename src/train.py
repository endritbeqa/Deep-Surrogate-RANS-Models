import torch
import os
import json
import torch.optim as optim

from src.models.swinV2_CNN import swin
from src.data import dataset
from src.losses import loss
from torch.utils.data import DataLoader
import utils
import config


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.model = swin.Swin_CNN(config)
        self.output_dir = config.output_dir
        self.train_dataset = dataset.Airfoil_Dataset(config, mode='train')
        self.val_dataset = dataset.Airfoil_Dataset(config, mode='validation')
        self.train_dataloader = DataLoader(self.train_dataset, config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, config.batch_size, shuffle=True)
        self.loss_func = loss.get_loss_function(config.loss_function)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        os.mkdir(self.output_dir)
        for dir in [os.path.join(self.output_dir, "checkpoints"),
                    os.path.join(self.output_dir, "logs"),
                    os.path.join(self.output_dir, "config"),
                    os.path.join(self.output_dir, "images")]:
            os.mkdir(dir)

    def train_model(self):

        train_curve = []
        val_curve = []

        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loss = 0.0

            for inputs, targets in self.train_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, targets)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss / len(self.train_dataloader.dataset)
            train_curve.append(train_loss)

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in self.val_dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_func(outputs, targets)
                    val_loss += loss.item()
            val_loss = val_loss / len(self.val_dataloader.dataset)
            val_curve.append(val_loss)

            with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
                file.write("{},{}\n".format(str(train_loss), str(val_loss)))

            if epoch % self.config.checkpoint_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss': train_loss,
                    'val_loss': val_loss,
                }
                torch.save(self.model.state_dict(), "{}/checkpoints/{}.pth".format(self.output_dir, epoch))

                utils.save_images(outputs, self.output_dir, epoch)

        loss_plot = utils.plot_losses(train_curve, val_curve)
        loss_plot.savefig("{}/logs/loss_curves.png".format(self.output_dir))
        loss_plot.close()

        config_dict = self.config.to_dict()

        # Save the dictionary as a JSON file
        with open("{}/config/config.json".format(self.output_dir), '+w') as json_file:
            json.dump(config_dict, json_file, indent=4)

        return val_curve[-1]


if __name__ == '__main__':
    train_config = config.get_config()
    trainer = Trainer(train_config)
    trainer.train_model()
