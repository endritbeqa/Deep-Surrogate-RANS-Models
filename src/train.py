import torch
import os
import json
import torch.optim as optim

from models.swin import Swin_CNN
from data.data_loader import Airfoil_Dataset
from losses.loss import get_loss_function
from torch.utils.data import DataLoader
from utils import plot_losses, save_images
from config import get_config

class Trainer(object):
    def __init__(self, config):
        self.model = Swin_CNN(config)
        self.output_dir = config.output_dir
        self.train_dataset = Airfoil_Dataset(config, mode='train')
        self.val_dataset = Airfoil_Dataset(config, mode='validation')
        self.train_dataloader = DataLoader(self.train_dataset, config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, config.batch_size, shuffle=True)
        self.loss_func = get_loss_function(config.loss_function)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        os.mkdir(self.output_dir)
        for dir in [os.path.join(self.output_dir, "checkpoints"),
                    os.path.join(self.output_dir,"logs"),
                    os.path.join(self.output_dir,"config"),
                    os.path.join(self.output_dir,"images")]:
            os.mkdir(dir)



    def train_model(self, config):

        train_curve = []
        val_curve = []

        for epoch in range(config.num_epochs):
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
                file.write("{},{}\n".format(str(train_loss),str(val_loss)))

            if epoch % config.checkpoint_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss': train_loss,
                    'val_loss': val_loss,
                }
                torch.save(self.model.state_dict(), "{}/checkpoints/{}.pth".format(self.output_dir,epoch))

                save_images(outputs,self.output_dir ,epoch)




        loss_plot = plot_losses(train_curve,val_curve)
        loss_plot.savefig("{}/logs/loss_curves.png".format(self.output_dir))
        loss_plot.close()

        config_dict = config.to_dict()

        # Save the dictionary as a JSON file
        with open("{}/config/config.json".format(self.output_dir), '+w') as json_file:
            json.dump(config_dict, json_file, indent=4)

        return val_curve[-1]
