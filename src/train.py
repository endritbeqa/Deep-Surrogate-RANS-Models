import math
import torch
import os
import json
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import tensorflow_datasets as tfds
from src.models.swin import  U_net_SwinV2, Config_UNet_Swin
from src.data import naca_dataset, dataset
from src.losses import loss
import utils
import config
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.model_config = Config_UNet_Swin.get_config()
        self.model = U_net_SwinV2.U_NET_Swin(self.model_config)
        self.output_dir = config.output_dir
        self.train_dataset = naca_dataset.get_data_from_tfds(self.config, mode='train')
        self.validation_dataset = naca_dataset.get_data_from_tfds(self.config, mode='validation')
        self.loss_func = loss.get_loss_function(config.loss_function)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs, eta_min=0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    def train_model(self):

        train_curve = []
        val_curve = []

        for epoch in range(self.config.num_epochs):
            print("Epoch:{}".format(epoch))
            self.model.train()
            train_loss = 0.0

            for batch in tfds.as_numpy(self.train_dataset):
                inputs, targets, label = naca_dataset.preprocess_data(batch)
                inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs= self.model(inputs)
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
                for batch  in tfds.as_numpy(self.validation_dataset):
                    inputs, targets, label = naca_dataset.preprocess_data(batch)
                    inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
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