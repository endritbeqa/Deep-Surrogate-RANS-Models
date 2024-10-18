import json
import math
import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.models import model_select
from src.data import dataset
from src import config, utils


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



def load_training(trainer,checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    train_config = checkpoint['train_config']
    trainer.start_epoch = checkpoint['epoch']
    trainer.model_config = checkpoint['model_config']
    trainer.model = model_select.load_model(trainer.config.model_name, trainer.model_config, checkpoint)
    trainer.model = trainer.model.to(trainer.device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=train_config.lr,weight_decay=train_config.weight_decay)
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])

class DiffusionTrainer(object):
    def __init__(self, train_config):
        self.config = train_config
        self.model_config, self.model = model_select.get_model(train_config.model_name)
        self.model_config.device = self.config.device # TODO this is a bit ugly look into fixing it
        self.output_dir = train_config.output_dir
        self.train_dataset = dataset.Airfoil_Dataset(train_config, mode='train')
        self.val_dataset = dataset.Airfoil_Dataset(train_config, mode='validation')
        self.train_dataloader = DataLoader(self.train_dataset, train_config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, train_config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=30)
        self.device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters())
        self.start_epoch = 0
        print("Model: {}, Num parameters: {}".format(self.config.model_name, self.num_model_parameters))
        os.makedirs(self.output_dir, exist_ok=True)
        for dir in [os.path.join(self.output_dir, "checkpoints"),
                    os.path.join(self.output_dir, "logs"),
                    os.path.join(self.output_dir, "configs")]:
            os.makedirs(dir, exist_ok=True)




    def train(self):
        with open("{}/configs/config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.config.to_dict(), json_file, indent=4)

        with open("{}/configs/model_config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.model_config.to_dict(), json_file, indent=4)

        with open("{}/configs/model_size.txt".format(self.output_dir), '+w') as file:
            file.write("Number of model parameters: {}".format(self.num_model_parameters))


        train_curve = []
        val_curve = []

        for epoch in range(self.start_epoch, self.config.num_epochs):
            print("Epoch:{}, Started at:{}".format(epoch, datetime.now()))
            self.model.train()
            epoch_loss = 0.0
            for conditions, targets, label in self.train_dataloader:
                B, C, H, W = targets.shape

                conditions = conditions.to(self.device)
                targets = targets.to(self.device)

                t = torch.randint(0, self.model_config.timesteps, (targets.size(0),))

                noisy_data, noise = self.model.noise_step(targets, t)

                t_emb = self.model.sinusoidal_embedding(t, math.prod(self.model_config.swin_decoder.time_embedding))
                t_emb = t_emb.view(B, *self.model_config.swin_decoder.time_embedding)

                predicted_noise = self.model(conditions, noisy_data, t_emb)
                loss = F.mse_loss(predicted_noise, noise)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = epoch_loss / len(self.train_dataloader)
            train_curve.append(train_loss)
            val_curve.append(train_loss)

            #self.scheduler.step()

            with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
                file.write("{},{}\n".format(str(train_loss), str(train_loss)))


            if epoch % self.config.checkpoint_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'train_config': self.config,
                    'model_config': self.model_config,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, "{}/checkpoints/{}.pth".format(self.output_dir, epoch))


                loss_plot = utils.plot_losses(train_curve, val_curve)
                loss_plot.savefig("{}/logs/loss_curves.png".format(self.output_dir))
                loss_plot.close()


if __name__ == '__main__':
    train_config = config.get_config()
    trainer = DiffusionTrainer(train_config)
    if train_config.load_training:
        load_training(trainer, train_config.checkpoint_path)
    trainer.train()
