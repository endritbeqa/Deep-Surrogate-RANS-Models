import json
import math
import os
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn.utils as nn_utils
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.models import model_select
from src.data import dataset
from src import config, utils

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

#TODO refactor this with the new checkpoint format
def load_training(trainer,checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    train_config = checkpoint['train_config']
    trainer.start_epoch = checkpoint['epoch']+1
    trainer.model_config = checkpoint['model_config']
    trainer.model = model_select.load_model(trainer.config.model_name, trainer.model_config, checkpoint)
    trainer.model = trainer.model.to(trainer.device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=train_config.lr,weight_decay=train_config.weight_decay)
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])

class DiffusionTrainer(object):
    def __init__(self, train_config):
        self.config = train_config
        self.model_config, self.model = model_select.get_model(train_config)
        self.output_dir = train_config.output_dir
        self.train_dataset = dataset.Airfoil_Dataset(train_config, mode='train')
        self.val_dataset = dataset.Airfoil_Dataset(train_config, mode='validation')
        self.train_dataloader = DataLoader(self.train_dataset, train_config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, train_config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
        #self.scheduler = CosineAnnealingLR(self.optimizer, T_max=train_config.cosine_anneling_TMax)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.linear_schedule(initial_lr=train_config.lr, final_lr=train_config.final_lr, total_steps=train_config.num_epochs))
        self.device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters())
        self.gradient_clip_norm = train_config.gradient_clip_norm
        self.start_epoch = 0
        print("Model: {}, Num parameters: {}".format(self.config.model_name, self.num_model_parameters))
        os.makedirs(self.output_dir, exist_ok=True)
        for dir in [os.path.join(self.output_dir, "checkpoints"),
                    os.path.join(self.output_dir, "logs"),
                    os.path.join(self.output_dir, "configs")]:
            os.makedirs(dir, exist_ok=True)

    def linear_schedule(self, initial_lr, final_lr, total_steps):
        def lr_lambda(current_step):
            return 1 - (current_step / total_steps) * (1 - final_lr / initial_lr)

        return lr_lambda

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
            train_loss = 0.0
            for conditions, targets, label in self.train_dataloader:
                B, C, H, W = targets.shape

                conditions = conditions.to(self.device)
                targets = targets.to(self.device)

                t = torch.randint(0, self.model_config.timesteps, (targets.size(0),))
                #uniform_samples = torch.rand(B)
                #cosine_samples = (1 + torch.cos(uniform_samples * math.pi)) / 2
                #t = (cosine_samples * (self.model_config.timesteps - 1)).long()

                noisy_data, noise = self.model.noise_step(targets, t)
                t_emb = self.model.sinusoidal_embedding(t, 100)

                predicted_noise = self.model(conditions, noisy_data, t_emb)
                loss = F.mse_loss(predicted_noise, noise)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                if not (self.gradient_clip_norm is None):
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
            self.scheduler.step()

            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for conditions, targets, label in self.val_dataloader:
                    B, C, H, W = targets.shape

                    conditions = conditions.to(self.device)
                    targets = targets.to(self.device)

                    t = torch.randint(0, self.model_config.timesteps, (targets.size(0),))

                    noisy_data, noise = self.model.noise_step(targets, t)
                    #TODO remove this hard coded stuff
                    t_emb = self.model.sinusoidal_embedding(t, 100)
                    #t_emb = t_emb.view(B, *self.model_config.time_embedding_shape)

                    predicted_noise = self.model(conditions, noisy_data, t_emb)
                    loss = F.mse_loss(predicted_noise, noise)
                    val_loss += loss.item()



            train_loss = train_loss / len(self.train_dataloader)
            val_loss = val_loss/ len(self.val_dataloader)
            train_curve.append(train_loss)
            val_curve.append(val_loss)


            with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
                file.write("{},{}\n".format(str(train_loss), str(val_loss)))


            if epoch % self.config.checkpoint_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'train_config': self.config,
                    'model_config': self.model_config,
                    'model_params': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'model': self.model,
                }
                torch.save(checkpoint, "{}/checkpoints/{}.pth".format(self.output_dir, epoch))

                utils.save_images(predicted_noise, self.output_dir, "predictions", epoch)
                utils.save_images(noise, self.output_dir, "targets", epoch)

                loss_plot = utils.plot_losses(train_curve, val_curve)
                loss_plot.savefig("{}/logs/loss_curves.png".format(self.output_dir))
                loss_plot.close()


if __name__ == '__main__':
    train_config = config.get_config()
    trainer = DiffusionTrainer(train_config)
    if train_config.load_training:
        load_training(trainer, train_config.checkpoint_path)
    trainer.train()
