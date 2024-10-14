import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.models import model_select
from src.data import dataset
from src import diffusion_config, utils
from src import diffusion_utils


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class DiffusionTrainer(object):
    def __init__(self, train_config):
        self.config = train_config
        self.model_config, self.model = model_select.get_model(train_config.model_name)
        self.output_dir = train_config.output_dir
        self.train_dataset = dataset.Airfoil_Dataset(train_config, mode='train')
        self.val_dataset = dataset.Airfoil_Dataset(train_config, mode='validation')
        self.train_dataloader = DataLoader(self.train_dataset, train_config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, train_config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_config.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=30)
        self.device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.noise_scheduler = diffusion_utils.get_noise_scheduler(self.config)
        self.noise_scheduler.alpha_bar.to(self.device)
        self.num_timesteps = train_config.timesteps
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters())
        print("Model: {}, Num parameters: {}".format(self.config.model_name, self.num_model_parameters))
        os.makedirs(self.output_dir, exist_ok=True)
        for dir in [os.path.join(self.output_dir, "checkpoints"),
                    os.path.join(self.output_dir, "logs"),
                    os.path.join(self.output_dir, "configs")]:
            os.makedirs(dir, exist_ok=True)


    def noise_step(self, x_0, t):
        alpha_bar_t = self.noise_scheduler.get_alpha_bar(t).to(self.device)
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
        noise = torch.randn_like(x_0).to(self.device)
        noisy_data = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_data, noise

    def train(self):
        train_curve = []
        val_curve = []

        for epoch in range(self.config.num_epochs):
            print("Epoch:{}, Started at:{}".format(epoch, datetime.now()))
            self.model.train()
            epoch_loss = 0.0
            for conditions, targets, label in self.train_dataloader:

                conditions = conditions.to(self.device)
                targets = targets.to(self.device)

                # Sample random time steps for each batch
                t = torch.randint(0, self.config.timesteps, (targets.size(0),))

                noisy_data, noise = self.noise_step(targets, t)

                predicted_noise = self.model(conditions, noisy_data, t)

                loss = F.mse_loss(predicted_noise, noise)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = epoch_loss / len(self.train_dataloader)
            train_curve.append(train_loss)
            val_curve.append(train_loss)

            self.scheduler.step()

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
    train_config = diffusion_config.get_config()
    trainer = DiffusionTrainer(train_config)
    trainer.train()
