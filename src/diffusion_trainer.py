import os
import json
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.models import model_select
from src.data import dataset
from src import config

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
        self.num_timesteps = train_config.diffusion.num_timesteps
        self.beta_start = train_config.diffusion.beta_start
        self.beta_end = train_config.diffusion.beta_end
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters())
        print("Model: {}, Num parameters: {}".format(self.config.model_name, self.num_model_parameters))
        os.makedirs(self.output_dir, exist_ok=True)
        for dir in [os.path.join(self.output_dir, "checkpoints"),
                    os.path.join(self.output_dir, "logs"),
                    os.path.join(self.output_dir, "configs")]:
            os.makedirs(dir, exist_ok=True)

    def train_model(self):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        torch.cuda.empty_cache()
        with open("{}/configs/config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.config.to_dict(), json_file, indent=4)

        with open("{}/configs/model_config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.model_config.to_dict(), json_file, indent=4)

        with open("{}/configs/model_size.txt".format(self.output_dir), '+w') as file:
            file.write("Number of model parameters: {}".format(self.num_model_parameters))

        with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
            file.write("train_loss, val_loss\n")

        train_curve = []
        val_curve = []

        for epoch in range(self.config.num_epochs):
            print("Epoch:{}, Started at:{}".format(epoch, datetime.now()))
            self.model.train()
            train_loss = 0.0

            for conditions, targets, labels in self.train_dataloader:
                conditions = conditions.to(self.device)
                targets = targets.to(self.device)
                t = torch.randint(0, self.num_timesteps, (targets.size(0),)).to(self.device)
                noise = torch.randn_like(targets).to(self.device)
                noisy_targets = self.add_noise(targets, t, noise)

                self.optimizer.zero_grad()

                predicted_noise = self.model(conditions, noisy_targets, t)
                loss = F.mse_loss(predicted_noise, noise)
                train_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            train_loss = train_loss / len(self.train_dataloader)
            train_curve.append(train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for conditions, targets, labels in self.val_dataloader:
                    conditions = conditions.to(self.device)
                    targets = targets.to(self.device)
                    t = torch.randint(0, self.num_timesteps, (targets.size(0),)).to(self.device)
                    noise = torch.randn_like(targets).to(self.device)

                    noisy_targets = self.add_noise(targets, t, noise)
                    predicted_noise = self.model(conditions, noisy_targets, t)

                    val_loss += F.mse_loss(predicted_noise, noise).item()

            val_loss = val_loss / len(self.val_dataloader)
            val_curve.append(val_loss)

            with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
                file.write("{},{}\n".format(str(train_loss), str(val_loss)))

            if epoch % self.config.checkpoint_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'train_config': self.config,
                    'model_config': self.model_config,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, "{}/checkpoints/{}.pth".format(self.output_dir, epoch))

        return val_curve[-1]

    def add_noise(self, x0, t, noise):
        """Implements the forward diffusion process."""
        alpha_t = self.compute_alpha_t(t)
        return torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise


    def compute_alpha_t(self, t):
        beta_t = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(self.device)
        alpha_t = torch.cumprod(1 - beta_t, dim=0)
        # Ensure proper indexing when t is a batch of indices
        return alpha_t[t.long()].view(-1, 1, 1, 1)


if __name__ == '__main__':
    train_config = config.get_config()
    trainer = DiffusionTrainer(train_config)
    trainer.train_model()
