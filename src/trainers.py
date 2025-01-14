import os
import json
from abc import abstractmethod
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from datetime import datetime

from src.models import model_select
from src.data import dataset
from src import utils


class Base_Trainer(object):
    def __init__(self, train_config):
        if train_config.load_training:
            self.load_training(train_config.checkpoint_path)
        else:
            self.train_config = train_config
            self.seed_everything(train_config.seed)
            self.model_config, self.model = model_select.get_model(train_config)
            self.output_dir = train_config.output_dir
            self.train_dataset = dataset.Airfoil_Dataset(train_config, mode='train')
            self.val_dataset = dataset.Airfoil_Dataset(train_config, mode='validation')
            self.train_dataloader = DataLoader(self.train_dataset, train_config.batch_size, shuffle=True, num_workers=2,
                                               prefetch_factor=2, pin_memory=True)
            self.val_dataloader = DataLoader(self.val_dataset, train_config.batch_size, shuffle=True, num_workers=2,
                                             prefetch_factor=2, pin_memory=True)
            self.loss_func = self.loss_select(self.train_config.loss_function)
            self.optimizer = self.optimizer_select(self.train_config)
            self.scheduler = self.scheduler_select(self.train_config)
            self.device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.num_model_parameters = sum(p.numel() for p in self.model.parameters())
            self.gradient_clip_norm = train_config.gradient_clip_norm
            self.start_epoch = 0

        print("Model: {}, Num parameters: {}".format(self.train_config.model_name, self.num_model_parameters))
        for dir in [self.output_dir,
                    os.path.join(self.output_dir, "checkpoints"),
                    os.path.join(self.output_dir, "logs"),
                    os.path.join(self.output_dir, "configs"),
                    os.path.join(self.output_dir, "samples")]:
            os.makedirs(dir, exist_ok=True)

    def seed_everything(self, seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def optimizer_select(self, train_config):
        if train_config.optimizer == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
        elif train_config.optimizer == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    def scheduler_select(self, train_config):
        if train_config.scheduler == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=train_config.cosine_anneling_TMax)
        elif train_config.scheduler == 'lambda':
            return LambdaLR(self.optimizer, lr_lambda=self.linear_schedule(initial_lr=train_config.lr, final_lr=train_config.final_lr, total_steps=train_config.num_epochs))

    def linear_schedule(self, initial_lr, final_lr, total_steps):
        def lr_lambda(current_step):
            return 1 - (current_step / total_steps) * (1 - final_lr / initial_lr)
        return lr_lambda


    def loss_select(self, loss: str):
        def mean_relative_loss_function(input, target):
            epsilon = 1e-10
            relative_difference = torch.abs(input - target) / torch.max(torch.abs(target), torch.tensor(epsilon, dtype=target.dtype, device=target.device))
            return relative_difference.mean()

        if loss == 'mse':
            return F.mse_loss
        elif loss == 'l1':
            return F.l1_loss
        elif loss == 'huber_loss':
            return F.smooth_l1_loss
        elif loss == 'mrl':
            return mean_relative_loss_function
        else:
            raise ValueError(f"Unknown loss function: {loss}, available are mse, l1, mrl, huber.")

    def load_training(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        train_config = checkpoint['train_config']
        self.train_config = train_config
        self.start_epoch = checkpoint['epoch'] + 1
        self.device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
        self.loss_func = self.loss_select(self.train_config.loss_function)
        self.model_config = checkpoint['model_config']
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['model_params'])
        self.model = self.model.to(self.device)
        self.optimizer = self.optimizer_select(train_config)
        self.scheduler = self.scheduler_select(train_config)
        self.optimizer.load_state_dict(checkpoint['optimizer_params'])
        self.scheduler.load_state_dict(checkpoint['scheduler_params'])
        self.output_dir = train_config.output_dir
        self.gradient_clip_norm = train_config.gradient_clip_norm
        self.train_dataset = dataset.Airfoil_Dataset(train_config, mode='train')
        self.val_dataset = dataset.Airfoil_Dataset(train_config, mode='validation')
        self.train_dataloader = DataLoader(self.train_dataset, train_config.batch_size, shuffle=True, num_workers=2,
                                           prefetch_factor=2, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, train_config.batch_size, shuffle=True, num_workers=2,
                                         prefetch_factor=2, pin_memory=True)
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters())

    def save_configs(self):
        with open("{}/configs/train_config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.train_config.to_dict(), json_file, sort_keys=False, indent=4)

        with open("{}/configs/model_config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.model_config.to_dict(), json_file, sort_keys=False, indent=4)

        with open("{}/configs/model_size.txt".format(self.output_dir), '+w') as file:
            file.write("Number of model parameters: {}".format(self.num_model_parameters))

        with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
            file.write("train_loss, val_loss\n")

    def save_checkpoint(self, epoch):

        checkpoints = [(checkpoint, os.path.getctime(os.path.join(self.output_dir, "checkpoints", checkpoint)))
                       for checkpoint in os.listdir(os.path.join(self.output_dir, "checkpoints"))]

        if len(checkpoints) > 20:
            checkpoints.sort(key=lambda x: x[1])
            last_checkpoint = os.path.join(self.output_dir, "checkpoints", checkpoints[0][0])
            os.remove(last_checkpoint)

        checkpoint = {
            'epoch': epoch,
            'train_config': self.train_config,
            'model_config': self.model_config,
            'model_params': self.model.state_dict(),
            'optimizer_params': self.optimizer.state_dict(),
            'scheduler_params': self.scheduler.state_dict(),
            'model': self.model,
        }
        torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoints', f"{epoch}.pth"))

    def plot_loss_curve(self, train_curve, val_curve):
        curves = {"Train loss": train_curve, "Validation Loss": val_curve}
        loss_plot = utils.plot_losses(curves, "Epoch", "{} loss".format(self.train_config.loss_function))
        loss_plot.savefig("{}/logs/loss_curves.png".format(self.output_dir))
        loss_plot.close()

    @abstractmethod
    def train_model(self):
        pass


class VAE_Trainer(Base_Trainer):
    def __init__(self, train_config):
        super().__init__(train_config)
        self.beta = self.model_config.KLD_beta

    def train_model(self):
        torch.cuda.empty_cache()
        self.save_configs()

        with open("{}/logs/recon_vs_KLD_curves.txt".format(self.output_dir), "+a") as file:
            file.write("train_reconstruction, train_KLD, val_reconstruction, val_KLD\n")

        train_curve = []
        val_curve = []
        train_reconstruction_curve = []
        train_KLD_curve = []
        val_reconstruction_curve = []
        val_KLD_curve = []

        for epoch in range(self.train_config.num_epochs):
            print("Epoch:{}, Started at:{}".format(epoch, datetime.now()))
            self.model.train()
            train_loss = 0.0
            train_reconstruction_loss = 0.0
            train_KLD_loss = 0.0
            val_loss = 0.0
            val_reconstruction_loss = 0.0
            val_KLD_loss = 0.0

            for conditions, targets, label in self.train_dataloader:
                self.optimizer.zero_grad()

                conditions = conditions.to(self.device)
                targets = targets.to(self.device)
                predictions, KLD_loss = self.model(conditions, targets)

                RE_loss = self.loss_func(predictions, targets)
                KLD_loss *= self.beta
                loss = RE_loss + KLD_loss
                loss.backward()

                train_loss += loss.item()
                train_reconstruction_loss += RE_loss.item()
                train_KLD_loss += KLD_loss.item()
                self.optimizer.step()
            self.scheduler.step()

            train_loss = train_loss / len(self.train_dataloader)
            train_reconstruction_loss = train_reconstruction_loss / len(self.train_dataloader)
            train_KLD_loss = train_KLD_loss / len(self.train_dataloader)
            train_curve.append(train_loss)
            train_reconstruction_curve.append(train_reconstruction_loss)
            train_KLD_curve.append(train_KLD_loss)

            self.model.eval()
            with torch.no_grad():
                for conditions, targets, label in self.val_dataloader:
                    conditions = conditions.to(self.device)
                    targets = targets.to(self.device)
                    predictions, KLD_loss = self.model(conditions, targets)
                    RE_loss = self.loss_func(predictions, targets)
                    KLD_loss *= self.beta
                    loss = RE_loss + KLD_loss

                    val_loss += loss.item()
                    val_reconstruction_loss += RE_loss.item()
                    val_KLD_loss += KLD_loss.item()

            val_loss = val_loss / len(self.val_dataloader)
            val_reconstruction_loss = val_reconstruction_loss / len(self.val_dataloader)
            val_KLD_loss = val_KLD_loss / len(self.val_dataloader)

            val_curve.append(val_loss)
            val_reconstruction_curve.append(val_reconstruction_loss)
            val_KLD_curve.append(val_KLD_loss)

            with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
                file.write("{},{}\n".format(str(train_loss), str(val_loss)))

            with open("{}/logs/recon_vs_KLD_curves.txt".format(self.output_dir), "+a") as file:
                file.write("{},{},{},{}\n".format(str(train_reconstruction_loss), str(train_KLD_loss),
                                                  str(val_reconstruction_loss), str(val_KLD_loss)))

            if epoch % self.train_config.checkpoint_every == 0:
                self.save_checkpoint(epoch)
                target_output_dir = os.path.join(self.output_dir,"samples", "target", epoch)
                prediction_output_dir = os.path.join(self.output_dir,"samples", "prediction", epoch)
                utils.plot_samples(targets, target_output_dir)
                utils.plot_samples(predictions, prediction_output_dir)

            self.plot_loss_curve(train_curve, val_curve)

            curves = {"Train recon": train_reconstruction_curve,
                      "Train KLD": train_KLD_curve,
                      "Validation recon": val_reconstruction_curve,
                      "Validation KLD": val_KLD_curve
                      }

            loss_plot = utils.plot_losses(curves, "Epoch", "{} loss".format(self.train_config.loss_function))
            loss_plot.savefig("{}/logs/recon_vs_KLD.png".format(self.output_dir))
            loss_plot.close()

        self.save_checkpoint("Final")
        return val_curve[-1]


class DiffusionTrainer(Base_Trainer):
    def __init__(self, train_config):
        super().__init__(train_config)

    def train_model(self):
        torch.cuda.empty_cache()
        self.save_configs()

        train_curve = []
        val_curve = []

        for epoch in range(self.start_epoch, self.train_config.num_epochs):
            print("Epoch:{}, Started at:{}".format(epoch, datetime.now()))
            train_loss = 0.0
            val_loss = 0

            self.model.train()
            for conditions, targets, label in self.train_dataloader:
                conditions, targets = conditions.to(self.device), targets.to(self.device)

                t = torch.randint(0, self.model_config.timesteps, (targets.shape[0],))
                noisy_data, noise = self.model.noise_step(targets, t)
                t_emb = self.model.sinusoidal_embedding(t, 100)
                predicted_noise = self.model(conditions, noisy_data, t_emb)
                loss = self.loss_func(predicted_noise, noise)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()

                if not (self.gradient_clip_norm is None):
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
            self.scheduler.step()

            self.model.eval()
            with torch.no_grad():
                for conditions, targets, label in self.val_dataloader:
                    conditions, targets = conditions.to(self.device), targets.to(self.device)

                    t = torch.randint(0, self.model_config.timesteps, (targets.size(0),))
                    noisy_data, noise = self.model.noise_step(targets, t)
                    t_emb = self.model.sinusoidal_embedding(t, 100)
                    predicted_noise = self.model(conditions, noisy_data, t_emb)
                    loss = self.loss_func(predicted_noise, noise)
                    val_loss += loss.item()

            train_loss = train_loss / len(self.train_dataloader)
            val_loss = val_loss / len(self.val_dataloader)
            train_curve.append(train_loss)
            val_curve.append(val_loss)

            with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
                file.write("{},{}\n".format(str(train_loss), str(val_loss)))

            if epoch % self.train_config.checkpoint_every == 0:
                self.save_checkpoint(epoch)
            self.plot_loss_curve(train_curve, val_curve)

        self.save_checkpoint("Final")
        return val_curve[-1]
