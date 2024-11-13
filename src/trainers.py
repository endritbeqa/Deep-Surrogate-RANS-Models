import os
import json
from abc import abstractmethod

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
        self.config = train_config
        self.model_config, self.model = model_select.get_model(train_config)
        self.output_dir = train_config.output_dir
        self.train_dataset = dataset.Airfoil_Dataset(train_config, mode='train')
        self.val_dataset = dataset.Airfoil_Dataset(train_config, mode='validation')
        self.train_dataloader = DataLoader(self.train_dataset, train_config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, train_config.batch_size, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True)
        self.loss_func = utils.get_loss_function(self.config.loss_function)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
        self.scheduler = self.scheduler_select(train_config)
        self.device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters())
        self.gradient_clip_norm = train_config.gradient_clip_norm
        self.start_epoch = 0
        print("Model: {}, Num parameters: {}".format(self.config.model_name, self.num_model_parameters))
        for dir in [self.output_dir,
                    os.path.join(self.output_dir, "checkpoints"),
                    os.path.join(self.output_dir, "logs"),
                    os.path.join(self.output_dir, "configs"),
                    os.path.join(self.output_dir, "images"),
                    os.path.join(self.output_dir, "images/predictions"),
                    os.path.join(self.output_dir, "images/targets")]:
            os.makedirs(dir, exist_ok=True)

    def scheduler_select(self, train_config):
        if train_config.scheduler == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=train_config.cosine_anneling_TMax)
        elif train_config.scheduler == 'lambda':
            return LambdaLR(self.optimizer, lr_lambda=self.linear_schedule(initial_lr=train_config.lr,
                                                                                 final_lr=train_config.final_lr,
                                                                                 total_steps=train_config.num_epochs))

    #TODO figure this out
    def load_training(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        train_config = checkpoint['train_config']
        self.start_epoch = checkpoint['epoch'] + 1
        self.model_config = checkpoint['model_config']
        self.model = model_select.load_model(self.config.model_name, self.model_config, checkpoint)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_config.lr,
                                          weight_decay=train_config.weight_decay)
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def linear_schedule(self, initial_lr, final_lr, total_steps):
        def lr_lambda(current_step):
            return 1 - (current_step / total_steps) * (1 - final_lr / initial_lr)

        return lr_lambda

    def save_configs(self):
        with open("{}/configs/config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.config.to_dict(), json_file, sort_keys=False, indent=4)

        with open("{}/configs/model_config.json".format(self.output_dir), '+w') as json_file:
            json.dump(self.model_config.to_dict(), json_file, sort_keys=False, indent=4)

        with open("{}/configs/model_size.txt".format(self.output_dir), '+w') as file:
            file.write("Number of model parameters: {}".format(self.num_model_parameters))

        with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
            file.write("train_loss, val_loss\n")


    def save_epoch(self, epoch, predictions, targets, train_curve, val_curve):
        if epoch % self.config.checkpoint_every == 0:
            checkpoint = {
                'epoch': epoch,
                'train_config': self.config,
                'model_config': self.model_config,
                'model_params': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'model': self.model,
            }
            torch.save(checkpoint, "{}/checkpoints/{}.pth".format(self.output_dir, epoch))

            utils.save_images(predictions, self.output_dir, "predictions", epoch)
            utils.save_images(targets, self.output_dir, "targets", epoch)

            loss_plot = utils.plot_losses(train_curve, val_curve)
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

        for epoch in range(self.config.num_epochs):
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

                RE_loss  = F.l1_loss(predictions, targets)
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
                    RE_loss = F.l1_loss(predictions, targets)
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
                file.write("{},{},{},{}\n".format(str(train_reconstruction_loss), str(train_KLD_loss),str(val_reconstruction_loss), str(val_KLD_loss)))

            self.save_epoch(epoch, predictions, targets, train_curve, val_curve)

            if epoch % self.config.checkpoint_every == 0:
                loss_plot = utils.plot_recon_vs_KLD(train_reconstruction_curve, train_KLD_curve,val_reconstruction_curve, val_KLD_curve)
                loss_plot.savefig("{}/logs/recon_vs_KLD.png".format(self.output_dir))
                loss_plot.close()

        return val_curve[-1]


class DiffusionTrainer(Base_Trainer):
    def __init__(self, train_config):
        super().__init__(train_config)

    def train_model(self):
        torch.cuda.empty_cache()
        self.save_configs()

        train_curve = []
        val_curve = []

        for epoch in range(self.start_epoch, self.config.num_epochs):
            print("Epoch:{}, Started at:{}".format(epoch, datetime.now()))
            train_loss = 0.0
            val_loss = 0

            self.model.train()
            for conditions, targets, label in self.train_dataloader:
                conditions, targets = conditions.to(self.device), targets.to(self.device)

                t = torch.randint(0, self.model_config.timesteps, (targets.shape[0],))
                #uniform_samples = torch.rand(targets.shape[0])
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
            with torch.no_grad():
                for conditions, targets, label in self.val_dataloader:
                    conditions, targets = conditions.to(self.device), targets.to(self.device)

                    t = torch.randint(0, self.model_config.timesteps, (targets.size(0),))
                    noisy_data, noise = self.model.noise_step(targets, t)
                    t_emb = self.model.sinusoidal_embedding(t, 100)
                    predicted_noise = self.model(conditions, noisy_data, t_emb)
                    loss = F.mse_loss(predicted_noise, noise)
                    val_loss += loss.item()



            train_loss = train_loss / len(self.train_dataloader)
            val_loss = val_loss / len(self.val_dataloader)
            train_curve.append(train_loss)
            val_curve.append(val_loss)


            with open("{}/logs/curves.txt".format(self.output_dir), "+a") as file:
                file.write("{},{}\n".format(str(train_loss), str(val_loss)))

            self.save_epoch(epoch, predicted_noise, noise, train_curve, val_curve)


