from datetime import datetime

import torch
import torch.nn.utils as nn_utils

from src.trainers.unceartainty.base_trainer import Base_Trainer


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
            val_loss = 0.0

            self.model.train()
            for conditions, targets, label in self.train_dataloader:
                conditions, targets = conditions.to(self.device), targets.to(
                    self.device
                )
                t = torch.randint(0, self.model_config.timesteps, (targets.shape[0],))
                noisy_data, noise = self.model.noise_step(targets, t)
                t_emb = self.model.sinusoidal_embedding(t, 100)
                predicted_noise = self.model(conditions, noisy_data, t_emb)
                loss = self.loss_func(predicted_noise, noise)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()

                if not (self.gradient_clip_norm is None):
                    nn_utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )
                self.optimizer.step()
            self.scheduler.step()

            self.model.eval()
            with torch.no_grad():
                for conditions, targets, label in self.val_dataloader:
                    conditions, targets = conditions.to(self.device), targets.to(
                        self.device
                    )

                    t = torch.randint(
                        0, self.model_config.timesteps, (targets.size(0),)
                    )
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
        print("Finished training")
        return val_curve[-1]
