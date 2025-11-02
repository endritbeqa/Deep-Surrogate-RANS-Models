import os

from datetime import datetime

import torch

from src import utils
from src.trainers.unceartainty.base_trainer import Base_Trainer


class VAE_Trainer(Base_Trainer):
    def __init__(self, train_config):
        super().__init__(train_config)
        self.beta = self.model_config.KLD_beta

    def train_model(self):
        torch.cuda.empty_cache()
        self.save_configs()

        with open(
            "{}/logs/recon_vs_KLD_curves.txt".format(self.output_dir), "+a"
        ) as file:
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
            train_reconstruction_loss = train_reconstruction_loss / len(
                self.train_dataloader
            )
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

            with open(
                "{}/logs/recon_vs_KLD_curves.txt".format(self.output_dir), "+a"
            ) as file:
                file.write(
                    "{},{},{},{}\n".format(
                        str(train_reconstruction_loss),
                        str(train_KLD_loss),
                        str(val_reconstruction_loss),
                        str(val_KLD_loss),
                    )
                )

            if epoch % self.train_config.checkpoint_every == 0:
                self.save_checkpoint(epoch)
                target_output_dir = os.path.join(
                    self.output_dir, "samples", "target", epoch
                )
                prediction_output_dir = os.path.join(
                    self.output_dir, "samples", "prediction", epoch
                )
                utils.plot_samples(targets, target_output_dir)
                utils.plot_samples(predictions, prediction_output_dir)

            self.plot_loss_curve(train_curve, val_curve)

            curves = {
                "Train recon": train_reconstruction_curve,
                "Train KLD": train_KLD_curve,
                "Validation recon": val_reconstruction_curve,
                "Validation KLD": val_KLD_curve,
            }

            loss_plot = utils.plot_losses(
                curves, "Epoch", "{} loss".format(self.train_config.loss_function)
            )
            loss_plot.savefig("{}/logs/recon_vs_KLD.png".format(self.output_dir))
            loss_plot.close()

        self.save_checkpoint("Final")
        print("Finished training")
        return val_curve[-1]
