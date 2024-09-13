import torch
import torch.nn as nn
from src.models.swin import Swin_VAE_decoder, Swin_VAE_encoder


class U_NET_Swin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Swin_VAE_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_VAE_decoder.Swin_VAE_decoder(config)


    def forward(self, condition, target):
        z, mu, log_var = self.encoder(condition, target)
        prediction = self.decoder(z)

        return prediction , torch.cat(mu, dim =1), torch.cat(log_var, dim =1)


    def inference(self, condition, random_tensors):
        condition = self.encoder.condition_encoder(condition, output_hidden_states=True)
        condition_hidden_states = condition.hidden_states
        condition_hidden_states = list(condition_hidden_states)[:-2]
        condition_hidden_states.append(condition.last_hidden_state)

        condition_latent = []

        for i, condition in enumerate(condition_hidden_states):
            condition = torch.flatten(condition, start_dim=1, end_dim=2)
            condition_latent.append(self.encoder.fc_condition[i](condition))

        [random_tensor.to(device='cpu') for random_tensor in random_tensors]

        for i, random_tensor in enumerate(random_tensors):
            random_tensors[i] = torch.cat([random_tensor, condition_latent[i]], dim=1)

        prediction = self.decoder(random_tensors)

        return prediction


