import torch
import torch.nn as nn
from src.models import Swin_VAE_encoder, Swin_VAE_decoder




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
        condition = self.encoder.condition_encoder(condition, output_hidden_states=False)
        condition = torch.flatten(condition.last_hidden_state, start_dim=1, end_dim=2)
        condition_latent = self.encoder.fc_condition(condition)

        [random_tensor.to(device='cpu') for random_tensor in random_tensors]

        for i, random_tensor in enumerate(random_tensors):
            random_tensors[i] = torch.cat([random_tensor, condition_latent], dim=1)

        prediction = self.decoder(random_tensors)

        return prediction