import torch
import torch.nn as nn
from src.models.swin_VAE import Swin_VAE_decoder, Swin_VAE_encoder


class U_NET_Swin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.encoder = Swin_VAE_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_VAE_decoder.Swin_VAE_decoder(config)

    #TODO ADD a skip connection at input resolution you dumbass
    def forward(self, condition, target):

        z, mu, log_var = self.encoder(condition, target)
        prediction = self.decoder(z)

        return prediction , torch.cat(mu, dim =1), torch.cat(log_var, dim =1)


    def inference(self, condition):
        condition_conv_block_output = self.encoder.conv_block_condition(condition)
        condition = self.encoder.condition_encoder(condition_conv_block_output, output_hidden_states=True)
        condition_hidden_states = condition.hidden_states
        condition_hidden_states = list(condition_hidden_states)[:-2]
        first_condition = torch.permute(torch.flatten(condition_conv_block_output, start_dim=2, end_dim=3),dims=(0, 2, 1))
        condition_hidden_states.insert(0, first_condition)
        condition_hidden_states.append(condition.last_hidden_state)
        random_tensors = [torch.unsqueeze(torch.rand(self.config.latent_dim[i]), dim=0) for i in
                          range(len(self.config.latent_dim))]

        condition_latent = []

        for i, condition in enumerate(condition_hidden_states):
            condition = torch.flatten(condition, start_dim=1, end_dim=2)
            condition_latent.append(self.encoder.fc_condition[i](condition))

        [random_tensor.to(device='cpu') for random_tensor in random_tensors]

        for i, random_tensor in enumerate(random_tensors):
            random_tensors[i] = torch.cat([random_tensor, condition_latent[i]], dim=1)

        prediction = self.decoder(random_tensors)

        return prediction


