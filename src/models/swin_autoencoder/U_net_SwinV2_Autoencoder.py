import torch
import torch.nn as nn
from src.models.swin_autoencoder import Swin_decoder, Swin_encoder


class U_NET_Swin_Autoencoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.encoder = Swin_encoder.Swin_encoder(config)
        self.decoder = Swin_decoder.Swin_decoder(config)


    def forward(self, condition, target):
        z = self.encoder(target)
        prediction = self.decoder(z)

        return prediction, 0, 0


    def inference(self, condition):
        condition = self.encoder.condition_encoder(condition, output_hidden_states=True)
        condition_hidden_states = condition.hidden_states
        condition_hidden_states = list(condition_hidden_states)[:-2]
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


