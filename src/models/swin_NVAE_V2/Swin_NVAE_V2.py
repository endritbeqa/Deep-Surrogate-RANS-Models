import copy

import torch
import torch.nn as nn
from src.models.swin_NVAE_V2 import Encoder, prior_select
from src.models.swin_NVAE_V2.layers import Swin_Decoder


class U_NET_Swin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = config.device
        self.encoder = Encoder.Encoder(config.encoder)
        condition_encoder_config = copy.deepcopy(config.encoder)
        condition_encoder_config.num_channels = 3
        self.condition_encoder = Encoder.Encoder(condition_encoder_config)
        self.decoder = Swin_Decoder(config.decoder)
        self.prior_class, self.prior_config = prior_select.get_Z_Cell(config)
        z_cells = [self.prior_class(self.prior_config, i_layer) for i_layer in range(len(self.prior_config.latent_dim))]
        self.z_cells = torch.nn.ModuleList(z_cells)
        C, H, W = config.decoder.skip_connection_shape[0]
        self.hidden_states = nn.Parameter(torch.randn(H*W, C))

    def move_to_device(self):
        self.to(self.device)

    def forward(self, condition, target):
        B, _,_,_ = target.shape
        KLDs = []

        input = torch.cat([condition, target], dim=1)
        skip_connections = self.encoder(input)
        conditions = self.condition_encoder(condition)
        skip_connections = list(reversed(skip_connections))
        conditions = list(reversed(conditions))

        hidden_state = self.hidden_states.repeat(B, 1, 1)

        for i, skip_connection in enumerate(skip_connections):

            skip_connection_flattened = skip_connection.flatten(1)
            condition_flattened = conditions[i].flatten(1)
            hidden_state_flattened = hidden_state.flatten(1)

            z, KLD_i = self.z_cells[i](skip_connection_flattened, hidden_state_flattened, condition_flattened)
            KLDs.append(KLD_i)
            C, H, W = self.config.decoder.skip_connection_shape[i]
            z = z.view(B, H*W, C)
            z = torch.cat((z, hidden_state), dim=2)
            hidden_state = self.decoder.layers[i](z, (H, W))[0]



        return hidden_state, sum(KLDs)

    def sample(self, condition, num_samples):
        condition = condition.unsqueeze(0).repeat(num_samples, 1, 1, 1)

        conditions = self.condition_encoder(condition)
        conditions = list(reversed(conditions))

        hidden_state = self.hidden_states.repeat(num_samples, 1, 1)

        for i, condition in enumerate(conditions):

            condition_flattened = torch.flatten(condition, start_dim=1, end_dim=-1)
            hidden_state_flattened = torch.flatten(hidden_state, start_dim=1, end_dim=-1)

            noise = self.z_cells[i].sample(num_samples=num_samples)
            noise = noise.to(self.device)

            condition_latent = self.z_cells[i].fc_condition(condition_flattened)
            hidden_state_latent = self.z_cells[i].fc_prev(hidden_state_flattened)

            z = torch.cat((noise, hidden_state_latent, condition_latent), dim=1)
            z = self.z_cells[i].fc_z(z)
            C, H, W = self.config.decoder.skip_connection_shape[i]
            z = z.view(num_samples, H * W, C)
            z = torch.cat((z, hidden_state), dim=2)
            hidden_state = self.decoder.layers[i](z, (H, W))[0]


        return hidden_state



