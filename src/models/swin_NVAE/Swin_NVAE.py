import torch
import torch.nn as nn
from src.models.swin_NVAE import Swin_VAE_decoder, Swin_VAE_encoder, prior_select


class U_NET_Swin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.encoder = Swin_VAE_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_VAE_decoder.Swin_VAE_decoder(config)
        self.prior_class, self.prior_config = prior_select.get_Z_Cell(config)

        z_cells = [self.prior_class(self.prior_config, i_layer) for i_layer in len(self.prior_config.latent_dim)]
        self.z_cells = torch.nn.ModuleList(z_cells)


    def forward(self, condition, target):

        hidden_states, conditions = self.encoder(condition, target)

        current_hidden_state = None
        mu, logvar = [], []

        for i, hidden_state in enumerate(hidden_states):
            z, mu_i, logvar_i = self.z_cells[i](hidden_state, current_hidden_state, conditions[i])
            mu.insert(0, mu_i)
            logvar.insert(0, logvar_i)
            shape = self.config.swin_decoder.skip_connection_shape[i]
            z = torch.reshape(z, shape)
            if current_hidden_state is not None:
                z = torch.cat((z, current_hidden_state), dim=1)
            current_hidden_state = self.decoder.layers[i](z, tuple(shape[1:-1]))

        return current_hidden_state, mu, logvar





