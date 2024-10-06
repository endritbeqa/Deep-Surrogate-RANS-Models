import torch
import torch.nn as nn
from src.models.swin_NVAE import Swin_VAE_decoder, Swin_VAE_encoder, prior_select, Z_cell


class U_NET_Swin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.encoder = Swin_VAE_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_VAE_decoder.Swin_VAE_decoder(config)
        self.prior_class, self.prior_config = prior_select.get_Z_Cell(config)

        z_cells = [self.prior_class(self.prior_config, i_layer) for i_layer in range(len(self.prior_config.latent_dim))]

        self.z_cells = torch.nn.ModuleList(z_cells)


    def forward(self, condition, target):
        B, _,_,_ = target.shape

        skip_connections, conditions = self.encoder(condition, target)

        skip_connections = list(reversed(skip_connections))
        conditions = list(reversed(conditions))

        hidden_state = None
        mu, logvar = [], []

        for i, skip_connection in enumerate(skip_connections):

            skip_connection_flattened = torch.flatten(skip_connection, start_dim=1, end_dim=-1)
            condition_flattened = torch.flatten(conditions[i], start_dim=1, end_dim=-1)
            if hidden_state is None:
                hidden_state_flattened = None
            else:
                hidden_state_flattened = torch.flatten(hidden_state, start_dim=1, end_dim=-1)


            z, mu_i, logvar_i = self.z_cells[i](skip_connection_flattened, hidden_state_flattened, condition_flattened)
            mu.insert(0, mu_i)
            logvar.insert(0, logvar_i)
            shape = self.config.swin_decoder.skip_connection_shape_pre_cat[i]
            z = z.view(B,*shape)
            if hidden_state is not None:
                z = torch.cat((z, hidden_state), dim=1)
            input_dimension = shape[1:3]
            hidden_state = self.decoder.layers[i](z, input_dimension)

        return hidden_state, mu, logvar



    def sample(self, condition):
        B, _, _, _ = condition.shape

        place_holder = torch.randn_like(condition)
        _, conditions = self.encoder(condition, place_holder)
        conditions = list(reversed(conditions))

        hidden_state = self.z_cells[0].H.repeat(B, 1)
        hidden_state = torch.reshape(hidden_state, (B,*self.config.swin_decoder.skip_connection_shape_pre_cat[0]))


        for i, condition in enumerate(conditions):

            condition_flattened = torch.flatten(condition, start_dim=1, end_dim=-1)
            hidden_state_flattened = torch.flatten(hidden_state, start_dim=1, end_dim=-1)

            noise = torch.unsqueeze(torch.randn(self.prior_config.latent_dim[i]), dim=0)

            condition_latent = self.z_cells[i].fc_condition(condition_flattened)
            hidden_state_latent = self.z_cells[i].fc_prev(hidden_state_flattened)

            z = torch.cat((noise, hidden_state_latent, condition_latent), dim=1)
            z = self.z_cells[i].fc_z(z)
            shape = self.config.swin_decoder.skip_connection_shape_pre_cat[i]
            z = z.view(B,*shape)
            if i!=0:
                z = torch.cat((z, hidden_state), dim=1)
            input_dimension = shape[1:3]
            hidden_state = self.decoder.layers[i](z, input_dimension)

        return hidden_state



