import torch
import torch.nn as nn
import math

from src.models.swin_UNet import Swin_decoder, Swin_encoder


class U_NET_Swin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.encoder = Swin_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_decoder.Swin_VAE_decoder(config)

        self.time_embed = nn.Linear(100, int(math.prod([*config.swin_encoder.skip_connection_shape[-1]])))

    def sinusoidal_embedding(self, timesteps, dim):
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -embeddings)
        embeddings = embeddings.to("cuda:0")
        timesteps = timesteps.to("cuda:0")
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

    def forward(self, condition, target, t):
        B, C, H, W = target.shape
        t_emb = self.time_embed(self.sinusoidal_embedding(t, 100))
        t_emb = t_emb.view(B, *self.config.swin_encoder.skip_connection_shape[-1])

        input = torch.cat([condition, target], dim=1)
        skip_connections = self.encoder(input)

        skip_connections = list(reversed(skip_connections))
        skip_connections[0] = torch.cat([skip_connections[0], t_emb], dim=1)

        prediction = self.decoder(skip_connections)

        return prediction



    def sample(self, condition):
        x_t = torch.randn_like(condition)

        # Compute alpha and  from betas
        alphas = 1 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        for t in reversed(range(timesteps)):
            # The model predicts noise (eps_t) given the current noisy image and conditioning info
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            predicted_noise = model(x_t, t_tensor, cond_info)

            # Compute the current estimate of the denoised image
            alpha_t = alpha_hat[t]
            alpha_t_1 = alpha_hat[t - 1] if t > 0 else alpha_hat[0]

            mean_x_t = (1 / torch.sqrt(alphas[t])) * (x_t - (betas[t] / torch.sqrt(1 - alpha_hat[t])) * predicted_noise)

            if t > 0:
                # Compute the noise term
                z = torch.randn_like(x_t) if t > 0 else 0  # sample noise z only for t > 0
                x_t = mean_x_t + torch.sqrt(betas[t]) * z
            else:
                # Final step: use only the mean (deterministic step)
                x_t = mean_x_t

        return x_t






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

        hidden_state = self.decoder.last_layer(hidden_state)

        return hidden_state



