import os

import torch
from torch.utils.data import DataLoader

import numpy as np

from src.data import dataset
from src.models.U_net_SwinV2 import U_NET_Swin
from src.models.Config_UNet_Swin import get_config






def sample_from_vae(model_config, checkpoint, condition):

    model = U_NET_Swin(model_config)
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    condition_encoder = model.encoder.condition_encoder
    decoder = model.decoder

    with torch.no_grad():
        z = torch.randn(1, model_config.latent_dim)
        condition = torch.unsqueeze(torch.from_numpy(condition), 0)

        
        z = z.to(device='cpu')
        condition = condition.to(device='cpu')


        condition = (condition_encoder(condition)).last_hidden_state
        shape = condition.shape
        condition = torch.flatten(condition, start_dim=1, end_dim=2)
        condition = model.fc_condition(condition)
        z = torch.cat((z, condition), dim =1)
        generated_samples = model.decoder(z, None, shape)
        print(generated_samples)

    return generated_samples


if __name__ == '__main__':
    config = get_config()
    checkpoint = "/home/blin/PycharmProjects/Thesis/src/Outputs/checkpoints/1.pth"
    condition_dir = "/home/blin/PycharmProjects/Thesis/src/data_res_32_uncertainty/data/test"
    conditions = []
    for data in os.listdir(condition_dir):
        data = np.load(os.path.join(condition_dir, data))
        data = data['a']
        conditions.append(data[0:3])


    sample_from_vae(config, checkpoint, conditions[0])