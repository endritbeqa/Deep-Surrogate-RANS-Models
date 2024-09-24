import torch

from src.models.swin_VAE import U_net_SwinV2_VAE, Config_UNet_Swin_VAE
from src.models.swin_autoencoder import U_net_SwinV2_Autoencoder, Config_UNet_Swin_Autoencoder
from src.models.swin_hierarchical_VAE import U_net_SwinV2_hierarchical_VAE, Config_UNet_Swin_hierarchical_VAE

def get_model(name: str):

    if name == 'swin_VAE':
        model_config = Config_UNet_Swin_VAE.get_config()
        model = U_net_SwinV2_VAE.U_NET_Swin(model_config)
    elif name == 'swin_autoencoder':
        model_config = Config_UNet_Swin_Autoencoder.get_config()
        model = U_net_SwinV2_Autoencoder.U_NET_Swin_Autoencoder(model_config)
    elif name == 'swin_hierarchical_VAE':
        model_config = Config_UNet_Swin_hierarchical_VAE.get_config()
        model = U_net_SwinV2_hierarchical_VAE.U_NET_Swin(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    return model_config, model

def load_model(name: str, model_config, checkpoint):

    if name == 'swin_VAE':
        model = U_net_SwinV2_VAE.U_NET_Swin(model_config)
    elif name == 'swin_autoencoder':
        model = U_net_SwinV2_Autoencoder.U_NET_Swin_Autoencoder(model_config)
    elif name == 'swin_hierarchical_VAE':
        model = U_net_SwinV2_hierarchical_VAE.U_NET_Swin(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")


    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    return model