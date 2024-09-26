import torch

from src.models.swin_VAE import U_net_SwinV2_VAE, Config_UNet_Swin_VAE
from src.models.swin_hierarchical_VAE import U_net_SwinV2_hierarchical_VAE, Config_UNet_Swin_hierarchical_VAE
from src.models.swin_VAE_condition import U_net_SwinV2_VAE_condition, Config_UNet_Swin_VAE_condition

def get_model(name: str):

    if name == 'swin_VAE':
        model_config = Config_UNet_Swin_VAE.get_config()
        model = U_net_SwinV2_VAE.U_NET_Swin(model_config)
    elif name == 'swin_NVAE':
        model_config = Config_UNet_Swin_hierarchical_VAE.get_config()
        model = U_net_SwinV2_hierarchical_VAE.U_NET_Swin(model_config)
    elif name == 'swin_VAE_condition':
        model_config = Config_UNet_Swin_VAE_condition.get_config()
        model = U_net_SwinV2_VAE_condition.U_NET_Swin(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    return model_config, model

def load_model(name: str, model_config, checkpoint):

    if name == 'swin_VAE':
        model = U_net_SwinV2_VAE.U_NET_Swin(model_config)
    elif name == 'swin_NVAE':
        model = U_net_SwinV2_hierarchical_VAE.U_NET_Swin(model_config)
    elif name == 'swin_VAE_condition':
        model = U_net_SwinV2_VAE_condition.U_NET_Swin(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")


    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    return model