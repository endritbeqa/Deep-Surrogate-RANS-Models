from src.models.swin_NVAE import Config_Swin_NVAE, Swin_NVAE
from src.models.NVAE import Config_NVAE, NVAE
from src.models.diffusion_swin_UNet_V2_0 import Config_Swin_UNet, Swin_UNet
from src.models.diffusion_ViT_UNet import Config_ViT_UNet, ViT_UNet
from src.models.swin_NVAE_V2 import Config_Swin_NVAE_V2, Swin_NVAE_V2



def get_model(config):

    if config.model_name == 'swin_NVAE':
        model_config = Config_Swin_NVAE.get_config()
        model_config.device = config.device
        model = Swin_NVAE.U_NET_Swin(model_config)
    elif config.model_name == 'swin_NVAE_V2':
        model_config = Config_Swin_NVAE_V2.get_config()
        model_config.device = config.device
        model = Swin_NVAE_V2.U_NET_Swin(model_config)
    elif config.model_name == 'NVAE':
        model_config = Config_NVAE.get_config()
        model_config.device = config.device
        model = NVAE.U_NET_Swin(model_config)
    elif config.model_name == 'diffusion_swin_UNet_V2':
        model_config = Config_Swin_UNet.get_config()
        model_config.device = config.device
        model = Swin_UNet.Swin_UNet(model_config)
    elif config.model_name == 'diffusion_ViT_UNet':
        model_config = Config_ViT_UNet.get_config()
        model_config.device = config.device
        model = ViT_UNet.DiffusionUNet(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    return model_config, model

def load_model(name: str, model_config, checkpoint):

    if name == 'swin_NVAE':
        model = Swin_NVAE.U_NET_Swin(model_config)
    elif name == 'swin_NVAE_V2':
        model = Swin_NVAE_V2.U_NET_Swin(model_config)
    elif name == 'NVAE':
        model = NVAE.U_NET_Swin(model_config)
    elif name == 'diffusion_swin_UNet_V2':
        model = Swin_UNet.Swin_UNet(model_config)
    elif name == 'diffusion_ViT_UNet':
        model = ViT_UNet.DiffusionUNet(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    model.load_state_dict(checkpoint['model'])

    return model