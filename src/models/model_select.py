from src.models.diffusion.Swin_UNet import Swin_UNet, Config_Swin_UNet
from src.models.diffusion.ViT_UNet import Config_ViT_UNet, ViT_UNet
from src.models.swin_NVAE import Config_Swin_NVAE, Swin_NVAE
from src.models.diffusion.DiT import DiT, Config_DiT
from src.models.diffusion.FactFormer import Config_FactFormer, FactFormer
from src.models.diffusion.Diffusion_wrapper import Diffuser


def get_model(config):

    if config.model_name == 'swin_NVAE':
        model_config = Config_Swin_NVAE.get_config()
        model_config.device = config.device
        model = Swin_NVAE.U_NET_Swin(model_config)
    elif config.model_name == 'Swin_UNet':
        model_config = Config_Swin_UNet.get_config()
        model_config.device = config.device
        model = Swin_UNet.Swin_UNet(model_config)
        model = Diffuser(model_config, model)
    elif config.model_name == 'DiT':
        model_config = Config_DiT.get_config()
        model_config.device = config.device
        model = DiT.DiT(model_config)
        model = Diffuser(model_config, model)
    elif config.model_name == 'FactFormer':
        model_config = Config_FactFormer.get_config()
        model_config.device = config.device
        model = FactFormer.FactFormer(model_config)
        model = Diffuser(model_config, model)
    elif config.model_name == 'ViT_UNet':
        model_config = Config_ViT_UNet.get_config()
        model_config.device = config.device
        model = ViT_UNet.DiffusionUNet(model_config)
        model = Diffuser(model_config, model)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    return model_config, model

def load_model(name: str, model_config, checkpoint):

    if name == 'swin_NVAE':
        model = Swin_NVAE.U_NET_Swin(model_config)
    elif name == 'Swin_UNet':
        model = Swin_UNet.Swin_UNet(model_config)
        model = Diffuser(model_config, model)
    elif name == 'DiT':
        model = DiT.DiT(model_config)
        model = Diffuser(model_config, model)
    elif name == 'FactFormer':
        model = FactFormer.FactFormer(model_config)
        model = Diffuser(model_config, model)
    elif name == 'ViT_UNet':
        model = ViT_UNet.DiffusionUNet(model_config)
        model = Diffuser(model_config, model)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    model.load_state_dict(checkpoint['model'])

    return model