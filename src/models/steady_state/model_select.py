from src.models.steady_state.Swin_UNet import Swin_UNet, Config_Swin_UNet
from src.models.steady_state.ViT_UNet import Config_ViT_UNet, ViT_UNet
from src.models.steady_state.Swin import Config_Swin, Swin
from src.models.steady_state.DiT import Config_DiT, DiT
from src.models.steady_state.FactFormer import Config_FactFormer, FactFormer


def get_model(config):

    if config.model_name == 'Swin_UNet':
        model_config = Config_Swin_UNet.get_config()
        model_config.device = config.device
        model_config.image_size = config.resolution
        model = Swin_UNet.Swin_UNet(model_config)
    elif config.model_name == 'DiT':
        model_config = Config_DiT.get_config()
        model_config.device = config.device
        model_config.image_size = config.resolution
        model = DiT.DiT(model_config)
    elif config.model_name == 'Swin':
        model_config = Config_Swin.get_config()
        model_config.device = config.device
        model_config.image_size = config.resolution
        model = Swin.Swin(model_config)
    elif config.model_name == 'FactFormer':
        model_config = Config_FactFormer.get_config()
        model_config.device = config.device
        model_config.resolution = config.resolution
        model = FactFormer.FactFormer(model_config)
    elif config.model_name == 'ViT_UNet':
        model_config = Config_ViT_UNet.get_config()
        model_config.device = config.device
        model_config.image_size = config.resolution
        model = ViT_UNet.DiffusionUNet(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    return model_config, model

def load_model(name: str, model_config, checkpoint):

    if name == 'Swin_UNet':
        model = Swin_UNet.Swin_UNet(model_config)
    elif name == 'Swin':
        model = Swin.Swin(model_config)
    elif name == 'DiT':
        model = DiT.DiT(model_config)
    elif name == 'FactFormer':
        model = FactFormer.FactFormer(model_config)
    elif name == 'ViT_UNet':
        model = ViT_UNet.DiffusionUNet(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    model.load_state_dict(checkpoint['model'])

    return model