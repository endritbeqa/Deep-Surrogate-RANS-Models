from src.models.swin_NVAE import Config_Swin_NVAE, Swin_NVAE
from src.models.NVAE import Config_NVAE, NVAE
from src.models.ViT import Config_ViT_VAE, ViT_VAE

def get_model(name: str):

    if name == 'swin_NVAE':
        model_config = Config_Swin_NVAE.get_config()
        model = Swin_NVAE.U_NET_Swin(model_config)
    elif name == 'NVAE':
        model_config = Config_NVAE.get_config()
        model = NVAE.U_NET_Swin(model_config)
    elif name == 'ViT_VAE':
        model_config = Config_ViT_VAE.get_config()
        model = ViT_VAE.AutoregressiveImageTransformer(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    return model_config, model

def load_model(name: str, model_config, checkpoint):

    if name == 'swin_NVAE':
        model = Swin_NVAE.U_NET_Swin(model_config)
    elif name == 'NVAE':
        model = NVAE.U_NET_Swin(model_config)
    elif name == 'ViT_VAE':
        model = ViT_VAE.AutoregressiveImageTransformer(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    model.load_state_dict(checkpoint['model'])

    return model