from src.models.swin import Autoregressive_Swin_Unet, Config_Autoregressive_Swin_Unet

def get_model(name: str):

    if name == 'swin':
        model_config = Config_Autoregressive_Swin_Unet.get_config()
        model = Autoregressive_Swin_Unet.U_NET_Swin_Sequence_Modeler(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    return model_config, model