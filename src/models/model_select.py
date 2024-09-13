from src.models.swin import U_net_SwinV2, Config_UNet_Swin

def get_model(name: str):

    if name == 'swin':
        model_config = Config_UNet_Swin.get_config()
        model = U_net_SwinV2.U_NET_Swin(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    return model_config, model