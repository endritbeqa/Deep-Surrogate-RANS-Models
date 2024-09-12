from src.models.swin import U_Net_SwinV2_Sequence_Modeler, Config_UNet_Swin

def get_model(name: str):

    if name == 'swin':
        model_config = Config_UNet_Swin.get_config()
        model = U_Net_SwinV2_Sequence_Modeler.U_NET_Swin_Sequence_Modeler(model_config)
    else:
        raise Exception("Model name not found.Check if model is implemented.")

    return model_config, model