import Config_UNet_Swin, Config_UNet_Swin_CNN
import U_net_SwinV2, U_net_SwinV2_CNN


def get_model(name: str):
    if name == 'swin_cnn':
        config = Config_UNet_Swin_CNN.get_config()
        return U_net_SwinV2_CNN.U_NET_Swin_CNN(config)
    elif name == "swin":
        config = Config_UNet_Swin.get_config()
        return U_net_SwinV2.U_NET_Swin(config)

    raise ValueError("Model selected is not available")


