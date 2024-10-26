from ml_collections import ConfigDict



def get_unet_config():

    config = ConfigDict()
    config.in_channels = 7
    config.out_channels = 3
    config.embed_dim = 256
    config.num_heads = 8
    config.mlp_dim = 512
    config.dropout = 0.1
    config.num_layers = 4
    config.patch_size = 16
    config.time_embed_dim = 128
    return config
