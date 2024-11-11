from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    config.device = 'cuda:0'
    config.noise_scheduler = 'cosine'
    config.timesteps = 200
    config.start_beta = 1e-4
    config.end_beta = 0.02

    config.image_size = 32
    config.input_dim = 6
    config.conv_output_dim = 16
    config.output_dim = 3
    config.patch_size = 2
    config.init_dim = 32
    config.depths = [2, 2, 4]  # Number of ViT blocks per stage
    config.num_heads = [2, 4, 8]  # Number of attention heads per stage
    config.mlp_ratio = 4.0


    return config
