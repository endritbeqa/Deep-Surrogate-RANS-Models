from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.device = "cuda:0"
    config.noise_scheduler = "cosine"
    config.timesteps = 200
    config.start_beta = 1e-6
    config.end_beta = 1

    config.image_size = 32
    config.input_dim = 6
    config.output_dim = 3
    config.patch_size = 2
    config.init_dim = 32
    config.depths = [2, 2, 2]  # Number of ViT blocks per stage
    config.num_heads = [4, 8, 16]  # Number of attention heads per stage
    config.mlp_ratio = 4.0

    return config
