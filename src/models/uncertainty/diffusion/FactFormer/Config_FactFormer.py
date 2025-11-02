from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.device = "cuda:0"
    config.noise_scheduler = "cosine"
    config.timesteps = 50
    config.start_beta = 1e-10
    config.end_beta = 0.01
    config.decoder = "CNN"

    config.patch_size = 4
    config.in_dim = 6
    config.out_dim = 3
    config.dim = 224
    config.heads = 4
    config.depth = 12
    config.dim_head = 64
    config.kernel_multiplier = 2
    config.pos_in_dim = 2
    config.pos_out_dim = 2
    config.positional_embedding = "rotary"
    config.resolution = 128

    config.MLP_decoder = ConfigDict()
    config.MLP_decoder.encoder_dim = (
        config.dim * (config.resolution // config.patch_size) ** 2
    )
    config.MLP_decoder.latent_dim_1 = config.MLP_decoder.encoder_dim // 2
    config.MLP_decoder.latent_dim_2 = config.MLP_decoder.latent_dim_1 // 2
    config.MLP_decoder.output_dim = (3, config.resolution, config.resolution)

    config.CNN_decoder = ConfigDict()
    config.CNN_decoder.upsample_res = (config.resolution, config.resolution)
    config.CNN_decoder.input_dim = [
        config.dim,
        config.dim,
        config.dim // 2,
        config.dim // 2,
        config.dim // 4,
        config.dim // 4,
    ]
    config.CNN_decoder.hidden_dim = [
        config.dim,
        config.dim,
        config.dim // 2,
        config.dim // 2,
        config.dim // 4,
        config.dim // 4,
    ]
    config.CNN_decoder.output_dim = [
        config.dim,
        config.dim // 2,
        config.dim // 2,
        config.dim // 4,
        config.dim // 4,
        3,
    ]
    config.CNN_decoder.kernel_size = [11, 11, 7, 7, 3, 3]

    return config
