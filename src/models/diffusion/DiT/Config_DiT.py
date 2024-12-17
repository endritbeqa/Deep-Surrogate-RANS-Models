from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()


    config.device = ''
    config.noise_scheduler = 'cosine'
    config.timesteps = 200
    config.start_beta = 1e-6
    config.end_beta = 1
    config.decoder = "CNN"

    config.image_size = 128
    config.input_dim = 6
    config.output_dim = 3
    config.patch_size = 8
    config.init_dim = 512
    config.depths = [4]  # Number of ViT blocks per stage
    config.num_heads = [32]  # Number of attention heads per stage
    config.mlp_ratio = 4.0


    config.MLP_decoder = ConfigDict()
    config.MLP_decoder.encoder_dim = (config.init_dim * (config.image_size//config.patch_size)**2)
    config.MLP_decoder.latent_dim_1 = config.MLP_decoder.encoder_dim // 2
    config.MLP_decoder.latent_dim_2 = config.MLP_decoder.latent_dim_1 // 2
    config.MLP_decoder.output_dim = (3, config.image_size, config.image_size)

    config.CNN_decoder = ConfigDict()
    config.CNN_decoder.input_size = (config.image_size // config.patch_size, config.image_size // config.patch_size)
    config.CNN_decoder.input_dim = config.init_dim
    config.CNN_decoder.conv_1_output_dim = config.init_dim // 2
    config.CNN_decoder.output_dim = (3, config.image_size, config.image_size)

    return config
