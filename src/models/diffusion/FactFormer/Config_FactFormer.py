from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()


    config.device = 'cuda:0'
    config.noise_scheduler = 'cosine'
    config.timesteps = 200
    config.start_beta = 1e-10
    config.end_beta = 0.01
    config.decoder = "CNN"

    config.patch_size = 4
    config.in_dim = 6
    config.out_dim = 3
    config.dim = 128
    config.heads = 12
    config.depth = 8
    config.dim_head = 64
    config.kernel_multiplier = 2
    config.pos_in_dim = 2
    config.pos_out_dim = 2
    config.positional_embedding = 'rotary'
    config.resolution = 64


    config.MLP_decoder = ConfigDict()
    config.MLP_decoder.encoder_dim = (config.dim * config.resolution**2)
    config.MLP_decoder.latent_dim_1 = config.MLP_decoder.encoder_dim // 2
    config.MLP_decoder.latent_dim_2 = config.MLP_decoder.latent_dim_1 // 2
    config.MLP_decoder.output_dim = (3, config.resolution, config.resolution)

    config.CNN_decoder = ConfigDict()
    config.CNN_decoder.input_size = (config.resolution, config.resolution)
    config.CNN_decoder.input_dim = config.dim
    #config.CNN_decoder.conv_1_output_dim = config.dim
    config.CNN_decoder.output_dim = (3, config.resolution, config.resolution)

    return config
