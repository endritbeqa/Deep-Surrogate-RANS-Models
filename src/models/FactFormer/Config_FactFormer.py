from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()


    config.device = 'cuda:0'
    config.decoder = "CNN"

    config.patch_size = 8
    config.in_dim = 3
    config.out_dim = 3
    config.dim = 64
    config.heads = 4
    config.depth = 6
    config.dim_head = 32
    config.kernel_multiplier = 2
    config.pos_in_dim = 2
    config.pos_out_dim = 2
    config.positional_embedding = 'rotary'
    config.resolution = 128


    config.MLP_decoder = ConfigDict()
    config.MLP_decoder.encoder_dim = (config.dim * (config.resolution//config.patch_size)**2)
    config.MLP_decoder.latent_dim_1 = config.MLP_decoder.encoder_dim // 2
    config.MLP_decoder.latent_dim_2 = config.MLP_decoder.latent_dim_1 // 2
    config.MLP_decoder.output_dim = (3, config.resolution, config.resolution)

    config.CNN_decoder = ConfigDict()
    config.CNN_decoder.upsample_res = (config.resolution, config.resolution)
    config.CNN_decoder.input_dim = [config.dim, config.dim, config.dim // 2, config.dim // 4]
    config.CNN_decoder.hidden_dim = [config.dim, config.dim, config.dim // 2, config.dim // 4]
    config.CNN_decoder.output_dim = [config.dim, config.dim // 2, config.dim // 4, 3]
    config.CNN_decoder.kernel_size = [11, 7, 3, 3]


    return config
