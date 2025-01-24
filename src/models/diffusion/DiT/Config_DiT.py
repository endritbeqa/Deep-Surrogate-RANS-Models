from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    config.device = ''
    config.noise_scheduler = 'cosine'
    config.timesteps = 200
    config.start_beta = 1e-6
    config.end_beta = 1
    config.decoder = "CNN"

    config.image_size = 64
    config.input_dim = 6
    config.output_dim = 3
    config.patch_size = 4
    config.init_dim = 160
    config.depths = [12]  # Number of ViT blocks per stage
    config.num_heads = [4]  # Number of attention heads per stage
    config.mlp_ratio = 4.0


    config.MLP_decoder = ConfigDict()
    config.MLP_decoder.encoder_dim = (config.init_dim * (config.image_size//config.patch_size)**2)
    config.MLP_decoder.latent_dim_1 = config.MLP_decoder.encoder_dim // 2
    config.MLP_decoder.latent_dim_2 = config.MLP_decoder.latent_dim_1 // 2
    config.MLP_decoder.output_dim = (3, config.image_size, config.image_size)

    config.CNN_decoder = ConfigDict()
    config.CNN_decoder.upsample_res = (config.image_size, config.image_size)
    config.CNN_decoder.input_dim = [config.init_dim, config.init_dim,config.init_dim//2, config.init_dim//4]
    config.CNN_decoder.hidden_dim = [config.init_dim, config.init_dim,config.init_dim//2, config.init_dim//4]
    config.CNN_decoder.output_dim = [config.init_dim, config.init_dim//2, config.init_dim//4, 3]
    config.CNN_decoder.kernel_size = [7, 7, 3, 3]



    return config
