from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()

    config.device = 'cuda:0'
    config.noise_scheduler = 'cosine'
    config.timesteps = 200
    config.start_beta = 1e-6
    config.end_beta = 1
    config.decoder = "CNN"
    config.image_size = 32

    config.encoder = config_dict.ConfigDict()
    config.encoder.image_size = config.image_size
    config.encoder.num_channels = 6
    config.encoder.embed_dim = 128
    config.encoder.patch_size = 2
    config.encoder.depths = [24]
    config.encoder.num_heads = [8]
    config.encoder.window_size = 4
    config.encoder.qkv_bias = True
    config.encoder.attention_probs_dropout_prob = 0.0
    config.encoder.layer_norm_eps = 1e-05
    config.encoder.drop_path_rate = 0.1
    config.encoder.mlp_ratio = 4.0
    config.encoder.hidden_act = 'gelu'
    config.encoder.hidden_dropout_prob = 0.0

    config.MLP_decoder = config_dict.ConfigDict()
    config.MLP_decoder.encoder_dim = (config.encoder.embed_dim * (config.image_size // config.encoder.patch_size) ** 2)
    config.MLP_decoder.latent_dim_1 = config.MLP_decoder.encoder_dim // 2
    config.MLP_decoder.latent_dim_2 = config.MLP_decoder.latent_dim_1 // 2
    config.MLP_decoder.output_dim = (3, config.image_size, config.image_size)

    config.CNN_decoder = config_dict.ConfigDict()
    config.CNN_decoder.input_size = (config.image_size // config.encoder.patch_size, config.image_size // config.encoder.patch_size)
    config.CNN_decoder.input_dim = config.encoder.embed_dim
    config.CNN_decoder.conv_1_output_dim = config.encoder.embed_dim // 2
    config.CNN_decoder.output_dim = (3, config.image_size, config.image_size)

    return config
