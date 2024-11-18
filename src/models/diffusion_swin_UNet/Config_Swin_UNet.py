from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()

    config.device = 'cuda:0'
    config.noise_scheduler = 'cosine'
    config.timesteps = 200
    config.start_beta = 1e-6
    config.end_beta = 0.1

    config.image_size = 32
    config.input_res_skip = False

    config.encoder = config_dict.ConfigDict()
    config.encoder.input_res_skip = config.input_res_skip
    config.encoder.image_size = config.image_size
    config.encoder.conv_input_dim = 6
    config.encoder.conv_output_dim = 16
    config.encoder.num_channels = config.encoder.conv_output_dim if config.encoder.input_res_skip else config.encoder.conv_input_dim
    config.encoder.embed_dim = 32
    config.encoder.patch_size = 2
    config.encoder.depths = [2, 2, 2]
    config.encoder.num_heads = [4, 8, 16]
    config.encoder.window_size = 4
    config.encoder.qkv_bias = True
    config.encoder.attention_probs_dropout_prob = 0.0
    config.encoder.layer_norm_eps = 1e-05
    config.encoder.drop_path_rate = 0.1
    config.encoder.mlp_ratio = 4.0
    config.encoder.hidden_act = 'gelu'
    config.encoder.hidden_dropout_prob = 0.0

    config.decoder = config_dict.ConfigDict()
    config.decoder.input_res_skip = config.input_res_skip
    config.decoder.image_size = config.image_size
    config.decoder.patch_size = config.encoder.patch_size
    config.decoder.conv_skip_dim = config.encoder.conv_output_dim
    config.decoder.num_output_channels = 3
    config.decoder.embed_dim = config.encoder.embed_dim
    config.decoder.depths = list(reversed(config.encoder.depths))
    config.decoder.num_heads = list(reversed(config.encoder.num_heads))
    config.decoder.window_size = config.encoder.window_size
    config.decoder.qkv_bias = True
    config.decoder.attention_probs_dropout_prob = 0.0
    config.decoder.layer_norm_eps = 1e-05
    config.decoder.drop_path_rate = 0.1
    config.decoder.mlp_ratio = 4.0
    config.decoder.hidden_act = 'gelu'
    config.decoder.hidden_dropout_prob = 0.0

    config.middle_block = config_dict.ConfigDict()
    config.middle_block.dim = config.encoder.embed_dim * (2**(len(config.encoder.depths)))
    config.middle_block.depth = 2
    config.middle_block.num_heads = 16
    config.middle_block.input_res = (config.image_size // (config.encoder.patch_size * 2**(len(config.encoder.depths))),
                                     config.image_size // (config.encoder.patch_size * 2**(len(config.encoder.depths))))
    config.middle_block.window_size = config.encoder.window_size
    config.middle_block.qkv_bias = True
    config.middle_block.attention_probs_dropout_prob = 0.0
    config.middle_block.layer_norm_eps = 1e-05
    config.middle_block.drop_path_rate = 0.1
    config.middle_block.mlp_ratio = 4.0
    config.middle_block.hidden_act = 'gelu'
    config.middle_block.hidden_dropout_prob = 0.0




    return config
