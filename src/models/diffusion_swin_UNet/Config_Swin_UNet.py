import copy
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()

    config.device = 'cuda:0'
    config.noise_scheduler = 'cosine'
    config.timesteps = 100
    config.start_beta = 1e-4
    config.end_beta = 0.02

    ####### Encoder CONV_BLOCK #######

    config.encoder_conv_block = config_dict.ConfigDict()
    config.encoder_conv_block.image_size = 128
    config.encoder_conv_block.num_channels = 6
    config.encoder_conv_block.embed_dim = 12
    config.encoder_conv_block.output_dim = 24



    ########  ENCODER  ########

    config.swin_encoder = config_dict.ConfigDict()
    config.swin_encoder.image_size = 128
    config.swin_encoder.num_channels = config.encoder_conv_block.output_dim
    config.swin_encoder.patch_size = 8
    config.swin_encoder.embed_dim = 64
    config.swin_encoder.depths = [2, 6, 2]
    config.swin_encoder.num_heads = [2, 2, 2]
    config.swin_encoder.window_size = 4
    config.swin_encoder.pretrained_window_sizes = [0, 0, 0]
    config.swin_encoder.mlp_ratio = 4.0
    config.swin_encoder.qkv_bias = True
    config.swin_encoder.hidden_dropout_prob = 0.0
    config.swin_encoder.attention_probs_dropout_prob = 0.0
    config.swin_encoder.drop_path_rate = 0.1
    config.swin_encoder.hidden_act = 'gelu'
    config.swin_encoder.use_absolute_embeddings = False
    config.swin_encoder.initializer_range = 0.02
    config.swin_encoder.layer_norm_eps = 1e-05
    config.swin_encoder.encoder_stride = 8
    config.swin_encoder.output_hidden_states = True
    config.swin_encoder.out_features = None
    config.swin_encoder.out_indices = None
    config.swin_encoder.skip_connection_shape = [[2**i * config.swin_encoder.embed_dim,
        int(config.swin_encoder.image_size / (config.swin_encoder.patch_size * 2**i)),
        int(config.swin_encoder.image_size/(config.swin_encoder.patch_size * 2**i))]
        for i in range(len(config.swin_encoder.depths))] #skip connections shape (C,W,H)
    config.swin_encoder.skip_connection_shape.insert(0, [config.encoder_conv_block.output_dim, config.encoder_conv_block.image_size,config.encoder_conv_block.image_size])


#########  DECODER  ##############

    config.swin_decoder = config_dict.ConfigDict()
    config.swin_decoder.image_size = 128
    config.swin_decoder.out_channels = 3
    config.swin_decoder.patch_size = 8
    config.swin_decoder.embed_dim = 64
    config.swin_decoder.depths = [2, 6, 2]
    config.swin_decoder.num_heads = [2, 2, 2]
    config.swin_decoder.window_size = 4
    config.swin_decoder.pretrained_window_sizes = [0, 0, 0]
    config.swin_decoder.channel_reduction_ratio = 2
    config.swin_decoder.mlp_ratio = 4.0
    config.swin_decoder.qkv_bias = True
    config.swin_decoder.hidden_dropout_prob = 0.0
    config.swin_decoder.attention_probs_dropout_prob = 0.0
    config.swin_decoder.drop_path_rate = 0.1
    config.swin_decoder.hidden_act = 'gelu'
    config.swin_decoder.use_absolute_embeddings = False
    config.swin_decoder.initializer_range = 0.02
    config.swin_decoder.layer_norm_eps = 1e-05
    config.swin_decoder.encoder_stride = 8
    config.swin_decoder.output_hidden_states = False
    config.swin_decoder.out_features = None
    config.swin_decoder.out_indices = None
    config.swin_decoder.num_channels_time_embedding = 8

    config.swin_decoder.skip_connection_shape_pre_cat = list(reversed(copy.deepcopy(config.swin_encoder.skip_connection_shape)))
    config.swin_decoder.skip_connection_shape_pre_cat[0][0] += config.swin_decoder.num_channels_time_embedding # time embedding concat

    config.swin_decoder.skip_connection_shape = list(reversed(copy.deepcopy(config.swin_encoder.skip_connection_shape))) #skip connections shape (C,H,W)
    config.swin_decoder.skip_connection_shape[0][0] += config.swin_decoder.num_channels_time_embedding # time embedding concat

    config.swin_decoder.stage_output_shape = list(reversed(copy.deepcopy(config.swin_encoder.skip_connection_shape))) #skip connections shape (C,H,W)
    for i, skip_connection_shape in enumerate(config.swin_decoder.skip_connection_shape):
        if i != 0:
            config.swin_decoder.stage_output_shape[i][0] = int(config.swin_decoder.skip_connection_shape[i - 1][0] / 4)
            config.swin_decoder.skip_connection_shape[i][0] += int(config.swin_decoder.skip_connection_shape[i - 1][0]/4)

    config.swin_decoder.time_embedding = (config.swin_decoder.num_channels_time_embedding, int(config.swin_decoder.skip_connection_shape[0][1]), int(config.swin_decoder.skip_connection_shape[0][2]))



##### Conv Block ##########

    config.conv_block = config_dict.ConfigDict()
    config.conv_block.res = 128
    config.conv_block.input_dim = int(config.swin_decoder.skip_connection_shape[-1][0])
    config.conv_block.hidden_dim_1 = 32
    config.conv_block.hidden_dim_2 = 24
    config.conv_block.out_dim = 3




    return config
