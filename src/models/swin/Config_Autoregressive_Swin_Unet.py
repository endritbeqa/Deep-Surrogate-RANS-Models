import copy
from functools import reduce
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()

########  ENCODER  ########

    config.swin_encoder = config_dict.ConfigDict()
    config.swin_encoder.image_size = 32
    config.swin_encoder.num_channels = 3
    config.swin_encoder.patch_size = 2
    config.swin_encoder.embed_dim = 32
    config.swin_encoder.depths = [2, 4, 8]
    config.swin_encoder.num_heads = [2, 4, 4]
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
    config.swin_encoder.skip_connection_shape = [[
        int(config.swin_encoder.image_size / (config.swin_encoder.patch_size * 2 ** i)),
        int(config.swin_encoder.image_size / (config.swin_encoder.patch_size * 2 ** i)),
        2 ** i * config.swin_encoder.embed_dim]
        for i in range(len(config.swin_encoder.depths))]  # skip connections shape (H,W,C)

    #########  DECODER  ##############

    config.swin_decoder = config_dict.ConfigDict()
    config.swin_decoder.image_size = 32
    config.swin_decoder.num_channels = 3
    config.swin_decoder.patch_size = 2
    config.swin_decoder.embed_dim = 32
    config.swin_decoder.depths = [2, 4, 8]
    config.swin_decoder.num_heads = [2, 4, 4]
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
    config.swin_decoder.skip_connection_shape = list(reversed(copy.deepcopy(config.swin_encoder.skip_connection_shape)))  # skip connections shape (H,W,C)ss

    ########  SEQUENCE DECODER  #################

    config.sequence_modeler = config_dict.ConfigDict()
    config.sequence_modeler.context_size = 8
    config.sequence_modeler.depths = [2, 2, 2]
    config.sequence_modeler.skip_connection_dim = [reduce(lambda x, y: x*y, skip_connection) for skip_connection in config.swin_encoder.skip_connection_shape]
    config.sequence_modeler.embed_dim = [512, 256, 128]
    config.sequence_modeler.num_heads = [2, 2, 2]
    config.sequence_modeler.dropout_rate = 0.1
    config.sequence_modeler.causal_mask = True
    config.sequence_modeler.use_bias = True


    return config