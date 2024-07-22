from ml_collections import config_dict



def get_config():

    config = config_dict.ConfigDict()
    config.patch_size = 16
    config.embed_dim = 768
    config.in_channels = 3
    config.num_heads = 12
    config.num_layers = 12
    config.emb_dropout = 0.1
    config.num_classes = 1000
    config.image_size = 224
    config.num_patches = (config.image_size // config.patch_size) ** 2

    config.swin_encoder = config_dict.ConfigDict()
    config.swin_encoder.image_size = 32
    config.swin_encoder.patch_size = 2
    config.swin_encoder.num_channels = 3
    config.swin_encoder.embed_dim = 48
    config.swin_encoder.depths = [2, 6, 2]
    config.swin_encoder.num_heads = [3, 8, 12]
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
    config.swin_encoder.encoder_stride = 32
    config.swin_encoder.out_features = None
    config.swin_encoder.out_indices = None

    config.CNN_decoder = config_dict.ConfigDict()
    config.CNN_decoder.num_layers = 3
    config.CNN_decoder.embedding_dim = config.swin_encoder.embed_dim * 2 ** (len(config.swin_encoder.depths) - 1)
    config.CNN_decoder.output_size = 32
    config.CNN_decoder.output_channels = 3
    config.CNN_decoder.num_channels = [128, 64, 32]
    config.CNN_decoder.activation_fns = ['relu', 'relu', 'relu']
    config.CNN_decoder.kernel_sizes = [2, 2, 2]
    config.CNN_decoder.strides = [2, 2, 2]






    return config