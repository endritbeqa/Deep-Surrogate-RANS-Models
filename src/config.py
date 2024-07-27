from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.data_dir = './data_res_32/data'
    config.output_dir = 'Outputs'
    config.num_epochs = 100
    config.batch_size = 600
    config.normalize = True
    config.optimizer = 'adam'
    config.lr = 1e-3
    config.loss_function = ['mae']
    config.checkpoint_every = 10


    config.swin_encoder = config_dict.ConfigDict()
    config.swin_encoder.image_size = 32
    config.swin_encoder.patch_size = 2
    config.swin_encoder.num_channels = 3
    config.swin_encoder.embed_dim = 16
    config.swin_encoder.depths = [2, 2, 2]
    config.swin_encoder.num_heads = [2, 2, 4]
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
    config.swin_encoder.output_hidden_states = True
    config.swin_encoder.out_features = None
    config.swin_encoder.out_indices = None


    config.CNN_decoder = config_dict.ConfigDict()
    config.CNN_decoder.num_layers = 3
    config.CNN_decoder.embedding_dim = config.swin_encoder.embed_dim * 2**(len(config.swin_encoder.depths)-1)
    config.CNN_decoder.output_size = 32
    config.CNN_decoder.output_channels = 3
    config.CNN_decoder.num_channels = [32, 16, 8]
    config.CNN_decoder.activation_fns = ['relu', 'relu', 'relu']
    config.CNN_decoder.kernel_sizes = [4, 4, 2]
    config.CNN_decoder.strides = [1, 2, 2]

    config.Interpolation_decoder = config_dict.ConfigDict()
    config.Interpolation_decoder.num_layers = 3
    config.Interpolation_decoder.embedding_dim = config.swin_encoder.embed_dim * 2 ** (len(config.swin_encoder.depths) - 1)
    config.Interpolation_decoder.interpolation_dims = [(8,8),(16,16),(33,33)]
    config.Interpolation_decoder.output_size = 32
    config.Interpolation_decoder.output_channels = 3
    config.Interpolation_decoder.num_channels = [128, 64, 32]
    config.Interpolation_decoder.activation_fns = ['relu', 'relu', 'relu']
    config.Interpolation_decoder.kernel_sizes = [4, 4, 2]
    config.Interpolation_decoder.strides = [1, 1, 1]


    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = True
    config.data_preprocessing.removePOffset = True



    return config