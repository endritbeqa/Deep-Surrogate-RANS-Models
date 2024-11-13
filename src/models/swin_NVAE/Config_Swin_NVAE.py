import copy
import math

from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()

    config.device = 'cuda:0'
    config.prior = 'gaussian'
    config.KLD_beta = 0.01

    config.image_size = 32

    config.encoder = config_dict.ConfigDict()
    config.encoder.image_size = config.image_size
    config.encoder.num_channels = 6
    config.encoder.embed_dim = 16
    config.encoder.patch_size = 2
    config.encoder.depths = [2, 2, 2]
    config.encoder.num_heads = [2, 4, 8]
    config.encoder.window_size = 4
    config.encoder.qkv_bias = True
    config.encoder.attention_probs_dropout_prob = 0.0
    config.encoder.layer_norm_eps = 1e-05
    config.encoder.drop_path_rate = 0.1
    config.encoder.mlp_ratio = 4.0
    config.encoder.hidden_act = 'gelu'
    config.encoder.hidden_dropout_prob = 0.0
    config.encoder.skip_connection = [(int(config.encoder.embed_dim*2**i),
                                       int(config.image_size/(config.encoder.patch_size*2**i)),
                                       int(config.image_size/(config.encoder.patch_size*2**i)))
                                      for i in range(len(config.encoder.depths)+1)] # C, H, W

    config.decoder = config_dict.ConfigDict()
    config.decoder.output_dims = 3
    config.decoder.image_size = config.image_size
    config.decoder.patch_size = config.encoder.patch_size
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
    config.decoder.skip_connection_shape = list(reversed(copy.deepcopy(config.encoder.skip_connection)))


    config.gaussian_prior = config_dict.ConfigDict()
    config.gaussian_prior.latent_dim = [64, 128, 256, 512]
    config.gaussian_prior.hidden_dim = [math.prod(skip) for skip in config.decoder.skip_connection_shape]

    config.gmm_prior = config_dict.ConfigDict()
    config.gmm_prior.device = config.device
    config.gmm_prior.latent_dim = [16, 32, 64, 128]
    config.gmm_prior.num_components = [4, 8, 16, 32]
    config.gmm_prior.hidden_dim = [math.prod(skip) for skip in config.decoder.skip_connection_shape]

    return config
