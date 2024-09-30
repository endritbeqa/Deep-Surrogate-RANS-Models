from ml_collections import ConfigDict


def get_config():

    config = ConfigDict()

##### Encoder ##########

    config.encoder = ConfigDict()
    config.encoder.img_size = 224
    config.encoder.patch_size = 16
    config.encoder.in_channels = 3
    config.encoder.embed_dim = 768
    config.encoder.num_layers = 12
    config.encoder.num_heads = 12
    config.encoder.mlp_dim = 3072
    config.encoder.dropout_rate = 0.1

##### Decoder ##########

    config.decoder = ConfigDict()
    config.decoder.embed_dim = 768
    config.decoder.num_layers = 12
    config.decoder.num_heads = 12
    config.decoder.mlp_dim = 3072
    config.decoder.dropout_rate = 0.1
    config.decoder.num_patches = 196
    config.decoder.patch_size = 16
    config.decoder.out_channels = 3

##### Bottleneck ######

    config.bottleneck = ConfigDict()
    config.bottleneck.latent_dim = 64
    config.bottleneck.hidden_dim = 256


    return config