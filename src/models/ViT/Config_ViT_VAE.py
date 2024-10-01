from ml_collections import ConfigDict


def get_config():

    config = ConfigDict()

##### Encoder ##########

    config.encoder = ConfigDict()
    config.encoder.img_size = 128
    config.encoder.patch_size = 16
    config.encoder.in_channels = 3
    config.encoder.embed_dim = 256
    config.encoder.num_layers = 4
    config.encoder.num_heads = 4
    config.encoder.mlp_dim = 1024
    config.encoder.dropout_rate = 0.1

##### Decoder ##########

    config.decoder = ConfigDict()
    config.decoder.embed_dim = 256
    config.decoder.num_layers = 4
    config.decoder.num_heads = 4
    config.decoder.mlp_dim = 1024
    config.decoder.dropout_rate = 0.1
    config.decoder.num_patches = 64
    config.decoder.patch_size = 16
    config.decoder.out_channels = 3

##### Gaussian Prior Bottleneck ######

    config.gaussian_prior = ConfigDict()
    config.gaussian_prior.latent_dim = 64
    config.gaussian_prior.hidden_dim = 256

##### Vamp Prior Bottleneck ##########

    config.vamp_prior = ConfigDict()
    config.vamp_prior.latent_dim = 64  # Dimensionality of the latent space
    config.vamp_prior.hidden_dim = 256  # Dimensionality of the input features (before the bottleneck)
    config.vamp_prior.num_pseudo_inputs = 10

##### GMM Prior Bottleneck ##########

    config.GMM_prior = ConfigDict()
    config.GMM_prior.latent_dim = 64  # Dimensionality of the latent space
    config.GMM_prior.hidden_dim = 256  # Dimensionality of the input features (before the bottleneck)
    config.GMM_prior.num_components = 10  # Number of Gaussian components in the mixture

    return config