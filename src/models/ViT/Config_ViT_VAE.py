from ml_collections import ConfigDict


def get_config():

    config = ConfigDict()
    config.prior = 'gaussian'

##### Encoder ##########

    config.encoder = ConfigDict()
    config.encoder.img_size = 32
    config.encoder.patch_size = 4
    config.encoder.in_channels = 6
    config.encoder.embed_dim = 128
    config.encoder.num_layers = 4
    config.encoder.num_heads = 4

##### Decoder ##########

    config.decoder = ConfigDict()
    config.decoder.img_size = 32
    config.decoder.patch_size = 4
    config.decoder.out_channels = 3
    config.decoder.embed_dim = 128 ## double the encoder because of the condition also
    config.decoder.num_layers = 4
    config.decoder.num_heads = 4

##### Gaussian Prior Bottleneck ######

    config.gaussian_prior = ConfigDict()
    config.gaussian_prior.latent_dim = 512
    config.gaussian_prior.hidden_dim = config.encoder.embed_dim * (config.encoder.img_size/config.encoder.patch_size)**2


##### Vamp Prior Bottleneck ##########

    config.vamp_prior = ConfigDict()
    config.vamp_prior.latent_dim = 64  # Dimensionality of the latent space
    config.vamp_prior.hidden_dim = 256  # Dimensionality of the input features (before the bottleneck)
    config.vamp_prior.num_pseudo_inputs = 10

##### GMM Prior Bottleneck ##########

    config.gmm_prior = ConfigDict()
    config.gmm_prior.latent_dim = 64  # Dimensionality of the latent space
    config.gmm_prior.hidden_dim = 256  # Dimensionality of the input features (before the bottleneck)
    config.gmm_prior.num_components = 10  # Number of Gaussian components in the mixture

    return config