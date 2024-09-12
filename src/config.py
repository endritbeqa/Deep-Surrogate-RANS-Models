from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.model = "swin"
    config.data_dir = ''
    config.output_dir = 'Outputs'
    config.context_size = 32
    config.num_epochs = 100
    config.batch_size = 600
    config.normalize = True
    config.optimizer = 'adam'
    config.lr = 1e-4
    config.weight_decay = 0.01
    config.loss_function = ['mae']
    config.checkpoint_every = 10

    config.dataset = config_dict.ConfigDict()
    config.dataset.normalize = True
    config.dataset.simulation_step = 1
    config.dataset.data_type = 'inc' # the modes are 'inc' (Incompressible Wake Flows) https://en.wikipedia.org/wiki/Wake_(physics)
                             # 'tra' (transonic Cylinder Flow) https://en.wikipedia.org/wiki/Transonic
                             # 'iso' (isotropic Turbulence) https://en.wikipedia.org/wiki/Homogeneous_isotropic_turbulence

    return config