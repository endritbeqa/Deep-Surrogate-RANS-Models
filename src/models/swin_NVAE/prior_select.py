from src.models.swin_NVAE import Z_cell, Z_cell_GMM #, Z_cell_VampPrior


def get_Z_Cell(config):

    if config.prior =='gaussian':
        return Z_cell.VAEBottleneck, config.gaussian_prior
    #elif train_config.prior =='vamp':
    #    return Z_cell_VampPrior.VampPriorVAEBottleneck, train_config.vamp_prior
    elif config.prior =='GMM':
        return Z_cell_GMM.GMM_VAEBottleneck, config.gmm_prior
    else:
        raise Exception("Prior name not found.Check if prior is implemented.")
