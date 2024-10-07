from src.models.NVAE import Z_cell  #, Z_cell_GMM_Prior, Z_cell_VampPrior


def get_Z_Cell(config):

    if config.prior =='gaussian':
        return Z_cell.VAEBottleneck, config.gaussian_prior
    #elif config.prior =='vamp':
    #    return Z_cell_VampPrior.VampPriorVAEBottleneck, config.vamp_prior
    #elif config.prior =='gaussian_GMM':
    #    return Z_cell_GMM_Prior.GMMVAEBottleneck, config.gmm_prior
    else:
        raise Exception("Prior name not found.Check if prior is implemented.")
