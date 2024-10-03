from src.models.ViT import Z_cell, Z_cell_GMM_Prior, Z_cell_VampPrior
from src.models.swin_NVAE import Config_Swin_NVAE


def get_Z_Cell(config: str):
    config = Config_Swin_NVAE.get_config()

    if config.prior =='gaussian':
        return Z_cell.VAEBottleneck, config.gaussian_prior
    elif config.prior =='vamp':
        return Z_cell_VampPrior.VampPriorVAEBottleneck, config.vamp_prior
    elif config.prior =='gaussian_GMM':
        return Z_cell_GMM_Prior.GMMVAEBottleneck, config.gmm_prior
    else:
        raise Exception("Prior name not found.Check if prior is implemented.")
