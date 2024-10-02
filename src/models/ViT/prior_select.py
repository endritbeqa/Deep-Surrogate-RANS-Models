from src.models.ViT import Z_cell, Z_cell_GMM_Prior, Z_cell_VampPrior


def get_Z_Cell(prior: str, config):
    if prior =='gaussian':
        return Z_cell.VAEBottleneck(config.gaussian_prior)
    elif prior =='vamp':
        return Z_cell_VampPrior.VampPriorVAEBottleneck(config.vamp_prior)
    elif prior =='gaussian_GMM':
        return Z_cell_GMM_Prior.GMMVAEBottleneck(config.gmm_prior)
    else:
        raise Exception("Prior name not found.Check if prior is implemented.")
