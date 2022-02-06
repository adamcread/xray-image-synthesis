# ========================================================
# Compositional GAN
# Models to be loaded
# By Samaneh Azadi
# ========================================================


def create_model(opt):
    model = None
    if opt.model == 'objCompose':
        # use own models
        if opt.dataset_mode == 'comp_decomp_aligned':
            from .objCompose_supervised_model import objComposeSuperviseModel
            model = objComposeSuperviseModel() # paired data
        elif opt.dataset_mode == 'comp_decomp_unaligned':
            from .objCompose_unsupervised_model import objComposeUnsuperviseModel
            model = objComposeUnsuperviseModel() # unpaired data
        else:
            raise ValueError("Model [%s] not recognized." % opt.model)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
        
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
