
from .generator import (
    RRDBNet,
    MSRResNet,
    RCAN,
    DCSRN,
    mDCSRN,
    MedSR
)
from .discriminator import (
    VGGStyleDiscriminator,
    UNetDiscriminatorSN,
    SimpleDiscriminator,
    RelativisticDisc
)

def define_network(model_type, params):
    if model_type == 'rrdbnet':
        return RRDBNet(**params)
    if model_type == 'msrresnet':
        return MSRResNet(**params)
    if model_type == 'rcan':
        return RCAN(**params)
    if model_type == 'dcsrn':
        return DCSRN(**params)
    if model_type == 'mdcsrn':
        return mDCSRN(**params)
    if model_type == 'medsr':
        return MedSR(**params)
    if model_type == 'vgg_disc':
        return VGGStyleDiscriminator(**params)
    if model_type == 'unet_disc':
        return UNetDiscriminatorSN(**params)
    if model_type == 'simple_disc':
        return SimpleDiscriminator(**params)
    if model_type == 'relativistic_disc':
        return RelativisticDisc(**params)