import importlib

from .vgg_disc import VGGStyleDiscriminator
from .unet_disc import UNetDiscriminatorSN
from .simple_disc import SimpleDiscriminator
from .relativistic_disc import RelativisticDisc

__all__ = ['VGGStyleDiscriminator', 'UNetDiscriminatorSN', 'SimpleDiscriminator', 'RelativisticDisc']