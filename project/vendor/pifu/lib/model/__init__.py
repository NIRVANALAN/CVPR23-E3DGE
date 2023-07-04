from .BasePIFuNet import BasePIFuNet
from .VhullPIFuNet import VhullPIFuNet
from .ConvPIFuNet import ConvPIFuNet
from .HGPIFuNet import HGPIFuNet
from .ResBlkPIFuNet import ResBlkPIFuNet
from .HGPIFuGANNet import HGPIFuNetGAN
from .HGPIFuGANNetResidualInput import HGPIFuNetGANResidual
# from .HGPIFuGANNetResidualInputResnetFC import HGPIFuNetGANInpaintingResidual
from .HGPIFuGANNetResidualInputResnetFC import HGPIFuNetGANResidualResnetFC 


def build_E1(opt_rendering, opt_pifu):
    # opt_rendering = None
    # opt_pifu = None
    if opt_rendering.netLocal_type == 'HGPIFuNetGAN':
        E1 = HGPIFuNetGAN(opt_pifu, opt_rendering, 'projection')
    elif opt_rendering.netLocal_type == 'HGPIFuNetGANResidual':
        E1 = HGPIFuNetGANResidual(opt_pifu, opt_rendering, 'projection')
    # elif opt_rendering.netLocal_type == 'HGPIFuNetGANInpaintingResidual':
    #     E1 = HGPIFuNetGANInpaintingResidual(opt_pifu, opt_rendering,
                                            # 'projection')
    else:
        raise NotImplementedError('netLocal_type')

    return E1
