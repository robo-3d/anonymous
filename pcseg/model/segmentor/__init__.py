
# voxel
from .voxel.minkunet.minkunet import MinkUNet




__all__ = {
    # raw point
    # ...



    # voxel

    'MinkUNet': MinkUNet,

}


def build_segmentor(model_cfgs, num_class):
    model = eval(model_cfgs.NAME)(
        model_cfgs=model_cfgs,
        num_class=num_class,
    )

    return model
