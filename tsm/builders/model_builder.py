import torch
from ..models import MobileNetV2, resnet50, TSN


def build_model(model_type, **kwargs):
    if model_type == 'mobilenetv2_online':
        # num_classes=1000, input_size=224, width_mult=1., shift_div=8
        return MobileNetV2(**kwargs).eval()
    elif model_type == 'resnet50_online':
        # num_classes
        return resnet50(**kwargs).eval()
    elif model_type in ['mobilenetv2', 'resnet50']:
        model = TSN(base_model=model_type, **kwargs)
        model = torch.nn.DataParallel(model).cuda()
        return model.eval()
    raise ValueError("unknown model type {}".format(model_type))


def build_buffer_shapes(model_type):
    if model_type == 'mobilenetv2_online':
        return [
            [1, 3, 56, 56],
            [1, 4, 28, 28],
            [1, 4, 28, 28],
            [1, 8, 14, 14],
            [1, 8, 14, 14],
            [1, 8, 14, 14],
            [1, 12, 14, 14],
            [1, 12, 14, 14],
            [1, 20, 7, 7],
            [1, 20, 7, 7]
        ]
    elif model_type == 'resnet50_online':
        return [
            [1, 8, 56, 56],
            [1, 32, 56, 56],
            [1, 32, 56, 56],

            [1, 32, 56, 56],
            [1, 64, 28, 28],
            [1, 64, 28, 28],
            [1, 64, 28, 28],

            [1, 64, 28, 28],
            [1, 128, 14, 14],
            [1, 128, 14, 14],
            [1, 128, 14, 14],
            [1, 128, 14, 14],
            [1, 128, 14, 14],

            [1, 128, 14, 14],
            [1, 256, 7, 7],
            [1, 256, 7, 7],
        ]
    raise ValueError("unknown model type {}".format(model_type))
