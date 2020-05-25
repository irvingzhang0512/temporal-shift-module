import torch
from tvm import relay


def build_pytorch_model(model_type,
                        num_classes=27,
                        ckpt_path=None):
    model = None
    if model_type == 'mobilenetv2_online':
        from tsm.models.mobilenet_v2_tsm_online import MobileNetV2
        model = MobileNetV2(num_classes).eval()
    elif model_type == 'resnet50_online':
        from tsm.models.resnet_tsm_online import resnet50
        model = resnet50(num_classes=num_classes).eval()
    if model is not None:
        if ckpt_path is not None:
            model.load_state_dict(torch.load(ckpt_path))
        return model
    raise ValueError("unknown model type {}".format(model_type))


def build_tvm_model(model_type, num_classes, ckpt_path=None):
    model = build_pytorch_model(model_type, num_classes, ckpt_path)

    x = torch.rand(1, 3, 224, 224)
    if model_type == 'mobilenetv2_online':
        shift_buffer = [torch.zeros([1, 3, 56, 56]),
                        torch.zeros([1, 4, 28, 28]),
                        torch.zeros([1, 4, 28, 28]),
                        torch.zeros([1, 8, 14, 14]),
                        torch.zeros([1, 8, 14, 14]),
                        torch.zeros([1, 8, 14, 14]),
                        torch.zeros([1, 12, 14, 14]),
                        torch.zeros([1, 12, 14, 14]),
                        torch.zeros([1, 20, 7, 7]),
                        torch.zeros([1, 20, 7, 7])]
    elif model_type == 'resnet50_online':
        shift_buffer = [
            torch.zeros([1, 8, 56, 56]),
            torch.zeros([1, 32, 56, 56]),
            torch.zeros([1, 32, 56, 56]),

            torch.zeros([1, 32, 56, 56]),
            torch.zeros([1, 64, 28, 28]),
            torch.zeros([1, 64, 28, 28]),
            torch.zeros([1, 64, 28, 28]),

            torch.zeros([1, 64, 28, 28]),
            torch.zeros([1, 128, 14, 14]),
            torch.zeros([1, 128, 14, 14]),
            torch.zeros([1, 128, 14, 14]),
            torch.zeros([1, 128, 14, 14]),
            torch.zeros([1, 128, 14, 14]),

            torch.zeros([1, 128, 14, 14]),
            torch.zeros([1, 256, 7, 7]),
            torch.zeros([1, 256, 7, 7]),
        ]
    scripted_model = torch.jit.trace(model, (x, *shift_buffer)).eval()
    shape_list = [("input0", (1, 3, 224, 224))]
    for i, buffer in enumerate(shift_buffer):
        shape_list.append(("input" + str(i + 1), buffer.size()))
    mod, params = relay.frontend.from_pytorch(scripted_model,
                                              shape_list)
    return mod, params


if __name__ == '__main__':
    import sys
    sys.path.append("/hdd02/zhangyiyang/temporal-shift-module")

    mod, params = build_tvm_model('mobilenetv2_online', 27)
    print('mobilenetv2 online model is successfully created')
    mod, params = build_tvm_model('resnet50_online', 27)
    print('mobilenetv2 online model is successfully created')
