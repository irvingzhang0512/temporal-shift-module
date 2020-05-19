import torch

MOBILENET_V2_ONLINE_TO_OFFLINE = 0
MOBILENET_V2_OFFLINE_TO_ONLINE = 1
RESNET50_OFFLINE_TO_ONLINE = 2
RESNET50_ONLINE_TO_OFFLINE = 3
AVAILABLE_CKPT_TRANSFORMATIONS = (
    MOBILENET_V2_OFFLINE_TO_ONLINE,
    MOBILENET_V2_ONLINE_TO_OFFLINE,
    RESNET50_OFFLINE_TO_ONLINE,
    RESNET50_ONLINE_TO_OFFLINE,
)


def _key_mobilenetv2_online_to_offline(key):
    shift_ids = [3, 5, 6, 8, 9, 10, 12, 13, 15, 16]
    if key.startswith("features"):
        splits = key.split(".")
        if key.endswith("conv.0.weight") and int(splits[1]) in shift_ids:
            # features.16.conv.0.weight
            # features.16.conv.0.net.weight
            key = key.replace("0.weight", "0.net.weight")
        key = "module.base_model." + key
    elif key.startswith("classifier"):
        key = key.replace("classifier.", "module.new_fc.")

    return key


def _key_mobilenetv2_offline_to_online(key):
    return key.replace("module.", "").replace("base_model.", "")\
        .replace("net.", "").replace("new_fc", "classifier")


def _key_resnet50_online_to_offline(key):
    if not key.startswith("fc"):
        key = "base_model." + key
    key = "module." + key
    if 'conv1' in key and 'layer' in key:
        key = key.replace('conv1', 'conv1.net')
    if 'fc' in key:
        key = key.replace('fc', 'new_fc')
    return key


def _key_resnet50_offline_to_online(key):
    return key.replace("module.", "").replace("base_model.", "")\
        .replace("net.", "").replace("new_", "")


MODEL_TO_KEY_TRANSFORMATION_FN = {
    MOBILENET_V2_OFFLINE_TO_ONLINE: _key_mobilenetv2_offline_to_online,
    MOBILENET_V2_ONLINE_TO_OFFLINE: _key_mobilenetv2_online_to_offline,
    RESNET50_OFFLINE_TO_ONLINE: _key_resnet50_offline_to_online,
    RESNET50_ONLINE_TO_OFFLINE: _key_resnet50_online_to_offline,
}


def transform_ckpt(src_ckpt, target_ckpt, mode):
    if mode not in AVAILABLE_CKPT_TRANSFORMATIONS:
        raise ValueError("unknown transformation mode")
    try:
        src_dict = torch.load(src_ckpt)
    except Exception:
        raise ValueError("unknown src ckpt {}".format(src_ckpt))

    key_transform_fn = MODEL_TO_KEY_TRANSFORMATION_FN[mode]
    if mode in [MOBILENET_V2_OFFLINE_TO_ONLINE, RESNET50_OFFLINE_TO_ONLINE]:
        src_dict = src_dict['state_dict']
    target_dict = {key_transform_fn(k): src_dict[k] for k in src_dict.keys()}
    torch.save(target_dict, target_ckpt)


if __name__ == '__main__':
    import os
    target_ckpt = "./test.pth.tar"

    # resnet offline to online
    from tsm.models import resnet50
    transform_ckpt("/hdd02/zhangyiyang/temporal-shift-module/checkpoint/TSM_jester_RGB_resnet50_shift8_blockres_avg_segment8_e50_online_default/ckpt.best.pth.tar",
                   target_ckpt, RESNET50_OFFLINE_TO_ONLINE)
    resnet50_online_model = resnet50(num_classes=27)
    target_dict = torch.load(target_ckpt)
    resnet50_online_model.load_state_dict(target_dict)
    os.remove(target_ckpt)
