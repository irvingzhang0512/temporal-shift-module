AVAILABLE_BACKBONES = (
    "mobilenetv2", "resnet50"
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


def convert_state_dict(src_dict,
                       backbone='mobilenetv2',
                       from_online_ckpt=True,
                       to_online_ckpt=True,
                       from_params_ckpt=False,):
    """
    转换权重文件中的字典，主要是各类state_dict转换

    backbone 输入模型的骨干网络
    from_online_ckpt 输入的权重文件是否是在线模型
    to_online_ckpt 期望输出的权重文件是否是在线模型
    from_params_ckpt 输入的模型是否是刚刚训练得到的
        PS：训练得到的权重文件不仅仅包括了 state_dict，还包括其他训练信息。
    """
    if from_params_ckpt:
        src_dict = src_dict['state_dict']
    if from_online_ckpt == to_online_ckpt:
        return src_dict

    if backbone not in AVAILABLE_BACKBONES:
        raise ValueError("unknown backbone {}".format(backbone))

    if backbone == 'mobilenetv2':
        if from_online_ckpt:
            fn = _key_mobilenetv2_online_to_offline
        else:
            fn = _key_mobilenetv2_offline_to_online
    elif backbone == 'resnet50':
        if from_online_ckpt:
            fn = _key_resnet50_online_to_offline
        else:
            fn = _key_resnet50_offline_to_online

    target_dict = {fn(k): src_dict[k] for k in src_dict.keys()}
    return target_dict
