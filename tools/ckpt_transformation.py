import torch
import argparse

from tsm.utils.ckpt_convert_utils import convert_state_dict


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--from-ckpt-path", type=str,
                        default="/hdd02/zhangyiyang/temporal-shift-module/checkpoint/TSM_ar_RGB_mobilenetv2_shift8_blockres_avg_segment8_e30_0608_generate_200_crop_reisze_bz64/best.pth.tar")
    parser.add_argument("--to-ckpt-path", type=str,
                        default="./no_generate.pth.tar")
    parser.add_argument("--backbone", type=str, default="mobilenetv2",
                        help="should be one of AVAILABLE_BACKBONES")
    parser.add_argument("--from-params-ckpt",
                        action='store_true', default=False)
    parser.add_argument("--from-online-ckpt",
                        action='store_true', default=False)
    parser.add_argument("--to-online-ckpt",
                        action='store_true', default=False)

    return parser.parse_args()


def main(args):
    src_dict = torch.load(args.from_ckpt_path)
    target_dict = convert_state_dict(
        src_dict,
        backbone=args.backbone,
        from_online_ckpt=args.from_online_ckpt,
        to_online_ckpt=args.to_online_ckpt,
        from_params_ckpt=args.from_params_ckpt,
    )
    torch.save(target_dict, args.to_ckpt_path)


if __name__ == '__main__':
    main(_parse_args())
