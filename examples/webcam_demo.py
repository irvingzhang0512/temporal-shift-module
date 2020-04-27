import argparse
import os
import time

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

categories = None


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category-file', type=str,
                        default="/ssd4/zhangyiyang/data/AR/category.txt")
    parser.add_argument('--online-mode', action="store_true", default=False)
    parser.add_argument('--not-params-model', action="store_true", default=False)
    parser.add_argument('--file-name-suffix', type=str, default=None)

    # filter label
    parser.add_argument('--prob-threshold', type=float, default=0.5)
    parser.add_argument('--doing-nothing-label-id', type=int, default=0)
    parser.add_argument('--doing-other-label-id', type=int, default=1)
    parser.add_argument('--probs-history-cumsum', type=int, default=4)

    # output
    parser.add_argument('--output-file-dir', type=str,
                        default="/ssd4/zhangyiyang/temporal-shift-module/data/output")
    parser.add_argument('--output-video-height', type=int, default=480)
    parser.add_argument('--output-video-width', type=int, default=640)
    parser.add_argument('--output-video-flip', action="store_true", default=False)
    parser.add_argument('--show', action="store_true", default=False)

    # video
    parser.add_argument("--tmp-video", type=str, default="./test.avi")
    parser.add_argument('--input-video', type=str,
                        default="/ssd4/zhangyiyang/temporal-shift-module/data/input/ar-4.mp4")
    parser.add_argument('--input-frame-interval', type=int, default=2)
    parser.add_argument('--output-video-dir', type=str,
                        default="/ssd4/zhangyiyang/temporal-shift-module/data/output")
    parser.add_argument('--output-video-fps', type=int, default=20)

    # model
    parser.add_argument('--online-ckpt-model',
                        action="store_true", default=False)
    parser.add_argument('--model-ckpt-path', type=str,
                        default="/ssd4/zhangyiyang/temporal-shift-module/checkpoint/TSM_ar_RGB_mobilenetv2_shift4_blockres_avg_segment8_e50_online_default/ckpt.best.pth.tar")
    parser.add_argument('--num-segments', type=int, default=8)
    parser.add_argument('--shift-div', type=int, default=4)

    return parser.parse_args()


def _draw_image(img, fps,
                label_id, doing_nothing_label_id, prob=0.,
                resize_shape=None, flip=True):
    if resize_shape is not None:
        img = cv2.resize(img, resize_shape)
    
    if flip:
        img = img[:, ::-1]
    height, width, _ = img.shape
    label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

    text = 'Prediction: ' + categories[label_id]
    if label_id != doing_nothing_label_id:
        text += " {:.2f}%".format(prob*100.)
    cv2.putText(label, text,
                (0, int(height / 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)
    cv2.putText(label, '{:.1f} Vid/s'.format(fps),
                (width - 170, int(height / 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)

    img = np.concatenate((img, label), axis=0)
    return img


def get_transform():
    class GroupScale(object):
        """ Rescales the input PIL.Image to the given 'size'.
        'size' will be the size of the smaller edge.
        For example, if height > width, then image will be
        rescaled to (size * height / width, size)
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
        """

        def __init__(self, size, interpolation=Image.BILINEAR):
            self.worker = torchvision.transforms.Resize(size, interpolation)

        def __call__(self, img_group):
            return [self.worker(img) for img in img_group]

    class GroupCenterCrop(object):
        def __init__(self, size):
            self.worker = torchvision.transforms.CenterCrop(size)

        def __call__(self, img_group):
            return [self.worker(img) for img in img_group]

    class Stack(object):

        def __init__(self, roll=False):
            self.roll = roll

        def __call__(self, img_group):
            if img_group[0].mode == 'L':
                return np.concatenate(
                    [np.expand_dims(x, 2) for x in img_group], axis=2)
            elif img_group[0].mode == 'RGB':
                if self.roll:
                    return np.concatenate([np.array(x)[:, :, ::-1]
                                           for x in img_group], axis=2)
                else:
                    return np.concatenate(img_group, axis=2)

    class ToTorchFormatTensor(object):
        """
        Converts a PIL.Image (RGB) or ndarray (H x W x C) in the range [0, 255]
        to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        """

        def __init__(self, div=True):
            self.div = div

        def __call__(self, pic):
            if isinstance(pic, np.ndarray):
                # handle numpy array
                img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
            else:
                # handle PIL Image
                img = torch.ByteTensor(
                    torch.ByteStorage.from_buffer(pic.tobytes()))
                img = img.view(pic.size[1], pic.size[0], len(pic.mode))
                # put it from HWC to CHW format
                # yikes, this transpose takes 80% of the loading time/CPU
                img = img.transpose(0, 1).transpose(0, 2).contiguous()
            return img.float().div(255) if self.div else img.float()

    class GroupNormalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
            rep_std = self.std * (tensor.size()[0] // len(self.std))

            # TODO: make efficient
            for t, m, s in zip(tensor, rep_mean, rep_std):
                t.sub_(m).div_(s)

            return tensor

    transform = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform


def predict(online_mode, model_inputs, model, args):
    with torch.no_grad():
        if online_mode:
            imgs, shift_buffer = model_inputs
            outputs = model(imgs, *shift_buffer)
            logits, shift_buffer = outputs[0], outputs[1:]
            probs = torch.softmax(logits, dim=1).squeeze(0)
            label_id = torch.argmax(probs)
            return label_id, probs.cpu().numpy(), shift_buffer
        else:
            imgs = torch.stack(model_inputs).view(
                1, args.num_segments, 3, 224, 224)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            label_id = torch.argmax(probs)
            return label_id, probs.cpu().numpy()


def _convert_state_dict(src_dict, online_mode=False, online_ckpt_model=False):
    if (online_mode and online_ckpt_model) or \
            (not online_mode and not online_ckpt_model):
        return src_dict
    if online_ckpt_model:
        # online dict to offline dict
        shift_ids = [3, 5, 6, 8, 9, 10, 12, 13, 15, 16]
        original_keys = list(src_dict.keys())
        target_keys = []
        for s in original_keys:
            if s.startswith("features"):
                splits = s.split(".")
                if s.endswith("conv.0.weight") and int(splits[1]) in shift_ids:
                    # features.16.conv.0.weight
                    # features.16.conv.0.net.weight
                    s = s.replace("0.weight", "0.net.weight")
                s = "module.base_model." + s
            elif s.startswith("classifier"):
                s = s.replace("classifier.", "module.new_fc.")
            target_keys.append(s)
    else:
        # offline dict to online ckpt
        original_keys = list(src_dict.keys())
        target_keys = [
            s.replace("module.", "")
            .replace("base_model.", "")
            .replace("net.", "")
            .replace("new_fc", "classifier")
            for s in original_keys
        ]
    target_dict = {target_keys[i]: src_dict[original_keys[i]]
                   for i in range(len(original_keys))}
    return target_dict


def load_model(ckpt_path, args,
               online_mode=False,
               online_ckpt_model=False,
               params_model=True,):
    if not params_model:
        return torch.load(ckpt_path)

    src_dict = torch.load(ckpt_path) if online_ckpt_model \
        else torch.load(ckpt_path)['state_dict']
    target_dict = _convert_state_dict(src_dict,
                                      online_mode, online_ckpt_model)
    if online_ckpt_model:
        num_classes = src_dict['classifier.weight'].size(0)
    else:
        num_classes = src_dict['module.new_fc.weight'].size(0)

    if online_mode:
        from tsm.models.mobilenet_v2_tsm_online import MobileNetV2
        online_model = MobileNetV2(num_classes).eval()
        online_model.load_state_dict(target_dict)
        return online_model
    else:
        from tsm.models.tsn import TSN
        model = TSN(
            num_class=num_classes,
            num_segments=args.num_segments,
            modality='RGB',
            base_model='mobilenetv2',
            consensus_type='avg',
            dropout=0.5,
            img_feature_dim=256,
            crop_num=1,
            partial_bn=False,
            print_spec=True,
            pretrain='imagenet',
            is_shift=True,
            shift_div=args.shift_div,
            shift_place='blockres',
            fc_lr5=False,
            temporal_pool=False,
            non_local=False,
            offline=False,
        )
        model = torch.nn.DataParallel(model).cuda().eval()
        model.load_state_dict(target_dict)
        return model


def create_model_input(online_mode, cur_img, transform,
                       imgs=None, num_segments=8,
                       shift_buffer=None,):
    if online_mode:
        img_transform = transform([cur_img])
        img_transform = torch.unsqueeze(img_transform, 0)
        return img_transform, shift_buffer
    else:
        img_transform = transform([cur_img])
        imgs.append(img_transform)
        if len(imgs) < num_segments:
            return None
        imgs = imgs[-num_segments:]
        return imgs


def _get_output_file_name(args, suffix=".txt"):
    file_name = os.path.basename(args.input_video)
    file_name = file_name[:file_name.rfind(".")]

    if args.online_mode:
        file_name += "-online"
    if args.prob_threshold:
        file_name += "-thres_{:.2f}".format(args.prob_threshold)
    if args.probs_history_cumsum > 1:
        file_name += "-cumsum-{}".format(args.probs_history_cumsum)
    if args.file_name_suffix is not None:
        file_name += "-" + args.file_name_suffix
    file_name += suffix
    return file_name


def _update_label_id(args,
                     label_id, probs,  # original data
                     probs_history, cur_probs,  # cumsum
                     ):
    # cumsum probs
    if args.probs_history_cumsum > 1:
        probs_history.append(probs)
        cur_probs = probs if cur_probs is None else (cur_probs + probs)
        if len(probs_history) > args.probs_history_cumsum:
            cur_probs = cur_probs - probs_history.pop(0)
        probs = cur_probs / len(probs_history)
        label_id = np.argmax(cur_probs)

    # filter label
    if label_id != args.doing_nothing_label_id and\
            probs[label_id] < args.prob_threshold:
        label_id = args.doing_nothing_label_id
    if args.doing_other_label_id > -1 and \
            label_id == args.doing_other_label_id:
        label_id = args.doing_nothing_label_id

    return label_id, probs, cur_probs


def main(args):
    global categories
    with open(args.category_file, 'r') as f:
        lines = f.readlines()
    categories = [l.strip() for l in lines]

    # load model
    model = load_model(args.model_ckpt_path, args,
                       online_mode=args.online_mode,
                       online_ckpt_model=args.online_ckpt_model,
                       params_model=not args.not_params_model,)
    print("model loaded.")

    # video reader
    try:
        cap_input = int(args.input_video)
    except ValueError:
        cmd = "ffmpeg -i {} -q:v 6 {}".format(args.input_video, args.tmp_video)
        os.system(cmd)
        cap_input = args.tmp_video
    cap = cv2.VideoCapture(cap_input)
    print("video capture created.")

    # video writer
    writer = None
    if args.output_video_dir is not None:
        file_name = _get_output_file_name(args, ".avi")
        fourcc = cv2.VideoWriter_fourcc(* 'XVID')
        writer = cv2.VideoWriter(
            os.path.join(args.output_video_dir, file_name),
            fourcc,
            args.output_video_fps,
            (args.output_video_width, args.output_video_height),
        )
        print("video writer created.")

    # create result txt file
    output_file = None
    if args.output_file_dir is not None:
        file_name = _get_output_file_name(args, ".txt")
        output_file = open(os.path.join(args.output_file_dir, file_name), "w")

    # run loop
    transform = get_transform()
    t1 = time.time()
    i_frame = -1
    target_img = None

    # online/offline params
    imgs = None
    shift_buffer = None
    if args.online_mode:
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
    else:
        imgs = []

    # cumsum probs
    probs_history = []
    cur_probs = None

    while True:
        flag, img = cap.read()
        if not flag:
            break
        i_frame += 1
        if i_frame % args.input_frame_interval == 0:
            # get model input
            cur_img = Image.fromarray(img).convert('RGB')
            model_input = create_model_input(
                args.online_mode, cur_img, transform,
                imgs=imgs, num_segments=args.num_segments,
                shift_buffer=shift_buffer,
            )
            if model_input is None:
                continue

            # get model outputs
            outputs = predict(args.online_mode, model_input, model, args)
            if args.online_mode:
                label_id, probs, shift_buffer = outputs
            else:
                label_id, probs = outputs
            label_id, probs, cur_probs = _update_label_id(
                args, label_id, probs,
                probs_history, cur_probs
            )
            cur_prob = probs[label_id]

            # generate target image
            t2 = time.time()
            fps = 1./(t2 - t1)
            t1 = t2
            target_img = _draw_image(
                img, fps,
                label_id, args.doing_nothing_label_id,
                cur_prob, flip=args.output_video_flip)
            target_img = cv2.resize(
                target_img,
                (args.output_video_width, args.output_video_height))

            if output_file is not None:
                output_file.write(
                    categories[label_id] + "," +
                    "{:.2f}".format(cur_prob*100.) + "\n")
        if target_img is not None:
            if writer is not None:
                writer.write(target_img)
            if args.show:
                cv2.imshow('OnlineDemo', target_img)
                cv2.waitKey(1)
    if writer is not None:
        writer.release()
    if os.path.exists(args.tmp_video):
        os.remove(args.tmp_video)
    if output_file is not None:
        output_file.close()

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main(_parse_args())
