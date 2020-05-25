"""
自定义视频，输出结果

支持的功能
+ 支持所有在线模型与离线模型。
+ 支持输入所有类型的权重。
+ 如果backbone相同，但输入的权重与模型不匹配，会自动转换。
+ 支持本地视频文件/cv2/本地文本文件三种展示方式。

使用细节
+ 通过 `--from-params-ckpt` 指定输入的ckpt是否是刚刚训练得到的。
+ 通过 `--from-online-model` 指定输入的ckpt类型。
+ 通过 `--model-type` 指定目标模型类型。
    + 当输入的是离线模型时，还需要指定 `--num-segments` 与 `--shift-div`

"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from tsm.builders import model_builder
from tsm.utils.ckpt_convert_utils import convert_state_dict

categories = None


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category-file', type=str,
                        default="/hdd02/zhangyiyang/data/AR/label/category.txt")
    parser.add_argument('--file-name-suffix', type=str, default=None)

    # filter label
    parser.add_argument('--prob-threshold', type=float, default=0.5)
    parser.add_argument('--doing-nothing-label-id', type=int, default=0)
    parser.add_argument('--doing-other-label-id', type=int, default=1)
    parser.add_argument('--probs-history-cumsum', type=int, default=4)

    # output
    parser.add_argument('--output-file-dir', type=str,
                        default="./data/videos/output")
    parser.add_argument('--output-video-dir', type=str,
                        default="./data/videos/output")
    parser.add_argument('--output-video-fps', type=int, default=20)
    parser.add_argument('--output-video-height', type=int, default=480)
    parser.add_argument('--output-video-width', type=int, default=640)
    parser.add_argument('--output-video-flip',
                        action="store_true", default=False)
    parser.add_argument('--show', action="store_true", default=False)

    # input video
    parser.add_argument("--tmp-video", type=str, default="./test.avi")
    parser.add_argument('--input-video', type=str,
                        default="./data/videos/input/ar.mp4")
    parser.add_argument('--input-frame-interval', type=int, default=2)

    # model
    parser.add_argument('--model-type', type=str,
                        default="mobilenetv2_online")
    parser.add_argument('--model-ckpt-path', type=str,
                        default="checkpoint/TSM_ar_RGB_mobilenetv2_shift8_blockres_avg_segment8_e50_5_9_dataset/ckpt.best.pth.tar")
    parser.add_argument('--from-online-model',
                        action="store_true", default=False)
    parser.add_argument('--from-params-ckpt',
                        action="store_true", default=False)
    parser.add_argument('--num-segments', type=int, default=8)
    parser.add_argument('--shift-div', type=int, default=8)

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


def predict(to_online_model, model_inputs, model, args):
    with torch.no_grad():
        if to_online_model:
            imgs, shift_buffer = model_inputs
            outputs = model(imgs, *shift_buffer)
            probs, shift_buffer = outputs[0], outputs[1:]
            probs = probs.squeeze(0)
            label_id = torch.argmax(probs)
            return label_id, probs.cpu().numpy(), shift_buffer
        else:
            imgs = torch.stack(model_inputs).view(
                1, args.num_segments, 3, 224, 224)
            probs = model(imgs)
            probs = probs.squeeze(0)
            label_id = torch.argmax(probs)
            return label_id, probs.cpu().numpy()


def load_model(model_type, ckpt_path, num_classes,
               to_online_model=False,
               from_online_model=False,
               from_params_ckpt=True,
               num_segments=8, shift_div=8):
    src_dict = torch.load(ckpt_path)
    target_dict = convert_state_dict(
        src_dict,
        backbone=model_type.replace("_online", ""),
        from_params_ckpt=from_params_ckpt,
        from_online_ckpt=from_online_model,
        to_online_ckpt=to_online_model,
    )

    kwargs = {}
    if not to_online_model:
        kwargs['num_segments'] = num_segments
        kwargs['shift_div'] = shift_div
    model = model_builder.build_model(
        model_type, num_classes=num_classes, **kwargs
    )
    model.load_state_dict(target_dict)
    return model


def create_model_input(to_online_model, cur_img, transform,
                       imgs=None, num_segments=8,
                       shift_buffer=None,):
    if to_online_model:
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


def _get_output_file_name(args, to_online_model, suffix=".txt"):
    file_name = os.path.basename(args.input_video)
    file_name = file_name[:file_name.rfind(".")]

    if to_online_model:
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
    categories = [line.strip() for line in lines]

    to_online_model = "online" in args.model_type

    # load model
    model = load_model(args.model_type,
                       args.model_ckpt_path,
                       len(categories),
                       to_online_model=to_online_model,
                       from_online_model=args.from_online_model,
                       from_params_ckpt=args.from_params_ckpt,
                       num_segments=args.num_segments,
                       shift_div=args.shift_div)
    print("model loaded.")

    # video reader
    try:
        cap_input = int(args.input_video)
    except ValueError:
        cmd = "ffmpeg -i {} -q:v 6 {}".format(args.input_video, args.tmp_video)
        if os.path.exists(args.tmp_video):
            os.remove(args.tmp_video)
        os.system(cmd)
        cap_input = args.tmp_video
    cap = cv2.VideoCapture(cap_input)
    print("video capture created.")

    # video writer
    writer = None
    if args.output_video_dir is not None:
        file_name = _get_output_file_name(args, to_online_model, ".avi")
        fourcc = cv2.VideoWriter_fourcc(* 'XVID')
        writer = cv2.VideoWriter(
            os.path.join(args.output_video_dir, file_name),
            fourcc,
            args.output_video_fps,
            (args.output_video_width, args.output_video_height),
        )
        print(os.path.join(args.output_video_dir, file_name))
        print("video writer created.")

    # create result txt file
    output_file = None
    if args.output_file_dir is not None:
        file_name = _get_output_file_name(args, to_online_model, ".txt")
        output_file = open(os.path.join(args.output_file_dir, file_name), "w")

    # run loop
    transform = get_transform()
    t1 = time.time()
    i_frame = -1
    target_img = None

    # online/offline params
    imgs = None
    shift_buffer = None
    if to_online_model:
        buffer_shapes = model_builder.build_buffer_shapes(args.model_type)
        shift_buffer = [torch.zeros(shape) for shape in buffer_shapes]
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
                to_online_model, cur_img, transform,
                imgs=imgs, num_segments=args.num_segments,
                shift_buffer=shift_buffer,
            )
            if model_input is None:
                continue

            # get model outputs
            outputs = predict(to_online_model, model_input, model, args)
            if to_online_model:
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
