import time
import numpy as np
import torch
import torchvision
import cv2
import argparse
from PIL import Image


catigories = [
    "Doing other things",  # 0
    "Drumming Fingers",  # 1
    "No gesture",  # 2
    "Pulling Hand In",  # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",  # 5
    "Pushing Two Fingers Away",  # 6
    "Rolling Hand Backward",  # 7
    "Rolling Hand Forward",  # 8
    "Shaking Hand",  # 9
    "Sliding Two Fingers Down",  # 10
    "Sliding Two Fingers Left",  # 11
    "Sliding Two Fingers Right",  # 12
    "Sliding Two Fingers Up",  # 13
    "Stop Sign",  # 14
    "Swiping Down",  # 15
    "Swiping Left",  # 16
    "Swiping Right",  # 17
    "Swiping Up",  # 18
    "Thumb Down",  # 19
    "Thumb Up",  # 20
    "Turning Hand Clockwise",  # 21
    "Turning Hand Counterclockwise",  # 22
    "Zooming In With Full Hand",  # 23
    "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",  # 25
    "Zooming Out With Two Fingers"  # 26
]


def _parse_args():
    parser = argparse.ArgumentParser()

    # video
    parser.add_argument('--input-video', type=str,
                        default="/ssd/zhangyiyang/temporal-shift-module/data/test.mp4")
    parser.add_argument('--input-frame-interval', type=int, default=2)
    parser.add_argument('--output-video', type=str,
                        default="/ssd/zhangyiyang/temporal-shift-module/data/test-output.mp4")
    parser.add_argument('--output-video-fps', type=int, default=20)
    parser.add_argument('--output-video-height', type=int, default=480)
    parser.add_argument('--output-video-width', type=int, default=640)

    # model
    parser.add_argument('--model-ckpt-path', type=str,
                        default="/ssd/zhangyiyang/temporal-shift-module/checkpoint/TSM_jester_RGB_mobilenetv2_shift8_blockres_avg_segment8_e50_online/ckpt.best.pth.tar")
    parser.add_argument('--num-segments', type=int, default=8)

    return parser.parse_args()


def _draw_image(img, fps, label_id, resize_shape=None,):
    if resize_shape is not None:
        img = cv2.resize(img, resize_shape)
    img = img[:, ::-1]
    height, width, _ = img.shape
    label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

    cv2.putText(label, 'Prediction: ' + catigories[label_id],
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
            self.worker = torchvision.transforms.Scale(size, interpolation)

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
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])
    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


def predict(imgs, model, args):
    with torch.no_grad():
        imgs = torch.stack(imgs).view(1, args.num_segments, 3, 224, 224)
        output = model(imgs).squeeze(0)
        return torch.argmax(output)


def main(args):
    # load model
    model = torch.load(args.model_ckpt_path)

    # create video reader & writer
    cap = cv2.VideoCapture(args.input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(
        args.output_video,
        fourcc,
        args.output_video_fps,
        (args.output_video_width, args.output_video_height),
    )

    # run loop
    transform = get_transform()
    imgs = []
    t1 = time.time()
    i_frame = -1
    while True:
        flag, img = cap.read()
        if not flag:
            break
        i_frame += 1
        if i_frame % args.input_frame_interval == 0:
            img_transform = transform([Image.fromarray(img).convert('RGB')])
            imgs.append(img_transform)
            if len(imgs) < args.num_segments:
                continue
            imgs = imgs[-args.num_segments:]

            # prediction label: categories[label_id]
            label_id = predict(imgs, model, args)

            t2 = time.time()
            fps = 1./(t2 - t1)
            t1 = t2
            target_img = _draw_image(img, fps, label_id)
            target_img = cv2.resize(
                target_img, (args.output_video_width, args.output_video_height))
            writer.write(target_img)
    writer.release()


if __name__ == '__main__':
    main(_parse_args())
