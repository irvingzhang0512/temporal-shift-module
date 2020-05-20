"""
有两种模式，本地运行模式以及服务器交叉编译模式。
+ 本地模式：默认选项。
+ 服务器交叉编译模式：通过 `--use-cross-compile` 选择。
    + 需要在边缘设备（如Jetbot）上运行 `python3 -m tvm.exec.rpc_server --host 0.0.0.0 --port 9090`
    + 通过 `--remote-ip` 与 `--remote-port` 指定远程设备的ip与端口号。

注意事项：
+ 指定的lib/graph/params文件必须是在指定设备上 Auto Tuning 后得到的。
    + 比如，如果要使用本地运行模式，则要求 Auto Tuning 必须是在以本地设备作为输入。
    + 换句话说，在Jetbot上Auto Tuning的结果放在Server上本地运行，是会报错的。
"""
import argparse
import os
import time
from typing import Tuple

import cv2
import numpy as np
import torchvision
from PIL import Image

import torch
import tvm
import tvm.contrib.graph_runtime as graph_runtime

WINDOW_NAME = 'Video Gesture Recognition'


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-video", type=str,
                        default="/hdd02/zhangyiyang/temporal-shift-module/data/videos/input/ar.mp4")
    parser.add_argument("--categories-file-path", type=str,
                        default="/hdd02/zhangyiyang/data/AR/label/category.txt")
    parser.add_argument("--doing-nothing-label-id", type=int, default=0)
    parser.add_argument("--doing-other-label-id", type=int, default=1)

    # model
    parser.add_argument("--model-type", type=str, default="mobilenetv2_online",
                        help="[mobilenetv2_online, resnet50_online]")
    parser.add_argument("--lib-file-path", type=str,
                        default="./logs-gpu-jetbot-mobilenetv2_online-1589948376552/deploy_lib.tar")
    parser.add_argument("--graph-file-path", type=str,
                        default="./logs-gpu-jetbot-mobilenetv2_online-1589948376552/deploy_graph.json")
    parser.add_argument("--params-file-path", type=str,
                        default="./logs-gpu-jetbot-mobilenetv2_online-1589948376552/deploy_param.params")

    # post preprocessing
    parser.add_argument("--use-history-logit",
                        action="store_true", default=False)
    parser.add_argument("--refine-output", action="store_true", default=False)
    parser.add_argument("--softmax-threshold", type=float, default=.5)

    # cross compile
    parser.add_argument("--use-cross-compile",
                        action="store_true", default=False)
    parser.add_argument("--remote-ip", type=str, default="10.1.171.16")
    parser.add_argument("--remote-port", type=int, default=9090)

    # cv2 show
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--output-flip", action="store_true", default=False)

    return parser.parse_args()


def _get_model_buffer(model_type, ctx):
    if model_type == "mobilenetv2_online":
        return (
            tvm.nd.empty((1, 3, 56, 56), ctx=ctx),
            tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
            tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
            tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
            tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
            tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
            tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
            tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
            tvm.nd.empty((1, 20, 7, 7), ctx=ctx),
            tvm.nd.empty((1, 20, 7, 7), ctx=ctx)
        )
    if model_type == 'resnet50_online':
        return (
            tvm.nd.empty((1, 8, 56, 56), ctx=ctx),
            tvm.nd.empty((1, 32, 56, 56), ctx=ctx),
            tvm.nd.empty((1, 32, 56, 56), ctx=ctx),

            tvm.nd.empty((1, 32, 56, 56), ctx=ctx),
            tvm.nd.empty((1, 64, 28, 28), ctx=ctx),
            tvm.nd.empty((1, 64, 28, 28), ctx=ctx),
            tvm.nd.empty((1, 64, 28, 28), ctx=ctx),

            tvm.nd.empty((1, 64, 28, 28), ctx=ctx),
            tvm.nd.empty((1, 128, 14, 14), ctx=ctx),
            tvm.nd.empty((1, 128, 14, 14), ctx=ctx),
            tvm.nd.empty((1, 128, 14, 14), ctx=ctx),
            tvm.nd.empty((1, 128, 14, 14), ctx=ctx),
            tvm.nd.empty((1, 128, 14, 14), ctx=ctx),

            tvm.nd.empty((1, 128, 14, 14), ctx=ctx),
            tvm.nd.empty((1, 256, 7, 7), ctx=ctx),
            tvm.nd.empty((1, 256, 7, 7), ctx=ctx),
        )
    raise ValueError("unknown model type {}".format(model_type))


def get_executor(args):
    if args.use_cross_compile:
        remote = tvm.rpc.connect(args.remote_ip, args.remote_port)
        remote.upload(args.lib_file_path)
        loaded_lib = remote.load_module(os.path.basename(args.lib_file_path))
        ctx = remote.gpu()
    else:
        loaded_lib = tvm.runtime.load_module(args.lib_file_path)
        ctx = tvm.gpu()

    with open(args.graph_file_path, 'rt') as f:
        graph = f.read()
    graph_module = graph_runtime.create(graph, loaded_lib, ctx)
    graph_module.load_params(
        bytearray(open(args.params_file_path, "rb").read()))

    def executor(inputs: Tuple[tvm.nd.NDArray]):
        for index, value in enumerate(inputs):
            graph_module.set_input("input"+str(index), value)
        graph_module.run()
        return tuple(graph_module.get_output(index)
                     for index in range(len(inputs)))

    return executor, ctx


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
                    return np.concatenate(
                        [np.array(x)[:, :, ::-1] for x in img_group], axis=2)
                else:
                    return np.concatenate(img_group, axis=2)

    class ToTorchFormatTensor(object):
        """
        Converts a PIL.Image (RGB) or numpy.ndarray
        (H x W x C)  in the range [0, 255] to a torch.FloatTensor of shape
        (C x H x W)  in the range [0.0, 1.0]
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


def process_output(idx_, history, refine_output):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not refine_output:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # use only single no action class
    if idx_ == 1:
        idx_ = 0

    # history smoothing
    if idx_ != history[-1] and len(history) > 1:
        # and history[-2] == history[-3]):
        if not (history[-1] == history[-2]):
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history


def main(args):

    # cv2 ops
    print("Open camera...")
    try:
        cap = cv2.VideoCapture(int(args.input_video))
    except Exception:
        cap = cv2.VideoCapture(args.input_video)

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # cv2 show configs
    if args.show:
        full_screen = False
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 640, 480)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    categories = open(args.categories_file_path, "r").readlines()
    categories = [category.strip() for category in categories]
    print("Build Executor...")
    executor, ctx = get_executor(args)
    print("Build transformer...")
    transform = get_transform()

    # prepare for loop
    t = None
    index = 0
    buffer = _get_model_buffer(args.model_type, ctx)
    idx = 0
    history = [2]
    history_logit = []
    i_frame = -1

    print("Ready!")
    while True:
        i_frame += 1
        flag, img = cap.read()  # (480, 640, 3) 0 ~ 255
        if not flag:
            break
        if i_frame % 2 == 0:
            t1 = time.time()

            # image preprocessing & model run
            img_tran = transform([Image.fromarray(img).convert('RGB')])
            input_var = torch.autograd.Variable(
                img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
            img_nd = tvm.nd.array(input_var.detach().numpy(), ctx=ctx)
            inputs: Tuple[tvm.nd.NDArray] = (img_nd,) + buffer
            outputs = executor(inputs)
            feat, buffer = outputs[0], outputs[1:]
            assert isinstance(feat, tvm.nd.NDArray)

            if args.softmax_threshold > 0:
                softmax = feat.asnumpy().reshape(-1)
                print(max(softmax))
                if max(softmax) > args.softmax_threshold:
                    idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
                else:
                    idx_ = args.doing_nothing_label_id
            else:
                idx_ = np.argmax(feat.asnumpy(), axis=1)[0]

            if args.use_history_logit:
                history_logit.append(feat.asnumpy())
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history, args.refine_output)

            t2 = time.time()
            print(f"{index} {categories[idx]}")

            current_time = t2 - t1

        if args.show:
            # draw image
            img = cv2.resize(img, (640, 480))
            if args.output_flip:
                img = img[:, ::-1]
            height, width, _ = img.shape
            label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

            cv2.putText(label, 'Prediction: ' + categories[idx],
                        (0, int(height / 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)
            cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                        (width - 170, int(height / 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)
            img = np.concatenate((img, label), axis=0)

            # show image
            cv2.imshow(WINDOW_NAME, img)
            key = cv2.waitKey(10)
            if key & 0xFF == ord('q') or key == 27:  # exit
                break
            elif key == ord('F') or key == ord('f'):  # full screen
                print('Changing full screen option!')
                full_screen = not full_screen
                if full_screen:
                    print('Setting FS!!!')
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_NORMAL)

        if t is None:
            t = time.time()
        else:
            nt = time.time()
            index += 1
            t = nt

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(_parse_args())
