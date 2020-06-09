import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch


class GroupColorJitter(object):
    # copy from mmaction

    def __init__(self, color_space_aug=False, alphastd=0.1,
                 eigval=None, eigvec=None):
        if eigval is None:
            # note that the data range should be [0, 255]
            self.eigval = np.array([55.46, 4.794, 1.148])
        if eigvec is None:
            self.eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                                    [-0.5808, -0.0045, -0.8140],
                                    [-0.5836, -0.6948, 0.4203]])
        self.alphastd = alphastd
        self.color_space_aug = color_space_aug

    @staticmethod
    def brightnetss(img, delta):
        if random.uniform(0, 1) > 0.5:
            # delta = np.random.uniform(-32, 32)
            delta = np.array(delta).astype(np.float32)
            img = img + delta
            # img_group = [img + delta for img in img_group]
        return img

    @staticmethod
    def contrast(img, alpha):
        if random.uniform(0, 1) > 0.5:
            # alpha = np.random.uniform(0.6,1.4)
            alpha = np.array(alpha).astype(np.float32)
            img = img * alpha
            # img_group = [img * alpha for img in img_group]
        return img

    @staticmethod
    def saturation(img, alpha):
        if random.uniform(0, 1) > 0.5:
            # alpha = np.random.uniform(0.6,1.4)
            gray = img * np.array([0.299, 0.587, 0.114]).astype(np.float32)
            gray = np.sum(gray, 2, keepdims=True)
            gray *= (1.0 - alpha)
            img = img * alpha
            img = img + gray
        return img

    @staticmethod
    def hue(img, alpha):
        if random.uniform(0, 1) > 0.5:
            # alpha = random.uniform(-18, 18)
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0],
                           [0.0, u, -w],
                           [0.0, w, u]])
            tyiq = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.321],
                             [0.211, -0.523, 0.311]])
            ityiq = np.array([[1.0, 0.956, 0.621],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.107, 1.705]])
            t = np.dot(np.dot(ityiq, bt), tyiq).T
            t = np.array(t).astype(np.float32)
            img = np.dot(img, t)
            # img_group = [np.dot(img, t) for img in img_group]
        return img

    def __call__(self, img_group):
        img_group = [np.array(img) for img in img_group]
        if self.color_space_aug:
            bright_delta = np.random.uniform(-32, 32)
            contrast_alpha = np.random.uniform(0.6, 1.4)
            saturation_alpha = np.random.uniform(0.6, 1.4)
            # hue_alpha = random.uniform(-18, 18)
            out = []
            for img in img_group:
                img = self.brightnetss(img, delta=bright_delta)
                if random.uniform(0, 1) > 0.5:
                    img = self.contrast(img, alpha=contrast_alpha)
                    img = self.saturation(img, alpha=saturation_alpha)
                    # img = self.hue(img, alpha=hue_alpha)
                else:
                    img = self.saturation(img, alpha=saturation_alpha)
                    # img = self.hue(img, alpha=hue_alpha)
                    img = self.contrast(img, alpha=contrast_alpha)
                out.append(img)
            img_group = out

        alpha = np.random.normal(0, self.alphastd, size=(3,))
        rgb = np.array(np.dot(self.eigvec * alpha, self.eigval)
                       ).astype(np.float32)
        # bgr = np.expand_dims(np.expand_dims(rgb[::-1], 0), 0)
        return [img + rgb for img in img_group]


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


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


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            if self.flip:
                oversample_group.extend(flip_group)
        return oversample_group


class GroupFullResSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == 'L' and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self,
                 input_size,
                 scales=None,
                 max_distort=1,
                 fix_crop=True,
                 more_fix_crop=True):
        """
        本质就是 crop + resize
        input_size 是 resize 后的结果
        widht/height 随机选择 self.scales 中的一种配对
        max_distort 绝对了选中的 width/height 编号差值的最大值，即 1/.875 是可行的，1/.75 是不可行的
        fix_crop 用于设置选择的 h/w offset 是否是从一些固定值中选择
        more_fix_crop 用于在 fix_crop 为 True 的情况下，增加一些选择
        """
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [
            input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):
        im_size = img_group[0].size

        # 根据输入参数 crop
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop(
            (offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in img_group]

        # 根据输入参数 resize
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]),
                                    self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(
            x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(
            x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """
    Random crop the given PIL.Image to a random size of (0.08 to 1.0) of
    the original size and and a random aspect ratio of 3/4 to 4/3 of
    the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize(
                    (self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if len(np.array(img_group[0]).shape) == 2:
            return np.concatenate([np.expand_dims(x, 2) for x in img_group],
                                  axis=2)
        else:
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1]
                                       for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """
    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C)
    in the range [0, 255] to a torch.FloatTensor of shape (C x H x W)
    in the range [0.0, 1.0]
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
        img = img.float()
        if self.div:
            img = img.div(255)
        img[img > 1.] = 1.
        img[img < 0.] = 0.
        return img


class IdentityTransform(object):

    def __call__(self, data):
        return data


if __name__ == "__main__":
    import cv2
    trans = torchvision.transforms.Compose([
        GroupMultiScaleCrop(224,
                            [1, .875, .75, .66],
                            fix_crop=False,),
        GroupColorJitter(),
        # Stack(False),
        # ToTorchFormatTensor(),
        # GroupNormalize(
        #     mean=[.485, .456, .406],
        #     std=[.229, .224, .225]
        # )
    ])

    im = Image.open(
        '/hdd02/zhangyiyang/temporal-shift-module/data/images/drinking/img_00003.jpg')

    color_group = [im] * 3
    rst = trans(color_group)
    cv2.imwrite("./test.jpg", rst[0].astype(np.uint8)[:, :, ::-1])
    print(rst[0]/255., rst[0].shape)
    # print(rst.shape, np.max(rst), np.min(rst))
