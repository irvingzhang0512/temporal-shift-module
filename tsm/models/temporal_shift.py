# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class TemporalShift(nn.Module):
    """TSM的核心，主要任务就是将输入进行shift操作，然后运行输入的net"""

    def __init__(self, net,
                 n_segment=3,
                 n_div=8,
                 inplace=False,
                 uni_direction=True,
                 ):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.uni_direction = uni_direction

    def forward(self, x):
        x = self.shift(x,
                       self.n_segment,
                       fold_div=self.fold_div,
                       inplace=self.inplace,
                       uni_direction=self.uni_direction,
                       )
        return self.net(x)

    @staticmethod
    def shift(x, n_segment,
              fold_div=3,
              inplace=False,
              uni_direction=True):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when
            # performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)

            if uni_direction:
                # shift right
                out[:, 1:, :fold] = x[:, :-1, :fold]
            else:
                # shift left
                out[:, :-1, :fold] = x[:, 1:, :fold]
                # shift right
                out[:, 1:, fold: 2*fold] = x[:, :-1, fold: 2*fold]

            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

    @staticmethod
    def shift2(x, n_segment,
               fold_div=3,
               inplace=False,
               uni_direction=True,
               batch_size=1,):
        nt, c, h, w = x.size()
        x = x.view(batch_size, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            raise NotImplementedError
        else:
            out1_1 = x[:, 1:, :fold]
            out1_2 = torch.zeros((batch_size, 1, fold, h, w), device=x.device)
            out1 = torch.cat((out1_1, out1_2), dim=1)

            if uni_direction:
                # shift right
                out2_1 = torch.zeros(
                    (batch_size, 1, fold, h, w), device=x.device)
                out2_2 = x[:, :-1, fold:2 * fold]
                out2 = torch.cat((out2_1, out2_2), dim=1)
            else:
                # shift left
                out2_1 = x[:, :-1, fold:2 * fold]
                out2_2 = torch.zeros(
                    (batch_size, 1, fold, h, w), device=x.device)
                out2 = torch.cat((out2_1, out2_2), dim=1)

            out3 = x[:, :, 2*fold:]

        return torch.cat((out1, out2, out3), dim=2).view(nt, c, h, w)


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(
            1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1),
                         stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, n_div=8,
                        place='blockres',
                        temporal_pool=False,
                        uni_direction=True):
    """为resnet设计的"""
    if temporal_pool:
        # 8, 4, 4, 4
        n_segment_list = [n_segment, n_segment //
                          2, n_segment // 2, n_segment // 2]
    else:
        # 8, 8, 8, 8
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0

    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    # 把ResNet中每个block都替换为TemporalShift对象
                    blocks[i] = TemporalShift(
                        b, n_segment=this_segment, n_div=n_div,
                        uni_direction=uni_direction)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    # 只替换resnet中每个block的第一个conv
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(
                            b.conv1, n_segment=this_segment, n_div=n_div,
                            uni_direction=uni_direction)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    if isinstance(net, torchvision.models.ResNet):
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
    tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5

    print('=> Testing GPU...')
    tsm1.cuda()
    tsm2.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5
    print('Test passed.')
