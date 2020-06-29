"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

import torch.nn as nn
import math
import torch



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SkipMobileNet_v3(nn.Module):
    def __init__(self, cfgs,num_classes, mode,  width_mult=1.):
        super(SkipMobileNet_v3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']
        # building inverted residual blocks
        block = InvertedResidual
        # building first layer
        input_channel = 16
        self.first_layer = conv_3x3_bn(3, input_channel, 2)
        self.base_block = block(input_channel, input_channel, input_channel, 3, 1, 0, 0)

        self.block1, self.block2, self.block3, self.block4 = None, None, None, None
        features = None
        for k, t, output_channel, use_se, use_hs, s, make_block in self.cfgs:
            if features == None:
                features = []
            exp_size = _make_divisible(input_channel * t, 8)
            features.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel

            if make_block:
                if self.block1 is None:
                    self.block1 = nn.Sequential(*features)
                    self.pyramid2 = InvertedResidual(16, 16 * t, output_channel,  k, s, 0, 0)
                    features = None
                    continue
                if self.block2 is None:
                    self.block2 = nn.Sequential(*features)
                    self.pyramid3 = InvertedResidual(16, 16 * t, output_channel,  k, s, 0, 0)
                    features = None
                    continue
                if self.block3 is None:
                    self.block3 = nn.Sequential(*features)
                    # self.pyramid4 = InvertedResidual(16, 16 * t, output_channel,  k, s, 0, 0)
                    features = None
                    continue

                if self.block4 is None:
                    self.block4 = nn.Sequential(*features)

                    features = None
                    continue


        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        output_channel = {'large': 1280, 'small': 1024}
        output_channel = output_channel[mode]

        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.first_layer(x)
        x_base = self.base_block(x)

        x = self.block1(x_base)

        x_p2 = nn.functional.adaptive_avg_pool2d(x_base, x.size()[2])
        x_p2 = self.pyramid2(x_p2)
        x = self.block2(x + x_p2)

        x_p3 = nn.functional.adaptive_avg_pool2d(x_base, x.size()[2])
        x_p3 = self.pyramid3(x_p3)
        x = self.block3(x + x_p3)

        # x_p4 = nn.functional.adaptive_avg_pool2d(x_base, x.size()[2])
        # x_p4 = self.pyramid4(x_p4)
        x = self.block4(x)

        x_features = self.conv(x)
        x = self.avgpool(x_features)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, x_features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def skip_v3(num_classes):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k,  t,  c,  SE,HS,s

        [3,   4,  24, 0, 0, 2, False],
        [3,   3,  24, 0, 0, 1, True],
        [5,   3,  40, 1, 0, 2, False],
        [5,   3,  40, 1, 0, 1, False],
        [5,   6,  40, 1, 1, 1, True],
        # [3,   6,  80, 0, 1, 2],
        # [3, 2.5,  80, 0, 1, 1],
        # [3, 2.3,  80, 0, 1, 1],
        # [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 2, False],
        [3,   6, 112, 1, 1, 1, True],
        [5,   6, 160, 1, 1, 2, False],
        [5,   6, 160, 1, 1, 1, False],
        [5,   6, 160, 1, 1, 1, True]
    ]
    return SkipMobileNet_v3(cfgs, num_classes, mode='large')
