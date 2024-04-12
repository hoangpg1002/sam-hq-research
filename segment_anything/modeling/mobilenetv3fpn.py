import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MobileNetV3', 'mobilenetv3']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3_forFPN(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3_forFPN, self).__init__()
        input_channel = 16
        last_channel = 1280

        if mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel

        self.layers_os4 = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]#after 1st bottle neck
        self.layers_os8 = [] #after 3rd bottle neck
        self.layers_os16 = [] #after 8th bottle neck
        self.layers_os32 = [] #last
        layers = [self.layers_os4, self.layers_os8, self.layers_os16, self.layers_os32]
        last_bneck = [1, 3, 8, 0]
        
        n_layer = 0
        layer = layers[n_layer]
        n_last_bneck = last_bneck[n_layer]
        # building mobile blocks
        for i, (k, exp, c, se, nl, s) in enumerate(mobile_setting):
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            layer.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

            if i+1 == n_last_bneck:
                n_layer +=1
                layer = layers[n_layer]
                n_last_bneck = last_bneck[n_layer]

        # make it nn.Sequential
        self.layers_os4, self.layers_os8, self.layers_os16, self.layers_os32 = [nn.Sequential(*layer) for layer in layers]

        self._initialize_weights()

    def forward(self, x):
        x1 = self.layers_os4(x)
        x2 = self.layers_os8(x1)
        x3 = self.layers_os16(x2)
        x4 = self.layers_os32(x3)

        return x4

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def mobilenetV3_fpn_backbone(pretrained_path=False):
    '''
    This function builds FPN on mobileNetV3  using torchvision utils
    '''
    from torchvision.models.detection.backbone_utils import BackboneWithFPN

    backbone = MobileNetV3_forFPN()   
        
    return_layers = {'layers_os4': 0, 'layers_os8': 1, 'layers_os16': 2, 'layers_os32': 3}
    # TO-DO: it should be possible to choose which layers to use 
    in_channels_list = [16, 24, 48, 96]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
if __name__ == '__main__':
    net = mobilenetV3_fpn_backbone()
    for p in net.parameters():
        p.requires_grad=True
    print(net)
    input_size=(1, 3, 1024, 1024)
    x = torch.randn(input_size)
    out = net(x)
    for i in range(0,4):
        print(out[i].shape)
    f1,f2,f3,f4=out[0],out[1],out[2],out[3]
    print(f2.shape)
    transformer_dim=256
    #128, 
    embedding_imagelocal = nn.Sequential(
                                            nn.Conv2d(transformer_dim,transformer_dim, kernel_size=3, stride=1,padding=1),
                                            LayerNorm2d(transformer_dim),
                                            nn.GELU(),
                                            nn.Conv2d(transformer_dim,transformer_dim//8,kernel_size=1,stride=1),
                                            LayerNorm2d(transformer_dim//8),
                                            nn.GELU(),
                                        )
    embedding_imagelocal2 = nn.Sequential(
                                            nn.Conv2d(transformer_dim,transformer_dim, kernel_size=3, stride=1,padding=1),
                                            LayerNorm2d(transformer_dim),
                                            nn.GELU(),
                                            nn.ConvTranspose2d(transformer_dim,transformer_dim//4,kernel_size=2,stride=2),
                                            LayerNorm2d(transformer_dim//4),
                                            nn.GELU(),
                                            nn.Conv2d(transformer_dim//4,transformer_dim//8,kernel_size=1,stride=1),
                                        )
    embedding_imagelocal3 = nn.Sequential(
                                            nn.Conv2d(transformer_dim,transformer_dim, kernel_size=3, stride=1,padding=1),
                                            LayerNorm2d(transformer_dim),
                                            nn.GELU(),
                                            nn.ConvTranspose2d(transformer_dim,transformer_dim//4,kernel_size=2,stride=2),
                                            LayerNorm2d(transformer_dim//4),
                                            nn.GELU(),
                                            nn.ConvTranspose2d(transformer_dim//4,transformer_dim//8,kernel_size=2,stride=2),
                                            LayerNorm2d(transformer_dim//8),
                                            nn.GELU(),
                                        )
    embedding_imagelocal4 = nn.Sequential(
                                            nn.Conv2d(transformer_dim,transformer_dim, kernel_size=3, stride=1,padding=1),
                                            LayerNorm2d(transformer_dim),
                                            nn.GELU(),
                                            nn.ConvTranspose2d(transformer_dim,transformer_dim//4,kernel_size=2,stride=2),
                                            LayerNorm2d(transformer_dim//4),
                                            nn.GELU(),
                                            nn.ConvTranspose2d(transformer_dim//4,transformer_dim//8,kernel_size=2,stride=2),
                                            LayerNorm2d(transformer_dim//8),
                                            nn.GELU(),
                                            nn.ConvTranspose2d(transformer_dim//8,transformer_dim//16,kernel_size=2,stride=2),
                                            LayerNorm2d(transformer_dim//16),
                                            nn.GELU(),
                                            nn.Conv2d(transformer_dim//16,transformer_dim//8,kernel_size=1,stride=1)
                                        )

    print(embedding_imagelocal4(f4).shape)
