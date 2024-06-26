import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(in_channels=inp, out_channels=inp * expand_ratio,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # depthwise convolution via groups
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio,
                      kernel_size=3, stride=stride, padding=1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pointwise linear convolution
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=oup,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_dynamicFPN(nn.Module):
    def __init__(self, width_mult=1.):
        super(MobileNetV2_dynamicFPN, self).__init__()

        self.input_channel = int(32 * width_mult)
        self.width_mult = width_mult

        # First layer
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, self.input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channel),
            nn.ReLU6(inplace=True)
        )

        # Inverted residual blocks (each n layers)
        self.inverted_residual_setting = [
            {'expansion_factor': 1, 'width_factor': 16, 'n': 1, 'stride': 1},
            {'expansion_factor': 6, 'width_factor': 24, 'n': 2, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 32, 'n': 3, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 64, 'n': 4, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 96, 'n': 3, 'stride': 1},
            {'expansion_factor': 6, 'width_factor': 160, 'n': 3, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 320, 'n': 1, 'stride': 1},
        ]
        self.inverted_residual_blocks = nn.ModuleList(
            [self._make_inverted_residual_block(**setting)
             for setting in self.inverted_residual_setting])

        # reduce feature maps to one pixel
        # allows to upsample semantic information of every part of the image
        self.average_pool = nn.AdaptiveAvgPool2d(1)

        # Top layer
        # input channels = last width factor
        self.top_layer = nn.Conv2d(
            int(self.inverted_residual_setting[-1]['width_factor'] * self.width_mult),
            256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        # exclude last setting as this lateral connection is the the top layer
        # build layer only if resulution has decreases (stride > 1)
        self.lateral_setting = [setting for setting in self.inverted_residual_setting[:-1]
                                if setting['stride'] > 1]
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(int(setting['width_factor'] * self.width_mult),
                      256, kernel_size=1, stride=1, padding=0)
            for setting in self.lateral_setting])

        # Smooth layers
        # n = lateral layers + 1 for top layer
        self.smooth_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)] *
                                           (len(self.lateral_layers) + 1))

        self._initialize_weights()

    def _make_inverted_residual_block(self, expansion_factor, width_factor, n, stride):
        inverted_residual_block = []
        output_channel = int(width_factor * self.width_mult)
        for i in range(n):
            # except the first layer, all layers have stride 1
            if i != 0:
                stride = 1
            inverted_residual_block.append(
                InvertedResidual(self.input_channel, output_channel, stride, expansion_factor))
            self.input_channel = output_channel

        return nn.Sequential(*inverted_residual_block)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=False) + y

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

    def forward(self, x):
        # bottom up
        x = self.first_layer(x)

        # loop through inverted_residual_blocks (mobile_netV2)
        # save lateral_connections to lateral_tensors
        # track how many lateral connections have been made
        lateral_tensors = []
        n_lateral_connections = 0
        for i, block in enumerate(self.inverted_residual_blocks):
            output = block(x)  # run block of mobile_net_V2
            if self.inverted_residual_setting[i]['stride'] > 1 \
                    and n_lateral_connections < len(self.lateral_layers):
                lateral_tensors.append(self.lateral_layers[n_lateral_connections](output))
                n_lateral_connections += 1
            x = output

        x = self.average_pool(x)

        # connect m_layer with previous m_layer and lateral layers recursively
        m_layers = [self.top_layer(x)]
        # reverse lateral tensor order for top down
        lateral_tensors.reverse()
        for lateral_tensor in lateral_tensors:
            m_layers.append(self._upsample_add(m_layers[-1], lateral_tensor))

        # smooth all m_layers
        assert len(self.smooth_layers) == len(m_layers)
        p_layers = [smooth_layer(m_layer) for smooth_layer, m_layer in zip(self.smooth_layers, m_layers)]

        return p_layers


# def test():
#     net = MobileNetV2_dynamicFPN()
#     print(net)
#     output = net(torch.randn(1, 3, 1024, 1024))
#     print(output[0].shape)
#     for feature_map in output:
#         print(feature_map.size())


# # test()
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
transformer_dim=256
embedding_imagelocal = nn.Sequential(
                                            nn.Conv2d(transformer_dim,transformer_dim//8, kernel_size=1, stride=1),
                                            LayerNorm2d(transformer_dim // 8),
                                            nn.GELU(),
                                            # nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                            # LayerNorm2d(transformer_dim // 8),
                                            # nn.GELU(),
                                            # nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 8, kernel_size=2, stride=2),
                                        )
embedding_imagemid = nn.Sequential(
                                    nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                    LayerNorm2d(transformer_dim // 4),
                                    nn.GELU(),
                                    nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    LayerNorm2d(transformer_dim // 8),
                                    nn.GELU(),
                                    nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 8, kernel_size=2, stride=2),
                                     )
embedding_imageglobal = nn.Sequential(
                                            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                            LayerNorm2d(transformer_dim // 4),
                                            nn.GELU(),
                                            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                            LayerNorm2d(transformer_dim // 8),
                                            nn.GELU(),
                                            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2),
                                            LayerNorm2d(transformer_dim // 16),
                                            nn.GELU(),
                                            nn.ConvTranspose2d(transformer_dim // 16, transformer_dim // 16, kernel_size=2, stride=2),
                                            nn.Conv2d(transformer_dim//16,32,1,1,0),
                                        )
net = MobileNetV2_dynamicFPN()
output = net(torch.randn(1, 3, 1024, 1024))
for feature_map in output:
    print(feature_map.size())
f0,f1,f2,f3,f4=output[0],output[1],output[2],output[3],output[4]
print(f4.shape)
test_shape=embedding_imagelocal(f4)

print(test_shape.shape)