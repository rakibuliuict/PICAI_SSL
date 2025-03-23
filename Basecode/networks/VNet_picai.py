
##VNet------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Conv, Norm, Dropout, split_args

def get_acti_layer(act, nchan=0):
    if act == "prelu":
        act = ("prelu", {"num_parameters": nchan})
    act_name, act_args = split_args(act)
    act_type = Act[act_name]
    return act_type(**act_args)

class InputTransition(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, act, bias=False):
        super().__init__()
        self.act_function = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv_block(x)
        x_repeated = x.repeat(1, out.size(1) // x.size(1), *([1] * (x.ndim - 2)))
        out = self.act_function(out + x_repeated)
        return out

class DownTransition(nn.Module):
    def __init__(self, spatial_dims, in_channels, nconvs, act, dropout_prob=None, bias=False):
        super().__init__()
        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type = Norm[Norm.BATCH, spatial_dims]
        self.down_conv = conv_type(in_channels, 2 * in_channels, kernel_size=2, stride=2, bias=bias)
        self.bn = norm_type(2 * in_channels)
        self.act_function = get_acti_layer(act, 2 * in_channels)
        self.ops = nn.Sequential(*[Convolution(spatial_dims, 2 * in_channels, 2 * in_channels, kernel_size=5, act=act, norm=Norm.BATCH, bias=bias) for _ in range(nconvs)])
        self.dropout = Dropout[Dropout.DROPOUT, 3](dropout_prob) if dropout_prob else nn.Identity()

    def forward(self, x):
        x = self.act_function(self.bn(self.down_conv(x)))
        x = self.dropout(x)
        x_out = self.ops(x)
        return self.act_function(x_out + x)

class UpTransition(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, nconvs, act, dropout_prob=(None, 0.5)):
        super().__init__()
        conv_trans_type = Conv[Conv.CONVTRANS, spatial_dims]
        norm_type = Norm[Norm.BATCH, spatial_dims]
        self.up_conv = conv_trans_type(in_channels, out_channels // 2, kernel_size=2, stride=2)
        self.bn = norm_type(out_channels // 2)
        self.act_function = get_acti_layer(act, out_channels // 2)
        self.ops = nn.Sequential(*[Convolution(spatial_dims, out_channels, out_channels, kernel_size=5, act=act, norm=Norm.BATCH) for _ in range(nconvs)])
        self.dropout = Dropout[Dropout.DROPOUT, 3](dropout_prob[1])

    def forward(self, x, skipx):
        x = self.act_function(self.bn(self.up_conv(x)))
        x = torch.cat((x, self.dropout(skipx)), 1)
        return self.ops(x) + x

class OutputTransition(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, act, bias=False):
        super().__init__()
        self.conv_block = Convolution(spatial_dims, in_channels, out_channels, kernel_size=5, act=None, norm=Norm.BATCH, bias=bias)
        self.conv2 = Conv[Conv.CONV, spatial_dims](out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.conv_block(x))

class VNet(nn.Module):
    def __init__(self, spatial_dims=3, in_channels=3, out_channels=1, act=("elu", {"inplace": True}), dropout_prob_down=0.5, dropout_prob_up=(0.5, 0.5), bias=False):
        super().__init__()
        self.in_tr = InputTransition(spatial_dims, in_channels, 16, act, bias=bias)
        self.down_tr32 = DownTransition(spatial_dims, 16, 1, act, bias=bias)
        self.down_tr64 = DownTransition(spatial_dims, 32, 2, act, bias=bias)
        self.down_tr128 = DownTransition(spatial_dims, 64, 3, act, dropout_prob=dropout_prob_down, bias=bias)
        self.down_tr256 = DownTransition(spatial_dims, 128, 2, act, dropout_prob=dropout_prob_down, bias=bias)
        self.up_tr256 = UpTransition(spatial_dims, 256, 256, 2, act, dropout_prob=dropout_prob_up)
        self.up_tr128 = UpTransition(spatial_dims, 256, 128, 2, act, dropout_prob=dropout_prob_up)
        self.up_tr64 = UpTransition(spatial_dims, 128, 64, 1, act)
        self.up_tr32 = UpTransition(spatial_dims, 64, 32, 1, act)
        self.out_tr = OutputTransition(spatial_dims, 32, out_channels, act, bias=bias)

    def forward(self, x):
        x16 = self.in_tr(x)
        x32 = self.down_tr32(x16)
        x64 = self.down_tr64(x32)
        x128 = self.down_tr128(x64)
        x256 = self.down_tr256(x128)
        x = self.up_tr256(x256, x128)
        x = self.up_tr128(x, x64)
        x = self.up_tr64(x, x32)
        x = self.up_tr32(x, x16)
        return self.out_tr(x)