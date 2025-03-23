from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from monai.networks.nets import UNet,AttentionUnet,RegUNet
from monai.networks.layers import Norm

import tqdm
import segmentation_models_pytorch_3d as smp
import torch



##SegResNet-----------------------------------------------------------------------------------------------------------------

__all__ = ["SegResNet"]

class SegResNet(nn.Module):

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final

        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        filters = self.init_filters
        for i, num_blocks in enumerate(self.blocks_down):
            in_channels = filters * 2**i
            pre_conv = get_conv_layer(self.spatial_dims, in_channels // 2, in_channels, stride=2) if i > 0 else nn.Identity()
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(self.spatial_dims, in_channels, norm=self.norm, act=self.act) for _ in range(num_blocks)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        filters = self.init_filters
        for i in range(len(self.blocks_up)):
            in_channels = filters * 2 ** (len(self.blocks_up) - i)
            up_layer = nn.Sequential(
                *[ResBlock(self.spatial_dims, in_channels // 2, norm=self.norm, act=self.act) for _ in range(self.blocks_up[i])]
            )
            up_layers.append(up_layer)

            up_sample = nn.Sequential(
                get_conv_layer(self.spatial_dims, in_channels, in_channels // 2, kernel_size=1),
                get_upsample_layer(self.spatial_dims, in_channels // 2, self.upsample_mode)
            )
            up_samples.append(up_sample)

        return up_layers, up_samples

    def _make_final_conv(self, out_channels):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x):
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = [x]
        for down_layer in self.down_layers:
            x = down_layer(x)
            down_x.append(x)
        return x, down_x

    def decode(self, x, down_x):
        down_x.reverse()
        for i, (up_sample, up_layer) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up_sample(x) + down_x[i + 1]
            x = up_layer(x)
        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, x):
        x, down_x = self.encode(x)
        x = self.decode(x, down_x)
        return x



#MRRN-----------------------------------------------------------------------------------------------------------------

class Residual_Unit3d(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, in_c, inter_c, out_c):
        super(Residual_Unit3d, self).__init__()
        self.unit = CNN_block3d(in_c, inter_c)

    def forward(self, x):
        x_ = self.unit(x)
        return x + x_

class CNN_block3d(nn.Module):
    """3D CNN Block with instance normalization."""
    def __init__(self, in_c, inter_c):
        super(CNN_block3d, self).__init__()
        self.conv1 = nn.Conv3d(in_c, inter_c, 3, 1, padding=1, bias=True)
        self.norm1 = nn.BatchNorm3d(inter_c)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.norm1(x1)
        x3 = self.activation(x2)
        return x3

class FRRU3d(nn.Module):
    """3D FRRU for the MRRN net."""
    def __init__(self, in_c, inter_c, up_scale, adjust_channel, max_p_size):
        super(FRRU3d, self).__init__()
        self.maxp = nn.MaxPool3d(kernel_size=(max_p_size, max_p_size, max_p_size))
        self.drop = nn.Dropout3d(p=0.5)
        self.cnn_block = nn.Sequential(
            nn.Conv3d(in_c, inter_c, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(inter_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_c, inter_c, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(inter_c),
            nn.ReLU(inplace=True)
        )
        self.channel_adjust = nn.Conv3d(inter_c, adjust_channel, 1, 1, padding=0, bias=True)
        self.upsample = nn.Upsample(scale_factor=up_scale, mode="trilinear")

    def forward(self, p_s, r_s):
        r_s1 = self.drop(self.maxp(r_s))
        merged_ = torch.cat((r_s1, p_s), dim=1)
        pool_sm_out = self.cnn_block(merged_)
        adjust_out1 = self.channel_adjust(pool_sm_out)
        adjust_out1_up_samp = self.upsample(adjust_out1)
        residual_sm_out = adjust_out1_up_samp + r_s
        return pool_sm_out, residual_sm_out

class Incre_MRRN_v2_3d(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Incre_MRRN_v2_3d, self).__init__()
        self.CNN_block1 = CNN_block3d(n_channels, 32)
        self.RU1 = Residual_Unit3d(32, 32, 32)
        self.RU2 = Residual_Unit3d(32, 32, 32)
        self.RU3 = Residual_Unit3d(32, 32, 32)
        self.Pool_stream1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.FRRU1 = FRRU3d(64, 64, 2, 32, 2)
        self.out_conv = nn.Conv3d(32, n_classes, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.CNN_block1(x)
        x2 = self.RU1(x1)
        x3 = self.RU2(x2)
        x4 = self.RU3(x3)
        rs_2 = self.Pool_stream1(x4)
        rs_1 = x4
        rs_2, rs_1 = self.FRRU1(rs_2, rs_1)
        out = self.out_conv(rs_1)
        return out
    

#R2UNet-----------------------------------------------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class U_Net3D(nn.Module):
    def __init__(self, img_ch=3, output_ch=2):
        super(U_Net3D, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.Conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.Conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.Conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.Conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.Up5 = UpConv(ch_in=1024, ch_out=512)
        self.Up_conv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.Up4 = UpConv(ch_in=512, ch_out=256)
        self.Up_conv4 = ConvBlock(ch_in=512, ch_out=256)

        self.Up3 = UpConv(ch_in=256, ch_out=128)
        self.Up_conv3 = ConvBlock(ch_in=256, ch_out=128)

        self.Up2 = UpConv(ch_in=128, ch_out=64)
        self.Up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.Final_conv = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        output = self.Final_conv(d2)
        return output
    
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
    def __init__(self, spatial_dims=3, in_channels=1, out_channels=1, act=("elu", {"inplace": True}), dropout_prob_down=0.5, dropout_prob_up=(0.5, 0.5), bias=False):
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