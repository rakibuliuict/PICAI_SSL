import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            input_channel = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError("Unknown normalization")
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=(2, 2, 1), normalization='none'):
        super(DownsamplingConvBlock, self).__init__()
        ops = [nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, stride=stride, padding=1)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError("Unknown normalization")
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


# class UpsamplingDeconvBlock(nn.Module):
#     def __init__(self, n_filters_in, n_filters_out, stride=(2, 2, 1), normalization='none'):
#         super(UpsamplingDeconvBlock, self).__init__()
#         ops = [
#             nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=3, stride=stride, padding=1, output_padding=1)
#         ]
#         if normalization == 'batchnorm':
#             ops.append(nn.BatchNorm3d(n_filters_out))
#         elif normalization == 'groupnorm':
#             ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
#         elif normalization == 'instancenorm':
#             ops.append(nn.InstanceNorm3d(n_filters_out))
#         elif normalization != 'none':
#             raise ValueError("Unknown normalization")
#         ops.append(nn.ReLU(inplace=True))
#         self.conv = nn.Sequential(*ops)

#     def forward(self, x):
#         return self.conv(x)

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=(2, 2, 1), normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                raise ValueError("Unsupported normalization type.")
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=stride, stride=stride))

        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)




class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, n_filters=16, normalization='instancenorm', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, n_filters * 2, stride=(2, 2, 1), normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, stride=(2, 2, 1), normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, stride=(2, 2, 1), normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, stride=(2, 2, 1), normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, stride=(2, 2, 1), normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, stride=(2, 2, 1), normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, stride=(2, 2, 1), normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, stride=(2, 2, 1), normalization=normalization)

        self.final_block = ConvBlock(1, n_filters, n_filters, normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, kernel_size=1)

        if has_dropout:
            self.dropout = nn.Dropout3d(p=0.5)

    def forward(self, x):
        x1 = self.block_one(x)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        x5_up = self.block_five_up(x5) + x4
        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6) + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7) + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8) + x1

        x9 = self.final_block(x8_up)
        out = self.out_conv(x9)
        return out