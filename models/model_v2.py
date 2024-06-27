import torch
import torch.nn as nn
#PCGC
import MinkowskiEngine as ME
from argparse import Namespace
import warnings
from .sh_utils import RGB2SH
import time

def convert_str_2_list(str_):
    words = str_.split(' ')
    trt = [int(x) for x in words]
    return trt

class InceptionResNet(torch.nn.Module):
    """Inception Residual Network
    """
    
    def __init__(self, channels):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//4,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels//4,
            out_channels=channels//2,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//4,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels//4,
            out_channels=channels//4,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_2 = ME.MinkowskiConvolution(
            in_channels=channels//4,
            out_channels=channels//2,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        out0 = self.conv0_1(self.relu(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu(self.conv1_1(self.relu(self.conv1_0(x)))))
        out = ME.cat(out0, out1) + x

        return out

class SparseUNet(torch.nn.Module):
    def __init__(self, channels=[1,16,32,64,32,8], feat_dim=32):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block0 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[2])

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block1 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[3])

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block2 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[4])

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=channels[4],
            out_channels=channels[5],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

        self.up0 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[5],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv_0 = ME.MinkowskiConvolution(
            in_channels=channels[3]*2,
            out_channels=channels[3],
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.block_0 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[3])

        self.up1 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[3],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv_1 = ME.MinkowskiConvolution(
            in_channels=channels[2]*2,
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block_1 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[2])

        self.up2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[1],
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv_2 = ME.MinkowskiConvolution(
            in_channels=channels[1]*2,
            out_channels=channels[1],
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.block_2 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[1])
        
        self.conv_3 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=feat_dim,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
    
    def forward(self, x):
        out_x = self.relu(self.conv0(x))
        out0 = self.relu(self.down0(out_x))
        out0 = self.block0(out0)
        out1 = self.relu(self.down1(self.relu(self.conv1(out0))))
        out1 = self.block1(out1)
        out2 = self.relu(self.down2(self.relu(self.conv2(out1))))
        out2 = self.block2(out2)
        out2 = self.conv3(out2)

        out = self.relu(self.up0(out2, out1.coordinate_map_key))
        out = ME.cat([out, out1])
        out = self.block_0(self.relu(self.conv_0(out)))

        out = self.relu(self.up1(out, out0.coordinate_map_key))
        out = ME.cat([out, out0])
        out = self.block_1(self.relu(self.conv_1(out)))

        out = self.relu(self.up2(out, x.coordinate_map_key))
        out = ME.cat([out, out_x])
        out = self.block_2(self.relu(self.conv_2(out)))

        out = self.conv_3(out)

        return [out]


def make_layer(block, block_layers, channels):
    """make stacked InceptionResNet layers.
    """
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))
        
    return torch.nn.Sequential(*layers)

class PCEncoder(nn.Module):
    # Use center points as primitives
    # Encoder color into spherical harmonics
    def __init__(self, args):
        super(PCEncoder, self).__init__()
        self.args = Namespace(**args) # the args
        color_conv_channels = convert_str_2_list(self.args.clr_encoder_channels)

        # calculate the feature dimension
        feat_dim = 0
        if self.args.use_rotation:
            feat_dim += 4
        if self.args.use_scale:
            feat_dim += 3
        if self.args.use_offset:
            feat_dim += 3
        if self.args.use_dc_offset:
            feat_dim += 3
        if self.args.use_opacity:
            feat_dim += 1
        if getattr(self.args, 'est_normal', False):
            feat_dim += 3
        sh_feat_deg = self.args.sh_feat_deg
        if sh_feat_deg > 0:
            feat_dim = feat_dim + (2 ** (sh_feat_deg + 1)) * 3
        if self.args.model_type == 'non_scale':
            raise NotImplementedError('Non-scale encoder not implemented!')
        elif self.args.model_type == 'unet':
            self.color_encoder = SparseUNet(color_conv_channels, feat_dim)
        else:
            raise NotImplementedError(
                f'Model type {self.args.model_type} not implemented!')
        self.interp = ME.MinkowskiInterpolation()
        self.pruning = ME.MinkowskiPruning()
        default_quaternion = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        self.register_buffer('default_quaternion', default_quaternion.unsqueeze(0))

    def forward(self, color_sparse,training=True):
        # The input [points_sparse] should be in shape same as pcgc
        ## PCGC compression
        if color_sparse.C.shape[0] < 100000:
            warnings.warn('The input point cloud contains too few points! There might be a mistake in the data preparation.')

        color_feat_decompressed = self.color_encoder(color_sparse)[0]

        decoded_primitives, decoded_color_feature = color_feat_decompressed.decomposed_coordinates_and_features
        _,dc_color_rgb = color_sparse.decomposed_coordinates_and_features
        # _, dc_color_rgb = dc.decomposed_coordinates_and_features
        dc_color_rgb = [
            dc_color_rgb[i][:, -3:] for i in range(len(dc_color_rgb))
        ]
        used_feat_dim = 0
        decoded_offset = None
        decoded_sh_ac = None
        if self.args.use_rotation:
            decoded_r = [
                decoded_color_feature[i][:, 0:4] + self.default_quaternion for i in range(len(decoded_color_feature))
            ]
            used_feat_dim += 4
        else:
            decoded_r = [
                self.default_quaternion.expand(decoded_color_feature[i].shape[0], 4) for i in range(len(decoded_color_feature))
            ]
        if self.args.use_scale:
            decoded_s = [
                decoded_color_feature[i][:, used_feat_dim:used_feat_dim+3] + torch.ones_like(decoded_color_feature[i][:, used_feat_dim:used_feat_dim+3]) for i in range(len(decoded_color_feature))
            ]
            used_feat_dim += 3
            decoded_s = [
                torch.clamp(decoded_s[i], min=0.) for i in range(len(decoded_s))
            ]
        else:
            decoded_s = [
                torch.ones_like(decoded_color_feature[i][:, 0:3]) for i in range(len(decoded_color_feature))
            ]
        if self.args.use_opacity:
            decoded_o = [
                decoded_color_feature[i][:, used_feat_dim:used_feat_dim+1] for i in range(len(decoded_color_feature))
            ]
            decoded_o = [
                torch.clamp(decoded_o[i], min=0., max=1.) for i in range(len(decoded_o))
            ]
            used_feat_dim += 1
        else:
            decoded_o = [
                torch.ones_like(decoded_color_feature[i][:, 0:1]) for i in range(len(decoded_color_feature))
            ]
        if self.args.use_offset:
            decoded_offset = [
                decoded_color_feature[i][:, used_feat_dim:used_feat_dim+3] for i in range(len(decoded_color_feature))
            ]
            used_feat_dim += 3
        if self.args.use_dc_offset:
            decoded_sh_dc = [
                (decoded_color_feature[i][:, used_feat_dim:used_feat_dim+3] + RGB2SH(dc_color_rgb[i])).unsqueeze(-2) for i in range(len(decoded_color_feature))
            ]
            used_feat_dim += 3
        else:
            decoded_sh_dc = [
                RGB2SH(dc_color_rgb[i]).unsqueeze(-2) for i in range(len(decoded_color_feature))
            ]
        if getattr(self.args, 'est_normal', False):
            decoded_n = [
                decoded_color_feature[i][:, used_feat_dim:used_feat_dim+3] for i in range(len(decoded_color_feature))
            ]
            used_feat_dim += 3
            if getattr(self.args, 'normalize_normal', True):
                decoded_n = [
                    torch.nn.functional.normalize(n, dim=-1) for n in decoded_n
                ]
        else:
            decoded_n = None
        if self.args.sh_deg > 0 and self.args.sh_feat_deg > 0:
            decoded_sh_ac = [
                decoded_color_feature[i][:, used_feat_dim:].reshape(
                    decoded_color_feature[i].shape[0], -1, 3) for i in range(len(decoded_color_feature))
            ]
            decoded_sh = [
                torch.cat([decoded_sh_dc[i], decoded_sh_ac[i]], dim=1) for i in range(len(decoded_sh_dc))
            ]
        elif self.args.sh_deg > 0 and self.args.sh_feat_deg == 0:
            pseudo_sh_dim = (2 ** (self.args.sh_deg + 1)) * 3
            decoded_sh_ac = [
                torch.zeros((decoded_color_feature[i].shape[0], pseudo_sh_dim, 3), device=decoded_color_feature[i].device) for i in range(len(decoded_color_feature))
            ]
            decoded_sh = [
                torch.cat([decoded_sh_dc[i], decoded_sh_ac[i]], dim=1) for i in range(len(decoded_sh_dc))
            ]
        else:
            decoded_sh = decoded_sh_dc
        
        if self.args.use_offset:
            decoded_primitives_aug = [
                decoded_primitives[i] + decoded_offset[i] for i in range(len(decoded_primitives))
            ]
        else:
            decoded_primitives_aug = decoded_primitives
        
        return decoded_primitives_aug, decoded_sh, decoded_r, decoded_s, decoded_o,0, decoded_primitives, decoded_offset, 0, 0, 0, decoded_n