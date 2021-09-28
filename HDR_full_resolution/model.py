import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")

try:
    from dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')


class get_offset(nn.Module):
    def __init__(self, filters):
        super(get_offset, self).__init__()
        self.conv_offset1 = nn.Conv2d(filters * 2, filters, 3, 1, 1, bias=True)
        self.conv_offset2 = nn.Conv2d(filters, filters, 3, 1, 1, bias=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ali, ref):
        offset = self.lrule(self.conv_offset1(torch.cat([ali, ref], dim=1)))
        offset = self.lrule(self.conv_offset2(offset))
        return offset


class Alignnet(nn.Module):
    def __init__(self, filters_in=64, nf=64, groups=8):
        super(Alignnet, self).__init__()
        self.GAMMA = 2.2

        self.L1_downsample_a = nn.Conv2d(filters_in, nf, 3, 2, 1, bias=True)
        self.L2_downsample_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L3_downsample_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L4_downsample_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.L1_downsample_r = nn.Conv2d(filters_in, nf, 3, 2, 1, bias=True)
        self.L2_downsample_r = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L3_downsample_r = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L4_downsample_r = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.L4_get_offset = get_offset(filters=nf)
        self.L3_get_offset = get_offset(filters=nf)
        self.L2_get_offset = get_offset(filters=nf)
        self.L1_get_offset = get_offset(filters=nf)
        self.L0_get_offset = get_offset(filters=nf)

        self.L3_Dconv = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                            extra_offset_mask=True)
        self.L2_Dconv = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                            extra_offset_mask=True)
        self.L1_Dconv = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                            extra_offset_mask=True)
        self.L0_Dconv = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                            extra_offset_mask=True)
        self.out_Dconv = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                            extra_offset_mask=True)

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ali, ref):
        ali_L1 = self.lrelu(self.L1_downsample_a(ali))
        ali_L2 = self.lrelu(self.L2_downsample_a(ali_L1))
        ali_L3 = self.lrelu(self.L3_downsample_a(ali_L2))
        ali_L4 = self.lrelu(self.L4_downsample_a(ali_L3))

        ref_L1 = self.lrelu(self.L1_downsample_r(ref))
        ref_L2 = self.lrelu(self.L2_downsample_r(ref_L1))
        ref_L3 = self.lrelu(self.L3_downsample_r(ref_L2))
        ref_L4 = self.lrelu(self.L4_downsample_r(ref_L3))

        L4_offset = self.L4_get_offset(ali_L4, ref_L4)

        L3_offset = F.interpolate(L4_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L3_aligned = self.L3_Dconv([ali_L3, L3_offset])
        L3_offset = self.L3_get_offset(L3_aligned, ref_L3) + L3_offset

        L2_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_aligned = self.L2_Dconv([ali_L2, L2_offset])
        L2_offset = self.L2_get_offset(L2_aligned, ref_L2) + L2_offset

        L1_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_aligned = self.L1_Dconv([ali_L1, L1_offset])
        L1_offset = self.L1_get_offset(L1_aligned, ref_L1) + L1_offset

        L0_offset = F.interpolate(L1_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L0_aligned = self.L0_Dconv([ali, L0_offset])
        L0_offset = self.L0_get_offset(L0_aligned, ref) + L0_offset

        output = self.out_Dconv([ali, L0_offset])
        return output


class make_res(nn.Module):
    def __init__(self, nFeat, kernel_size=3):
        super(make_res, self).__init__()
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out) + x
        return out


class Res_block(nn.Module):
    def __init__(self, nFeat, nReslayer):
        super(Res_block, self).__init__()
        modules = []
        for i in range(nReslayer):
            modules.append(make_res(nFeat))
        self.dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        out = self.dense_layers(x)
        return out


class Mergenet(nn.Module):
    def __init__(self, nFeat=64, nReslayer=3, filters_out=3):
        super(Mergenet, self).__init__()
        # fusion1
        self.conv_in1 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv_in3 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.res1 = Res_block(nFeat, nReslayer)
        self.res3 = Res_block(nFeat, nReslayer)

        # fusion2
        self.conv_stage2 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.res_satge2 = Res_block(nFeat, nReslayer)
        self.conv_out = nn.Conv2d(nFeat, filters_out, kernel_size=3, padding=1, bias=True)

        # activation
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img1_aligned, img2, img3_aligned):
        img2_1 = self.relu(self.conv_in1(torch.cat([img1_aligned, img2], dim=1)))
        img2_3 = self.relu(self.conv_in3(torch.cat([img3_aligned, img2], dim=1)))

        img2_1 = self.res1(img2_1) + img2
        img2_3 = self.res3(img2_3) + img2

        output = self.relu(self.conv_stage2(torch.cat([img2_1, img2_3], dim=1)))
        output = self.res_satge2(output)
        output = self.sigmoid(self.conv_out(output))
        return output


class HDR(nn.Module):
    def __init__(self, args):
        filters_in = args.filters_in
        nReslayer = args.nReslayer
        nFeat = args.nFeat
        filters_out = args.filters_out
        groups = args.groups
        self.args = args
        super(HDR, self).__init__()
        self.GAMMA = 2.2

        self.conv_img1 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv_img2 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv_img3 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)

        self.alignnet1 = Alignnet(filters_in=nFeat, nf=nFeat, groups=groups)
        self.alignnet3 = Alignnet(filters_in=nFeat, nf=nFeat, groups=groups)
        self.mergenet = Mergenet(nFeat=nFeat, nReslayer=nReslayer, filters_out=filters_out)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, imgs):
        img1 = imgs[:, 0:6, :, :]
        img2 = imgs[:, 6:12, :, :]
        img3 = imgs[:, 12:18, :, :]

        img1 = self.lrelu(self.conv_img1(img1))
        img2 = self.lrelu(self.conv_img2(img2))
        img3 = self.lrelu(self.conv_img3(img3))

        img1_aligned = self.alignnet1(img1, img2)
        img3_aligned = self.alignnet3(img3, img2)
        out = self.mergenet(img1_aligned, img2, img3_aligned)
        return out

