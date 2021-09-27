from ops import *

try:
    from dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')


class Alignnet(nn.Module):
    def __init__(self, filters_in=64, nf=64, groups=8):
        super(Alignnet, self).__init__()
        self.GAMMA = 2.2

        self.L1_downsample_a = nn.Conv2d(filters_in, nf, 3, 2, 1, bias=True)
        self.L2_downsample_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L3_downsample_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.L1_downsample_r = nn.Conv2d(filters_in, nf, 3, 2, 1, bias=True)
        self.L2_downsample_r = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L3_downsample_r = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.L3_get_offset = get_offset(filters=nf)
        self.L2_get_offset = get_offset(filters=nf)
        self.L1_get_offset = get_offset(filters=nf)
        self.L0_get_offset = get_offset(filters=nf)

        self.L2_Dconv = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                            extra_offset_mask=True)
        self.L1_Dconv = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                            extra_offset_mask=True)
        self.L0_Dconv = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                            extra_offset_mask=True)
        self.out_Dconv = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                             extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ali, ref):
        ali_L1 = self.lrelu(self.L1_downsample_a(ali))
        ali_L2 = self.lrelu(self.L2_downsample_a(ali_L1))
        ali_L3 = self.lrelu(self.L3_downsample_a(ali_L2))

        ref_L1 = self.lrelu(self.L1_downsample_r(ref))
        ref_L2 = self.lrelu(self.L2_downsample_r(ref_L1))
        ref_L3 = self.lrelu(self.L3_downsample_r(ref_L2))

        L3_offset = self.L3_get_offset(ali_L3, ref_L3)

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

        output_f = self.relu(self.conv_stage2(torch.cat([img2_1, img2_3], dim=1)))
        output_f = self.res_satge2(output_f)

        output = self.sigmoid(self.conv_out(output_f))
        return output_f, output


class Residual_Estimator(nn.Module):
    def __init__(self, nFeat, nDenselayer, growthRate, filters_out, scale_factor):
        super(Residual_Estimator, self).__init__()

        # F-1
        self.conv_in = nn.Conv2d(nFeat * 3, nFeat, kernel_size=3, padding=1, bias=True)

        # DRDBs 3
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)

        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)
        self.conv_merge = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)

        # upsample
        self.upsample = Upsampler(scale=scale_factor, n_feats=nFeat, act='relu')
        self.conv_out = nn.Conv2d(nFeat, filters_out, kernel_size=3, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, imgs, merge_f):
        F_0 = self.relu(self.conv_in(imgs))
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.relu(self.GFF_1x1(FF))
        merge = self.relu(self.conv_merge(torch.cat([FdLF, merge_f], dim=1)))
        HR = self.upsample(merge)
        residual = self.conv_out(HR)

        return residual


class Deep_SR_HDR(nn.Module):
    def __init__(self, args):
        self.scale_factor = args.scale_factor
        filters_in = args.filters_in
        nReslayer = args.nReslayer
        nDenselayer = args.nDenselayer
        growthRate = args.growthRate
        nFeat = args.nFeat
        filters_out = args.filters_out
        groups = args.groups
        super(Deep_SR_HDR, self).__init__()
        self.GAMMA = 2.2

        self.conv_img1 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv_img2 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv_img3 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)

        self.alignnet1 = Alignnet(filters_in=nFeat, nf=nFeat, groups=groups)
        self.alignnet3 = Alignnet(filters_in=nFeat, nf=nFeat, groups=groups)

        self.mergenet = Mergenet(nFeat=nFeat, nReslayer=nReslayer, filters_out=filters_out)

        self.RE = Residual_Estimator(nFeat=nFeat, nDenselayer=nDenselayer, growthRate=growthRate,
                                     filters_out=filters_out, scale_factor=self.scale_factor)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, imgs):
        img1 = imgs[:, 0:6, :, :]
        img2 = imgs[:, 6:12, :, :]
        img3 = imgs[:, 12:18, :, :]

        # shallow features extraction
        img1 = self.lrelu(self.conv_img1(img1))
        img2 = self.lrelu(self.conv_img2(img2))
        img3 = self.lrelu(self.conv_img3(img3))

        # feature alignment
        img1_aligned = self.alignnet1(img1, img2)
        img3_aligned = self.alignnet3(img3, img2)

        # generate LR_LDR
        merge_f, LR_HDR = self.mergenet(img1_aligned, img2, img3_aligned)

        # extimate residual (extracte high frequency information)
        residual = self.RE(torch.cat([img1_aligned, img2, img3_aligned], dim=1), merge_f)

        # generate HR_LDR
        HR_HDR = F.interpolate(LR_HDR, scale_factor=self.scale_factor, mode='bilinear', align_corners=False) + residual
        return LR_HDR, HR_HDR

