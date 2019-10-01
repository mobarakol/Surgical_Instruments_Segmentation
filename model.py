import torch
import torch.nn as nn
from torchvision.models import resnet
import torch.nn.functional as F

class conv2DBN(nn.Module):
    def __init__( self, inc, outc, k_size, stride, padding, bias=True, dilation=1, with_bn=True,):
        super(conv2DBN, self).__init__()

        if dilation > 1:
            conv2d = nn.Conv2d(int(inc), int(outc), kernel_size=k_size, padding=padding, stride=stride,
                bias=bias, dilation=dilation,)
        else:
            conv2d = nn.Conv2d( int(inc), int(outc), kernel_size=k_size, padding=padding, stride=stride,
                bias=bias, dilation=1,)

        self.cb_unit = nn.Sequential(conv2d, nn.BatchNorm2d(int(outc)))

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class multiresolutionFeatureFusion(nn.Module):
    def __init__( self, n_classes, low_inc, high_inc, outc, with_bn=True ):
        super(multiresolutionFeatureFusion, self).__init__()

        bias = not with_bn

        self.decoder4_x2 = Decoder(512, 128, 3, 2, 1, 1)
        self.conv2_x2 = nn.Sequential(nn.Conv2d(128, 32, 3, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)
        self.mff_class = nn.Sequential( self.decoder4_x2, self.conv2_x2, self.lsm)
        self.high_convbn = conv2DBN(high_inc, outc, 3, stride=1, padding=2, bias=bias, dilation=2, with_bn=with_bn, )
        self.low_convbn = conv2DBN( low_inc, outc, 1, stride=2, padding=0, bias=bias, with_bn=with_bn,)

    def forward(self, x_low, x_high):
        low_fm = self.low_convbn(x_low)
        high_fm = self.high_convbn(x_high)
        high_fused_fm = F.relu(low_fm + high_fm, inplace=True)
        mff_cls = self.mff_class(high_fused_fm)
        return high_fused_fm, mff_cls


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 4, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(in_planes // 4),
                                   nn.ReLU(inplace=True), )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_planes // 4, in_planes // 4, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes // 4, out_planes, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(out_planes),
                                   nn.ReLU(inplace=True), )

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        return x


class spatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes):
        super(spatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        h, w = x.shape[2:]
        k_sizes = []
        strides = []
        for pool_size in self.pool_sizes:
            k_sizes.append((int(h / pool_size), int(w / pool_size)))
            strides.append((int(h / pool_size), int(w / pool_size)))

        pp_sum = x

        for i in range(len(self.pool_sizes)):
            out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
            out = F.upsample(out, size=(h, w), mode="bilinear")
            pp_sum = pp_sum + out

        return pp_sum


class InstrumentsMFF(nn.Module):

    def __init__(self, n_classes=21):
        """
        Initialization
        """
        super(InstrumentsMFF, self).__init__()
        self.spp = spatialPyramidPooling(pool_sizes=[16, 8, 4, 2])
        self.mff_sub24_x2 = multiresolutionFeatureFusion(n_classes, 128, 512, 512, with_bn=True )
        base = resnet.resnet18(pretrained=True)
        self.in_block = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        # decoder
        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)
        self.decoder4_x2 = Decoder(512, 128, 3, 2, 1, 1)

        # Classifier
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True), )
        self.conv2_x2 = nn.Sequential(nn.Conv2d(128, 32, 3, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.deconv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Initial block
        x_size = x.shape  # 3, 1024, 1280
        x1 = self.in_block(x)  # 64, 256, 320
        # Encoder blocks
        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)  # 128, 128, 160
        e3 = self.encoder3(e2)  # 256, 64, 80
        e4 = self.encoder4(e3)  # 512, 32, 40
        e4 = self.spp(e4)  # 512, 32, 40

        # Auxiliary layer
        x2 = F.upsample(x, (int(x_size[2] / 2), int(x_size[3] / 2)), mode='bilinear')  # 3, 512, 640
        x2 = self.in_block(x2)  # 64, 128, 160
        x2_e1 = self.encoder1(x2)  # 64, 128, 160
        x2_e2 = self.encoder2(x2_e1)  # 128, 64, 80
        # MFF
        x_sub24, mff_cls = self.mff_sub24_x2(x2_e2, e4)  # 256, 32, 40
        y2 = x2_e2 + self.decoder4_x2(x_sub24)  # 128 64 80
        y2 = self.conv2_x2(y2)  # 32 64 80
        y2 = self.deconv2(y2)  # 4 128 160

        # Decoder
        d4 = e3 + self.decoder4(x_sub24)  # 256, 64, 80
        d3 = e2 + self.decoder3(d4)  # 128, 128, 160
        d2 = e1 + self.decoder2(d3)  # 64, 256, 320
        d1 = x1 + self.decoder1(d2)  # 64, 256, 320

        # Classifier
        y = self.deconv1(d1)  # 32, 512, 640
        y = self.conv2(y)  # 32, 512, 640
        y = self.deconv2(y)  # 4, 1024, 1280
        # y = self.lsm(y)

        if self.training:
            return y, y2
        else:
            return y