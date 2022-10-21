
from .Model_parts_upload import *

class Cascade_UNet(nn.Module):
    """
    The Cascade U model presented for pv/d WMHs segmentation
    """
    def __init__(self, n_channels, n_classes, bilinear = True):
        super(Cascade_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        C = 64
        filter_seg = [C, 2 * C, 4 * C, 8 * C, 16 * C]
        filter_dif = [C // 2, C, 2 * C, 4 * C, 8 * C]
        """
        Segmentation model
        """
        self.inc = DoubleConv(n_channels, filter_seg[0])
        self.down1 = Down(filter_seg[0], filter_seg[1])
        self.down2 = Down(filter_seg[1], filter_seg[2])
        self.down3 = Down(filter_seg[2], filter_seg[3])
        self.down4 = Down(filter_seg[3], filter_seg[4])
        self.up1 = Up(filter_seg[4] + filter_seg[3], filter_seg[3], bilinear = bilinear)
        self.up2 = Up(filter_seg[3] + filter_seg[2], filter_seg[2], bilinear = bilinear)
        self.up3 = Up(filter_seg[2] + filter_seg[1], filter_seg[1], bilinear = bilinear)
        self.up4 = Up(filter_seg[1] + filter_seg[0], filter_seg[0], bilinear = bilinear)
        self.outc = OutConv(filter_seg[0], 2)

        """
        Differentiation model
        """
        self.inc2 = DoubleConv(2, filter_dif[0])
        self.down11 = Down(filter_dif[0], filter_dif[1])
        self.down22 = Down(filter_dif[1], filter_dif[2])
        self.down33 = Down(filter_dif[2], filter_dif[3])
        self.down44 = Down(filter_dif[3], filter_dif[4])
        self.up11 = Up(filter_dif[4] + filter_dif[3], filter_dif[3], bilinear = bilinear)
        self.up22 = Up(filter_dif[3] + filter_dif[2], filter_dif[2], bilinear = bilinear)
        self.up33 = Up(filter_dif[2] + filter_dif[1], filter_dif[1], bilinear = bilinear)
        self.up44 = Up(filter_dif[1] + filter_dif[0], filter_dif[0], bilinear = bilinear)
        self.outc2 =  OutConv(filter_dif[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_seg = self.up1(x5, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x1)
        out_seg = self.outc(x_seg)
        logit = torch.softmax(out_seg, dim = 1)
        new_x = torch.cat([x,logit[:,1,:,:].unsqueeze(1)], dim = 1)
        cx1 = self.inc2(new_x)
        cx2 = self.down11(cx1)
        cx3 = self.down22(cx2)
        cx4 = self.down33(cx3)
        cx5 = self.down44(cx4)
        x_dif = self.up11(cx5, cx4)
        x_dif = self.up22(x_dif, cx3)
        x_dif = self.up33(x_dif, cx2)
        x_dif = self.up44(x_dif, cx1)
        out_dif = self.outc2(x_dif)
        return out_seg, out_dif


class Pipleline_UNet_seg(nn.Module):
    """
    The segmentation model of Pipeline U. This model was trained to segment WMHs without differentiation
    """
    def __init__(self, n_channels=1, n_classes=2, bilinear=True):
        super(Pipleline_UNet_seg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        C = 64
        filter_seg = [C, 2 * C, 4 * C, 8 * C, 16 * C]
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(filter_seg[0], filter_seg[1])
        self.down2 = Down(filter_seg[1], filter_seg[2])
        self.down3 = Down(filter_seg[2], filter_seg[3])
        self.down4 = Down(filter_seg[3], filter_seg[4])
        self.up1 = Up(filter_seg[4] + filter_seg[3], filter_seg[3], bilinear = bilinear)
        self.up2 = Up(filter_seg[3] + filter_seg[2], filter_seg[2], bilinear = bilinear)
        self.up3 = Up(filter_seg[2] + filter_seg[1], filter_seg[1], bilinear = bilinear)
        self.up4 = Up(filter_seg[1] + filter_seg[0], filter_seg[0], bilinear = bilinear)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out


class Pipleline_UNet_dif(nn.Module):
    """
    The differentiation model of Pipeline U. The combination of FLAIR image and the prediction of  'Pileline_UNet_seg'
    was fed into this model for pv/d WMHs differentiation
    """
    def __init__(self, n_channels = 2, n_classes = 2, bilinear = True):
        super(Pipleline_UNet_dif, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        C = 32
        filter_dif = [C, 2 * C, 4 * C, 8 * C, 16 * C]
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(filter_dif[0], filter_dif[1])
        self.down2 = Down(filter_dif[1], filter_dif[2])
        self.down3 = Down(filter_dif[2], filter_dif[3])
        self.down4 = Down(filter_dif[3], filter_dif[4])
        self.up1 = Up(filter_dif[4] + filter_dif[3], filter_dif[3],bilinear = bilinear)
        self.up2 = Up(filter_dif[3] + filter_dif[2], filter_dif[2], bilinear = bilinear)
        self.up3 = Up(filter_dif[2] + filter_dif[1], filter_dif[1], bilinear = bilinear)
        self.up4 = Up(filter_dif[1] + filter_dif[0], filter_dif[0], bilinear = bilinear)
        self.outc = nn.Conv2d(32, n_classes, kernel_size = 1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

class Separate_UNet(nn.Module):
    """
     Two Separate U models were trained to segment pvWMHs and dWMHs, respectively
    """
    def __init__(self, n_channels=1, n_classes=2, bilinear=True):
        super(Separate_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        C = 32
        filter_seg = [C, 2 * C, 4 * C, 8 * C, 16 * C]
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(filter_seg[0], filter_seg[1])
        self.down2 = Down(filter_seg[1], filter_seg[2])
        self.down3 = Down(filter_seg[2], filter_seg[3])
        self.down4 = Down(filter_seg[3], filter_seg[4])
        self.up1 = Up(filter_seg[4] + filter_seg[3], filter_seg[3], bilinear=bilinear)
        self.up2 = Up(filter_seg[3] + filter_seg[2], filter_seg[2], bilinear=bilinear)
        self.up3 = Up(filter_seg[2] + filter_seg[1], filter_seg[1], bilinear=bilinear)
        self.up4 = Up(filter_seg[1] + filter_seg[0], filter_seg[0], bilinear=bilinear)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out
