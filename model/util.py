import torch

from torch import nn


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, local_info=True):
        super(SELayer, self).__init__()
        self.local_info = local_info
        if (self.local_info == True):
            self.avg_pool = nn.AdaptiveAvgPool2d(3)
            self.max_pool = nn.AdaptiveMaxPool2d(3)
            self.pool_conv = nn.Sequential(
                nn.Conv2d(channel,channel,3,1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            # torch.sigmoid()
        )


    def forward(self, x):
        b, c, _, _ = x.size()
        
        if (self.local_info == True):
            y_avg = self.avg_pool(x)
            y_max = self.max_pool(x)
            y_avg = self.pool_conv(y_avg).view(b, c)
            y_max = self.pool_conv(y_max).view(b, c)
        else:
            y_avg = self.avg_pool(x).view(b, c)
            y_max = self.max_pool(x).view(b, c)

        y_avg = torch.sigmoid(self.fc(y_avg)).view(b, c, 1, 1)
        y_max = torch.sigmoid(self.fc(y_max)).view(b, c, 1, 1)

        y_mix = y_avg + y_max

        return x * y_mix.expand_as(x)

class Spatial_att(nn.Module):
    def __init__(self, in_channel, inter_channle, pooling_size, fc_reduce, SE_local_info, out_channel):
        super(Spatial_att, self).__init__()

        self.per_in = Conv_Bn_Activation(in_channel, inter_channle,1,1,'relu')
        self.max1d=nn.AdaptiveMaxPool1d(pooling_size)
        self.avg1d=nn.AdaptiveAvgPool1d(pooling_size) 
        self.sp_conv = Conv_Bn_Activation(pooling_size*2,1,1,1,'relu')
        # self.sig = torch.sigmoid()
        self.se = SELayer(pooling_size*2, reduction=fc_reduce, local_info=SE_local_info)
        self.per_out = Conv_Bn_Activation(inter_channle,out_channel,1,1,'relu')

    def forward(self,x):
        x_sp = self.per_in(x)
        x_1 = x_sp.view(x_sp.size(0), x_sp.size(1), -1) # B,C,H,W--> B,C,HxW
        x_perm = x_1.permute(0,2,1) #B,HxW,C
        x_max = self.max1d(x_perm)
        x_avg = self.avg1d(x_perm)
        x_ma = torch.cat((x_max, x_avg),2) #B,HxW,2C
        x_maori = x_ma.view(x_sp.size(0), x_sp.size(2),x_sp.size(3),-1) #B,H,W,2C
        x_maori = x_maori.permute(0,3,1,2) #B,2C,H,W
        x_maori = self.se(x_maori)
        # x_spatt = self.sig(self.sp_conv(x_maori))
        x_spatt = torch.sigmoid(self.sp_conv(x_maori))
        x = x_sp*x_spatt.expand_as(x_sp)
        x = self.per_out(x)
        return x

class CCAM(nn.Module):
    def __init__(self, channels):
        super(CCAM, self).__init__()
        self.cross_region = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding = 3, dilation=3),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.adjacent_region = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding = 3, dilation=3),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.g_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        cross_region = self.cross_region(x)
        adjacent_region = self.adjacent_region(x)
        cat_g = torch.cat([cross_region, adjacent_region], dim=1)
        x = self.g_conv(cat_g)
        return x


