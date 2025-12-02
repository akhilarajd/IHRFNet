import torch
import torch.nn as nn
import torch.nn.functional as F


from hsab import HSAB



class Up(nn.Module):

    def __init__(self, nc, bias):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=nc, out_channels=nc, kernel_size=2, stride=2, bias=bias)

    def forward(self, x1, x):
        x2 = self.up(x1)

        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x3


## Spatial Attention
class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        groups = 1
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


## Channel Attention Layer
class CAB(nn.Module):
    def __init__(self, nc, reduction=8, bias=False):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ERAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(ERAB, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        layers = []
        #layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        self.res = nn.Sequential(*layers)
        self.sab = SAB()

    def forward(self, x):
        x1 = x + self.res(x)
        x2 = x1 + self.res(x1)
        x3 = x2 + self.res(x2)

        x3_1 = x1 + x3
        x4 = x3_1 + self.res(x3_1)
        x4_1 = x + x4

        x5 = self.sab(x4_1)
        x5_1 = x + x5

        return x5_1


class HDRAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(HDRAB, self).__init__()
        kernel_size = 3
        reduction = 8

        self.cab = CAB(in_channels, reduction, bias)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)

        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv_tail = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)

    def forward(self, y):
        y1 = self.conv1(y)
        y1_1 = self.relu1(y1)
        y2 = self.conv2(y1_1)
        y2_1 = y2 + y

        y3 = self.conv3(y2_1)
        y3_1 = self.relu3(y3)
        y4 = self.conv4(y3_1)
        y4_1 = y4 + y2_1

        y5 = self.conv3_1(y4_1)
        y5_1 = self.relu3_1(y5)
        y6 = self.conv2_1(y5_1+y3)
        y6_1 = y6 + y4_1

        y7 = self.conv1_1(y6_1+y2_1)
        y7_1 = self.relu1_1(y7)
        y8 = self.conv_tail(y7_1+y1)
        y8_1 = y8 + y6_1

        y9 = self.cab(y8_1)
        y9_1 = y + y9

        return y9_1
        



class IHRFNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, bias=True):
        super(IHRFNet, self).__init__()
        kernel_size = 3
        
        self.hsab = HSAB(in_nc, num_of_layers=15)
        
       

        self.conv_head = nn.Conv2d(in_nc, nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.erab = ERAB(nc, nc, bias)

        self.hdrab = HDRAB(nc, nc, bias)

        self.conv_tail = nn.Conv2d(nc, out_nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.dual_tail = nn.Conv2d(2*out_nc, out_nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.down = nn.Conv2d(nc, nc, kernel_size=2, stride=2, bias=bias)

        self.up = Up(nc, bias)
        self.red_chan = torch.nn.Conv2d(9, 6, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
    	
    	
        hsab = self.hsab(x)
        #print('hsab',hsab.shape)
       
        x1 = self.conv_head(x)

        #print('ihrfnet_input',x.shape)
        #print('ihrfnet_conv_head',x1.shape)
        x2 = self.erab(x1)
        #print('ihrfnet_ERAB1',x2.shape)
        x2_1 = self.down(x2)
        #print('ihrfnet_down1',x2_1.shape)
        x3 = self.erab(x2_1)
        #print('ihrfnet_ERAB2',x3.shape)
        x3_1 = self.down(x3)
        #print('ihrfnet_down2',x3_1.shape)
        x4 = self.erab(x3_1)
        #print('ihrfnet_ERAB3',x4.shape)
        x4_1 = self.up(x4, x3)
        #print('ihrfnet_up1',x4_1.shape)
        x5 = self.erab(x4_1 + x3)
        #print('ihrfnet_ERAB4',x5.shape)
        x5_1 = self.up(x5, x2)
        #print('ihrfnet_up2',x5_1.shape)
        x6 = self.erab(x5_1 + x2)
        #print('ihrfnet_ERAB5',x6.shape)
        x7 = self.conv_tail(x6 + x1)
        #print('ihrfnet_conv_tail',x2.shape)
        X = x - x7
        #print('ihrfnet_ERAB_out',X.shape)

        y1 = self.conv_head(x)
        #print('ihrfnet_conv_head',y1.shape)
        y2 = self.hdrab(y1)
        #print('ihrfnet_HDRAB1',y2.shape)
        c1 = torch.cat([x1, y1], dim=1)
        #print('ihrfnet_HDRAB_RAB_CAT1',c1.shape)
        y3 = self.hdrab(y2)
        #print('ihrfnet_HDRAB2',y3.shape)
        y4 = self.hdrab(y3)
        #print('ihrfnet_HDRAB3',y4.shape)
        y5 = self.hdrab(y4 + y3)
        #print('ihrfnet_HDRAB4',y5.shape)
        y6 = self.hdrab(y5 + y2)
        #print('ihrfnet_HDRAB5',y6.shape)
        y7 = self.conv_tail(y6 + y1)
        #print('ihrfnet_conv_tail_HDRAB',y7.shape)
        Y = x -y7
        #print('ihrfnet_HDRAB_out',Y.shape)
        
       
        
        
        # Concatenate outputs from HSAB and IHRFNet intermediate outputs
        z1 = torch.cat([X, Y, hsab], dim=1)
        #print('cat',z1.shape)
        
        z1 = self.red_chan(z1)
        #print('red',z1.shape)
        
        z = self.dual_tail(z1)
        #print('dual_tail',z.shape)
       
        Z = x - z
        #print('out',Z.shape)
        return Z
'''
        z1 = torch.cat([X, Y], dim=1)
        print('ihrfnet_HDRAB_ERAB_OUT_cat',z1.shape)
        z = self.dual_tail(z1)
        print('ihrfnet_dual_tail',z.shape)
        Z = x - z
        print('ihrfnet_out',Z.shape)
'''

       
