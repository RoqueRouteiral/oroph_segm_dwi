import torch
import torch.nn as nn
import torch.nn.functional as F
import torch 
import torch.nn as nn

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class YNet(nn.Module):
    def __init__(self, in_channel, n_classes=1):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(YNet, self).__init__()
        
        #Downsampling path A
        self.ec0_a = self.conv_block(3, 16, padding=1)
        self.ec1_a = self.conv_block(16, 32, padding=1)
        self.pool0_a = nn.MaxPool3d(2, stride=2)
        self.ec2_a = self.conv_block(32, 32, padding=1)
        self.ec3_a = self.conv_block(32, 64, padding=1)
        self.pool1_a = nn.MaxPool3d(2, stride=2)
        self.ec4_a = self.conv_block(64, 64, padding=1)
        self.ec5_a = self.conv_block(64, 128, padding=1)
        self.pool2_a = nn.MaxPool3d(2, stride=2)
        
        #Downsampling path B
        self.ec0_b = self.conv_block(1, 16, padding=1)
        self.ec1_b = self.conv_block(16, 32, padding=1)
        self.pool0_b = nn.MaxPool3d(2, stride=2)
        self.ec2_b = self.conv_block(32, 32, padding=1)
        self.ec3_b = self.conv_block(32, 64, padding=1)
        self.pool1_b = nn.MaxPool3d(2, stride=2)
        self.ec4_b = self.conv_block(64, 64, padding=1)
        self.ec5_b = self.conv_block(64, 128, padding=1)
        self.pool2_b = nn.MaxPool3d(2, stride=2)
        
        #Bottleneck
        self.ec6 = self.conv_block(256, 128, padding=1, dropout=True)
        self.ec7 = self.conv_block(128, 256, padding=1, dropout=True)

        #Upsampling path
        self.dc9 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.conv_block(256 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.conv_block(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.conv_block(128 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.conv_block(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(64, 64, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.conv_block(64 + 64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.conv_block(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.final = self.finalLayer(32, n_classes, kernel_size=1, stride=1, bias=False)
        

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=True,dropout=False):
        if dropout:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Dropout3d(p=0.2),
                ContBatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                #nn.Dropout3d(p=0.2),
                ContBatchNorm3d(out_channels),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias))
        return layer
       
    
    def finalLayer(self, in_channels, out_channels, kernel_size, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Sigmoid())
        return layer 
        

    def forward(self, a, b):
        #First encoder
        e0_a = self.ec0_a(a)
        syn0_a = self.ec1_a(e0_a)
        e1_a = self.pool0_a(syn0_a)
        e2_a = self.ec2_a(e1_a)
        syn1_a = self.ec3_a(e2_a)

        del e0_a, e1_a, e2_a

        e3_a = self.pool1_a(syn1_a)
        e4_a = self.ec4_a(e3_a)
        syn2_a = self.ec5_a(e4_a)
        del e3_a, e4_a

        e5_a = self.pool2_a(syn2_a)
        
        #Second encoder
        e0_b = self.ec0_b(b)
        syn0_b = self.ec1_b(e0_b)
        e1_b = self.pool0_b(syn0_b)
        e2_b = self.ec2_b(e1_b)
        syn1_b = self.ec3_b(e2_b)

        del e0_b, e1_b, e2_b

        e3_b = self.pool1_b(syn1_b)
        e4_b = self.ec4_b(e3_b)
        syn2_b = self.ec5_b(e4_b)
        del e3_b, e4_b
        e5_b = self.pool2_b(syn2_b)  
        
        #Bottleneck
        e6 = self.ec6(torch.cat((e5_a, e5_b), dim=1)) ## concat here
        e7 = self.ec7(e6)
        del e5_a, e5_b, e6
        # Shared decoder
        d9 = torch.cat((self.dc9(e7), syn2_a, syn2_b), dim=1)
        del e7, syn2_a, syn2_b
        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1_a, syn1_b), dim=1)
        del d7, syn1_a, syn1_b

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0_a, syn0_b), dim=1)
        del d4, syn0_a, syn0_b
        
        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.final(d1)
        del d1
        return d0