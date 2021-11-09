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

class Xlate(nn.Module):
    def __init__(self, in_channel, n_classes=1):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(Xlate, self).__init__()
        
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

        #Upsampling path A
        self.dc9_a = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc8_a = self.conv_block(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7_a = self.conv_block(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6_a = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc5_a = self.conv_block(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4_a = self.conv_block(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3_a = self.decoder(64, 64, kernel_size=2, stride=2, bias=False)
        self.dc2_a = self.conv_block(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1_a = self.conv_block(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        #Upsampling path B
        self.dc9_b = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc8_b = self.conv_block(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7_b = self.conv_block(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6_b = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc5_b = self.conv_block(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4_b = self.conv_block(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3_b = self.decoder(64, 64, kernel_size=2, stride=2, bias=False)
        self.dc2_b = self.conv_block(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1_b = self.conv_block(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
       #Final layer 
        self.final = self.finalLayer(64, n_classes, kernel_size=1, stride=1, bias=False)

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
        #print(x.size())
        
        # Pathway A - Encoder        
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
        
        #Pathway B - Encoder        
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
        
        # Bottleneck
        e5_cat = torch.cat((e5_a, e5_b), dim=1)
        e6 = self.ec6(e5_cat)
        e7 = self.ec7(e6)
        del e5_a, e5_b, e6
        
        # Pathway A - Decoder
        d9_a = torch.cat((self.dc9_a(e7), syn2_a), dim=1)
        del syn2_a

        d8_a = self.dc8_a(d9_a)
        d7_a = self.dc7_a(d8_a)
        del d9_a, d8_a

        d6_a = torch.cat((self.dc6_a(d7_a), syn1_a), dim=1)
        del d7_a, syn1_a

        d5_a = self.dc5_a(d6_a)
        d4_a = self.dc4_a(d5_a)
        del d6_a, d5_a

        d3_a = torch.cat((self.dc3_a(d4_a), syn0_a), dim=1)
        del d4_a, syn0_a

        d2_a = self.dc2_a(d3_a)
        d1_a = self.dc1_a(d2_a)
        del d3_a, d2_a
        
        #Pathway B
        d9_b = torch.cat((self.dc9_b(e7), syn2_b), dim=1)
        del e7, syn2_b

        d8_b = self.dc8_b(d9_b)
        d7_b = self.dc7_b(d8_b)
        del d9_b, d8_b

        d6_b = torch.cat((self.dc6_b(d7_b), syn1_b), dim=1)
        del d7_b, syn1_b

        d5_b = self.dc5_b(d6_b)
        d4_b = self.dc4_b(d5_b)
        del d6_b, d5_b

        d3_b = torch.cat((self.dc3_b(d4_b), syn0_b), dim=1)
        del d4_b, syn0_b

        d2_b = self.dc2_b(d3_b)
        d1_b = self.dc1_b(d2_b)
        del d3_b, d2_b

        d1 = torch.cat((d1_a,d1_b),dim=1)

        d0 = self.final(d1)
        return d0