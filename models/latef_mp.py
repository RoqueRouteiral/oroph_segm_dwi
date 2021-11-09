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

class Xlate_mp(nn.Module):
    def __init__(self, in_channel, n_classes=1):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(Xlate_mp, self).__init__()
        
        #Downsampling path A
        self.ec0_a = self.conv_block(1, 16, padding=1)
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
        
        #Downsampling path C
        self.ec0_c = self.conv_block(1, 16, padding=1)
        self.ec1_c = self.conv_block(16, 32, padding=1)
        self.pool0_c = nn.MaxPool3d(2, stride=2)
        self.ec2_c = self.conv_block(32, 32, padding=1)
        self.ec3_c = self.conv_block(32, 64, padding=1)
        self.pool1_c = nn.MaxPool3d(2, stride=2)
        self.ec4_c = self.conv_block(64, 64, padding=1)
        self.ec5_c = self.conv_block(64, 128, padding=1)
        self.pool2_c = nn.MaxPool3d(2, stride=2)
        
        #Downsampling path D
        self.ec0_d = self.conv_block(1, 16, padding=1)
        self.ec1_d = self.conv_block(16, 32, padding=1)
        self.pool0_d = nn.MaxPool3d(2, stride=2)
        self.ec2_d = self.conv_block(32, 32, padding=1)
        self.ec3_d = self.conv_block(32, 64, padding=1)
        self.pool1_d = nn.MaxPool3d(2, stride=2)
        self.ec4_d = self.conv_block(64, 64, padding=1)
        self.ec5_d = self.conv_block(64, 128, padding=1)
        self.pool2_d = nn.MaxPool3d(2, stride=2)
        
        #Bottleneck
        self.ec6 = self.conv_block(512, 128, padding=1, dropout=True)
        self.ec7 = self.conv_block(128, 256, padding=1, dropout=True)
        
        #Upsampling path Seg
        self.dc9_seg = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc8_seg = self.conv_block(384 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7_seg = self.conv_block(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6_seg = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc5_seg = self.conv_block(192 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4_seg = self.conv_block(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3_seg = self.decoder(64, 64, kernel_size=2, stride=2, bias=False)
        self.dc2_seg = self.conv_block(96 + 64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1_seg = self.conv_block(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        #Upsampling path Det
        self.dc9_det = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc8_det = self.conv_block(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7_det = self.conv_block(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6_det = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc5_det = self.conv_block(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4_det = self.conv_block(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3_det = self.decoder(64, 64, kernel_size=2, stride=2, bias=False)
        self.dc2_det = self.conv_block(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1_det = self.conv_block(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
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
        

    def forward(self, a, b,c,d):
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
        
        #Third encoder
        e0_c = self.ec0_c(c)
        syn0_c = self.ec1_c(e0_c)
        e1_c = self.pool0_c(syn0_c)
        e2_c = self.ec2_c(e1_c)
        syn1_c = self.ec3_c(e2_c)

        del e0_c, e1_c, e2_c

        e3_c = self.pool1_c(syn1_c)
        e4_c = self.ec4_c(e3_c)
        syn2_c = self.ec5_c(e4_c)
        del e3_c, e4_c

        e5_c = self.pool2_c(syn2_c)
        
        #Forth encoder
        e0_d = self.ec0_d(d)
        syn0_d = self.ec1_d(e0_d)
        e1_d = self.pool0_d(syn0_d)
        e2_d = self.ec2_d(e1_d)
        syn1_d = self.ec3_d(e2_d)
        
        del e0_d, e1_d, e2_d

        e3_d = self.pool1_d(syn1_d)
        e4_d = self.ec4_d(e3_d)
        syn2_d = self.ec5_d(e4_d)
        del e3_d, e4_d
        
        e5_d = self.pool2_d(syn2_d)  
        
        #Bottleneck
        e6 = self.ec6(torch.cat((e5_a, e5_b, e5_c, e5_d), dim=1)) ## concat here
        e7 = self.ec7(e6)
        
        del e5_a, e5_b, e5_c, e5_d, e6
        
        # Pathway A - Decoder
        d9_seg = torch.cat((self.dc9_seg(e7), syn2_a, syn2_b, syn2_c), dim=1)
#        print((self.dc9_seg(e7).shape, syn2_a.shape, syn2_b.shape, syn2_c.shape))
        del syn2_a, syn2_b, syn2_c

        d8_seg = self.dc8_seg(d9_seg)
        d7_seg = self.dc7_seg(d8_seg)
        del d9_seg, d8_seg

        d6_seg = torch.cat((self.dc6_seg(d7_seg), syn1_a, syn1_b, syn1_c), dim=1)
        del d7_seg, syn1_a, syn1_b, syn1_c

        d5_seg = self.dc5_seg(d6_seg)
        d4_seg = self.dc4_seg(d5_seg)
        del d6_seg, d5_seg

        d3_seg = torch.cat((self.dc3_seg(d4_seg), syn0_a, syn0_b, syn0_c), dim=1)
        del d4_seg, syn0_a, syn0_b, syn0_c

        d2_seg = self.dc2_seg(d3_seg)
        d1_seg = self.dc1_seg(d2_seg)
        del d3_seg, d2_seg
        
        #Pathway B
        d9_det = torch.cat((self.dc9_det(e7), syn2_d), dim=1)
        del e7, syn2_d

        d8_det = self.dc8_det(d9_det)
        d7_det = self.dc7_det(d8_det)
        del d9_det, d8_det

        d6_det = torch.cat((self.dc6_det(d7_det), syn1_d), dim=1)
        del d7_det, syn1_d

        d5_det = self.dc5_det(d6_det)
        d4_det = self.dc4_det(d5_det)
        del d6_det, d5_det

        d3_det = torch.cat((self.dc3_det(d4_det), syn0_d), dim=1)
        del d4_det, syn0_d

        d2_det = self.dc2_det(d3_det)
        d1_det = self.dc1_det(d2_det)
        del d3_det, d2_det

        d1 = torch.cat((d1_seg,d1_det),dim=1)
        d0 = self.final(d1)
        return d0