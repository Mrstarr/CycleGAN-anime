import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, nf, normal='instance',stride = 1, padding = True, padding_mode = 'reflect', activation ="relu",transpose = False):     
        super(ConvBlock,self).__init__()
        if padding:
            pad = int((nf - 1)/2)    # keep the size unchanged, 3x3: pad 1   7x7: pad 3
            if not transpose:
                ConvModel = [nn.Conv2d(input_nc, output_nc, nf, stride, pad, padding_mode = padding_mode)]
            else:
                ConvModel = [nn.ConvTranspose2d(input_nc, output_nc, nf, stride, padding=pad, padding_mode = 'zeros',output_padding= pad)]
        else:
            if not transpose:
                ConvModel = [nn.Conv2d(input_nc, output_nc, nf, stride)]
            else:
                ConvModel = [nn.ConvTranspose2d(input_nc, output_nc, nf, stride, output_padding= pad)]

        # normalization block
        if normal is not None:
            if normal == 'instance':
                norm_b = [nn.InstanceNorm2d(output_nc)]
            elif normal == 'batch':
                norm_b = [nn.BatchNorm2d(output_nc)]
            else:
                raise NameError("no such normalization method")

            ConvModel += norm_b

        # activation function
        if activation is not None:
            if activation == 'relu':
                af_b = [nn.ReLU(inplace=True)]
            elif activation == 'lrelu':
                af_b = [nn.LeakyReLU(0.2, inplace=True)]
            elif activation == 'sigmoid':
                af_b = [nn.Sigmoid()]
            elif activation == 'tanh':
                af_b = [nn.Tanh()]
            else:
                raise NameError("no such activation method")

            ConvModel += af_b
        
        self.ConvModel = nn.Sequential(*ConvModel)
    
    def forward(self,x):
        return self.ConvModel(x)


class ResBlock(nn.Module):
    def __init__(self, input_nc):
        super(ResBlock,self).__init__()

        res_b = [ConvBlock(input_nc,input_nc,3,activation ='relu'),
                    ConvBlock(input_nc,input_nc,3,activation = None)]
        self.res_b = nn.Sequential(*res_b)
    

    def forward(self,x):
        return x + self.res_b(x)


class Generator(nn.Module):
    def __init__(self,input_nc, output_nc, n_resblock = 6):
        super(Generator,self).__init__()

        model = [ConvBlock(input_nc,64,7)]
        model += [ConvBlock(64,128,3,stride=2,padding_mode='zeros')]
        model += [ConvBlock(128,256,3,stride=2,padding_mode='zeros')]

        for _ in range(n_resblock):
            model += [ResBlock(256)]

        model += [ConvBlock(256,128,3,stride=2,padding='zeros',transpose=True)]
        model += [ConvBlock(128,64,3,stride=2,padding='zeros',transpose=True)]
        model += [ConvBlock(64,output_nc,7,activation='tanh')]

        self.model = nn.Sequential(*model)
    
    def forward(self,x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator,self).__init__()

        conv1 = ConvBlock(input_nc,64,3,normal = None,stride = 2,padding_mode='zeros',activation='lrelu')
        conv2 = ConvBlock(64,128,3,stride=2,padding_mode='zeros',activation='lrelu') 
        conv3 = ConvBlock(128,256,3,stride=2,padding_mode='zeros',activation='lrelu')
        conv4 = ConvBlock(256,512,3,stride=2,padding_mode='zeros',activation='lrelu')

        fc_layer = nn.Conv2d(512,1,3,padding = 1)
        model = [conv1, conv2, conv3, conv4, fc_layer]
        self.model = nn.Sequential(*model)
    
    def forward(self,x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
