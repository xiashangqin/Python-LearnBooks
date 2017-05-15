import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import torch
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print m.weight.data.size()
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
         norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    print gpu_ids
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(device_id=gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    # cfg = [(64, 4, 2, 1), 'LR', (128, 4, 2, 1), 'B', 'LR', (256, 4, 2, 1), 'B', 'LR', (512, 4, 1, 1), 'B', 'LR', (1, 4, 1, 1), 'R', (1, 30, 1, 0)]
    # layers = create_convnets_D(cfg, input_nc)
    cfg = [(64, 4, 2, 1), 'LR', (128, 4, 2, 1), 'B', 'LR', (256, 4, 2, 1), 'B', 'LR', (512, 4, 1, 1), 'B', 'LR', (1, 4, 1, 1)]
    layers = create_convnets_D(cfg, input_nc)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        #netD = vgg16_bn(num_classes=1)
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, layers=layers)
        #netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, layers=layers)
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    #netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                 norm_layer(ngf, affine=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2), affine=True),
                      nn.ReLU(True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        cnt = 1
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        print '{}:unet_block'.format(cnt)
        print unet_block
        cnt +=1
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            print '{}:unet_block'.format(cnt)
            print unet_block
            cnt +=1
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        print '{}:unet_block'.format(cnt)
        print unet_block
        cnt +=1
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        print '{}:unet_block'.format(cnt)
        print unet_block
        cnt +=1
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        print '{}:unet_block'.format(cnt)
        print unet_block
        cnt +=1
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        print '{}:unet_block'.format(cnt)
        print unet_block
        cnt +=1

        self.model = unet_block
        print self.model.outermost

    def forward(self, input):
        if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], layers=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        # last
        # self.model = nn.ModuleList(layers)
        #v1.9
        self.pre = nn.Sequential(*layers)
        self.feature30 =  nn.Sequential(
            nn.Sigmoid()
        )
        self.feature10 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=3),
            nn.Sigmoid()
        )
        self.feature6 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, stride=5),
            nn.Sigmoid()
        )
        self.feature3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=10, stride=10),
            nn.Sigmoid()
        )
        self.feature2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=15, stride=15),
            nn.Sigmoid()
        )


    def forward(self, input):
        # last
        # x = input
        # for index in range(len(self.model)):
        #     x = self.model[index](x)
        # return x
        # 1.9 
        x = input 
        x = self.pre(x)
        feature30 = self.feature30(x)
        feature10 = self.feature10(x)
        feature6 = self.feature60(x)
        feature3 = self.feature3(x)
        feature2 = self.feature2(x)
        return feature30, feature10, feature6, feature3, feature2


def create_convnets_D(cfg, x_dim = 0, c_dim = 0, batch_norm = False):
    layers = []
    i_dim = x_dim + c_dim
    for v in cfg:
        if v == 'R':
            layers += [nn.ReLU(inplace=True)]
        elif v == 'LR':
            layers += [nn.LeakyReLU(0.2, inplace=True)]
        elif v == 'S':
            layers += [nn.Sigmoid()]
        elif v == 'B':
            layers += [nn.BatchNorm2d(i_dim, affine=True)]
        elif type(v) == tuple:
            o_dim, k, s, p = v
            layers += [nn.Conv2d(i_dim, o_dim, kernel_size=k, stride=s, padding=p, bias=False)]
            i_dim = o_dim
        else:
            if v[-1] == 'd':
                o_dim = int(v[:-1])
                layers += [nn.Linear(i_dim, o_dim, bias=False)]
                i_dim = o_dim + c_dim
            else:
                o_dim = int(v)
                layers += [nn.Linear(900, o_dim, bias=True)]
                i_dim = o_dim
    return layers


# vgg
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 6
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'SR': [(64, 3, 1, 1), 'LR', (64, 3, 2, 1), 'B', 'LR', (128, 3, 1, 1), 'B', 'LR', (128, 3, 2, 1), 'B', 'LR', (256, 3, 1, 1), 'B', 'LR', (256, 3, 2, 1), 'B', 'LR', (512, 3, 1, 1), 'B', 'LR', (512, 3, 2, 1), 'B', 'LR' ]
}

def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    #return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return VGG(nn.Sequential(*create_convnets_D(cfg['SR'], x_dim = 6)), **kwargs)

if __name__  == '__main__':
    # norm_layer = get_norm_layer(norm_type='batch')
    # netG = UnetGenerator(3, 3, 8, 64, norm_layer, True, [])
    # net = vgg16_bn(num_classes=1)
    # print net
    cfg = [(64, 4, 2, 1), 'LR', (128, 4, 2, 1), 'B', 'LR', (256, 4, 2, 1), 'B', 'LR', (512, 4, 1, 1), 'B', 'LR', (1, 4, 1, 1)]
    layers = create_convnets_D(cfg, 3)
    norm_layer = get_norm_layer(norm_type='batch')
    netD = NLayerDiscriminator(3, 64, 3, norm_layer=norm_layer, use_sigmoid=True, gpu_ids=0, layers=layers)
    print netD




    import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.pred_weights = [0.5, 0.05, 0.2, 0.2, 0.05]
        self.gamma = 1
        self.gamma_decay = 0.999
        self.lambda_D = 0.0002
        self.batchSize = opt.batchSize
        self.fineSize = opt.fineSize
        self.nc = opt.output_nc
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.real_B_copy = Variable(self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize))
        self.fake_B_copy = Variable(self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize))

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)
        if self.isTrain:
            #use_sigmoid = opt.no_lsgan
            use_sigmoid = False
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1Pair = torch.nn.PairwiseDistance(1)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    #get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)#.detach()
        self.pred_real30, self.pred_real10, self.pred_real6, self.pred_real3, self.pred_real2 = self.netD.forward(real_AB)
        self.loss_D_real30 = self.criterionGAN(self.pred_real30, True) * self.pred_weights[0]
        self.loss_D_real10 = self.criterionGAN(self.pred_real10, True) * self.pred_weights[1]
        self.loss_D_real6 = self.criterionGAN(self.pred_real6, True) * self.pred_weights[2]
        self.loss_D_real3 = self.criterionGAN(self.pred_real3, True) * self.pred_weights[3]
        self.loss_D_real2 = self.criterionGAN(self.pred_real2, True) * self.pred_weights[4]
        self.loss_D_real = self.loss_D_real30 + self.loss_D_real10 + self.loss_D_real6 + self.loss_D_real3 + self.loss_D_real2
        self.loss_D_real.backward(retain_variables=True)

        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake30, self.pred_fake10, self.pred_fake6, self.pred_fake3, self.pred_fake2 = self.netD.forward(fake_AB.detach())
        self.loss_D_fake30 = self.criterionGAN(self.pred_fake30, False) * self.pred_weights[0]
        self.loss_D_fake10 = self.criterionGAN(self.pred_fake10, False) * self.pred_weights[1]
        self.loss_D_fake6 = self.criterionGAN(self.pred_fake6, False) * self.pred_weights[2]
        self.loss_D_fake3 = self.criterionGAN(self.pred_fake3, False) * self.pred_weights[3]
        self.loss_D_fake2 = self.criterionGAN(self.pred_fake2, False) * self.pred_weights[4]
        self.loss_D_fake = self.loss_D_fake30 + self.loss_D_fake10 + self.loss_D_fake6 + self.loss_D_fake3 + self.loss_D_fake2
        self.loss_D_fake.backward(retain_variables=True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real)*0.5

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake30, pred_fake10, pred_fake6, pred_fake3, pred_fake2 = self.netD.forward(fake_AB)
        self.loss_G_fake30 = self.criterionGAN(pred_fake30, True) * self.pred_weights[0]
        self.loss_G_fake10 = self.criterionGAN(pred_fake10, True) * self.pred_weights[1]
        self.loss_G_fake6 = self.criterionGAN(pred_fake6, True) * self.pred_weights[2]
        self.loss_G_fake3 = self.criterionGAN(pred_fake3, True) * self.pred_weights[3]
        self.loss_G_fake2 = self.criterionGAN(pred_fake2, True) * self.pred_weights[4]
        self.loss_G_GAN  = self.loss_G_fake30 + self.loss_G_fake10 + self.loss_G_fake6 + self.loss_G_fake3 + self.loss_G_fake2

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                ('G_L1', self.loss_G_L1.data[0]),
                ('D_real', self.loss_D_real.data[0]),
                ('D_fake', self.loss_D_fake.data[0])
        ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


            


