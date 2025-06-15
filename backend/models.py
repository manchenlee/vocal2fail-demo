# SOURCE: https://github.com/mingyuliutw/UNIT

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch.autograd import Variable
import numpy as np

import activations
from san_modules import SANConv2d
from utils import init_weights, get_padding
from alias_free_torch import *

#set_seed(1111)

LRELU_SLOPE = 0.1

class AMPBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()
        self.h = h

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None):
        super(AMPBlock2, self).__init__()
        self.h = h

        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs) # total number of conv layers

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        for c, a in zip (self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class BigVSAN(torch.nn.Module):
    # this is our main BigVSAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, h):
        super(BigVSAN, self).__init__()
        self.h = h

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # pre conv
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        # define which AMPBlock to use. BigVSAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == '1' else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                            h.upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2))
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation))

        # post conv
        if h.activation == "snake": # periodic nonlinearity with snake function and anti-aliasing
            activation_post = activations.Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif h.activation == "snakebeta": # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x
    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Encoder_qian(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2, pitch_emb_dim=None):
        super(Encoder, self).__init__()
        self.pitches_embed_dim = 0
        if pitch_emb_dim is not None:
            self.pitches_embed_dim = pitch_emb_dim
        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(4):
            layers += [ResidualBlock(dim)]

        self.model_blocks = nn.Sequential(*layers)

    def reparameterization(self, mu):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return z + mu

    def forward(self, x, p=None):
        if p is not None:
            #pitches_embed = _get_pitches(p, self.pitches_embed_dim)
            #print(pitches_embed.shape)
            #pitches_embed = 
            pitches_embed_c = p.view(p.size(0), 1, p.size(1), p.size(2))
            #pitches_embed_c = pitches_embed.repeat(1, 1, x.shape[2], 1)
            #pitches_embed_c = pitches_embed_c.repeat(1, , x.shape[2], 1)
            x = torch.cat((x, pitches_embed_c), 2)
            #print(x.shape)
        #print(x.shape)
        mu = self.model_blocks(x)
        z = self.reparameterization(mu)
        #if p is not None:
        #    return mu, z, pitches_embed
        #else:
        return mu, z

class Encoder_lstm(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2):
        super(Encoder_lstm, self).__init__()
        #self.use_pitch_emb = use_pitch_emb
        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.conv_blocks = nn.Sequential(*layers)
        self.blstm = nn.LSTM(800, 48, batch_first=True, bidirectional=True)
        # Downsampling
        down_layers = []
        for _ in range(n_downsample):
            down_layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2
        self.down_blocks = nn.Sequential(*down_layers)
        res_layers = []
        for _ in range(4):
            res_layers += [ResidualBlock(dim)]
        self.res_blocks = nn.Sequential(*res_layers)
    
    def reparameterization(self, mu):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return z + mu
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.down_blocks(x)
        x = x.view(4, 128, -1)
        self.blstm.flatten_parameters()
        x, _ = self.blstm(x)
        #print(x.shape)
        x= x.view(4, 128, 3, 32)
        mu = self.res_blocks(x)
        z = self.reparameterization(mu)
        return mu, z
    
class Generator_lstm(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None):
        super(Generator_lstm, self).__init__()

        self.shared_block = shared_block
        layers = []
        dim = dim * 2 ** n_upsample
        res_layers = []
        for _ in range(3):
            res_layers += [ResidualBlock(dim)]
        self.res_blocks = nn.Sequential(*res_layers)
        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = dim // 2
        self.up_blocks = nn.Sequential(*layers)
        self.lstm1 = nn.LSTM(96, 800, batch_first=True)
        # Output layer
        output_layers = []
        output_layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]

        self.output_blocks = nn.Sequential(*output_layers)

    def forward(self, x):
        x = self.shared_block(x)
        x = self.res_blocks(x)
        x = x.view(4, 128, -1)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.contiguous().view(4, 128, 25, 32)
        x = self.up_blocks(x)
        #print(x.shape)
        
        x = self.output_blocks(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2):
        super(Encoder, self).__init__()
        #self.use_pitch_emb = use_pitch_emb
        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(4):
            layers += [ResidualBlock(dim)]
        
        self.model_blocks = nn.Sequential(*layers)
        #self.output_layer = nn.Conv2d(128, 2 * dim, kernel_size=1)
        
    def reparameterization(self, mu):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return z + mu
    """
    
    def reparameterization(self, mu):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return z + mu

    def reparameterization(self, mu, sigma=None):
        if sigma is not None:
            Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
            z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
            return z * sigma + mu
        else:
            Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
            z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
            return z + mu
    
    def forward(self, x):
        mu = self.model_blocks(x)
        z = self.reparameterization(mu)
        return mu, z

    def forward(self, x, id):
        mu = self.model_blocks(x)
        if id == 1:
            x = self.output_layer(mu)
            mu, logvar = torch.chunk(x, 2, dim=1)
            sigma = torch.exp(0.5 * logvar)
            z = self.reparameterization(mu, sigma)
            return mu, sigma, z
        else:
            z = self.reparameterization(mu)
            return mu, 0, z
    
"""
    def forward(self, x):
        mu = self.model_blocks(x)
        z = self.reparameterization(mu)
        return mu, z
    
        

class Generator(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None):
        super(Generator, self).__init__()

        self.shared_block = shared_block

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.shared_block(x)
        x = self.model_blocks(x)
        return x

class Generator_qian(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None):
        super(Generator, self).__init__()

        self.shared_block = shared_block

        layers_1 = []
        layers_2 = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(3):
            layers_1 += [ResidualBlock(dim)]

        # Upsampling
        for _ in range(n_upsample):
            layers_1 += [
                nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers_2 += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]

        self.model_blocks_1 = nn.Sequential(*layers_1)
        #self.lstm = nn.LSTM(356, 712, 2, batch_first=True)
        self.linear = nn.Linear(356, 100)
        self.model_blocks_2 = nn.Sequential(*layers_2)

    def forward(self, x):
        #print(x.shape)
        if self.shared_block is not None:
            x = self.shared_block(x)
        #print(x.shape)
        x = self.model_blocks_1(x)
        #print(x.shape)
        x = x.permute(0, 1, 3, 2)
        x = self.linear(x)
        x = x.permute(0, 1, 3, 2)
        #print(x.shape)
        x = self.model_blocks_2(x)
        #print(x.shape)
        #x, _ = torch.split(x, [100, 256], dim=2)
        #print(x.shape)
        return x

##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape, pitch_emb_dim=None):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, 6, width // 2 ** 4)
        self.pitches_embed_dim = 0
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 3, padding=1)
        )

    def forward(self, img):
        out = self.model(img)
        if self.pitches_embed_dim > 0:
            pitches_pred = self.pitch_predictor(img)
            return out, pitches_pred
        return out
    
if __name__ == "__main__":
    print("test encoder")
    encoder = Encoder_lstm(in_channels=1, dim=32, n_downsample=2).cuda()
    #encoder_p = Encoder(in_channels=1, dim=32, n_downsample=2).cuda()
    x = torch.randn(4, 1, 100, 128)
    #p = torch.rand(4, 257, 128)
    x = x.cuda()
    #p = p.cuda()
    #p = p.view(p.size(0), 1, p.size(1), p.size(2))
    with torch.no_grad():
        mu, z = encoder(x)
        #mu_,z_ = encoder_p(p)
        #print(mu.shape)
        #print(z.shape)
        #print(mu_.shape)
        #print(z_.shape)
        #print(p_e.shape)
    print(mu.shape)
    feat_1 = mu.contiguous().view(mu.size()[0], -1).mean(dim=0)
    
    print("test decoder")
    decoder = Generator_lstm(
        out_channels=1, 
        dim=32, 
        n_upsample=2, 
        shared_block=None).cuda()
    #print(z)
    with torch.no_grad():
        y = decoder(z)
        print(y.shape)
    
    print("test discriminator")
    discriminator = Discriminator((1, 100, 128), pitch_emb_dim=None)
    x = torch.randn(4, 1, 100, 128)
    with torch.no_grad():
        y = discriminator(x)
        print(y.shape)
        """
        """
    """
    """