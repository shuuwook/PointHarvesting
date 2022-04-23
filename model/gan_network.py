import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform as init_
import numpy as np
import math

from math import ceil

class Identity(nn.Module):
    def forward(self, input):
        return input

class Discriminator(nn.Module):
    def __init__(self, feature):
        super(Discriminator, self).__init__()

        # layers for pushforward
        self.disc_layers = {'pre': [], 'pool': [], 'post': []}

        for inx in range(len(feature)-1):
            self.disc_layers['pre'].append(nn.Linear(feature[inx], feature[inx+1]))
            self.disc_layers['pre'].append(nn.LeakyReLU(0.2))
        self.disc_layers['pre'] = nn.Sequential(*self.disc_layers['pre'][:-1])
        
        self.disc_layers['pool'] = nn.Linear(2*feature[-1], feature[-1])

        for inx in range(1, len(feature)):
            self.disc_layers['post'].append(nn.Linear(feature[-inx], feature[-inx-1]))
        self.disc_layers['post'].append(nn.Linear(feature[0], 1))
        self.disc_layers['post'] = nn.Sequential(*self.disc_layers['post'])

        self.disc_layers = nn.ModuleDict(self.disc_layers)


    def forward(self, input, flag=True):
        # points : (B,N,3) / label : (B,N,C)
        N = input.size(1)

        x = self.disc_layers['pre'](input)
        
        max_pool = F.max_pool1d(x.transpose(1,2), kernel_size=N).squeeze(-1) # (B,D)
        mean_pool = F.avg_pool1d(x.transpose(1,2), kernel_size=N).squeeze(-1) # (B,D)
        pool = torch.cat([max_pool, mean_pool], axis=-1) # (B,2D)
        pool = self.disc_layers['pool'](pool)

        out = self.disc_layers['post'](pool)
        
        return out


class Generator(nn.Module):
    def __init__(self, feature, noise_scale=0.1):
        self.feature = feature
        self.noise_scale = noise_scale
        super(Generator, self).__init__()

        # layer for code
        self.code_layers = nn.Sequential(nn.Linear(64, 512),
                                         nn.LeakyReLU(0.2),
                                         nn.Linear(512, 1024))

        # layers for pushforward
        self.pushforward_layers = {'pre': [], 'post': []}

        for inx in range(len(feature)-1):
            self.pushforward_layers['pre'].append(nn.Linear(feature[inx], feature[inx+1]))
            self.pushforward_layers['pre'].append(nn.LeakyReLU(0.2))                           
        self.pushforward_layers['pre'] = nn.Sequential(*self.pushforward_layers['pre'][:-1])
        
        for inx in range(1, len(feature)):
            self.pushforward_layers['post'].append(nn.Linear(feature[-inx], feature[-inx-1]))
            self.pushforward_layers['post'].append(nn.LeakyReLU(0.2))
        self.pushforward_layers['post'].append(nn.Linear(feature[0], 3))
        self.pushforward_layers['post'] = nn.ModuleList(self.pushforward_layers['post'])

        self.pushforward_layers = nn.ModuleDict(self.pushforward_layers)

            
    def forward(self, input, sample_num):
        B = input.size(0)
        N = sample_num

        # Latent
        code = self.code_layers(input).view(B,2,1,self.feature[-1])
        code = code + self.noise_scale*torch.randn_like(code)

        # Sampling
        z = torch.randn(B,N,self.feature[0]).to(input.device)
        z = z / torch.norm(z, dim=2, keepdim=True)

        # Pushforward
        x = self.pushforward_layers['pre'](z)
        
        # IN
        x = code[:,0] * F.instance_norm(x) + code[:,1]
        skip_x = x

        # Pushforward
        for inx in range(len(self.pushforward_layers['post'])-2):
            if inx % 2:
                x = self.pushforward_layers['post'][inx](x) + skip_x
                skip_x = x
            else:
                x = self.pushforward_layers['post'][inx](x)
        
        x = self.pushforward_layers['post'][-2](x)
        x = self.pushforward_layers['post'][-1](x)

        return x