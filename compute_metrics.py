from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


from parser import *
import numpy as np
import random
import sys
import time
import os
import json

levels = []


mapping = {
    0: "X",
    1: "S",
    2: "-",
    3: "?",
    4: "Q",
    5: "E",
    6: "<",
    7: ">",
    8: "[",
    9: "]",
    10: "o",
    11: "T",
    12: "M",
    13: "D",
    14: "#",
    15: "H"
    }

mapping_rev = {
    "X": 0,
    "S": 1,
    "-": 2,
    "?": 3,
    "Q": 4,
    "E": 5,
    "<": 6,
    ">": 7,
    "[": 8,
    "]": 9,
    "o": 10,
    "T": 11,
    "M": 12,
    "D": 13,
    "#": 14,
    "H": 15
    }


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, nc=16, h_dim=512, z_dim=64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,kernel_size=4,stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            Flatten()
        )
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, nc, kernel_size=4, stride=2,padding=1),
            nn.Sigmoid()
        )

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

nz = 64
nc = 16


def get_z_from_file(f):
    chunk_1 = open(f, 'r').read().splitlines()
    chunk_1 = [line.replace('\r\n','') for line in chunk_1]
    out_1 = []
    for line in chunk_1:
        line_list = list(line)
        line_list_map = [mapping_rev[x] for x in line_list]
        out_1.append(line_list_map)
    out_1 = np.asarray(out_1)
    out1_onehot = np.eye(nc, dtype='uint8')[out_1]
    out1_onehot = np.rollaxis(out1_onehot, 2, 0)

    out1_onehot = out1_onehot[None, :, :]

    data_1 = torch.FloatTensor(out1_onehot)
    z_1, _, _ = model.encode(data_1)

    return z_1

path = 'vae_ng_64/vae_ng_64_final.pth'
model = VAE(nc)
model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

def compute_density(z):
    z_decoded = model.decode(z)
    level = z_decoded.data.cpu().numpy()
    level = np.argmax(level, axis=1)
    total = 0
    for line in level[0]:
        total += len(line[line == 0]) # X
        total += len(line[line == 1]) # S
        total += len(line[line == 3]) # Q
        total += len(line[line == 4]) # ?
        total += len(line[line == 11]) # T
        total += len(line[line == 12]) # M
        total += len(line[line == 14]) # #
    return ((total*100)/128)

def compute_difficulty(z):
    z_decoded = model.decode(z)
    level = z_decoded.data.cpu().numpy()
    level = np.argmax( level, axis = 1)
    num_eh, num_gap = 0, 0
    for i, line in enumerate(level[0]):
        num_eh += len(line[line == 5]) + len(line[line == 15])
        
    return((num_eh*100)/16)

def compute_proportions(z):
    z_decoded = model.decode(z)
    level = z_decoded.data.cpu().numpy()
    level = np.argmax(level, axis=1)
    smb, ki, back = 0, 0, 0
    for line in level[0]:
        for tile in line:
            if tile != 2:
                if tile < 11:
                    smb += 1
                else:
                    ki += 1
            else:
                back += 1
    non_back = 256 - back
    return (smb/non_back, ki/non_back)

"""
outfile = open('density_metrics.csv','w')
for f in os.listdir('levels/chunks_ng/'):
    # print('levels/chunks_ng/'+f)
    #print(f)
    z = get_z_from_file('levels/chunks_ng/'+f)
    #print(f,compute_density(z))
    outfile.write(f + ',' + str(compute_density(z)) + '\n')
outfile.close()
"""

def get_metrics(f):
    density, difficulty, smb, ki, back = 0, 0, 0, 0, 0
    for line in open(f,'r'):
        for l in line:
            if l in ['X','S','Q','?','T','M','#']:
                density += 1
            if l in ['E','H']:
                difficulty += 1
            if l == '-':
                back += 1
            if l in ['X','S','Q','?','E','<','>','[',']','o']:
                smb += 1
            if l in ['#','T','M','D','H']:
                ki += 1
    density = (density*100)/128
    difficulty = (difficulty*100)/16
    smb = smb/(256-back)
    ki = ki/(256-back)
    return (density, difficulty, smb, ki)


outfile = open('metrics_new.csv','w')
outfile.write('Chunk,Density,Difficulty,SMB,KI\n')
for f in os.listdir('levels/chunks_ng/'):
    # print(f)
    z = get_z_from_file('levels/chunks_ng/'+f)
    density, difficulty, smb, ki = get_metrics('levels/chunks_ng/'+f)
    outfile.write(f + ',' + str(density) + ',' + str(difficulty) + ',' + str(smb) + ',' + str(ki) + '\n')
outfile.close()

outfile = open('random_newest.csv','w')
outfile.write('Vector,Density,Difficulty,SMB,KI\n')
for i in range(10000):
    z = torch.FloatTensor(1, nz).normal_(0,1)
    density, difficulty, prop = compute_density(z), compute_difficulty(z), compute_proportions(z)
    outfile.write(str(i) + ',' + str(density) + ',' + str(difficulty) + ',' + str(prop[0]) + ',' + str(prop[1]) + '\n')
outfile.close()
