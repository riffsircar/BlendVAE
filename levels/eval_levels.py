from __future__ import print_function
from parser import *
import numpy as np
import random
import sys
import time
import os
import argparse
import json

levels = []
patterns_smb = []
patterns_ki = []
orientation = 'C'


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
            #nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            #nn.ReLU(),
            nn.ConvTranspose2d(64, nc, kernel_size=4, stride=2,padding=1),
            nn.Sigmoid()
        )

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
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

for file in os.listdir('chunks_ng/'):
    print(file)

sys.exit()

def leniency(level):
    e, g = 0, 0
    for seq in level:
        e += seq.count('E')
        if seq.count('-') == len(seq):
            g += 0.5
    return((e+g)/len(level))

def density(level):
    total = 0
    for seq in level:
        total += seq.count('X')
        total += seq.count('S')
        total += seq.count('Q')
        total += seq.count('?')
    return(total/len(level))

def pattern_density(level,patterns):
    total = 0
    for seq in level:
        if seq in patterns:
            total += 1
    return(total/len(level))

def pattern_variation(level, patterns):
    variety = []
    for seq in level:
        if seq in patterns:
            if seq not in variety:
                variety.append(seq)

    return(len(variety)/len(level))

def calculate_metrics(level):
    global patterns_smb, patterns_ki
    leniency, d, pd, pv = 0, 0, 0, 0
    e, g = 0, 0
    variety = []
    seqs = len(level)
    print(seqs)
    for lev in level:
        seq, orient = lev[0], lev[1]
        e -= seq.count('E')
        if seq.count('-') == len(seq) and orient == 'C':
            g -= 0.5
        d += seq.count('X')+seq.count('S')+seq.count('Q')+seq.count('?')
        if orient == 'C':
            if seq in patterns_smb:
                pd += 1
                if seq not in variety:
                    variety.append(seq)
        else:
            if seq in patterns_ki:
                pd += 1
                if seq not in variety:
                    variety.append(seq)

    leniency = (e+g)/seqs
    d /= seqs
    pd /= seqs
    pv = len(variety)/seqs
    return (leniency, d, pd, pv)

smb_levels, ki_levels = {}, {}
for file in os.listdir("Generic/SMB/"):
    with open(os.path.join("Generic/SMB/",file),'rb') as infile:
        level = []
        for line in infile:
            level.append(list(str(line.rstrip(),'utf-8')))
            #level.append(str(line.rstrip(),'utf-8'))
        #levels.append(level)
        width = len(level[0])
        height = len(level)
        level_str = []
        for column in range(width):
            line_str = ''
            for row in range(height):
                line_str += str(level[row][column])
            if line_str not in patterns_smb:
                patterns_smb.append(line_str)
            level_str.append([line_str,'C'])
        smb_levels[file[:-4]] = level_str    


for file in os.listdir("Generic/KI/"):
    with open(os.path.join("Generic/KI/",file),'rb') as infile:
        level = []
        for line in infile:
            level.append(list(str(line.rstrip(),'utf-8')))
            #level.append(str(line.rstrip(),'utf-8'))
        #levels.append(level)
        width = len(level[0])
        height = len(level)
        level_str = []
        for row in range(height):
            line_str = ''
            for column in range(width):
                line_str += str(level[row][column])
            if line_str not in patterns_ki:
                patterns_ki.append(line_str)
            level_str.append([line_str,'R'])
        ki_levels[file[:-4]] = level_str


res_file = open("smb.csv","w")
res_file.write("Level,Leniency,Density,PD,PV\n")
for name in smb_levels:
    level = smb_levels[name]
    leniency, d, pd, pv = calculate_metrics(level)
    res_file.write(name + "," + str(leniency) + "," + str(d) + "," + str(pd) + "," + str(pv) + "\n")
res_file.close()


res_file = open("ki.csv","w")
res_file.write("Level,Leniency,Density,PD,PV\n")
for name in ki_levels:
    level = ki_levels[name]
    leniency, d, pd, pv = calculate_metrics(level)
    res_file.write(name + "," + str(leniency) + "," + str(d) + "," + str(pd) + "," + str(pv) + "\n")
res_file.close()

sys.exit()


for level in levels:
    level_str = []
    width = len(level[0])
    height = len(level)
    for row in range(height):
        line_str = ''
        for column in range(width):
            line_str += str(level[row][column])
        if line_str not in patterns_ki:
            patterns_ki.append(line_str)
        level_str.append(line_str)
    level_strs.append(level_str)


print (len(patterns_smb))
print (len(patterns_ki))



def calculate_metrics2(level):
    global patterns
    leniency, d, pd, pv = 0, 0, 0, 0
    e, g = 0, 0
    variety = []
    seqs = len(level)
    for seq in level:
        e += seq.count('E')
        if seq.count('-') == len(seq):
            g += 0.5
        d += seq.count('X')+seq.count('S')+seq.count('Q')+seq.count('?')
        if seq in patterns:
            pd += 1
            if seq not in variety:
                variety.append(seq)

    leniency = (e+g)/seqs
    d /= seqs
    pd /= seqs
    pv = len(variety)/seqs
    return (leniency, d, pd, pv)
        

for i in range(0,len(level_strs)):
    #print(level_str)
    #print(X)
    #linearity(X)
    level = level_strs[i]
    print("Level ", i, ":")
    leniency, d, pd, pv = calculate_metrics(level)
    print("Leniency: ", leniency)
    print("Density: ", d)
    print("Pattern density: ", pd)
    print("Pattern variation: ", pv)
    print()
    
