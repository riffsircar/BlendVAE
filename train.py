from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import sys
import 
from PIL import Image

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

device = torch.device('cpu')
nz = 64 # size of latent vector
batch_size = 32 # input batch size
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

examples_json = "levels/levels_ng_16.json"
X = np.array(json.load(open(examples_json)))
#print(X)
#print(X.shape)
nc = 16

num_batches = X.shape[0]/batch_size
#print("SHAPE: ", X.shape)
#print("Num batch: ", num_batches)
X_onehot = np.eye(nc, dtype='uint8')[X]
#print(X_onehot.shape)
X_onehot = np.rollaxis(X_onehot, 3, 1)
#print("SHAPE onehot: ", X_onehot.shape)
#print(X[0])
#print(X_onehot[0])
X_train = X_onehot
#print(X_train.shape)

"""
np.random.shuffle(X_onehot)
split_idx = int(0.8*X_onehot.shape[0])
X_train, X_test = X_onehot[:split_idx], X_onehot[split_idx:]
print(X_train.shape, X_test.shape)
num_train_batches, num_test_batches = X_train.shape[0] / batch_size, X_test.shape[0] / batch_size
print(num_train_batches, num_test_batches)
"""
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
        
vae = VAE(nc).to(device)
print(vae)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

epochs = 10000 # num epochs to train for
training = False
X_original = X_train
if training:
    loss_file = open('loss_file_ng_64.csv','w')
    loss_file.write('Epoch,Loss,BCE,KLD\n')
    loss_h, bce_h, kld_h = [], [], []
    for epoch in range(epochs):
        X_train = X_train[torch.randperm(len(X_train))]
        i = 0
        tot_loss, tot_bce, tot_kld = 0, 0, 0
        #print("Epoch: ", epoch)
        while i < num_batches:
            data = X_train[i*batch_size:(i+1)*batch_size]
            data = torch.FloatTensor(data)
            recon, mu, logvar = vae(data)
            #print("Data: ", data.shape)
            #print("recon: ", recon.shape)
            loss, bce, kld = loss_fn(recon, data, mu, logvar)
            tot_loss += loss.data.item()/batch_size
            tot_bce += bce.data.item()/batch_size
            tot_kld += kld.data.item()/batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
            
        if epoch % 50 == 0:
            to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                epochs, loss.data.item()/batch_size, bce.data.item()/batch_size, kld.data.item()/batch_size)
            print(to_print)
            torch.save(vae.state_dict(), 'vae_' + str(epoch) + '.pth')
        l = (tot_loss/num_batches)
        b = (tot_bce/num_batches)
        k = (tot_kld/num_batches)
        loss_h.append(l)
        bce_h.append(b)
        kld_h.append(k)
        loss_file.write(str(epoch+1) + ',' + str(l) + ',' + str(b) + ',' + str(k) + '\n')

    #print("Loss: ", loss_h)
    #print("BCE: ", bce_h)
    #print("KLD: ", kld_h)
    
    torch.save(vae.state_dict(), 'vae_final.pth')

    loss_file.close()
