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
from PIL import Image
from model import load_model, get_model

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
    15: "H",
    16: "*"
    }

device = torch.device('cpu')
nz = 32 # size of latent vector
batch_size = 32 # input batch size
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

examples_json = "levels/levels_smb.json"
X = np.array(json.load(open(examples_json)))
nc = 17

num_batches = X.shape[0]/batch_size
X_onehot = np.eye(nc, dtype='uint8')[X]
X_onehot = np.rollaxis(X_onehot, 3, 1)
X_train = X_onehot

vae, opt = get_model(device, nc, nz, 1e-3)
print(vae)

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

epochs = 5000 # num epochs to train for
training = True
X_original = X_train
if training:
    loss_file = open('loss_file_smb.csv','w')
    loss_file.write('Epoch,Loss,BCE,KLD\n')
    loss_h, bce_h, kld_h = [], [], []
    for epoch in range(epochs):
        X_train = X_train[torch.randperm(len(X_train))]
        i = 0
        tot_loss, tot_bce, tot_kld = 0, 0, 0
        while i < num_batches:
            data = X_train[i*batch_size:(i+1)*batch_size]
            data = torch.FloatTensor(data)
            recon, mu, logvar = vae(data)
            loss, bce, kld = loss_fn(recon, data, mu, logvar)
            tot_loss += loss.data.item()/batch_size
            tot_bce += bce.data.item()/batch_size
            tot_kld += kld.data.item()/batch_size
            opt.zero_grad()
            loss.backward()
            opt.step()
            i += 1
            
        if epoch % 50 == 0:
            to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                epochs, loss.data.item()/batch_size, bce.data.item()/batch_size, kld.data.item()/batch_size)
            print(to_print)
            torch.save(vae.state_dict(), 'vae_smb_' + str(epoch) + '.pth')
        l = (tot_loss/num_batches)
        b = (tot_bce/num_batches)
        k = (tot_kld/num_batches)
        loss_h.append(l)
        bce_h.append(b)
        kld_h.append(k)
        loss_file.write(str(epoch+1) + ',' + str(l) + ',' + str(b) + ',' + str(k) + '\n')
    torch.save(vae.state_dict(), 'vae_smb_final.pth')
    loss_file.close()
