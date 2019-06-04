# This generator program expands a low-dimentional latent vector into a 2D array of tiles.
# Each line of input should be an array of z vectors (which are themselves arrays of floats -1 to 1)
# Each line of output is an array of 32 levels (which are arrays-of-arrays of integer tile ids)

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

import cma
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import random

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


images = {
    # TODO: Get T, D, M tiles from Icarus
    "E": Image.open('tiles/E.png'),
    "H": Image.open('tiles/H.png'),
    "G": Image.open('tiles/G.png'),
    "M": Image.open('tiles/M.png'),
    "o": Image.open('tiles/o.png'),
    "S": Image.open('tiles/S.png'),
    "T": Image.open('tiles/T.png'),
    "?": Image.open('tiles/Q.png'),
    "Q": Image.open('tiles/Q.png'),
    "X": Image.open('tiles/X1.png'),
    "#": Image.open('tiles/X.png'),
    "-": Image.open('tiles/-.png'),
    "0": Image.open('tiles/0.png'),
    "D": Image.open('tiles/D.png'),
    "<": Image.open('tiles/PTL.png'),
    ">": Image.open('tiles/PTR.png'),
    "[": Image.open('tiles/[.png'),
    "]": Image.open('tiles/].png')
    }

level_data = "levels_ng_16.json"
X = np.array(json.load(open(level_data)))
nc = 16
# print("SHAPE: ", X.shape)
X_onehot = np.eye(nc, dtype='uint8')[X]
# print(X_onehot.shape)
X_onehot = np.rollaxis(X_onehot, 3, 1)
# print("SHAPE onehot: ", X_onehot.shape)
X_train = X_onehot
# print(X_train.shape)

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

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = np.zeros((height*shape[0], width*shape[1],shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img
    return image


nz = 64
out_folder = 'results/'

def get_z_from_file(f):
    chunk_1 = open(f, 'r').read().splitlines()
    chunk_1 = [line.replace('\r\n','') for line in chunk_1]
    out_1 = []
    for line in chunk_1:
        line_list = list(line)
        line_list_map = [mapping_rev[x] for x in line_list]
        out_1.append(line_list_map)
    out_1 = np.asarray(out_1)
    print(out_1, out_1.shape)
    out1_onehot = np.eye(nc, dtype='uint8')[out_1]
    out1_onehot = np.rollaxis(out1_onehot, 2, 0)

    out1_onehot = out1_onehot[None, :, :]

    data_1 = torch.FloatTensor(out1_onehot)
    z_1, _, _ = model.encode(data_1)

    return z_1

def blend_chunks(smb_file, ki_file, prop):
    z_smb, z_ki = get_z_from_file(smb_file), get_z_from_file(ki_file)
    blended_vector = z_smb*prop + z_ki*(1-prop)
    level = model.decode(blended_vector)
    im = level.data.cpu().numpy()
    im = np.argmax(im, axis=1)
    output = Image.new('RGB', (16*16, 16*16))
    for i in im:
        for row, seg in enumerate(i):
            for col, tile in enumerate(seg):
                output.paste(images[mapping[tile]],(col*16, row*16))
        output.save(out_folder + "blended_" + str(prop) + ".png")

def interpolate_chunks(smb_file, ki_file, num_linp=30):
    z_smb, z_ki = get_z_from_file(smb_file), get_z_from_file(ki_file)

    alpha_values = np.linspace(0, 1, num_linp)

    vectors = []
    for alpha in alpha_values:
        vector = z_smb*(1-alpha) + z_ki*alpha
        vectors.append(vector)

    for idx, vector in enumerate(vectors):
        level = model.decode(vector)
        im = level.data.cpu().numpy()
        im = np.argmax(im, axis=1)
        output = Image.new('RGB',(16*16, 16*16))
        for i in im:
            for row, seg in enumerate(i):
                for col, tile in enumerate(seg):
                    output.paste(images[mapping[tile]],(col*16, row*16))
            output.save(out_folder + "sinp_" + str(idx) + ".png")

def interpolate_random(num_linp=30):
    a = torch.FloatTensor(1, nz).normal_(0,1)
    b = torch.FloatTensor(1, nz).normal_(0,1)

    alpha_values = np.linspace(0, 1, num_linp)

    vectors = []
    for alpha in alpha_values:
        vector = a*(1-alpha) + b*alpha
        vectors.append(vector)

    for idx, vector in enumerate(vectors):
        level = model.decode(vector)
        im = level.data.cpu().numpy()
        im = np.argmax(im, axis=1)
        output = Image.new('RGB',(16*16, 16*16))
        for i in im:
            for row, seg in enumerate(i):
            #print(row,seg)
                for col, tile in enumerate(seg):
                #print(col, tile)
                    output.paste(images[mapping[tile]],(col*16, row*16))
            output.save(out_folder + "rand_chunk_" + str(idx) + ".png")


def output_image(z, name):
    level = model.decode(z)
    im = level.data.cpu().numpy()
    im = np.argmax(im, axis=1)
    # print("Im: ", im)
    output = Image.new('RGB',(16 * 16, 16 * 16))
    for i in im:
        for row, seg in enumerate(i):
            for col, tile in enumerate(seg):
                output.paste(images[mapping[tile]],(col*16, row*16))
        output.save(out_folder + name + ".png")
    

def add_chunks(chunk_1, chunk_2):
    z_1, z_2 = get_z_from_file(chunk_1), get_z_from_file(chunk_2)
    z_3 = z_1 + z_2
    output_image(z_1, "z1")
    output_image(z_2, "z2")
    output_image(z_3, "z1+z2")


def sub_chunks(chunk_1, chunk_2):
    z_1, z_2 = get_z_from_file(chunk_1), get_z_from_file(chunk_2)
    z_3 = z_1 - z_2
    output_image(z_1, "z1")
    output_image(z_2, "z2")
    output_image(z_3, "z1-z2")

def add_chunks_random():
    a = torch.FloatTensor(1, nz).normal_(0,1)
    b = torch.FloatTensor(1, nz).normal_(0,1)

    c = a + b
    output_image(a, "rand_a")
    output_image(b, "rand_b")
    output_image(c, "rand_a+b")


def maximize_tile_type(x, tile):
    x = np.array(x)
    latent_vector = torch.FloatTensor(x).view(1, nz)  # torch.from_numpy(lv)# torch.FloatTensor( torch.from_numpy(lv) )
    levels = model.decode(latent_vector)
    im = levels.data.cpu().numpy()
    im = np.argmax( im, axis = 1)
    num_tiles =  (len (im[im == tile]))
    return 256 - num_tiles

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

"""
def gan_fitness_function(x):
    x = np.array(x)
    latent_vector = torch.FloatTensor(x).view(batchSize, nz, 1, 1)  # torch.from_numpy(lv)# torch.FloatTensor( torch.from_numpy(lv) )
    with torch.no_grad():
        levels = generator(latent_vector)
    levels.data = levels.data[:, :, :16, :16]
    return solid_blocks_fraction(levels.data, 0.4)*ground_blocks_fraction(levels.data,0.8)
"""

def optimize(fn):
    es = cma.CMAEvolutionStrategy(nz * [0], 0.5)
    es.optimize(fn, args=[5])
    best = np.array(es.best.get()[0])
    print ("BEST ", best)
    latent_vector = torch.FloatTensor(best).view(1, nz)
    output_image(latent_vector, "optimized")
    cma.plot()
    

def density(x, p=1.0):
    total = 0
    z = torch.FloatTensor(x).view(1, nz)
    z_decoded = model.decode(z)
    level = z_decoded.data.cpu().numpy()
    level = np.argmax( level, axis = 1)
    for line in level[0]:
        total += len(line[line == 0]) # X
        total += len(line[line == 1]) # S
        total += len(line[line == 3]) # Q
        total += len(line[line == 4]) # ?
        total += len(line[line == 11]) # T
        total += len(line[line == 12]) # M
        total += len(line[line == 14]) # #
        
    return math.fabs(128*p - total)


def eval_density(p):
    out = open('density_new_'+str(p)+'.csv','w')
    out.write('N,Density\n')
    for i in range(100):
        print ("Eval Dense: ", i)
        es = cma.CMAEvolutionStrategy(nz * [0], 0.5)
        es.optimize(density, args=[p])
        best = np.array(es.best.get()[0])
        lv = torch.FloatTensor(best).view(1,nz)
        z_decoded = model.decode(lv)
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
        print((total*100)/128)
        out.write(str(i) + ',' + str((total*100)/128) + '\n')
    out.close()

def eval_diff(p):
    out = open('difficulty_'+str(p)+'.csv','w')
    out.write('N,Difficulty\n')
    for i in range(100):
        print ("Eval Diff: ", i)
        es = cma.CMAEvolutionStrategy(nz * [0], 0.5)
        es.optimize(difficulty, args=[p])
        best = np.array(es.best.get()[0])
        lv = torch.FloatTensor(best).view(1,nz)
        z_decoded = model.decode(lv)
        level = z_decoded.data.cpu().numpy()
        level = np.argmax(level, axis=1)
        num_eh = 0
        for line in level[0]:
            num_eh += len(line[line == 5]) + len(line[line == 15])
        print((num_eh*100)/16)
        out.write(str(i) + ',' + str((num_eh*100)/16) + '\n')
    out.close()
    
def eval_both(p):
    out = open('both_new_'+str(p)+'.csv','w')
    out.write('N,Mario,Icarus\n')
    for i in range(100):
        print ("Eval Both: ", i)
        es = cma.CMAEvolutionStrategy(nz * [0], 0.5)
        es.optimize(both_percent, args=[p])
        best = np.array(es.best.get()[0])
        lv = torch.FloatTensor(best).view(1,nz)
        z_decoded = model.decode(lv)
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
        prop_ki = 0 if non_back == 0 else (ki/non_back)
        prop_smb = 0 if non_back == 0 else (smb/non_back)
        out.write(str(i) + ',' + str(prop_smb) + ',' + str(prop_ki) + '\n')
    out.close()

    
def eval_both_dense(p):
    out = open('both_dense_new_'+str(p)+'.csv','w')
    out.write('N,Mario,Icarus\n')
    for i in range(50):
        print ("Eval Both Dense: ", i)
        es = cma.CMAEvolutionStrategy(nz * [0], 0.5)
        es.optimize(blend_dense, args=[p])
        best = np.array(es.best.get()[0])
        lv = torch.FloatTensor(best).view(1,nz)
        z_decoded = model.decode(lv)
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
        prop_ki = 0 if non_back == 0 else (ki/non_back)
        prop_smb = 0 if non_back == 0 else (smb/non_back)
        out.write(str(i) + ',' + str(prop_smb) + ',' + str(prop_ki) + '\n')
    out.close()

def optimize_density(p):
    es = cma.CMAEvolutionStrategy(nz * [0], 0.5)
    es.optimize(density, args=[p])
    best = np.array(es.best.get()[0])
    print ("BEST ", best)
    latent_vector = torch.FloatTensor(best).view(1, nz)
    output_image(latent_vector, "dens_" + str(p))

def optimize_difficulty(p):
    es = cma.CMAEvolutionStrategy(nz * [0], 0.5)
    es.optimize(difficulty, args=[p])
    best = np.array(es.best.get()[0])
    print ("BEST ", best)
    latent_vector = torch.FloatTensor(best).view(1, nz)
    output_image(latent_vector, "diff_" + str(p))

def optimize_tile_type(p):
    es = cma.CMAEvolutionStrategy(nz * [0], 0.5)
    es.optimize(maximize_tile_type, args=[p])
    best = np.array(es.best.get()[0])
    print ("BEST ", best)
    latent_vector = torch.FloatTensor(best).view(1, nz)
    output_image(latent_vector, "tile_type_" + str(p))
    

def difficulty(x, p=1.0):
    e, g = 0, 0
    z = torch.FloatTensor(x).view(1, nz)
    z_decoded = model.decode(z)
    level = z_decoded.data.cpu().numpy()
    level = np.argmax( level, axis = 1)
    num_eh, num_gap = 0, 0
    for i, line in enumerate(level[0]):
        num_eh += len(line[line == 5]) + len(line[line == 15])
    return math.fabs(16*p - (num_eh))

def optimize_both(p):
    es = cma.CMAEvolutionStrategy(nz * [0], 0.5)
    es.optimize(both_percent, args=[p])
    best = np.array(es.best.get()[0])
    print ("BEST ", best)
    latent_vector = torch.FloatTensor(best).view(1, nz)
    output_image(latent_vector, "smb_prop_" + str(p))

def optimize_both_dense(p):
    es = cma.CMAEvolutionStrategy(nz * [0], 0.5)
    es.optimize(blend_dense, args=[p])
    best = np.array(es.best.get()[0])
    print ("BEST ", best)
    latent_vector = torch.FloatTensor(best).view(1, nz)
    output_image(latent_vector, "smb_dense_prop_" + str(p))

def optimize_smb(p):
    es = cma.CMAEvolutionStrategy(nz * [0], 0.5)
    es.optimize(smb_percent, args=[p])
    best = np.array(es.best.get()[0])
    print ("BEST ", best)
    latent_vector = torch.FloatTensor(best).view(1, nz)
    output_image(latent_vector, "smb_" + str(p))

def optimize_ki(p):
    es = cma.CMAEvolutionStrategy(nz * [0], 0.5)
    es.optimize(ki_percent, args=[p])
    best = np.array(es.best.get()[0])
    print ("BEST ", best)
    latent_vector = torch.FloatTensor(best).view(1, nz)
    output_image(latent_vector, "icarus_" + str(p))
    

def both_percent(x, p=1.0):
    z = torch.FloatTensor(x).view(1,nz)
    z_decoded = model.decode(z)
    level = z_decoded.data.cpu().numpy()
    level = np.argmax( level, axis = 1)
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
    prop_ki = 0 if non_back == 0 else (ki/non_back)
    prop_smb = 0 if non_back == 0 else (smb/non_back)
    #return -1.0*((1-p)*prop_ki + (p*prop_smb))
    #return math.fabs(non_back - ((1-p)*prop_ki + (p*prop_smb)))
    #return math.fabs((1-p)*prop_ki - (p*prop_smb))
    return (1.0 - ((1-p)*prop_ki + (p*prop_smb)))

def blend_dense(x, p=1.0):
    z = torch.FloatTensor(x).view(1,nz)
    #z = torch.FloatTensor(1, nz).normal_(0,1)
    z_decoded = model.decode(z)
    level = z_decoded.data.cpu().numpy()
    level = np.argmax( level, axis = 1)
    smb, ki, back, total = 0, 0, 0, 0
    density = 0.25
    for line in level[0]:
        total += len(line[line == 0]) # X
        total += len(line[line == 1]) # S
        total += len(line[line == 3]) # Q
        total += len(line[line == 4]) # ?
        total += len(line[line == 11]) # T
        total += len(line[line == 12]) # M
        total += len(line[line == 14]) # #
        for tile in line:
            if tile != 2:
                if tile < 11:
                    smb += 1
                else:
                    ki += 1
            else:
                back += 1
    non_back = 256 - back
    prop_ki = 0 if non_back == 0 else (ki/non_back)
    prop_smb = 0 if non_back == 0 else (smb/non_back)
    return (1.0 - ((1-p)*prop_ki + (p*prop_smb))) + math.fabs(128*density - total)

def smb_percent(x, p=1.0):
    z = torch.FloatTensor(x).view(1,nz)
    #z = torch.FloatTensor(1, nz).normal_(0,1)
    z_decoded = model.decode(z)
    level = z_decoded.data.cpu().numpy()
    level = np.argmax( level, axis = 1)
    smb, ki, back = 0, 0, 0
    for line in level[0]:
        #print(line)
        for tile in line:
            if tile != 2:
                if tile < 11:
                    smb += 1
                else:
                    ki += 1
            else:
                back += 1
    non_back = 256 - back
    # print((smb*100)/non_back, (ki*100)/non_back)
    #print(math.fabs(p-(smb/non_back)))
    return math.fabs(p-(smb/non_back))

def ki_percent(x, p=1.0):
    z = torch.FloatTensor(x).view(1,nz)
    #z = torch.FloatTensor(1, nz).normal_(0,1)
    z_decoded = model.decode(z)
    level = z_decoded.data.cpu().numpy()
    level = np.argmax( level, axis = 1)
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
    # print((smb*100)/non_back, (ki*100)/non_back)
    #print(math.fabs(p-(smb/non_back)))
    return math.fabs(p-(ki/non_back))


if __name__ == '__main__':
    path = 'vae_ng_64_final.pth'
    #path = 'vae_beta/vae_beta_final.pth'
    #path = 'vae_20K/vae_20K_final.pth'
    model = VAE(nc)
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    z = torch.FloatTensor(1, nz).normal_(0,1)
    level = model.decode(z)
    im = level.data.cpu().numpy()
    im = np.argmax( im, axis = 1)
    #print(im)
    #leniency(im)
    #density(im)
    #optimize_density(1)
    #optimize_leniency(1)
    #optimize_density(0.25)
    #optimize_both(1)
    #optimize_both_dense(1)
    #interpolate_chunks('smb_chunk_15.txt','ki_chunk_187.txt',6)
    #optimize_both(0.5)

    #optimize_tile_type(4)
    #optimize_both(1)
    #blend_chunks('smb_chunk_15.txt', 'ki_chunk_187.txt',0)
    #optimize_both_dense(0.5)
    """
    interpolate_chunks('smb_chunk_15.txt','levels/chunks_ng/smb_chunk_132.txt',20)
    z = get_z_from_file('levels/chunks_ng/smb_chunk_15.txt')
    output_image(z, 'smb_chunk_15_test')
    z = get_z_from_file('levels/chunks_ng/smb_chunk_132.txt')
    output_image(z, 'smb_chunk_132_test')
    sys.exit()
    interpolate_chunks('smb_chunk_15.txt','smb_chunk_20.txt',30)
    """
    for i in [1]:
        eval_both_dense(i)

    """
    optimize_leniency(0.75)
    #game_percents(1)
    optimize_smb(1)
    optimize_ki(0)
    interpolate_chunks('custom_smb.txt','custom_ki.txt',30)
    
    z = get_z_from_file('custom_smb.txt')
    output_image(z, 'custom_smb_2')
    z = get_z_from_file('custom_ki.txt')
    output_image(z, 'custom_ki')
    
    interpolate_chunks('smb_chunk_15.txt','ki_chunk_187.txt',30)
    z = get_z_from_file('smb_chunk_20.txt')
    output_image(z, 'smb_chunk_20_test')
    z = get_z_from_file('smb_chunk_15.txt')
    output_image(z, 'smb_chunk_15_test')
    z = get_z_from_file('ki_chunk_187.txt')
    output_image(z, 'ki_chunk_187_chunktest2')
    interpolate_random()
    add_chunks_random()
    """        
    #optimize(gan_maximize_tile_type)
    

