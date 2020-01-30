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
import os

import cma
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import random

from PIL import Image
from scipy.spatial import distance
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from igraph import *
import rasterfairy

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
    "H": 15,
    "*": 16
    }


images = {
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
    "]": Image.open('tiles/].png'),
    "*": Image.open('tiles/0.png')
    }

nc = 17

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


nz = 32
out_folder = 'mlcd/'

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
                for col, tile in enumerate(seg):
                    output.paste(images[mapping[tile]],(col*16, row*16))
            output.save(out_folder + "rand_chunk_" + str(idx) + ".png")


def output_image(z, name):
    level = model.decode(z)
    im = level.data.cpu().numpy()
    im = np.argmax(im, axis=1)
    output = Image.new('RGB',(16 * 16, 16 * 16))
    for i in im:
        for row, seg in enumerate(i):
            for col, tile in enumerate(seg):
                output.paste(images[mapping[tile]],(col*16, row*16))
        output.save(out_folder + name + ".png")
    
def get_image(z):
    latent_vector = torch.FloatTensor(z).view(1, nz)
    level = model.decode(latent_vector)
    im = level.data.cpu().numpy()
    im = np.argmax(im, axis=1)
    output = Image.new('RGB',(16 * 16, 16 * 16))
    for i in im:
        for row, seg in enumerate(i):
            for col, tile in enumerate(seg):
                output.paste(images[mapping[tile]],(col*16, row*16))
    return output


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
    #z_decoded = model.decode(z)
    #level = z_decoded.data.cpu().numpy()
    #level = np.argmax( level, axis = 1)
    level = np.argmax(model.decoder(model.fc3(z)).detach().numpy(), axis=1)
    for line in level[0]:
        total += len(line[line == 0]) # X
        total += len(line[line == 1]) # S
        total += len(line[line == 3]) # Q
        total += len(line[line == 4]) # ?
        total += len(line[line == 11]) # T
        total += len(line[line == 12]) # M
        total += len(line[line == 14]) # #
    return math.fabs(256*p - total)

def nonlinearity(x,p=1.0,flag=False):
    z = torch.FloatTensor(x).view(1,nz)
    level = np.argmax(model.decoder(model.fc3(z)).detach().numpy(), axis=1)
    level_t = level[0].transpose()
    x = np.arange(16)
    y = []
    for i, arr in enumerate(level_t):
        appended = False
        for j, a in enumerate(arr):
            if a in [0,1,3,4,11,12,14]:
                y.append(15-j)
                appended = True
                break
        if not appended:
            y.append(0)
    x = x.reshape(-1,1)
    y = np.asarray(y)
    
    reg = linear_model.LinearRegression()
    reg.fit(x,y)
    y_pred = reg.predict(x)
    mse = mean_squared_error(y,y_pred)
    return math.fabs(56.25*p - mse)

def difficulty(x, p=1.0):
    e, g = 0, 0
    z = torch.FloatTensor(x).view(1, nz)
    level = np.argmax(model.decoder(model.fc3(z)).detach().numpy(), axis=1)
    num_eh, num_gap = 0, 0
    for i, line in enumerate(level[0]):
        num_eh += len(line[line == 5]) + len(line[line == 15])
    return math.fabs(16*p - num_eh)


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
    return math.fabs(p-(smb/non_back))

def ki_percent(x, p=1.0):
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
    return math.fabs(p-(ki/non_back))


if __name__ == '__main__':
    path = 'vae_smb_final.pth'
    model = load_model(path,nc)

    path = 'vae_ki_final.pth'
    model_ki = load_model(path,nc)
    model_ki.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    path = 'vae_both_final.pth'
    model_both = load_model(path,nc)
    model_both.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    model = model_ki
    model.eval()
    i = 0
    ki_chunks = {}
    smb_chunks = {}
    
    # TSNE Visualization
    for file in os.listdir('levels/chunks_icarus/'):
        z = get_z_from_file('levels/chunks_icarus/' + file)
        ki_chunks[file] = z
        #smb_chunks[file] = z
        #output_image(z, 'test_' + str(i) + '.png')
        i += 1
    #print(ki_chunks)
    #interpolate_chunks('levels/chunks_icarus/ki_chunk_0.txt','levels/chunks_icarus/ki_chunk_75.txt',5)
    #sys.exit()
    ki_array = []
    for kc in ki_chunks:
        ki_array.append(ki_chunks[kc].view(32).detach().numpy())
    #print(len(ki_array), type(ki_array[0]), ki_array[0], ki_array[0].size, ki_array[0].shape)
    ki_array = np.array(ki_array)
    print(type(ki_array[0]), ki_array[0].shape, ki_array[0])
    print(type(ki_array), ki_array.shape)
    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(ki_array)                                                                                                   
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 2000
    height = 2000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    for z, x, y in zip(ki_array, tx, ty):
        tile = get_image(z)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    #print(full_image)
    #full_image.save('full_image.png')
    plt.figure(figsize = (16,12))
    plt.imshow(full_image)
    plt.imsave('plot_ki.png',full_image)
    nx, ny = 20, 10
    grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))

    tile_width = 72
    tile_height = 56

    full_width = tile_width * nx
    full_height = tile_height * ny
    aspect_ratio = float(tile_width) / tile_height

    grid_image = Image.new('RGB', (full_width, full_height))

    for z, grid_pos in zip(ki_array, grid_assignment[0]):
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y
        tile = get_image(z)
        tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
        if (tile_ar > aspect_ratio):
            margin = 0.5 * (tile.width - aspect_ratio * tile.height)
            tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
        else:
            margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
            tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
        tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
        grid_image.paste(tile, (int(x), int(y)))

    plt.figure(figsize = (16,12))
    plt.imshow(grid_image)
    plt.imsave('grid_ki.png',grid_image)

    sampled_chunks = torch.FloatTensor(200, 32).normal_().mul_(1).to('cpu')
    sampled_ki_array = []
    for sc in sampled_chunks:
        sampled_ki_array.append(sc.view(32).detach().numpy())
    sampled_ki_array = np.array(sampled_ki_array)
    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(sampled_ki_array)                                                                                                   
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 2000
    height = 2000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    for z, x, y in zip(sampled_ki_array, tx, ty):
        tile = get_image(z)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    print(full_image)
    plt.figure(figsize = (16,12))
    plt.imshow(full_image)
    plt.imsave('plot_ki_sampled.png',full_image)

    nx, ny = 20, 10
    grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))

    tile_width = 72
    tile_height = 56

    full_width = tile_width * nx
    full_height = tile_height * ny
    aspect_ratio = float(tile_width) / tile_height

    grid_image = Image.new('RGB', (full_width, full_height))

    for z, grid_pos in zip(sampled_ki_array, grid_assignment[0]):
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y
        tile = get_image(z)
        tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
        if (tile_ar > aspect_ratio):
            margin = 0.5 * (tile.width - aspect_ratio * tile.height)
            tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
        else:
            margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
            tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
        tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
        grid_image.paste(tile, (int(x), int(y)))

    plt.figure(figsize = (16,12))
    plt.imshow(grid_image)
    plt.imsave('grid_ki_sampled.png',grid_image)

    
    # CONSTRUCTING KNN GRAPH
    kNN = 5
    ki_graph = Graph(directed=True)
    for kc in ki_chunks:
        ki_graph.add_vertex(kc)

    for i in ki_chunks:
        distances, names = [], []
        for j in ki_chunks:
            z_i, z_j = ki_chunks[i], ki_chunks[j]
            z_i_d, z_j_d = z_i.detach().numpy(), z_j.detach().numpy()
            distances.append(distance.cosine(z_i_d, z_j_d))
            names.append(j)
        idx_knn = sorted(range(len(distances)),key=lambda k: distances[k])[1:kNN+1]
        for idx in idx_knn:
            ki_graph.add_edge(i, names[idx], weight=distances[idx])
        
    print("KI_GRAPH: ", ki_graph)
    summary(ki_graph)
    idx1 = random.choice(list(ki_chunks))
    idx2 = random.choice(list(ki_chunks))
    print(idx1, idx2)
    path = ki_graph.get_shortest_paths(idx1, to=idx2, mode=OUT, output='vpath',weights='weight')[0]
    for i,p in enumerate(path):
        z = ki_chunks[names[p]]
        output_image(z, 'ki_path_' + str(i))

    
    # CLOSEST FURTHEST ANALYSES
    cosine_dists, nonlins, denses, diffs = [], [], [], []
    sampled_cosine_dists, sampled_nonlins, sampled_denses, sampled_diffs = [], [], [], []
    max_cd, max_nl, max_dens, max_diff = float('-inf'),float('-inf'),float('-inf'),float('-inf')
    min_cd, min_nl, min_dens, min_diff = float('inf'),float('inf'),float('inf'),float('inf')
    max_cd_z, min_cd_z, max_nl_z, min_nl_z, max_dens_z, min_dens_z, max_diff_z, min_diff_z = None,None,None,None,None,None,None,None
    z_0 = ki_chunks['ki_chunk_0.txt']
    z_orig = z_0
    z_0 = z_0.detach().numpy()
    nl_0 = nonlinearity(z_0)
    diff_0 = difficulty(z_0)
    dens_0 = density(z_0)
    for k in ki_chunks:
        if k == 'ki_chunk_0.txt':
            continue
        z = ki_chunks[k]
        z = z.detach().numpy()
        nl, diff, dens = nonlinearity(z), difficulty(z), density(z)
        nl_diff, diff_diff, dens_diff = math.fabs(nl - nl_0), math.fabs(diff - diff_0), math.fabs(dens - dens_0)
        cd = distance.cosine(z_0,z)
        if cd > max_cd:
            max_cd = cd
            max_cd_z = ki_chunks[k]
        elif cd < min_cd:
            min_cd = cd
            min_cd_z = ki_chunks[k]

        if nl_diff > max_nl:
            max_nl = nl_diff
            max_nl_z = ki_chunks[k]
        elif nl_diff < min_nl:
            min_nl = nl_diff
            min_nl_z = ki_chunks[k]

        if diff_diff > max_diff:
            max_diff = diff_diff
            max_diff_z = ki_chunks[k]
        elif diff_diff < min_diff:
            min_diff = diff_diff
            min_diff_z = ki_chunks[k]

        if dens_diff > max_dens:
            max_dens = dens_diff
            max_dens_z = ki_chunks[k]
        elif dens_diff < min_dens:
            min_dens = dens_diff
            min_dens_z = ki_chunks[k]
        

    output_image(z_orig, 'ki_0')
    print(type(z_orig))
    print(type(min_cd_z), min_cd_z, min_cd_z.shape)
    output_image(min_cd_z, 'ki_0_cosine_closest')
    output_image(max_cd_z, 'ki_0_cosine_farthest')
    output_image(min_nl_z, 'ki_0_nonlinearity_closest')
    output_image(max_nl_z, 'ki_0_nonlinearity_farthest')
    output_image(min_dens_z, 'ki_0_density_closest')
    output_image(max_dens_z, 'ki_0_density_farthest')
    output_image(min_diff_z, 'ki_0_difficulty_closest')
    output_image(max_diff_z, 'ki_0_difficulty_farthest')

    max_cd, max_nl, max_dens, max_diff = float('-inf'),float('-inf'),float('-inf'),float('-inf')
    min_cd, min_nl, min_dens, min_diff = float('inf'),float('inf'),float('inf'),float('inf')
    max_cd_z, min_cd_z, max_nl_z, min_nl_z, max_dens_z, min_dens_z, max_diff_z, min_diff_z = None,None,None,None,None,None,None,None
    sampled_chunks = torch.FloatTensor(10000, 32).normal_().mul_(1).to('cpu')
    for sc in sampled_chunks:
        sc = sc.view(1,nz)
        nl, diff, dens = nonlinearity(sc), difficulty(sc), density(sc)
        nl_diff, diff_diff, dens_diff = math.fabs(nl - nl_0), math.fabs(diff - diff_0), math.fabs(dens - dens_0)
        sc_d = sc.detach().numpy()
        cd = distance.cosine(z_0,sc_d)
        if cd > max_cd:
            max_cd = cd
            max_cd_z = sc
        elif cd < min_cd:
            min_cd = cd
            min_cd_z = sc

        if nl_diff > max_nl:
            max_nl = nl_diff
            max_nl_z = sc
        elif nl_diff < min_nl:
            min_nl = nl_diff
            min_nl_z = sc

        if diff_diff > max_diff:
            max_diff = diff_diff
            max_diff_z = sc
        elif diff_diff < min_diff:
            min_diff = diff_diff
            min_diff_z = sc

        if dens_diff > max_dens:
            max_dens = dens_diff
            max_dens_z = sc
        elif dens_diff < min_dens:
            min_dens = dens_diff
            min_dens_z = sc

    output_image(min_cd_z, 'ki_0_sampled_cosine_closest')
    output_image(max_cd_z, 'ki_0_sampled_cosine_farthest')
    output_image(min_nl_z, 'ki_0_sampled_nonlinearity_closest')
    output_image(max_nl_z, 'ki_0_sampled_nonlinearity_farthest')
    output_image(min_dens_z, 'ki_0_sampled_density_closest')
    output_image(max_dens_z, 'ki_0_sampled_density_farthest')
    output_image(min_diff_z, 'ki_0_sampled_difficulty_closest')
    output_image(max_diff_z, 'ki_0_sampled_difficulty_farthest')

    
    sampled_chunks = torch.FloatTensor(1000, 32).normal_().mul_(1).to('cpu')
    kNN = 10
    ki_sampled_graph = Graph(directed=True)
    ki_sampled_graph.add_vertices(len(sampled_chunks))

    for i in range(len(sampled_chunks)):
        distances = []
        for j in range(len(sampled_chunks)):
            z_i, z_j = sampled_chunks[i].view(1,32), sampled_chunks[j].view(1,32)
            z_i_d, z_j_d = z_i.detach().numpy(), z_j.detach().numpy()
            distances.append(distance.cosine(z_i_d, z_j_d))
        idx_knn = sorted(range(len(distances)),key=lambda k: distances[k])[1:kNN+1]
        for idx in idx_knn:
            ki_sampled_graph.add_edge(i, idx, weight=distances[idx])
        
    print("KI_SAMP_GRAPH: ", ki_sampled_graph)
    summary(ki_sampled_graph)
    idx1 = int(len(sampled_chunks) * random.random())
    idx2 = int(len(sampled_chunks) * random.random())
    print(idx1, idx2)
    path = ki_sampled_graph.get_shortest_paths(idx1, to=idx2, mode=OUT, output='vpath',weights='weight')[0]
    for i,p in enumerate(path):
        z = sampled_chunks[p].view(1,nz)
        output_image(z, 'ki_sampled_path_' + str(i))
    
    # REPEAT FOR SMB
    model = model_smb
    model.eval()
    i = 0
    for file in os.listdir('levels/chunks_mario/'):
        z = get_z_from_file('levels/chunks_mario/' + file)
        smb_chunks[file] = z
        i += 1
    
    smb_array = []
    for kc in smb_chunks:
        smb_array.append(smb_chunks[kc].view(32).detach().numpy())
    smb_array = np.array(smb_array)
    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(smb_array)                                                                                                   
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 2000
    height = 2000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    for z, x, y in zip(smb_array, tx, ty):
        tile = get_image(z)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    plt.figure(figsize = (16,12))
    plt.imshow(full_image)
    plt.imsave('plot_mario.png',full_image)
    
    nx, ny = 20, 10
    grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))

    tile_width = 72
    tile_height = 56

    full_width = tile_width * nx
    full_height = tile_height * ny
    aspect_ratio = float(tile_width) / tile_height

    grid_image = Image.new('RGB', (full_width, full_height))

    for z, grid_pos in zip(smb_array, grid_assignment[0]):
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y
        tile = get_image(z)
        tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
        if (tile_ar > aspect_ratio):
            margin = 0.5 * (tile.width - aspect_ratio * tile.height)
            tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
        else:
            margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
            tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
        tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
        grid_image.paste(tile, (int(x), int(y)))

    plt.figure(figsize = (16,12))
    plt.imshow(grid_image)
    plt.imsave('grid_smb.png',grid_image)

    sampled_chunks = torch.FloatTensor(200, 32).normal_().mul_(1).to('cpu')
    sampled_smb_array = []
    for sc in sampled_chunks:
        sampled_smb_array.append(sc.view(32).detach().numpy())
    sampled_smb_array = np.array(sampled_smb_array)
    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(sampled_smb_array)                                                                                                   
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 2000
    height = 2000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    for z, x, y in zip(sampled_smb_array, tx, ty):
        tile = get_image(z)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    print(full_image)
    plt.figure(figsize = (16,12))
    plt.imshow(full_image)
    plt.imsave('plot_smb_sampled.png',full_image)

    nx, ny = 20, 10
    grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))

    tile_width = 72
    tile_height = 56

    full_width = tile_width * nx
    full_height = tile_height * ny
    aspect_ratio = float(tile_width) / tile_height

    grid_image = Image.new('RGB', (full_width, full_height))

    for z, grid_pos in zip(sampled_smb_array, grid_assignment[0]):
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y
        tile = get_image(z)
        tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
        if (tile_ar > aspect_ratio):
            margin = 0.5 * (tile.width - aspect_ratio * tile.height)
            tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
        else:
            margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
            tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
        tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
        grid_image.paste(tile, (int(x), int(y)))

    plt.figure(figsize = (16,12))
    plt.imshow(grid_image)
    plt.imsave('grid_smb_sampled.png',grid_image)

    # REPEAT FOR COMBINED MODEL
    model = model_both
    model.eval()
    i = 0
    all_chunks = {}
    for file in os.listdir('levels/chunks_mario/'):
        z = get_z_from_file('levels/chunks_mario/' + file)
        all_chunks[file] = z
        i += 1

    for file in os.listdir('levels/chunks_icarus/'):
        z = get_z_from_file('levels/chunks_icarus/' + file)
        all_chunks[file] = z
        i += 1
    
    all_array = []
    for kc in all_chunks:
        all_array.append(all_chunks[kc].view(32).detach().numpy())
    all_array = np.array(all_array)
    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(all_array)                                                                                                   
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 2000
    height = 2000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    for z, x, y in zip(all_array, tx, ty):
        tile = get_image(z)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    plt.figure(figsize = (16,12))
    plt.imshow(full_image)
    plt.imsave('plot_both.png',full_image)
    
    nx, ny = 27, 14
    print("Transforming...")
    grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))
    print("After transforming...")
    tile_width = 72
    tile_height = 56

    full_width = tile_width * nx
    full_height = tile_height * ny
    aspect_ratio = float(tile_width) / tile_height

    grid_image = Image.new('RGB', (full_width, full_height))

    for z, grid_pos in zip(all_array, grid_assignment[0]):
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y
        tile = get_image(z)
        tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
        if (tile_ar > aspect_ratio):
            margin = 0.5 * (tile.width - aspect_ratio * tile.height)
            tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
        else:
            margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
            tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
        tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
        grid_image.paste(tile, (int(x), int(y)))

    plt.figure(figsize = (16,12))
    plt.imshow(grid_image)
    plt.imsave('grid_both.png',grid_image)

    sampled_chunks = torch.FloatTensor(200, 32).normal_().mul_(1).to('cpu')
    sampled_both_array = []
    for sc in sampled_chunks:
        sampled_both_array.append(sc.view(32).detach().numpy())
    sampled_both_array = np.array(sampled_both_array)
    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(sampled_both_array)                                                                                                   
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 2000
    height = 2000
    max_dim = 100

    full_image = Image.new('RGBA', (width, height))
    for z, x, y in zip(sampled_both_array, tx, ty):
        tile = get_image(z)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    plt.figure(figsize = (16,12))
    plt.imshow(full_image)
    plt.imsave('plot_both_sampled.png',full_image)

    nx, ny = 20, 10
    grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))

    tile_width = 72
    tile_height = 56

    full_width = tile_width * nx
    full_height = tile_height * ny
    aspect_ratio = float(tile_width) / tile_height

    grid_image = Image.new('RGB', (full_width, full_height))

    for z, grid_pos in zip(sampled_both_array, grid_assignment[0]):
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y
        tile = get_image(z)
        tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
        if (tile_ar > aspect_ratio):
            margin = 0.5 * (tile.width - aspect_ratio * tile.height)
            tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
        else:
            margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
            tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
        tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
        grid_image.paste(tile, (int(x), int(y)))

    plt.figure(figsize = (16,12))
    plt.imshow(grid_image)
    plt.imsave('grid_both_sampled.png',grid_image)
    
    kNN = 5
    smb_graph = Graph(directed=True)
    for kc in smb_chunks:
        smb_graph.add_vertex(kc)

    for i in smb_chunks:
        distances, names = [], []
        for j in smb_chunks:
            z_i, z_j = smb_chunks[i], smb_chunks[j]
            z_i_d, z_j_d = z_i.detach().numpy(), z_j.detach().numpy()
            distances.append(distance.cosine(z_i_d, z_j_d))
            names.append(j)
        idx_knn = sorted(range(len(distances)),key=lambda k: distances[k])[1:kNN+1]
        for idx in idx_knn:
            smb_graph.add_edge(i, names[idx], weight=distances[idx])
        
    print("SMB_GRAPH: ", smb_graph)
    summary(smb_graph)
    idx1 = random.choice(list(smb_chunks))
    idx2 = random.choice(list(smb_chunks))
    print(idx1, idx2)
    path = smb_graph.get_shortest_paths(idx1, to=idx2, mode=OUT, output='vpath',weights='weight')[0]
    print(path)
    for i,p in enumerate(path):
        z = smb_chunks[names[p]]
        output_image(z, 'smb_path_' + str(i))
    sys.exit()
    """
    """
    max_cd, max_nl, max_dens, max_diff = float('-inf'),float('-inf'),float('-inf'),float('-inf')
    min_cd, min_nl, min_dens, min_diff = float('inf'),float('inf'),float('inf'),float('inf')
    max_cd_z, min_cd_z, max_nl_z, min_nl_z, max_dens_z, min_dens_z, max_diff_z, min_diff_z = None,None,None,None,None,None,None,None
    z_0 = smb_chunks['smb_chunk_10.txt']
    z_orig = z_0
    z_0 = z_0.detach().numpy()
    nl_0 = nonlinearity(z_0)
    diff_0 = difficulty(z_0)
    dens_0 = density(z_0)
    for k in smb_chunks:
        if k == 'smb_chunk_10.txt':
            continue
        z = smb_chunks[k]
        z = z.detach().numpy()
        nl, diff, dens = nonlinearity(z), difficulty(z), density(z)
        nl_diff, diff_diff, dens_diff = math.fabs(nl - nl_0), math.fabs(diff - diff_0), math.fabs(dens - dens_0)
        cd = distance.cosine(z_0,z)
        if cd > max_cd:
            max_cd = cd
            max_cd_z = smb_chunks[k]
        elif cd < min_cd:
            min_cd = cd
            min_cd_z = smb_chunks[k]

        if nl_diff > max_nl:
            max_nl = nl_diff
            max_nl_z = smb_chunks[k]
        elif nl_diff < min_nl:
            min_nl = nl_diff
            min_nl_z = smb_chunks[k]

        if diff_diff > max_diff:
            max_diff = diff_diff
            max_diff_z = smb_chunks[k]
        elif diff_diff < min_diff:
            min_diff = diff_diff
            min_diff_z = smb_chunks[k]

        if dens_diff > max_dens:
            max_dens = dens_diff
            max_dens_z = smb_chunks[k]
        elif dens_diff < min_dens:
            min_dens = dens_diff
            min_dens_z = smb_chunks[k]
        

    output_image(z_orig, 'smb_10')
    output_image(min_cd_z, 'smb_10_cosine_closest')
    output_image(max_cd_z, 'smb_10_cosine_farthest')
    output_image(min_nl_z, 'smb_10_nonlinearity_closest')
    output_image(max_nl_z, 'smb_10_nonlinearity_farthest')
    output_image(min_dens_z, 'smb_10_density_closest')
    output_image(max_dens_z, 'smb_10_density_farthest')
    output_image(min_diff_z, 'smb_10_difficulty_closest')
    output_image(max_diff_z, 'smb_10_difficulty_farthest')

    max_cd, max_nl, max_dens, max_diff = float('-inf'),float('-inf'),float('-inf'),float('-inf')
    min_cd, min_nl, min_dens, min_diff = float('inf'),float('inf'),float('inf'),float('inf')
    max_cd_z, min_cd_z, max_nl_z, min_nl_z, max_dens_z, min_dens_z, max_diff_z, min_diff_z = None,None,None,None,None,None,None,None
    
    
    sampled_chunks = torch.FloatTensor(1000, 32).normal_().mul_(1).to('cpu')

    kNN = 10
    smb_sampled_graph = Graph(directed=True)
    smb_sampled_graph.add_vertices(len(sampled_chunks))

    for i in range(len(sampled_chunks)):
        distances = []
        for j in range(len(sampled_chunks)):
            z_i, z_j = sampled_chunks[i].view(1,32), sampled_chunks[j].view(1,32)
            z_i_d, z_j_d = z_i.detach().numpy(), z_j.detach().numpy()
            distances.append(distance.cosine(z_i_d, z_j_d))
        idx_knn = sorted(range(len(distances)),key=lambda k: distances[k])[1:kNN+1]
        for idx in idx_knn:
            smb_sampled_graph.add_edge(i, idx, weight=distances[idx])
        
    print("SMB_SAMP_GRAPH: ", smb_sampled_graph)
    summary(smb_sampled_graph)
    idx1 = int(len(sampled_chunks) * random.random())
    idx2 = int(len(sampled_chunks) * random.random())
    print(idx1, idx2)
    path = smb_sampled_graph.get_shortest_paths(idx1, to=idx2, mode=OUT, output='vpath',weights='weight')[0]
    for i,p in enumerate(path):
        z = sampled_chunks[p].view(1,nz)
        output_image(z, 'smb_sampled_path_' + str(i))
    
    for sc in sampled_chunks:
        sc = sc.view(1,nz)
        nl, diff, dens = nonlinearity(sc), difficulty(sc), density(sc)
        nl_diff, diff_diff, dens_diff = math.fabs(nl - nl_0), math.fabs(diff - diff_0), math.fabs(dens - dens_0)
        sc_d = sc.detach().numpy()
        cd = distance.cosine(z_0,sc_d)
        if cd > max_cd:
            max_cd = cd
            max_cd_z = sc
        elif cd < min_cd:
            min_cd = cd
            min_cd_z = sc

        if nl_diff > max_nl:
            max_nl = nl_diff
            max_nl_z = sc
        elif nl_diff < min_nl:
            min_nl = nl_diff
            min_nl_z = sc

        if diff_diff > max_diff:
            max_diff = diff_diff
            max_diff_z = sc
        elif diff_diff < min_diff:
            min_diff = diff_diff
            min_diff_z = sc

        if dens_diff > max_dens:
            max_dens = dens_diff
            max_dens_z = sc
        elif dens_diff < min_dens:
            min_dens = dens_diff
            min_dens_z = sc

    output_image(min_cd_z, 'smb_10_sampled_cosine_closest')
    output_image(max_cd_z, 'smb_10_sampled_cosine_farthest')
    output_image(min_nl_z, 'smb_10_sampled_nonlinearity_closest')
    output_image(max_nl_z, 'smb_10_sampled_nonlinearity_farthest')
    output_image(min_dens_z, 'smb_10_sampled_density_closest')
    output_image(max_dens_z, 'smb_10_sampled_density_farthest')
    output_image(min_diff_z, 'smb_10_sampled_difficulty_closest')
    output_image(max_diff_z, 'smb_10_sampled_difficulty_farthest')
    
    print(len(cosine_dists))
    
    cosine_dists = sorted(cosine_dists)
    nonlins = sorted(nonlins)
    denses = sorted(denses)
    diffs = sorted(diffs)

    sampled_cosine_dists = sorted(sampled_cosine_dists)
    sampled_nonlins = sorted(sampled_nonlins)
    sampled_denses = sorted(sampled_denses)
    sampled_diffs = sorted(sampled_diffs)
    
    output_image(sampled_cosine_dists[0], 'ki_0_sampled_cosine_closest.png')
    output_image(sampled_cosine_dists[189], 'ki_0_sampled_cosine_farthest.png')
    output_image(sampled_nonlins[0], 'ki_0_sampled_nonlinearity_closest.png')
    output_image(sampled_nonlins[189], 'ki_0_sampled_nonlinearity_farthest.png')
    output_image(sampled_denses[0], 'ki_0_sampled_density_closest.png')
    output_image(sampled_denses[189], 'ki_0_sampled_density_farthest.png')
    output_image(sampled_diffs[0], 'ki_0_sampled_difficulty_closest.png')
    output_image(sampled_diffs[189], 'ki_0_sampled_difficulty_farthest.png')

    sys.exit()

    z = torch.FloatTensor(1, nz).normal_(0,1)
    level = model.decode(z)
    im = level.data.cpu().numpy()
    im = np.argmax( im, axis = 1)
    
    interpolate_chunks('levels/chunks_ng/smb_chunk_15.txt','levels/chunks_ng/ki_chunk_187.txt',6)
