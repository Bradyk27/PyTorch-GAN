import argparse
import os
import numpy as np
import math
import pandas as pd

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import pickle
from numpy.lib.format import open_memmap

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=9, help="number of classes for dataset")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

class Feeder(torch.utils.data.Dataset): #Borrowed from HCN-PyTorch
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 num_frame_path,
                 random_valid_choose=False,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 normalization=False,
                 debug=False,
                 origin_transfer=False,
                 p_interval=1,
                 crop_resize=False,
                 rand_rotate=0,
                 mmap=False,
                 ):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.num_frame_path = num_frame_path
        self.random_choose = random_choose
        self.random_valid_choose = random_valid_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.origin_transfer = origin_transfer
        self.p_interval = p_interval
        self.crop_resize = crop_resize
        self.rand_rotate = rand_rotate
        self.mmap = mmap

        self.load_data()
        self.coordinate_transfer()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        # load label
        if '.pkl' in self.label_path:
            try:
                with open(self.label_path) as f:
                    self.sample_name, self.label = pickle.load(f)
            except:
                # for pickle file from python2
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(
                        f, encoding='latin1')
        # old label format
        elif '.npy' in self.label_path:
            self.label = list(np.load(self.label_path))
            self.sample_name = [str(i) for i in range(len(self.label))]
        else:
            raise ValueError()

        # load data
        if self.mmap == True:
            self.data = np.load(self.data_path,mmap_mode='r')
        else:
            self.data = np.load(self.data_path,mmap_mode=None) # directly load all data in memory, it more efficient but memory resource consuming for big file

        # load num of valid frame length
        self.valid_frame_num = np.load(self.num_frame_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.num_frame_path = self.num_frame_path[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def coordinate_transfer(self):
        data_numpy = self.data

        if self.origin_transfer == 2:
            #  take joints 2  of each person, as origins of each person
            origin = data_numpy[:, :, :, 1, :]
            data_numpy = data_numpy - origin[:, :, :, None, :]
        elif self.origin_transfer == 0:
            #  take joints 2  of first person, as origins of each person
            origin = data_numpy[:, :, :, 1, 0]
            data_numpy = data_numpy - origin[:, :, :, None, None]
        elif self.origin_transfer == 1:
            #  take joints 2  of second person, as origins of each person
            origin = data_numpy[:, :, :, 1, 1]
            data_numpy = data_numpy - origin[:, :, :, None, None]
        else:
            # print('no origin transfer')
            pass

        self.data = data_numpy

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(
            axis=2, keepdims=True).mean(
            axis=4, keepdims=True).mean(axis=0)
        # mean_map 3,1,25,1
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
            (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

        self.min_map = (data.transpose(0, 2, 3, 4, 1).reshape(N * T * V * M, C)).min(axis=0)
        self.max_map = (data.transpose(0, 2, 3, 4, 1).reshape(N * T * V * M, C)).max(axis=0)
        self.mean_mean = (data.transpose(0, 2, 3, 4, 1).reshape(N * T * V * M, C)).mean(axis=0)

        print('min_map', self.min_map, 'max_map', self.max_map, 'mean', self.mean_mean)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get data
        # input: C, T, V, M
        data_numpy = self.data[index]
        # if self.mmap = True, the loaded data_numpy is read-only, and torch.utils.data.DataLoader could load type 'numpy.core.memmap.memmap'
        if self.mmap:
            data_numpy = np.array(data_numpy) # convert numpy.core.memmap.memmap to numpy

        label = self.label[index]
        valid_frame_num = self.valid_frame_num[index]

        # normalization
        if self.normalization:
            # data_numpy = (data_numpy - self.mean_map) / self.std_map
            # be careful the value is for NTU_RGB-D, for other dataset, please replace with value from function 'get_mean_map'
            if self.origin_transfer == 0:
                min_map, max_map = np.array([-4.9881773, -2.939787, -4.728529]), np.array(
                    [5.826573, 2.391671, 4.824233])
            elif self.origin_transfer == 1:
                min_map, max_map = np.array([-5.836631, -2.793758, -4.574943]), np.array([5.2021008, 2.362596, 5.1607])
            elif self.origin_transfer == 2:
                min_map, max_map = np.array([-2.965678, -1.8587272, -4.574943]), np.array(
                    [2.908885, 2.0095677, 4.843938])
            else:
                min_map, max_map = np.array([-3.602826, -2.716611, 0.]), np.array([3.635367, 1.888282, 5.209939])

            data_numpy = np.floor(255 * (data_numpy - min_map[:, None, None, None]) / \
                                  (max_map[:, None, None, None] - min_map[:, None, None, None])) / 255

        # processing
        if self.crop_resize:
            data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        if self.rand_rotate > 0:
            data_numpy = tools.rand_rotate(data_numpy, self.rand_rotate)

        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size, auto_pad=True)


        if self.random_valid_choose:
            data_numpy = tools.valid_choose(data_numpy, self.window_size, random_pad = True)
        elif self.window_size > 0 and (not self.crop_resize) and (not self.random_choose):
            data_numpy = tools.valid_choose(data_numpy, self.window_size, random_pad=False)

        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)

        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def crop(seff, data, tw, th):
        _, w, h, _ = data.shape
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return data[:, x1:x1 + tw, y1:y1 + th, :]

def weights_init_normal(m): #Are these good values? I have no clue
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)
        
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * 3 * 13)) #128 channels by 3 frames by 13 joints

        self.conv_blocks = nn.Sequential( #Scaling up to 128 channels, then tuning down to 3 while upsampling
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor = .25) ## Couldn't find a compatible downsampling, so I'm just doing it here after the activation function & upsampling twice. Need to check logic on this.
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, 3, 13)
        out = self.conv_blocks(out)
        # print("Generator: ", out.shape)
        return out ## 64, 3, 3, 13 tensor

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential( #Scaling up from 3 channels to 128
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128, opt.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        # print("Discriminator: ", out.shape)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataloader = torch.utils.data.DataLoader( #Hardcoded feeder cuz I'm feeling lazy
            dataset=Feeder(data_path="../../data/lidar_mocap/train_data.npy", label_path="../../data/lidar_mocap/train_label.pkl", num_frame_path="../../data/lidar_mocap/train_num_frame.npy"),
            batch_size=64,
            shuffle=True)
'''
train_features, train_labels = next(iter(dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
print(train_features.shape)
print("STOP")
'''

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def save_skel(n_row, batches_done): #Saving as pngs? Really have no clue how to save / output these. Will save for later. Little curious what these pngs will look like lol
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row**2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)]) ## Can change this in order to get more samples, if we'd like. Probably a good idea tbh.
    labels = Variable(LongTensor(labels))
    gen_skel = generator(z, labels)
    #print(gen_skel.detach().numpy())

    sample_name = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9']
    for i in range(8): ## Change this in order to get more samples
        sample_name.extend(['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9'])

    with open('skels/{}_label.pkl'.format(batches_done), 'wb') as f:
        pickle.dump((sample_name, list([label+1 for label in labels.tolist()])), f)

    fp = open_memmap(
        'skels/{}_data.npy'.format(batches_done),
        dtype='float32',
        mode='w+',
        shape=(len(labels), 3, 3, 13, 1)) # Num samples, Num channels, Num Frames (might need to make 20...), Num Joints, Num Bodies

    fl = open_memmap(
        'skels/{}_num_frame.npy'.format(batches_done),
        dtype='int',
        mode='w+',
        shape=(len(labels),))
    
    gen_skel = gen_skel.unsqueeze(4)
    if cuda:
        gen_skel = gen_skel.cpu()
    for i in range(len(labels)):
        fp[i, :, :, :, :] = gen_skel[i].detach().numpy()
        fl[i] = 3 #Num frames

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (skel, labels) in enumerate(dataloader):
        skel = skel.squeeze()
        batch_size = skel.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_skel = Variable(skel.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_skel = generator(z, gen_labels)
        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_skel)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_skel)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_skel.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d]\n[D Adv_Real loss: %f, D Aux_Real loss: %f, D Adv_Fake loss: %f, D Aux_Fake loss: %f, acc: %d%%]\n[G Adv loss: %f, G Aux loss: %f]\n"
            % (epoch, opt.n_epochs, i, len(dataloader), adversarial_loss(real_pred, valid), auxiliary_loss(real_aux, labels),
             adversarial_loss(fake_pred, fake), auxiliary_loss(fake_aux, gen_labels), 100 * d_acc, adversarial_loss(validity, valid), auxiliary_loss(pred_label, gen_labels))
        )
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_skel(n_row=opt.n_classes, batches_done=batches_done) #Fix this!! I have no clue if it's actually saving or not
