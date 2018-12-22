import os, time, json, re
import itertools, argparse, pickle, random

import numpy as np
import pandas as pd
import nltk
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader, sampler

import torchvision.transforms as T
from torchvision import models

from tokenize_caption import *
from encode_image import *


def split_image_files(path):
    # load file names
    im_files = os.listdir(path + 'Flicker8k_Dataset')
    trn_files = open(path+'Flickr_8k.trainImages.txt', 'r').read().strip().split('\n')
    dev_files = open(path+'Flickr_8k.devImages.txt', 'r').read().strip().split('\n')
    test_files = open(path+'Flickr_8k.testImages.txt', 'r').read().strip().split('\n')
    trn_files += list(set(im_files) - set(trn_files) - set(dev_files) - set(test_files))
    return trn_files, dev_files, test_files


def split_captions(path, trn_files, dev_files, test_files):
    # load raw captions
    raw_f = open(path + 'Flickr8k.token.txt', 'r').read().strip().split('\n')
    raw_captions = {}
    for line in raw_f:
        line = line.split('\t')
        im_id, cap = line[0][:len(line[0])-2], line[1]
        if im_id not in raw_captions:
            raw_captions[im_id] = ['<start> ' + cap + ' <end>']
        else:
            raw_captions[im_id].append('<start> ' + cap + ' <end>')
    trn_raw_captions, dev_raw_captions, test_raw_captions = {}, {}, {}
    for im_id in trn_files: trn_raw_captions[im_id] = raw_captions[im_id]
    for im_id in dev_files: dev_raw_captions[im_id] = raw_captions[im_id]
    for im_id in test_files: test_raw_captions[im_id] = raw_captions[im_id]
    return trn_raw_captions, dev_raw_captions, test_raw_captions


def decode_captions(tokens, idx_to_word):
    '''
    Inputs:
    - tokens: (N, ) or (N, T) array
    - idx_to_word: mapping from index to word
    Returns:
    - decoded: list of decoded sentences
    '''
    singleton = False
    if tokens.ndim == 1:
        singleton = True
        tokens = tokens[None]
    decoded = []
    N, T = tokens.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[tokens[i, t]]
            if word != '<pad>':
                words.append(word)
            if word == '<end>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def shuffle_data(data, split='train'):
    size = data['%s_captions' % split].shape[0]
    mask = np.random.permutation(size)
    data['%s_captions' % split] = data['%s_captions' % split][mask]
    data['%s_image_ids' % split] = data['%s_image_ids' % split][mask]


def get_batch(data, idx, batch_size, split='train'):
    b_targets = data['%s_captions' % split][idx:idx+batch_size]
    b_ids = data['%s_image_ids' % split][idx:idx+batch_size]
    b_features = [torch.FloatTensor(data['%s_features' % split][id]) for id in b_ids]
    return torch.LongTensor(b_targets), torch.stack(b_features)


def sample_batch(data, batch_size, split='train'):
    size = data['%s_captions' % split].shape[0]
    mask = np.random.choice(size, batch_size)
    b_targets = data['%s_captions' % split][mask]
    b_ids = data['%s_image_ids' % split][mask]
    b_features = [torch.FloatTensor(data['%s_features' % split][id]) for id in b_ids]
    return torch.LongTensor(b_targets), torch.stack(b_features), b_ids


# for exploiting data.Dataset and data.DataLoader, but time too long during training
def get_captions_ids_mapping(path, data_part, maxlen=40, threshold=1):
    # word_idx_map(), tokenize() defined in tokenize_caption.py
    trn_files, dev_files, test_files = split_image_files(path)
    trn_raw_captions, dev_raw_captions, test_raw_captions = \
        split_captions(path, trn_files, dev_files, test_files)
    idx_to_word, word_to_idx = word_idx_map(trn_raw_captions, threshold)
    if data_part == 'train':
        captions, image_ids = tokenize(trn_raw_captions, word_to_idx, maxlen)
    if data_part == 'val':
        captions, image_ids = tokenize(dev_raw_captions, word_to_idx, maxlen)
    if data_part == 'test':
        captions, image_ids = tokenize(test_raw_captions, word_to_idx, maxlen)
    return captions, image_ids, idx_to_word, word_to_idx


class Flikr8k(Dataset):

    def __init__(self, path, data_part, transform=None):
        self.path = path
        self.captions, self.image_ids, self.idx_to_word, self.word_to_idx = \
            get_captions_ids_mapping(path, data_part)
        self.transform = transform
        '''
        self.pre_net = pre_net
        if pre_net == 'inception_v3':
            self.net = models.inception_v3(pretrained=True)
        if pre_net == 'densenet161':
            self.net = models.densenet161(pretrained=True)
        if pre_net == 'resnet101':
            self.net = models.resnet101(pretrained=True)
        if pre_net == 'vgg16':
            self.net = models.vgg16_bn(pretrained=True)
        '''

    def __getitem__(self, index):
        # feed_forward_net() defined in encode_image.py
        caption = self.captions[index]
        im_id = self.image_ids[index]
        im = Image.open(self.path+'Flicker8k_Dataset/'+im_id)
        if self.transform is not None:
            im = self.transform(im)
        '''
        with torch.no_grad():
            self.net.type(dtype)
            self.net.eval()
            im = im.type(dtype)
            im = feed_forward_net(im, self.net, self.pre_net)
        '''
        return im, torch.LongTensor(caption)

    def __len__(self):
        return len(self.image_ids)


def prepare_loader(path, batch_size, data_part, shuffle=True):
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]

    transform_train = T.Compose([
                    T.Resize((224, 224)),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    T.RandomResizedCrop(224, scale=(0.75, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(rgb_mean, rgb_std),
                ])
    transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(rgb_mean, rgb_std),
                ])

    if data_part == 'train':
        data_set = Flikr8k(path, data_part, transform=transform_train)
        loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    else:
        data_set = Flikr8k(path, data_part, transform=transform)
        dset_sampler = sampler.SubsetRandomSampler(range(len(data_set))) if shuffle else None
        loader = DataLoader(data_set, batch_size=batch_size, sampler=dset_sampler)
    return loader


# plot log of loss and bleu during training
def plot_history(history, fname):
    bleus, val_bleus, losses, val_losses = history
    epochs = range(len(bleus))
    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs, bleus, '-o')
    ax1.plot(epochs, val_bleus, '-o')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Bleu')
    ax1.legend(['train', 'val'], loc='lower right')
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(epochs, losses, '-o')
    ax2.plot(epochs, val_losses, '-o')
    #ax2.set_ylim(bottom=-0.1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(['train', 'val'], loc='upper right')
    fig.savefig(fname)
