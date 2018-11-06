import os, time, json, re
import itertools, argparse, pickle

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

import torchvision.transforms as T
from torchvision import models


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
    return trn_raw_captions, dev_raw_captions, test_raw_captions, raw_captions


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
    ax2.set_ylim(bottom=-0.1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(['train', 'val'], loc='upper right')
    fig.savefig(fname)