#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 22:40:05 2019

@author: vijay
"""

from io import open

import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import trainmodel as tm 
import wordindex as preparedata 
import config as config
#check for the nvidia graphics
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hidden_size = 256

#input_lang, output_lang, pairs = preparedata.prepareData('eng', 'fra', True)
#encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
#attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

tm.trainIters(config.encoder1, config.attn_decoder1, 75000, print_every=5000)