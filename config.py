
"""
Created on Thu Jan 24 22:40:05 2019

@author: vijay
"""






from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import EncoderRNN, AttnDecoderRNN
import trainmodel as tm 
import wordindex as preparedata 

#check for the nvidia graphics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 256

input_lang, output_lang, pairs = preparedata.prepareData('eng', 'fra', True)
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
