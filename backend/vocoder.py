# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import torch
from scipy.io.wavfile import write
from models import BigVSAN as Generator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = False

MAX_WAV_VALUE = 32768.0

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def mel2audio(mel, h):
    generator = Generator(h).to(device)
    #print(device)
    #print(mel.shape, type(mel))
    state_dict_g = load_checkpoint('./backend/models/g_10020000', device)
    generator.load_state_dict(state_dict_g['generator'])
    #print(generator)
    generator.eval()
    generator.remove_weight_norm()
    
    with torch.no_grad():
        # load the mel spectrogram in .npy format
        x = torch.FloatTensor(mel).to(device)
        if len(x.shape) == 2:
                x = x.unsqueeze(0)
        #print('vocoder - generator start.')
        y_g_hat = generator(x)
        #print('vocoder - generator complete.')
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        return audio
