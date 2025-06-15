from torch.autograd import Variable
from models import *

from utils import ls, preprocess_wav, to_numpy, denormalize
from spec_utils import mel_spectrogram, CONFIG

from tqdm import tqdm

shared_dim = 32 * 2 ** 2
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

encoder = Encoder(dim=32, in_channels=1, n_downsample=2)
G_trg= Generator(dim=32, out_channels=1, n_upsample=2, shared_block=ResidualBlock(features=shared_dim))

# load path
encoder.load_state_dict(torch.load(".\\models\\encoder_99.pth", map_location=torch.device('cpu')))
G_trg.load_state_dict(torch.load(".\\models\\G2_99.pth", map_location=torch.device('cpu')))

encoder.eval()
G_trg.eval()

def infer(S, multi):
    """Takes in a standard sized spectrogram, returns timbre converted version"""
    S = torch.from_numpy(S)
    S = S.view(1, 1, 100, 128)
    X = Variable(S.type(Tensor))
    
    ret = {} # just stores inference output

    mu, Z = encoder(X)
    attr_vector_pp = np.load('.\\models\\attr.npy')
    Z = Z + multi * torch.from_numpy(attr_vector_pp)
        
    fake_X = G_trg(Z)
    ret['fake'] = to_numpy(fake_X)
    
    return ret

def audio_infer(wav, multi):
    #print(multi)
    # Load audio and preprocess
    sample = preprocess_wav(wav)
    #spect_src = melspectrogram(sample)
    
    spect_src = mel_spectrogram(torch.tensor(sample, dtype=torch.float32).unsqueeze(0), 
                        n_fft=CONFIG['n_fft'],
                        num_mels=CONFIG['num_mels'],
                        sampling_rate=CONFIG['sampling_rate'],
                        hop_size=CONFIG['hop_size'],
                        win_size=CONFIG['win_size'],
                        fmin=CONFIG['fmin'],
                        fmax=CONFIG['fmax'])
    #print(spect_src.shape)

    spect_src = np.pad(spect_src, ((0,0),(128,128)), 'constant')  # padding for consistent overlap
    spect_trg = np.zeros(spect_src.shape)
    
    length = spect_src.shape[1]
    hop = 128 // 4

    for i in tqdm(range(0, length, hop)):
        x = i + 128

        # Get cropped spectro of right dims
        if x <= length:
            S = spect_src[:, i:x]
        else:  # pad sub on right if includes sub segments out of range
            S = spect_src[:, i:]
            S = np.pad(S, ((0,0),(x-length,0)), 'constant')
        ret = infer(S, multi) # perform inference from trained model
        T = ret['fake']

        # Add parts of target spectrogram with an average across overlapping segments    
        for j in range(0, 128, hop):
            y = j + hop
            if i+y > length: break  # neglect sub segments out of range
                
            # select subsegments to consider for overlap
            t = T[:, j:y]
            
            # add average element
            spect_trg[:, i+j:i+y] += t/4


    # remove initial padding
    spect_src = spect_src[:, 128:-128]
    spect_trg = spect_trg[:, 128:-128]
    spect_trg = denormalize(spect_trg)

    return spect_trg



