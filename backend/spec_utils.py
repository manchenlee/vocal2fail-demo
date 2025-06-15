import torch
import torchaudio
import librosa
import numpy as np

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

MEI_BASIS = {}
HANN_WINDOW = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global MEI_BASIS, HANN_WINDOW
    if fmax not in MEI_BASIS:
        mel = librosa.filters.mel(
            sr=sampling_rate, 
            n_fft=n_fft, 
            n_mels=num_mels, 
            fmin=fmin, 
            fmax=fmax
        )
        MEI_BASIS[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        HANN_WINDOW[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=HANN_WINDOW[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(MEI_BASIS[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    #print(type(spec))
    spec_norm = normalize(spec.squeeze().numpy())
    #spec = spec.squeeze(0)
    #print(spec_norm)
    return spec_norm

def normalize(S, min_level_db=-80):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def denormalize(S, min_level_db=-80):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

CONFIG = {
        'n_fft': 1024,
        'num_mels': 100,
        'sampling_rate': 24000,
        'hop_size': 256,
        'win_size': 1024,
        'fmin': 0,
        'fmax': 12000,
    }

def main():
    # Main function for simple testing

    # config for mel spectrogram

    # Sample audio y: (N, ), assume we load audio with librosa
    # y, sr = librosa.load('path/to/audio.wav', sr=CONFIG['sampling_rate'])
    y = np.random.randn(100000)

    # Convert numpy array to torch tensor
    y = torch.tensor(y, dtype=torch.float32)

    # (N, ) => (1, N)
    y = y.unsqueeze(0)

    # Mel spectrogram
    # n_frames = (N - n_fft) // hop_size + 1
    # (1, N) => (1, num_mels, n_frames)
    mel_spec = mel_spectrogram(
        y,
        n_fft=CONFIG['n_fft'],
        num_mels=CONFIG['num_mels'],
        sampling_rate=CONFIG['sampling_rate'],
        hop_size=CONFIG['hop_size'],
        win_size=CONFIG['win_size'],
        fmin=CONFIG['fmin'],
        fmax=CONFIG['fmax']
    )
    print(mel_spec.shape)
    print(mel_spec)

    # you can scale the mel spectrogram to [0, 1] range with using normalize function for training
    # I used it for TT-VAE-GAN
    mel_spec_norm = normalize(mel_spec.squeeze().numpy())
    print(mel_spec_norm.shape)
    print(mel_spec_norm)

if __name__ == "__main__":
    main()