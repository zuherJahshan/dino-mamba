import torch
import glob
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

datapath = '/workspace/mamba_based_slm/data/LibriSpeech/train-*/**/*.flac'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class AudioDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform.squeeze(0)  # Remove channel dimension if it is 1

def collate_fn(batch):
    # Move data to CUDA device
    
    # Pad all waveforms to the maximum length in the batch
    batched_lengths = torch.tensor([waveform.shape[0] for waveform in batch]).to(device)
    padded_waveforms = pad_sequence(batch, batch_first=True, padding_value=0).to(device)
    
    padded_waveforms = padded_waveforms.to(device, non_blocking=True)
    batched_lengths = batched_lengths.to(device, non_blocking=True)
    
    return padded_waveforms, batched_lengths

def get_dataloader(batch_size, device='cuda', num_workers=8, sample_rate=16000):
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    train_files = glob.glob(datapath, recursive=True)
    dataset = AudioDataset(train_files, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
