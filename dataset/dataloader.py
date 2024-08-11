import torch
import glob
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

datapaths = {
    "train": "/workspace/mamba_based_slm/data/LibriSpeech/train-*/**/*.flac",
    "abx": "/workspace/mamba_based_slm/data/zerospeech/datasets/abxLS-dataset/**/*.wav"
}

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

def get_dataloader(
    batch_size,
    data_type='train',
    device='cuda',
    num_workers=1,
    sample_rate=16000
):
    datapath = datapaths[data_type]
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


# import torch
# import glob
# import torchaudio
# from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# datapaths = {
#     "train": "/workspace/mamba_based_slm/data/LibriSpeech/train-*/**/*.flac",
#     "abx": "/workspace/mamba_based_slm/data/zerospeech/datasets/abxLS-dataset/**/*.wav"
# }

# class AudioDataset(Dataset):
#     def __init__(self, file_list, transform=None, return_filepaths=False):
#         self.file_list = file_list
#         self.transform = transform
#         self.return_filepaths = return_filepaths  # Flag to return file paths

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         audio_path = self.file_list[idx]
#         waveform, sample_rate = torchaudio.load(audio_path)
#         if self.transform:
#             waveform = self.transform(waveform)
#         waveform = waveform.squeeze(0)  # Remove channel dimension if it is 1
#         if self.return_filepaths:
#             return waveform, audio_path  # Return waveform and filepath
#         return waveform

# def collate_fn(batch):
#     # Separate waveforms and file paths if they are returned
#     if isinstance(batch[0], tuple):
#         waveforms, filepaths = zip(*batch)
#     else:
#         waveforms = batch
#         filepaths = None

#     # Pad all waveforms to the maximum length in the batch
#     batched_lengths = torch.tensor([waveform.shape[0] for waveform in waveforms]).to(device)
#     padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0).to(device)
    
#     batched_lengths = batched_lengths.to(device, non_blocking=True)
    
#     if filepaths is not None:
#         return padded_waveforms, batched_lengths, filepaths
#     return padded_waveforms, batched_lengths

# def get_dataloader(
#     batch_size,
#     data_type='train',
#     device='cuda',
#     num_workers=8,
#     sample_rate=16000,
#     return_filepaths=False
# ):
#     datapath = datapaths[data_type]
#     return_filepaths = data_type != 'train'
#     transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#     train_files = glob.glob(datapath, recursive=True)
#     dataset = AudioDataset(train_files, transform=transform, return_filepaths=return_filepaths)
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         collate_fn=collate_fn,
#     )
