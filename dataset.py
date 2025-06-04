import pandas as pd

import torchaudio
from torchaudio.transforms import Resample
from torch.utils.data import Dataset, DataLoader

SR = 16000

class MuTox(Dataset):
    def __init__(self, partition, lang=None):
        super().__init__()
        self.csv = pd.read_csv('mutox_clean.tsv', delimiter='\t')
        self.csv = self.csv[self.csv['partition']==partition]   # train/dev/devtest
        if lang:
            self.csv = self.csv[self.csv['lang']==lang]
        self.target_n_samples = 5 * SR
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        r = self.csv.iloc[index]
        id = r['id']
        lang = r['lang']
        transcription = r['audio_file_transcript']
        is_toxic = r['label']
        
        wav_path = f'waveforms/{id}.wav'
        try:
            signal, sr = torchaudio.load(wav_path)
            wav = self.format_wav(signal, sr)
            return {
                'id': id,
                'lang': lang,
                'wav': wav,
                'text': transcription,
                'is_toxic': is_toxic,
            }
        except Exception as e:
            print(wav_path, e)
            return {
                None
            }

    def format_wav(self, wav, sr):
        """
        Pad/Trim the waveform
        Audio length
            Mean: 7.46, Median: 4.35
        """
        if sr != SR:                                # Resample to SR=44100
            wav = Resample(sr, SR)
        if wav.shape[-1] < self.target_n_samples:   # Pad if shorter
            wav = F.pad(wav, (0, self.target_n_samples - wav.shape[-1]))
        else:                                       # Trim if longer
            wav = wav[:, :self.target_n_samples]
        return wav


def get_loader(partition, lang, batch_size, shuffle, num_workers):
    return DataLoader(MuTox(partition=partition, lang=lang), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

