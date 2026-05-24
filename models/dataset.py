import os
import glob
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'pleasant_surprised', 'sad', 'neutral']
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(EMOTIONS)}

# TESS filenames often use 'ps' for pleasant_surprised
def parse_emotion(filename):
    basename = os.path.basename(filename).lower()
    basename = basename.replace('.wav', '')
    parts = basename.split('_')
    emo = parts[-1]
    if emo == 'ps':
        return 'pleasant_surprised'
    return emo

def parse_word(filename):
    basename = os.path.basename(filename).lower()
    parts = basename.split('_')
    if len(parts) >= 3:
        return parts[1]
    return ""

class TESSDataset(Dataset):
    def __init__(self, data_dir, split='train', max_audio_len=64000, max_text_len=16):
        """
        data_dir: Path to the TESS dataset folder (should contain wav files, possibly in subfolders).
        split: 'train' or 'test'.
        max_audio_len: Number of samples for audio. Default is 64000 (4 seconds at 16kHz).
        max_text_len: Maximum length of tokenized text sequence.
        """
        self.data_dir = data_dir
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.target_sample_rate = 16000
        
        # Load file paths
        self.file_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    self.file_paths.append(os.path.join(root, file))
                    
        # Simple split (80/20) based on deterministic sorted order
        self.file_paths.sort()
        np.random.seed(42)
        indices = np.random.permutation(len(self.file_paths))
        split_idx = int(0.8 * len(indices))
        
        if split == 'train':
            self.file_paths = [self.file_paths[i] for i in indices[:split_idx]]
        else:
            self.file_paths = [self.file_paths[i] for i in indices[split_idx:]]
            
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # 1. Parse Emotion Label
        emotion_str = parse_emotion(file_path)
        label = EMOTION_TO_IDX.get(emotion_str, 6) # default to neutral if parsing fails
        
        # 2. Process Speech
        waveform_np, sr = librosa.load(file_path, sr=self.target_sample_rate, mono=True)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        
        # Pad or truncate waveform
        if waveform.shape[1] > self.max_audio_len:
            waveform = waveform[:, :self.max_audio_len]
        else:
            pad_amount = self.max_audio_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        # Compute Mel Spectrogram (Features: 64 mel bins)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_mels=64,
            n_fft=1024,
            hop_length=512
        )
        mel_spec = mel_transform(waveform) # [1, n_mels, time_steps]
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1) # [time_steps, n_mels]
        
        # 3. Process Text
        word = parse_word(file_path)
        text = f"say the word {word}"
        
        text_encoded = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = text_encoded['input_ids'].squeeze(0)
        attention_mask = text_encoded['attention_mask'].squeeze(0)
        
        return {
            'speech': mel_spec,
            'text_input_ids': input_ids,
            'text_attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_dataloaders(data_dir, batch_size=32):
    train_dataset = TESSDataset(data_dir, split='train')
    test_dataset = TESSDataset(data_dir, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
