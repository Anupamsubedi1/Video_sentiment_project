from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio

class MELDDataset(Dataset):
    def __init__(self,csv_path, video_dir):
        self.data = pd.read_csv(csv_path)

        self.video_dir = video_dir

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }

        self.sentiment_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2

        }
    
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file {video_path}")
            #try and read first frame to verify video is readable

            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Cannot read frames from video file {video_path}")
            
            #reset to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while len(frames)<30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame= cv2.resize(frame, (224, 224))
                frame = frame/255.0
                frames.append(frame)

        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
        
        finally:
            cap.release()

        if (len(frames) == 0):
            raise ValueError(f"No frames extracted from video file {video_path}")
        
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])]*(30-len(frames))

        else:
            frames = frames[:30]

        #before prmute:frames shape: (num_frames, height, width, channels)
        #after permute: (num_frames, channels, height, width)

        return torch.FloatTensor(np.array(frames)).permute(0,3,1,2)

    def __len__(self):
        return len(self.data)
    
    def _extract_audio_features(self, video_path):
        audio_path = video_path.replace(".mp4", ".wav")
        try :
            subprocess.run(["ffmpeg",
                            "-i", 
                            video_path,
                            "-vn",
                            "-acodec",
                            "pcm_s16le",
                            "-ar", "16000",
                            "-ac", "1", 
                            "-map",
                            "a",
                            audio_path], check=True,stdout= subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64,n_fft=1024,hop_length=512)
            
            mel_spec = mel_spectrogram(waveform)

            #normalize
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))

            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"FFmpeg error while extracting audio from : {e}")

        except Exception as e:  
            raise ValueError(f"Error extracting audio from {video_path}: {e}")
        
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        
    
    def __getitem__(self, idx):

        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]
        try:
            video_filename =f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            path = os.path.join(self.video_dir, video_filename)
            video_path_exist = os.path.exists(path)

            if not video_path_exist:
                raise FileNotFoundError(f"Video file {video_filename} not found in {self.video_dir}")
            
            # print("file found")

            text_inputs = self.tokenizer(row['Utterance'],
                                        padding='max_length',
                                        truncation=True,
                                        max_length=128,
                                        return_tensors="pt")
            
            video_frames = self._load_video_frames(path)
            audio_feature = self._extract_audio_features(path)

            emotion_label = self.emotion_map[row['Emotion'].lower()]        
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]


            return {
                "text_input":{
                    "input_ids": text_inputs['input_ids'].squeeze(),
                    "attention_mask": text_inputs['attention_mask'].squeeze()
                },
                "video_frames":video_frames,
                "audio_feature": audio_feature,
                "emotion_label": torch.tensor(emotion_label),
                "sentiment_label":torch.tensor(sentiment_label) 

            }
            # print(audio_feature)\
        except Exception as e:
            print(f"Error : {e}")
            return None


def collate_fn(batch):
    batch = list(filter(None, batch))
    return default_collate(batch)

def prepare_data_loader(train_csv,train_video_dir,dev_csv,dev_video_dir,test_csv,test_video_dir,batch_size=32):
    train_dataset = MELDDataset(train_csv,train_video_dir)
    dev_dataset = MELDDataset(dev_csv,dev_video_dir)
    test_dataset = MELDDataset(test_csv,test_video_dir)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, collate_fn=collate_fn)
    dev_loader =DataLoader(dev_dataset,batch_size=batch_size,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":

    train_loader, dev_loader, test_loader = prepare_data_loader(
        "../dataset/train/train_sent_emo.csv","../dataset/train/train_splits",
        "../dataset/dev/dev_sent_emo.csv","../dataset/dev/dev_splits_complete",
        "../dataset/test/test_sent_emo.csv","../dataset/test/output_repeated_splits_test")
    
    # meld = MELDDataset("../dataset/dev/dev_sent_emo.csv", "../dataset/dev/dev_splits_complete")
    for batch in train_loader:
        print(batch['text_input'])
        print(batch['video_frames'].shape)
        print(batch['audio_feature'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        
        break
    # print(meld[0])

