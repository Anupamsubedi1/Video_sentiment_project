from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess

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
        
        except Exception as e:  
            raise ValueError(f"Error extracting audio from {video_path}: {e}")

        
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_filename =f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        path = os.path.join(self.video_dir, video_filename)
        video_path_exist = os.path.exists(path)

        if not video_path_exist:
            raise FileNotFoundError(f"Video file {video_filename} not found in {self.video_dir}")
        
        print("file found")

        text_inputs = self.tokenizer(row['Utterance'],
                                     padding='max_length',
                                     truncation=True,
                                     max_length=128,
                                     return_tensors="pt")
        
        # video_frames = self._load_video_frames(path)
        self._extract_audio_features(path)
        # print(video_frames)

if __name__ == "__main__":
    meld = MELDDataset("../dataset/dev/dev_sent_emo.csv", "../dataset/dev/dev_splits_complete")

    print(meld[0])

