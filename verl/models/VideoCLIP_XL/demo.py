import os
from typing import List
import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from modeling import VideoCLIP_XL
from utils.text_encoder import text_encoder


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break
        
        
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data / 255.0 - v_mean) / v_std
        
    
def video_preprocessing(video_path, fnum=8):
    video = cv2.VideoCapture(video_path)
    frames = [x for x in _frame_from_video(video)]
    step = len(frames) // fnum
    frames = frames[::step][:fnum]

    vid_tube = []
    for fr in frames:
        fr = fr[:,:,::-1]
        fr = cv2.resize(fr, (224, 224))
        fr = np.expand_dims(normalize(fr), axis=(0, 1))
        vid_tube.append(fr) 
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube)
    
    return vid_tube


videoclip_xl = VideoCLIP_XL()
state_dict = torch.load("./VideoCLIP-XL.bin", map_location="cpu")
videoclip_xl.load_state_dict(state_dict)
videoclip_xl.cuda().eval()

       
videos = [
    "/gemini/platform/public/jiangshiqi/video_evaluator/wan_data_resized/A piece of sodium chloride is ignited, emitting a vivid and unique flame as it burns steadily.\\n.mp4",
]

texts = [
    "A piece of sodium chloride is ignited emitting a vivid and unique flame as it burns steadily.",
]

with torch.no_grad():
    video_inputs = torch.cat([video_preprocessing(video) for video in videos], 0).float().cuda()
    video_features = videoclip_xl.vision_model.get_vid_features(video_inputs).float()
    video_features = video_features / video_features.norm(dim=-1, keepdim=True)

    text_inputs = text_encoder.tokenize(texts, truncate=True).cuda()
    text_features = videoclip_xl.text_model.encode_text(text_inputs).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

Tmp = 100.

sim_matrix = (text_features @ video_features.T) * Tmp

print(sim_matrix)