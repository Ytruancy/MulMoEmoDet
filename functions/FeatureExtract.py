import torch
from functions.baseline.VideoExtract import video_extract
from functions.baseline.AudioExtract import extract_audio_features
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import subprocess
import os
from sklearn.preprocessing import LabelEncoder




def FeatureExtract(video_path,audio_path,openface_path,opensmile_path,opensmile_config, Video_processed = True):
    """
    Args:
        video_path: path to saved video file, or in my case, I already processed the video then it's the path to the csv file
        audio_path: path to saved audio file
        openface_path: path to openface executor, for instance "xxxxxx\openface\FeatureExtraction.exe" on windows
        opensmile_path: path to opensmile 
        opensmile_config: config file of opensmile
        video_status: whether video is already processed using openface
    Returns:
        Extracted audio and video features
    """
    audio_feature_lengths = {
        'pcm_loudness': 42,
        'pcm_fftMag': 630,
        'logMelFreqBand': 672,
        'lspFreq': 336,
        'F0finEnv': 42,
        'voicingFinalUnclipped': 42,
        'lspFreq_sma': 336,
        'F0final_sma': 38,
        'jitterLocal_sma': 38,
        'jitterDDP_sma': 38,
        'shimmerLocal_sma': 38
    }

    if not Video_processed:
        video_root = "data/VideoExtracted"
        os.makedirs("data/VideoExtracted",exist_ok = True)
        file_basename = os.path.basename(video_file)
        output_dir = os.path.join(video_root, os.path.basename(video_file).replace(".mp4", ""))
        output_csv = os.path.join(output_dir,file_basename.replace(".mp4", ".csv"))
        os.system(f"{openface_path} -f {output_video} -out_dir {output_dir}")
        video_path = output_csv
    
    video_features = video_extract(video_path)
    audio_features = extract_audio_features(audio_path, opensmile_path, opensmile_config)
    audio_features_vector = np.array([float(value) for value in audio_features.values()])
    audio_feature_collection = defaultdict()
    for key,value in audio_feature_lengths.items():
        audio_feature_collection[key] = torch.tensor(audio_features_vector[:value],dtype =torch.float32 )

    return video_features, audio_feature_collection
    
