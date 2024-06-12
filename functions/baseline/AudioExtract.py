import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import subprocess
import os
import torch
import ffmpeg
from sklearn.preprocessing import LabelEncoder

def parse_opensmile_output(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract attribute names, skipping the first line which is the name id
    attribute_lines = [line for line in lines if line.startswith('@attribute')]
    attributes = [line.split()[1] for line in attribute_lines][1:-3]
    
    # Extract the data lines (only the last line in this case)
    data = lines[-1].replace('\n', '').split(",")[1:-3]
    data = [float(value) for value in data]

    # Prepare the data string for pandas
    
    return dict(zip(attributes,data))



def extract_audio_features(wav_file,opensmile_path,config_file):
    if not wav_file.endswith(".wav"):
        print("Converting audio file to wav")
        _ , file_extension = os.path.splitext(wav_file)
        converted_audio_file = wav_file.replace(file_extension,".wav")
        ffmpeg.input(wav_file).output(converted_audio_file,format='wav').run()
        wav_file = converted_audio_file
    output_root = "data/AudioExtracted/"
    os.makedirs(output_root, exist_ok = True)
    output_file = output_root+os.path.basename(wav_file).replace(".wav",".csv")
    print(f"Processing audio, saving to {output_file}")
    if not os.path.exists(output_file):
        # Command to run OpenSMILE
        command = [
            opensmile_path,
            "-C", config_file,
            "-I", wav_file,
            "-O", output_file,
            "-loglevel", "0"
        ]
        
        # Run the command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print(f"Error in OpenSMILE execution: {result.stderr.decode()}")
        else:
            print(f"Feature extraction completed successfully. Output saved to {output_file}")

    data = parse_opensmile_output(output_file)

    return data
