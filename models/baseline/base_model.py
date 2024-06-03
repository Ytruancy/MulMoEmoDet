import torch
import torch.nn as nn

class Audio_FFN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(Audio_FFN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

class Video_FFN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Video_FFN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class EmotionClassifier(nn.Module):
    def __init__(self, audio_feature_lengths, video_input_size=1088, audio_final_size=128, video_final_size=128, num_classes=5):
        super(EmotionClassifier, self).__init__()
        self.device = torch.device("cuda")
        self.audio_models = self.create_models(audio_feature_lengths)
        self.audio_final_size = audio_final_size
        
        # Combined video features [68, 16]
        self.video_model = Video_FFN(input_size=68*10, hidden_size1=768, hidden_size2=512, output_size=video_final_size).to(torch.device("cuda"))

        self.audio_projection = nn.Linear(11*10, self.audio_final_size)
        
        # Final combined model
        self.fc1 = nn.Linear(audio_final_size + video_final_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, num_classes)

    def create_models(self,feature_lengths):
        models = {}
        for feature, length in feature_lengths.items():
            hidden_size1 = length * 3  # Project to a higher space
            hidden_size2 = length // 2  # Reduce dimension
            hidden_size3 = length // 4  # Further reduce dimension
            models[feature] = Audio_FFN(
                input_size=length, 
                hidden_size1=hidden_size1, 
                hidden_size2=hidden_size2, 
                hidden_size3=hidden_size3, 
                output_size=10
            ).to(torch.device("cuda"))
        return models
    
    def forward(self, audio_features, video_features):
        # Process each audio feature through its respective model
        audio_outputs = []
        for feature, model in self.audio_models.items():
            audio_feature = audio_features[feature]
            audio_output = model(audio_feature)
            audio_outputs.append(audio_output)
        
        # Concatenate all audio feature outputs
        audio_output = torch.cat(audio_outputs, dim=1)
        #print(audio_output.shape)

        #print(audio_output.shape)
        # Project audio output to final audio feature vector
        audio_output = self.audio_projection(audio_output)
        
        # Process video features
        video_output = video_features.view(video_features.size(0), -1)  # Flatten the video features
        video_output = self.video_model(video_output)
        
        # Concatenate audio and video outputs
        combined_output = torch.cat((audio_output, video_output), dim=1)
        
        # Final classification layers
        x = self.relu1(self.fc1(combined_output))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        
        return x
    

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


device = torch.device("cuda")
print("device specified")
model_weight = "models/baseline/checkpoints/model_emotionRec.pth"
model = EmotionClassifier(audio_feature_lengths)
model.load_state_dict(torch.load(model_weight))
model.to(device)
print("model loaded")
device = torch.device("cuda")
model.eval()
emotion_mapping = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happy/Joy', 4: 'Sad'}



def basemodel_predict(video_features,audio_features):
    """
    Provide video features and audio feature collection
    """
    video_features = video_features.unsqueeze(0)
    audio_features = {key: value.unsqueeze(0) for key, value in audio_features.items()}
    with torch.no_grad():
        video_features = video_features.to(device)
        audio_features = {key: value.to(device) for key, value in audio_features.items()}
        outputs = model(audio_features, video_features)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_index = torch.max(outputs, 1)
        confidence = probabilities[0][predicted_index.item()].item()
        emotion_label = emotion_mapping[predicted_index.item()]
        if confidence<0.4:
            emotion_label = "Unspecified"
    return emotion_label , confidence