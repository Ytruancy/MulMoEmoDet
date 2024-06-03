from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from functions.FeatureExtract import FeatureExtract
from models.baseline.base_model import basemodel_predict
import os
import shutil

app = Flask(__name__)
openface_path = "E:\Anaconda\envs\OpenFace\openface\FeatureExtraction.exe"
opensmile_path = "../opensmile/bin/SMILExtract" 
opensmile_config = "../opensmile/config/emobase/emobase2010.conf"

# video_path = "/mnt/g/MultiModalDetect/CREMA-D/VideoExtracted/1001_DFA_ANG_XX/1001_DFA_ANG_XX.csv"
# audio_path = "/mnt/g/MultiModalDetect/CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav"

# video_features, audio_features = FeatureExtract(video_path,audio_path,openface_path,opensmile_path,opensmile_config)
# print("feature extracted successfully")
# emotion_label, confidence = basemodel_predict(video_features,audio_features)
# print(emotion_label,confidence)



@app.route('/detect-segmentation', methods=['POST'])
def predict():
    #Create folder to store video&audio in app local directory
    video_path = request.form.get('video',type=str)
    audio_path = request.form.get('audio',type=str)
    user_id = request.form.get('user_id',type=str)
    user_id = str(user_id)
    os.makedirs(os.path.join("data",user_id),exist_ok = True)
    session_id = request.form.get('session_id',type=str)
    os.makedirs(os.path.join("data",user_id,session_id),exist_ok = True)
    question_id = request.form.get('segmentation_id',type=str)
    data_folder = os.path.join("data",user_id,session_id,question_id)
    os.makedirs(data_folder,exist_ok = True)
    desination_video = os.path.join(data_folder, os.path.basename(video_path))
    desination_audio = os.path.join(data_folder, os.path.basename(audio_path))
    shutil.copy(video_path,desination_video)
    shutil.copy(audio_path,desination_audio)
    total_segmentations = request.form.get('total_questions', type=str)

    if not all([video_path, audio_path, user_id, session_id, question_id, total_segmentations]):
        return jsonify(status="error", message="Invalid input or missing parameters."), 400
    try:
        video_features, audio_features = FeatureExtract(video_path,audio_path,openface_path,opensmile_path,opensmile_config)
        # return jsonify(
        # status="success",
        # message = "feature extracted successfully"   
        #  )
        emotion_label, confidence = basemodel_predict(video_features,audio_features)
    except FileNotFoundError:
        return jsonify(status="error", message="One or more file paths do not exist."), 400
    except Exception as e:
        return jsonify(status="error", message=str(e)), 500
    return jsonify(
        status="success",
        segmentation_id=question_id,
        emotional_detection=emotion_label,
        stress_detection="Will be updated in the future",
        confidence = confidence    
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    