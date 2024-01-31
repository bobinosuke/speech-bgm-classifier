import os
import numpy as np
import librosa
from keras.models import load_model

# モデルとデータのパス
model_path = 'speech_classification_model.h5'
input_dir = 'input'  # 仕分けるファイルがあるフォルダのパス
output_dir = 'output'  # 仕分けたファイルを保存するフォルダのパス

# 出力フォルダ内にspeechとspeech_withBGMフォルダを作成
speech_dir = os.path.join(output_dir, 'speech')
speech_withBGM_dir = os.path.join(output_dir, 'speech_withBGM')

os.makedirs(speech_dir, exist_ok=True)
os.makedirs(speech_withBGM_dir, exist_ok=True)

# モデルのロード
model = load_model(model_path)

# 音声データの前処理関数
def preprocess_file(file_path, n_mels=128):
    signal, sr = librosa.load(file_path, sr=None)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db_mean = np.mean(mel_db, axis=1)
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    zcr_mean = np.mean(zcr)
    X_mel = (mel_db_mean - np.min(mel_db_mean)) / (np.max(mel_db_mean) - np.min(mel_db_mean))
    X_zcr = (zcr_mean - np.min(zcr_mean)) / (np.max(zcr_mean) - np.min(zcr_mean))
    return np.array([X_mel]), np.array([[X_zcr]])

# ファイルの仕分け
for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        file_path = os.path.join(input_dir, filename)
        X_mel, X_zcr = preprocess_file(file_path)
        prediction = model.predict([X_mel, X_zcr])
        predicted_label = np.argmax(prediction, axis=1)[0]
        
        # 予測されたラベルに基づいてファイルを移動
        if predicted_label == 0:
            # クリーンな発話はspeechフォルダに移動
            os.rename(file_path, os.path.join(speech_dir, filename))
        else:
            # BGMがかぶさった発話はspeech_withBGMフォルダに移動
            os.rename(file_path, os.path.join(speech_withBGM_dir, filename))

print("ファイルの仕分けが完了しました。")
