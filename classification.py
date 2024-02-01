import os
import numpy as np
import librosa
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

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
def preprocess_file(file_path, n_mels=128, max_pad_len=32):
    signal, sr = librosa.load(file_path, sr=None)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    zcr = librosa.feature.zero_crossing_rate(signal)[0]

    # パディング処理
    mel_db_padded = pad_sequences([mel_db.T], maxlen=max_pad_len, dtype='float32', padding='post', truncating='post', value=0)
    zcr_padded = pad_sequences([zcr.reshape(-1, 1)], maxlen=max_pad_len, dtype='float32', padding='post', truncating='post', value=0)

    # 正規化処理
    mel_db_padded = (mel_db_padded - np.min(mel_db_padded)) / (np.max(mel_db_padded) - np.min(mel_db_padded))
    zcr_padded = (zcr_padded - np.min(zcr_padded)) / (np.max(zcr_padded) - np.min(zcr_padded))

    return mel_db_padded[0], zcr_padded[0]

# ファイルの仕分け
for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        file_path = os.path.join(input_dir, filename)
        X_mel, X_zcr = preprocess_file(file_path)
        prediction = model.predict([np.expand_dims(X_mel, axis=0), np.expand_dims(X_zcr, axis=0)])
        predicted_label = np.argmax(prediction, axis=1)[0]
        
        # 予測されたラベルに基づいてファイルを移動
        if predicted_label == 0:
            # クリーンな発話はspeechフォルダに移動
            os.rename(file_path, os.path.join(speech_dir, filename))
        else:
            # BGMがかぶさった発話はspeech_withBGMフォルダに移動
            os.rename(file_path, os.path.join(speech_withBGM_dir, filename))

print("ファイルの仕分けが完了しました。")
