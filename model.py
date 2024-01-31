import numpy as np
import librosa
from keras.layers import Dense, Dropout, Input, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 音声データの前処理関数
def preprocess_data(audio_files, labels, n_mels=128, max_pad_len=32):
    # メルスペクトログラム特徴量とZCR特徴量を格納するリスト
    mel_features = []
    zcr_features = []
    augmented_labels = []
    
    for file, label in zip(audio_files, labels):
        # 音声ファイルを読み込み
        signal, sr = librosa.load(file, sr=None)
    
        # メルスペクトログラムを計算
        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # メルスペクトログラムの時間平均を計算
        mel_db_mean = np.mean(mel_db, axis=1)
        
        # ZCRを計算
        zcr = librosa.feature.zero_crossing_rate(signal)[0]
        
        # ZCRの時間平均を計算
        zcr_mean = np.mean(zcr)
        
        # 特徴量をリストに追加
        mel_features.append(mel_db_mean)
        zcr_features.append([zcr_mean])  # 2D配列として扱うためにリストに入れる
        augmented_labels.append(label)
    
    # リストをNumPy配列に変換
    X_mel = np.array(mel_features)
    X_zcr = np.array(zcr_features)
    y = np.array(augmented_labels)
    
    # 正規化処理
    X_mel = (X_mel - np.min(X_mel)) / (np.max(X_mel) - np.min(X_mel))
    X_zcr = (X_zcr - np.min(X_zcr)) / (np.max(X_zcr) - np.min(X_zcr))
    
    # データセットを訓練用とテスト用に分割
    X_mel_train, X_mel_test, y_train, y_test = train_test_split(
        X_mel, y, test_size=0.2, random_state=42, stratify=y
    )
    X_zcr_train, X_zcr_test, _, _ = train_test_split(
        X_zcr, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_mel_train, X_mel_test, X_zcr_train, X_zcr_test, y_train, y_test

# モデル構築関数の修正
def build_model(input_shape_mel, input_shape_zcr, num_classes):
    # メルスペクトログラム特徴量のための入力層
    input_mel = Input(shape=(input_shape_mel,))
    # ZCR特徴量のための入力層
    input_zcr = Input(shape=(input_shape_zcr,))
    
    # メルスペクトログラム特徴量のための処理層
    x_mel = Dense(128, activation='relu')(input_mel)
    x_mel = Dense(128, activation='relu')(x_mel)
    
    # ZCR特徴量のための処理層
    x_zcr = Dense(128, activation='relu')(input_zcr)
    x_zcr = Dense(128, activation='relu')(x_zcr)
    
    # メルスペクトログラム特徴量とZCR特徴量の結合
    x = Concatenate()([x_mel, x_zcr])
    
    # 全結合層
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # 出力層
    output = Dense(num_classes, activation='softmax')(x)
    
    # モデルの構築
    model = Model(inputs=[input_mel, input_zcr], outputs=output)
    
    # モデルのコンパイル
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
