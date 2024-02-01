import numpy as np
import librosa
from keras.layers import Input, Conv1D, GRU, Dense, Dropout, Concatenate, MaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 音声データの前処理関数
def preprocess_data(audio_files, labels, n_mels=128, max_pad_len=32):
    mel_features = []
    zcr_features = []
    augmented_labels = []
    
    for file, label in zip(audio_files, labels):
        signal, sr = librosa.load(file, sr=None)
        
        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        zcr = librosa.feature.zero_crossing_rate(signal)[0]
        
        mel_features.append(mel_db.T)  # 転置して時間軸を最初にする
        zcr_features.append(zcr.reshape(-1, 1))  # 2D配列に変換
        augmented_labels.append(label)
    
    # パディング処理
    X_mel = pad_sequences(mel_features, maxlen=max_pad_len, dtype='float32', padding='post', truncating='post', value=0)
    X_zcr = pad_sequences(zcr_features, maxlen=max_pad_len, dtype='float32', padding='post', truncating='post', value=0)
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
    input_mel = Input(shape=(input_shape_mel[0], input_shape_mel[1]))  # チャネル次元を追加しない
    # ZCR特徴量のための入力層
    input_zcr = Input(shape=(input_shape_zcr[0], input_shape_zcr[1]))  # チャネル次元を追加しない
    
    # メルスペクトログラム特徴量のための処理層
    x_mel = Conv1D(64, kernel_size=3, activation='relu')(input_mel)
    x_mel = MaxPooling1D(pool_size=2)(x_mel)
    x_mel = Conv1D(64, kernel_size=3, activation='relu')(x_mel)
    x_mel = GRU(128)(x_mel)
    
    # ZCR特徴量のための処理層
    x_zcr = Conv1D(64, kernel_size=3, activation='relu')(input_zcr)
    x_zcr = MaxPooling1D(pool_size=2)(x_zcr)
    x_zcr = Conv1D(64, kernel_size=3, activation='relu')(x_zcr)
    x_zcr = GRU(128)(x_zcr)
    
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
