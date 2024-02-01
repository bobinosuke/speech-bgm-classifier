from model import preprocess_data, build_model
import os
from keras.callbacks import EarlyStopping

# データセットのパス
speech_dir = 'training_data/speech'
speech_withBGM_dir = 'training_data/speech_withBGM'

# データセットの読み込みとラベル付け
audio_files = []
labels = []

# クリーンな発話の読み込み
for filename in os.listdir(speech_dir):
    if filename.endswith('.wav'):
        audio_files.append(os.path.join(speech_dir, filename))
        labels.append(0)  # クリーンな発話はラベル0

# BGMがかぶさった発話の読み込み
for filename in os.listdir(speech_withBGM_dir):
    if filename.endswith('.wav'):
        audio_files.append(os.path.join(speech_withBGM_dir, filename))
        labels.append(1)  # BGMがかぶさった発話はラベル1

# データの前処理
X_mel_train, X_mel_test, X_zcr_train, X_zcr_test, y_train, y_test = preprocess_data(audio_files, labels)

# モデルの構築
# メルスペクトログラム特徴量とZCR特徴量の形状を取得
# 形状の指定には、時間軸を含む全ての次元を指定する
input_shape_mel = X_mel_train.shape[1:]  # 時間軸を含む形状
input_shape_zcr = X_zcr_train.shape[1:]  # 時間軸を含む形状
num_classes = 2  # クリーンな発話とBGMがかぶさった発話の2クラス

# モデル構築関数の呼び出し
model = build_model(input_shape_mel, input_shape_zcr, num_classes)

# 早期停止のコールバックを設定
early_stopping = EarlyStopping(
    monitor='val_loss',  # 監視する値
    patience=10,        # 指定したエポック数以上改善がない場合に訓練を停止
    verbose=1,           # 早期停止時にメッセージを出力
    restore_best_weights=True  # 最も良いモデルの重みを復元
)

# モデルの訓練（早期停止のコールバックを追加）
model.fit(
    [X_mel_train, X_zcr_train], y_train, 
    epochs=1000, 
    batch_size=32, 
    validation_data=([X_mel_test, X_zcr_test], y_test), 
    callbacks=[early_stopping]
)

# モデルの評価
test_loss, test_accuracy = model.evaluate([X_mel_test, X_zcr_test], y_test)
print(f"Test accuracy: {test_accuracy}")

# モデルの保存
model.save('speech_classification_model.h5')
