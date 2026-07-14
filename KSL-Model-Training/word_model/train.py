"""
단어 인식 모델 학습.

dataset/sequences/*.npy (각 파일 shape: [SEQ_LENGTH, 150], 한 파일 = 한 개의 단어 동작 샘플)를
모아 LSTM 분류기를 학습하고 models/word_model.h5 로 저장한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from config import ACTIONS, FEATURE_DIM, MODEL_DIR, MODEL_PATH, SEQ_LENGTH, SEQUENCE_DIR
from modules.utils import createDirectory


def load_dataset():
    label_index = {action: i for i, action in enumerate(ACTIONS)}
    sequences, labels = [], []

    for npy_path in sorted(SEQUENCE_DIR.glob("*.npy")):
        action = npy_path.stem.split("_")[0]
        if action not in label_index:
            print(f"경고: '{npy_path.name}' 의 라벨 '{action}' 을 ACTIONS에서 찾을 수 없어 건너뜁니다.")
            continue
        data = np.load(npy_path)  # (SEQ_LENGTH, 150)
        if data.shape != (SEQ_LENGTH, FEATURE_DIM):
            print(f"경고: '{npy_path.name}' shape={data.shape} 이 기대값과 달라 건너뜁니다.")
            continue
        sequences.append(data)
        labels.append(label_index[action])

    if not sequences:
        raise SystemExit(
            f"{SEQUENCE_DIR} 에 학습할 시퀀스가 없습니다. "
            "build_dataset_from_videos.py 를 먼저 실행하세요."
        )

    X = np.array(sequences, dtype=np.float32)
    y = to_categorical(labels, num_classes=len(ACTIONS))
    return X, y


def build_model(num_classes: int) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, activation="relu", input_shape=(SEQ_LENGTH, FEATURE_DIM)),
        LSTM(64, return_sequences=False, activation="relu"),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    X, y = load_dataset()
    print(f"데이터셋: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    model = build_model(len(ACTIONS))
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=1000,
        batch_size=16,
        callbacks=[early_stop],
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n테스트 정확도: {acc:.4f} (loss={loss:.4f})")

    createDirectory(str(MODEL_DIR))
    model.save(MODEL_PATH)
    print(f"모델 저장됨: {MODEL_PATH}")


if __name__ == "__main__":
    main()
