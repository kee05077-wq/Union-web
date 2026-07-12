"""
음운(지문자) 인식 모델 학습.

dataset/sequences/*.npy (build_dataset_from_videos.py 또는 build_dataset_from_aihub.py 로 생성)를
모아 LSTM 분류기를 학습하고 models/phoneme_model.h5 로 저장한다.

Union-web(app.py)이 그대로 로드할 수 있도록 입력 shape (SEQ_LENGTH, 55), 출력 31 클래스,
compile=False 로 불러도 동작하도록 저장한다.
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

from config import ACTIONS, MODEL_DIR, MODEL_PATH, SEQ_LENGTH, SEQUENCE_DIR
from modules.utils import createDirectory


def load_dataset():
    label_index = {action: i for i, action in enumerate(ACTIONS)}
    sequences, labels = [], []

    for npy_path in sorted(SEQUENCE_DIR.glob("*.npy")):
        action = npy_path.stem.split("_")[0]
        if action not in label_index:
            print(f"경고: '{npy_path.name}' 의 라벨 '{action}' 을 ACTIONS에서 찾을 수 없어 건너뜁니다.")
            continue
        data = np.load(npy_path)  # (N, SEQ_LENGTH, 55)
        sequences.append(data)
        labels.extend([label_index[action]] * len(data))

    if not sequences:
        raise SystemExit(
            f"{SEQUENCE_DIR} 에 학습할 시퀀스가 없습니다. "
            "build_dataset_from_videos.py / build_dataset_from_aihub.py 를 먼저 실행하세요."
        )

    X = np.concatenate(sequences, axis=0).astype(np.float32)
    y = to_categorical(labels, num_classes=len(ACTIONS))
    return X, y


def build_model(num_classes: int) -> Sequential:
    model = Sequential([
        LSTM(64, activation="relu", input_shape=(SEQ_LENGTH, 55)),
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

    early_stop = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=300,
        batch_size=32,
        callbacks=[early_stop],
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n테스트 정확도: {acc:.4f} (loss={loss:.4f})")

    createDirectory(str(MODEL_DIR))
    model.save(MODEL_PATH)
    print(f"모델 저장됨: {MODEL_PATH}")


if __name__ == "__main__":
    main()
