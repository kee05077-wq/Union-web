"""[ST-GCN 실험용] 음운(지문자) 인식 모델 학습 — LSTM 대신 ST-GCN 채택 검토용.

dataset/sequences_graph/*.npy (build_dataset_graph_from_videos.py로 생성)를 모아
ST-GCN 분류기를 학습하고 models/phoneme_model_stgcn.h5 로 저장한다.

train.py(LSTM)와 같은 데이터/라벨을 쓰므로, 두 스크립트의 테스트 정확도를
직접 비교해서 채택 여부를 판단하면 된다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import activations, layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from config import ACTIONS, MODEL_DIR, MODEL_STGCN_PATH, SEQ_LENGTH, SEQUENCE_GRAPH_DIR
from modules.hand_graph import NUM_HAND_JOINTS, build_spatial_partitions
from modules.utils import createDirectory


@tf.keras.utils.register_keras_serializable(package="stgcn")
class GraphConv(layers.Layer):
    """ST-GCN의 공간(그래프) 컨볼루션. 인접행렬을 3개 부분집합(hand_graph.build_spatial_partitions)으로
    나눠 각 부분집합마다 별도 가중치를 학습한다 (Yan et al. 2018 spatial configuration partitioning)."""

    def __init__(self, units, adjacency_matrices, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self._adjacency_np = [np.asarray(a, dtype=np.float32) for a in adjacency_matrices]
        self.activation = activations.get(activation)

    def build(self, input_shape):
        c_in = input_shape[-1]
        self.adjacency = [tf.constant(a) for a in self._adjacency_np]
        self.kernels = [
            self.add_weight(
                name=f"kernel_{k}", shape=(c_in, self.units),
                initializer="glorot_uniform", trainable=True,
            )
            for k in range(len(self._adjacency_np))
        ]
        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, T, V, C_in)
        out = self.bias
        for adjacency, kernel in zip(self.adjacency, self.kernels):
            aggregated = tf.einsum("vw,btwc->btvc", adjacency, inputs)  # 이웃 관절 정보 집계
            out = out + tf.matmul(aggregated, kernel)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "adjacency_matrices": [a.tolist() for a in self._adjacency_np],
            "activation": activations.serialize(self.activation),
        })
        return config


def st_gcn_block(x, units, adjacency_matrices, temporal_kernel=5, use_residual=True):
    residual = x
    x = GraphConv(units, adjacency_matrices)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(units, (temporal_kernel, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)

    if use_residual:
        if residual.shape[-1] != units:
            residual = layers.Conv2D(units, 1, padding="same")(residual)
        x = layers.Add()([x, residual])
    return layers.ReLU()(x)


def build_model(num_classes: int) -> models.Model:
    adjacency_matrices = build_spatial_partitions()

    inputs = layers.Input(shape=(SEQ_LENGTH, NUM_HAND_JOINTS, 2))
    x = st_gcn_block(inputs, 32, adjacency_matrices, use_residual=False)
    x = st_gcn_block(x, 64, adjacency_matrices)
    x = st_gcn_block(x, 64, adjacency_matrices)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="phoneme_stgcn")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def load_dataset():
    label_index = {action: i for i, action in enumerate(ACTIONS)}
    sequences, labels = [], []

    for npy_path in sorted(SEQUENCE_GRAPH_DIR.glob("*.npy")):
        action = npy_path.stem.split("_")[0]
        if action not in label_index:
            print(f"경고: '{npy_path.name}' 의 라벨 '{action}' 을 ACTIONS에서 찾을 수 없어 건너뜁니다.")
            continue
        data = np.load(npy_path)  # (N, SEQ_LENGTH, 21, 2)
        sequences.append(data)
        labels.extend([label_index[action]] * len(data))

    if not sequences:
        raise SystemExit(
            f"{SEQUENCE_GRAPH_DIR} 에 학습할 시퀀스가 없습니다. "
            "build_dataset_graph_from_videos.py 를 먼저 실행하세요."
        )

    X = np.concatenate(sequences, axis=0).astype(np.float32)
    y = to_categorical(labels, num_classes=len(ACTIONS))
    return X, y


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
    model.save(MODEL_STGCN_PATH)
    print(f"모델 저장됨: {MODEL_STGCN_PATH}")


if __name__ == "__main__":
    main()
