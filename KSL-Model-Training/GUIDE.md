# 코드 설명서

이 문서는 `KSL-Model-Training` 저장소의 현재 코드 구조와 각 파일의 역할을 설명합니다.
설치·사용법 요약은 [README.md](README.md)를 참고하고, 이 문서는 각 모듈이 내부적으로
어떻게 동작하는지, 왜 그렇게 설계했는지를 다룹니다.

## 1. 전체 구조

```
KSL-Model-Training/
├── modules/                     # 두 모델이 공유하는 랜드마크 추출·특징 변환 코드
│   ├── holistic_module.py       # MediaPipe HolisticLandmarker 래퍼
│   ├── utils.py                 # 랜드마크 -> 특징 벡터 변환
│   └── hand_graph.py            # ST-GCN용 손 골격 그래프(인접행렬) 정의
├── phoneme_model/                # 지문자(자음/모음 낱자) 인식 모델
│   ├── config.py                 # 클래스 목록, 시퀀스 길이, 경로 상수
│   ├── record_dataset.py         # 웹캠으로 클래스별 영상 녹화
│   ├── build_dataset_from_videos.py       # 영상 -> LSTM용 55차원 특징 시퀀스(.npy)
│   ├── build_dataset_graph_from_videos.py # 영상 -> ST-GCN용 (21,2) 그래프 시퀀스(.npy)
│   ├── train.py                  # LSTM 학습 -> models/phoneme_model.h5
│   ├── train_stgcn.py            # ST-GCN 학습(실험) -> models/phoneme_model_stgcn.h5
│   ├── test_webcam.py            # 웹캠 실시간 테스트
│   └── dataset/, models/         # 데이터·모델 산출물 (git 추적 안 함)
├── word_model/                   # 단어 수어 인식 모델
│   ├── config.py, record_dataset.py, build_dataset_from_videos.py,
│   │   train.py, test_webcam.py  # phoneme_model과 동일한 구조, 입력/모델만 다름
│   └── dataset/, models/
├── mediapipe_models/              # HolisticLandmarker 모델 번들(.task, git 추적 안 함)
├── download_mediapipe_models.py   # 위 번들을 내려받는 스크립트
└── requirements.txt
```

`phoneme_model`과 `word_model`은 서로 독립된 파이프라인이지만 파일 구성(설정 →
녹화 → 특징 추출 → 학습 → 테스트)과 코드 패턴이 동일합니다. 차이는 아래 표를
참고하세요.

| | phoneme_model | word_model |
|---|---|---|
| 인식 대상 | 지문자 31종(자음 14 + 모음 17) | 단어 4종(예시, `ACTIONS`에서 자유롭게 추가) |
| 입력 관절 | 오른손 21점 | 양손 21점×2 + pose 6점(어깨/팔꿈치/손목) |
| 특징 차원 | 55 (단위벡터 20×2 + 관절각 15) | 150 (pose 24 + 왼손 63 + 오른손 63) |
| 시퀀스 길이 | 10프레임 (슬라이딩 윈도우) | 30프레임 (촬영 1회 = 샘플 1개) |
| 모델 | LSTM(64)→Dropout→Dense(32)→Dense(softmax) | LSTM(64,seq)→LSTM(64)→Dropout→Dense(32)→Dense(softmax) |
| Union-web 연동 | 됨 (`gesture_classifier.h5` 교체) | 안 됨 (특징 추출/추론 백엔드 별도 필요) |

## 2. `modules/` — 공유 모듈

### `holistic_module.py`

MediaPipe의 신규 Tasks API(`HolisticLandmarker`)를 감싼 `HolisticDetector` 클래스입니다.
`mediapipe==0.10.35`부터 레거시 `mp.solutions.holistic`이 제거되었기 때문에, 이 프로젝트는
`mediapipe_models/holistic_landmarker.task` 모델 번들을 로드해서 사용합니다
(`download_mediapipe_models.py`로 최초 1회 다운로드).

- `process(img_bgr)`: BGR 이미지를 받아 `RunningMode.IMAGE` 모드로 랜드마크를 추론하고
  결과(`self.results`, `left_hand_landmarks`/`right_hand_landmarks`/`pose_landmarks` 포함)를 반환합니다.
- `draw(img_bgr)`: 최근 추론 결과를 이미지 위에 원으로 그려줍니다(디버깅/시각화용). 왼손은
  보라색, 오른손은 주황색, pose 6점(`POSE_LANDMARK_INDICES = [11,12,13,14,15,16]`,
  어깨·팔꿈치·손목)은 남색으로 표시됩니다.
- `close()`: 랜드마커 리소스를 해제합니다.

IMAGE 모드이기 때문에 `min_tracking_confidence`는 옵션에 없습니다(프레임 간 추적을 하지
않고 매 프레임 독립적으로 추론).

### `utils.py`

랜드마크 좌표를 각 모델이 학습에 쓰는 고정 차원 특징 벡터로 변환하는 함수 모음입니다.

- `createDirectory(directory)`: 없으면 디렉터리를 생성하는 헬퍼.
- `Vector_Normalization(joint)` / `extract_phoneme_keypoints(right_hand_landmarks)`:
  오른손 21개 랜드마크(x, y)를 받아 55차원 벡터를 만듭니다.
  1. 관절을 잇는 20개의 단위 벡터(뼈 방향)를 계산 (`v1`→`v2` 쌍, 정규화).
  2. 인접한 벡터 15쌍 사이의 각도(도 단위, `arccos(dot product)`)를 계산.
  3. `concatenate([벡터 20×2=40, 각도 15]) = 55차원`.
  이 공식은 Union-web에 이미 배포된 `gesture_classifier.h5`와 동일하므로, 이 함수로
  뽑은 특징으로 학습한 모델은 Union-web `app.py`에 그대로 꽂아 넣을 수 있습니다.
  손이 인식되지 않으면(빈 리스트) `None`을 반환합니다.
- `extract_phoneme_graph_keypoints(right_hand_landmarks)`: ST-GCN 실험용. 55차원으로
  가공하지 않고, 손목(관절 0번)을 원점으로 평행이동 + 손 크기로 스케일 정규화한
  `(21, 2)` 좌표 배열을 그대로 반환합니다. 손이 인식되지 않으면 `None`.
- `extract_word_keypoints(results)`: pose 6점(x,y,z,visibility, 없으면 0) + 왼손 21점(x,y,z) +
  오른손 21점(x,y,z)을 이어붙여 150차원 벡터를 만듭니다. 손/포즈가 인식되지 않은
  프레임은 해당 구간을 0으로 채웁니다(phoneme과 달리 `None`을 반환하지 않음 — 단어
  모델은 시퀀스 전체 프레임이 필요하므로 빠지는 프레임 없이 항상 값을 채웁니다).

### `hand_graph.py`

`train_stgcn.py`(ST-GCN 실험 모델)가 쓰는 손 골격 그래프 정의입니다. ST-GCN
(Yan et al., 2018)의 "spatial configuration partitioning" 방식을 구현합니다.

- `HAND_CONNECTIONS`: MediaPipe `HAND_CONNECTIONS`과 동일한 21점 뼈대 연결 목록(21개 엣지).
- `_bfs_distances(...)`: 지정한 중심 노드(손목, 0번)로부터 각 노드까지의 BFS 최단거리를 구합니다.
- `build_spatial_partitions(...)`: 각 엣지를 손목으로부터의 거리 기준 3개 부분집합으로 나눠
  정규화된 인접행렬 3개를 반환합니다.
  - `subset0`: 자기 자신 + 손목으로부터 같은 거리인 이웃(예: 손가락 시작 관절끼리의 손바닥 연결)
  - `subset1`: 손목에 더 가까운 이웃 (centripetal)
  - `subset2`: 손목에서 더 먼 이웃 (centrifugal)
  각 부분집합은 Kipf & Welling 정규화(`Λ^-1/2 A Λ^-1/2`)를 거칩니다. 이 3개 행렬이
  `train_stgcn.py`의 `GraphConv` 레이어에서 서로 다른 학습 가능 가중치와 곱해집니다.

## 3. `phoneme_model/` — 지문자 인식

### `config.py`

`ACTIONS`(31개 자모, Union-web 배포 모델과 동일한 순서), `SEQ_LENGTH=10`, 데이터/모델
경로 상수를 정의합니다. `SEQUENCE_GRAPH_DIR`, `MODEL_STGCN_PATH`는 ST-GCN 실험 전용 경로입니다.

### `record_dataset.py`

웹캠으로 `ACTIONS`를 순서대로 촬영합니다. 클래스별로 안내 화면을 띄운 뒤 아무 키나
누르면 `--seconds`(기본 20초) 동안 녹화하고 `dataset/videos/<자모>/<자모>_<n>.avi`로
저장합니다. `next_take_number`가 기존 파일을 보고 다음 take 번호를 이어서 매기므로,
여러 번 실행해 촬영을 누적할 수 있습니다. 녹화 중 ESC를 누르면 해당 클래스 촬영을 건너뜁니다.

### `build_dataset_from_videos.py`

`dataset/videos/*/*.avi`의 각 프레임에서 `HolisticDetector`로 오른손 랜드마크를 뽑고
`extract_phoneme_keypoints`로 55차원 특징을 계산합니다. 영상 하나에서 나온 프레임별
특징을 `SEQ_LENGTH(10)` 프레임 단위로 슬라이딩 윈도우(stride=1)로 잘라 여러 시퀀스를
만들고, `dataset/sequences/<자모>_<take>.npy`(shape `(N, 10, 55)`)로 저장합니다. 인식된
프레임이 10개 미만인 영상은 건너뜁니다.

### `build_dataset_graph_from_videos.py`

위와 동일한 영상을 재사용하되, 55차원 대신 `extract_phoneme_graph_keypoints`로 얻은
`(21, 2)` 좌표 그래프를 그대로 시퀀스화해 `dataset/sequences_graph/<자모>_<take>.npy`
(shape `(N, 10, 21, 2)`)에 저장합니다. ST-GCN 학습(`train_stgcn.py`) 전용입니다.

### `train.py`

`dataset/sequences/*.npy`를 모두 모아(`load_dataset`) `LSTM(64)→Dropout(0.5)→Dense(32)
→Dense(31, softmax)` 모델을 학습합니다. `test_size=0.05`, `EarlyStopping(patience=30,
monitor="val_loss")`, 최대 300 epoch. 학습이 끝나면 `models/phoneme_model.h5`로 저장합니다.
Union-web `app.py`가 `compile=False`로 그대로 로드할 수 있도록 순수 Keras H5 포맷입니다.
파일명에서 언더스코어 앞부분(`<자모>_<take>.npy` → `<자모>`)을 라벨로 사용하므로,
`ACTIONS`에 없는 라벨은 경고 후 스킵됩니다.

### `train_stgcn.py`

LSTM 대신 ST-GCN 구조를 채택할지 검토하기 위한 실험용 스크립트입니다.

- `GraphConv`: `hand_graph.build_spatial_partitions()`가 만든 인접행렬 3개를 상수로 갖고,
  각 부분집합마다 별도의 학습 가능 가중치(`kernel_0/1/2`)로 이웃 관절 정보를 집계
  (`einsum("vw,btwc->btvc", ...)`)한 뒤 더하는 공간 그래프 컨볼루션 레이어입니다.
  `@register_keras_serializable`로 등록되어 있어 모델 저장/로드 시 커스텀 레이어로 복원됩니다.
- `st_gcn_block`: `GraphConv`(공간) → `BatchNorm` → `ReLU` → `Conv2D`(시간 축 커널, 기본 5)
  → `BatchNorm` → (옵션) residual 연결 → `ReLU` 순서의 블록.
- `build_model`: 입력 `(SEQ_LENGTH, 21, 2)` → st_gcn_block ×3(32→64→64 채널) →
  `GlobalAveragePooling2D` → `Dense(32)` → `Dropout(0.5)` → `Dense(num_classes, softmax)`.
- `load_dataset`은 `train.py`와 동일한 라벨링 규칙으로 `sequences_graph/*.npy`를 읽습니다.
- 결과는 `models/phoneme_model_stgcn.h5`로 저장되며, `train.py`가 만든 LSTM 모델과
  테스트 정확도를 비교해 채택 여부를 판단하는 용도입니다.

### `test_webcam.py`

`models/phoneme_model.h5`를 `compile=False`로 로드하고, 웹캠 프레임마다 오른손 특징을
뽑아 길이 10짜리 `deque`에 쌓습니다. 큐가 다 차면 모델로 예측하고, confidence가
`CONFIDENCE_THRESHOLD(0.6)`를 넘을 때만 화면에 라벨을 표시합니다. ESC로 종료합니다.

## 4. `word_model/` — 단어 수어 인식

`phoneme_model`과 파일 구조·패턴은 동일하고, 다루는 특징과 시퀀스 길이만 다릅니다.

### `config.py`

`ACTIONS = ['나는', '맛있다', '고양이', '식사']`(예시 목록, 자유롭게 추가/수정),
`SEQ_LENGTH=30`, `FEATURE_DIM=150`.

### `record_dataset.py`

단어 하나의 동작을 정확히 `SEQ_LENGTH(30)` 프레임으로 고정 녹화합니다(촬영 1회 =
학습 샘플 1개). `--takes`(기본 20)만큼 단어별로 반복 촬영하며, `phoneme_model`과 달리
녹화 시간이 아니라 프레임 수로 종료 시점을 정합니다.

### `build_dataset_from_videos.py`

영상의 모든 프레임에서 `extract_word_keypoints`로 150차원 특징을 뽑습니다.
`fit_to_seq_length`가 녹화 오차(카메라 FPS 흔들림 등)로 프레임 수가 30과 다를 경우
잘라내거나 마지막 프레임을 반복해 정확히 30프레임으로 맞춥니다. 영상 하나 = npy 파일
하나(shape `(30, 150)`)로 `dataset/sequences/<단어>_<take>.npy`에 저장합니다
(phoneme_model처럼 슬라이딩 윈도우로 여러 시퀀스를 뽑지 않음).

### `train.py`

`dataset/sequences/*.npy`를 모아(각 파일이 이미 샘플 1개, shape `(30, 150)`이 아니면
스킵) `LSTM(64, return_sequences=True)→LSTM(64)→Dropout(0.5)→Dense(32)→Dense(4, softmax)`
모델을 학습합니다. `EarlyStopping(patience=50)`, 최대 1000 epoch, `batch_size=16`.
결과는 `models/word_model.h5`. **아직 Union-web에 연동되어 있지 않습니다** — 양손+pose
특징 추출 및 30프레임 시퀀스 처리를 위한 백엔드 로직이 Union-web 쪽에 별도로 필요합니다.

### `test_webcam.py`

`phoneme_model`과 달리 매 프레임 특징을 무조건 큐에 쌓습니다(단어 모델은 손이 안
잡혀도 0벡터로 채워 프레임을 유지해야 하므로). 예측이 최근 `STABLE_FRAMES(10)`개 동안
동일하고 confidence가 `CONFIDENCE_THRESHOLD(0.5)`를 넘으면 "확정"으로 보고, 직전
단어와 다를 때만 문장(`sentence`, 최대 `MAX_SENTENCE_WORDS=5`개)에 이어붙여 화면에
표시합니다.

## 5. 루트 스크립트

### `download_mediapipe_models.py`

`mediapipe_models/holistic_landmarker.task`가 없을 때만 Google Storage에서 모델 번들을
내려받습니다(이미 있으면 스킵). `requirements.txt` 설치 후 한 번만 실행하면 됩니다.

## 6. 데이터 흐름 요약

```
record_dataset.py                 build_dataset_from_videos.py            train.py
(웹캠 -> .avi)          ──────▶   (.avi -> HolisticDetector -> 특징 -> .npy) ──▶ (.npy 모아 LSTM 학습 -> .h5)
                                                                                      │
                                                                                      ▼
                                                                          test_webcam.py (웹캠 실시간 추론)
```

`phoneme_model`은 여기에 더해 `build_dataset_graph_from_videos.py` → `train_stgcn.py`로
이어지는 별도 실험 경로가 있습니다(같은 영상, 다른 특징 표현·모델 구조).

## 7. Union-web 연동 메모

Union-web `app.py`는 `gesture_classifier.h5`를 루트에서 로드해 오른손 55차원 특징 ×
10프레임 시퀀스로 지문자를 추론합니다. `phoneme_model/train.py`로 학습한
`phoneme_model.h5`는 입력/출력 shape이 동일하므로 그대로 교체해 쓸 수 있습니다.
`word_model.h5`는 특징 차원(150)과 시퀀스 길이(30)가 달라 Union-web 쪽 추론 코드를
별도로 작성해야 연동할 수 있습니다.
