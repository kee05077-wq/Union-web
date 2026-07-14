# KSL-Model-Training

Union-web(화상회의 수어 인식)에서 쓰는 모델을 직접 학습시키기 위한 데이터셋 구축 + 학습 + 테스트 프로젝트입니다.
현재 Union-web이 쓰고 있는 `gesture_classifier.h5`는 외부에서 내려받은 모델이며, 이 프로젝트는 그것을
**우리가 직접 촬영/가공한 데이터로 대체 학습**시키기 위한 전 과정을 담습니다.

## Python 버전

Union-web과 동일하게 **Python 3.11** 기준입니다 (`tensorflow==2.15.1`, `mediapipe==0.10.35` 고정).

```
pip install -r requirements.txt
python download_mediapipe_models.py   # HolisticLandmarker 모델 번들(.task) 다운로드
```

> `mediapipe==0.10.35`부터 레거시 `mp.solutions.holistic` API가 제거되어, 이 프로젝트는
> 신규 Tasks API(`HolisticLandmarker`)를 사용합니다. 그래서 별도 모델 번들 파일이 필요합니다.

## 두 가지 모델

| | phoneme_model | word_model |
|---|---|---|
| 인식 대상 | 지문자 (자음/모음 낱자, 31종) | 단어 (나는/맛있다/고양이/식사 등) |
| 입력 관절 | 오른손 21점 | 양손 21점x2 + pose(어깨/팔꿈치/손목) 6점 |
| 특징 차원 | 55 (관절 벡터+각도) | 150 (pose 24 + 왼손 63 + 오른손 63) |
| 시퀀스 길이 | 10프레임 | 30프레임 |
| 모델 구조 | LSTM(64) → Dropout → Dense(32) → Dense(softmax) | LSTM(64,seq) → LSTM(64) → Dropout → Dense(32) → Dense(softmax) |
| 결과물 | `phoneme_model/models/phoneme_model.h5` | `word_model/models/word_model.h5` |

지문자는 정지된 손 모양 하나로 판별되는 반면, 단어 수어는 시간에 따른 동작(양손 + 팔 움직임)으로
판별되기 때문에 입력 관절과 시퀀스 길이, 모델 구조를 다르게 가져갑니다.

## 데이터 수집

각 모델 폴더(`phoneme_model/`, `word_model/`) 안에 동일한 이름의 스크립트가 있습니다.

```
python record_dataset.py          # 웹캠으로 클래스별 영상 녹화 -> dataset/videos/
python build_dataset_from_videos.py   # 영상 -> 특징 시퀀스 npy -> dataset/sequences/
```

## 학습 및 테스트

```
python train.py          # dataset/sequences/*.npy 를 모아 학습, models/*.h5 저장
python test_webcam.py     # 웹캠으로 학습된 모델 실시간 테스트 (ESC로 종료)
```

## Union-web 연동

`phoneme_model.h5`는 Union-web의 `app.py`가 그대로 로드할 수 있도록 입력 shape `(10, 55)`,
출력 31 클래스, `compile=False`로도 로드되는 순수 Keras H5로 저장됩니다. 학습이 끝나면
`phoneme_model/models/phoneme_model.h5` 파일을 Union-web 프로젝트 루트의 `gesture_classifier.h5`로
교체하면 됩니다.

`word_model.h5`는 아직 Union-web에 연동되어 있지 않습니다 (양손+pose 특징 추출 및 30프레임 시퀀스
처리를 위한 백엔드 로직 추가가 별도로 필요합니다 - 향후 작업).
