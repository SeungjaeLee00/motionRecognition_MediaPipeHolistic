
# MediaPipe Holistic을 활용한 모션 인식 및 특정 행동 수행 여부 판단

## 개요

이 프로젝트는 MediaPipe Holistic을 활용하여 얼굴, 손, 포즈 추적을 결합한 종합 솔루션을 통해 사람의 동작을 인식하고 특정 행동을 감지합니다. 목표는 트래킹된 랜드마크를 기반으로 특정 동작을 식별하고, 해당 동작이 올바르게 수행되었는지 판단하는 시스템을 구현하는 것입니다.

## 주요 기능

- **실시간 모션 추적**: MediaPipe Holistic을 사용하여 얼굴, 손, 몸의 랜드마크를 실시간으로 추적합니다.
- **행동 인식**: 추적된 랜드마크를 기반으로 특정 동작을 식별하고 분류합니다.
- **행동 수행 여부 판단**: 미리 정의된 동작이 올바르게 수행되었는지 여부를 평가합니다.

## 요구 사항

- Python 3.8 이상
- OpenCV
- MediaPipe
- NumPy

## 설치 방법

1. 저장소를 클론합니다:

   ```bash
   git clone https://github.com/yourusername/mediapipe-holistic-motion-recognition.git
   cd mediapipe-holistic-motion-recognition
   ```

2. 필요한 패키지를 설치합니다:

   ```bash
   pip install opencv-python mediapipe numpy
   ```

## 사용 방법

### 모션 인식

1. `pose_estimation` 디렉토리로 이동합니다:

   ```bash
   cd pose_estimation
   ```

2. 상황에 맞는 코드를 실행합니다. 예를 들어, 비디오 파일을 사용하려면 `singlePerson_mediaPipe_Holistic_video.py`를, 웹캠을 사용하려면 `singlePerson_mediaPipe_Holistic_cam.py`를, 이미지를 사용하려면 `singlePerson_mediaPipe_Holistic.py`를 실행합니다.

   - 비디오 파일로 모션 인식을 실행하려면:

     ```bash
     python singlePerson_mediaPipe_Holistic_video.py
     ```

   - 웹캠으로 모션 인식을 실행하려면:

     ```bash
     python singlePerson_mediaPipe_Holistic_cam.py
     ```

   - 이미지로 모션 인식을 실행하려면:

     ```bash
     python singlePerson_mediaPipe_Holistic.py
     ```

### 특정 행동 수행 여부 판단

1. `pose_accuracy` 디렉토리로 이동합니다:

   ```bash
   cd pose_accuracy
   ```

2. `pose_accuracy.py` 스크립트를 실행합니다:

   ```bash
   python pose_accuracy.py
   ```

이렇게 하면 특정 동작이 올바르게 수행되었는지 여부를 평가할 수 있습니다.
