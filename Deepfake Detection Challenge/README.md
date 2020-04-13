# **Competition**
* video data analysis and face detection
* 진짜 영상과 가짜 영상 구분하기. 
* https://www.kaggle.com/c/deepfake-detection-challenge


# **Deep fake?**
- 딥페이크란, AI 기술을 이용해 특정 사람의 얼굴이나 목소리를 합성 구현한 것.


# **Fake video detection pipline**
**1. Pre-train**
- 외부에서 받아온 real face image와 fake face image 데이터를 이용해 사전훈련모델(VGG, ResNet, DenseNet, EfficientNet,...)을 학습시킨 뒤 해당 모델을 저장한다.
- face image 데이터는 kaggle플랫폼에서 쉽게 구할 수 있다.

**2. Face detection**
- 예측에 사용할 비디오 영상에서 face image를 추출하여 데이터셋을 구성한다.

**3. Learning**
- 사전 훈련된 모델을 불러와 다시 학습.

**4. Prediction**
- 예측 결과는 'FAKE'일 경우 1, 'REAL'일 경우 0에 가까운 값을 가진다.
- LogLoss를 이용해 scoring


# **Files**
* video-data-eda-face-detection-visualization.ipynb
  * video 데이터 시각화 및 face detection 도구 사용법.
  * OpenCV Haar cascade. face recognition package, MTCNN face detector.
* deepfake-detection-with-mtcnn-pretrain-densenet.ipynb
  * how to build a deepfake detection workflow around the Keras DenseNet.
