# **Competiton**
* 자동차 모델 분류 문제
* https://www.kaggle.com/c/2019-3rd-ml-month-with-kakr

# **References**
* https://www.kaggle.com/tmheo74/3rd-ml-month-car-image-cropping-updated-7-10
  - 이미지 crop 참고.
* https://www.kaggle.com/fulrose/3rd-ml-month-car-model-classification-baseline
  - 참고하여 **ResNet50** baseline 완성
* https://www.kaggle.com/janged/3rd-ml-month-xception-stratifiedkfold-ensemble
  - image generator, callback 함수, KFold 구현 방법 참고.
* https://www.kaggle.com/c/2019-3rd-ml-month-with-kakr/discussion/99538#latest-594363
  - 제공해주신 팁을 기반으로 metric 튜닝
  
# STEP
**1. EDA**
- 데이터 구성, 클래스 분포 등 분석

**2. Image load & pre-processing**
- 영상에서 불필요한 배경 부분 잘라내기
- 영상 크기를 224로 resize. (ResNet50 default image size로 맞춤.)

**3. Image generator**
- 입력 데이터에 여러 변형을 가해 증식.
- 데이터 scale을 0 ~ 1사이로 rescale. (디지털 영상은 값이 0~255인 픽셀들로 이루어짐.신경망은 작은 값을 좋아하므로 rescale.)
- 데이터 증식은 훈련셋에만 적용. 검증 및 테스트셋에 적용하면 안됨.

**4. Modeling**
- pretrain model 중 하나인 ResNet50 사용. https://keras.io/applications/#resnet
- Imagenet 데이터베이스의 1백만개가 넘는 이미지에 대해 훈련된 네트워크로, 영상을 1000가지 범주로 분류할 수 있음.
- 본 문제에서는 분류해야할 범주가 196가지이므로, include top 옵션을 False로 설정하여 pretrain 모델의 FC층은 사용하지 않고 최종 출력이 196이 되도록 따로 층을 쌓아 구성.
- ResNet50의 conv layer로부터 영상의 특징만 추출한 뒤 따로 추가한 FC layer에 연결하는 방식. 

**5. Learning**
- K-Fold cross validation. 
- 훈련셋을 K번 tran/validation set으로 나누어 훈련.
- epoch 30(1Fold) / batch size 64
- K개의 모델 생성.

**6. Prediction**
- K개의 모델로 각 각 예측 결과내어 평균값을 최종 prediction값으로 도출.
