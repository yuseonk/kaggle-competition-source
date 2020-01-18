# **Competition**
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
 
# ResNet?
* 아주 깊은 신경망을 학습하지 못하는 이유 중 하나가 학습 과정에서 gradient가 손실되거나 폭발적으로 증가하기 때문.
* ResNet은 **skip connection**으로 이 문제를 해결.
  skip connection(short cut)이란 한 층의 활성값을 더 깊은 층에 적용하는 방식을 말함. 이를 통해 훨씬 더 깊은 훈령망을 훈련시킬 수 있다.
* ResNet은 여러개의 residual block으로 구성.
  residual block이란 다음 그림과 같이 현재 층의 선형방정식에 이전 층의 입력 값을 더하고 이를 활성화 함수(ReLU)에 적용하는 것을 말함.


![ResNet](./img/RB.JPG)


  * 기존에는 입력 a가 한 층을 거쳐 다음층 a2로 가기까지 
  * (선형성적용 z1 = aW+b -> 활성 a1 = g(z1)) --> (선형성적용 z2 = a1W+b -> 활성 a2 = g(z2))
  * 이러한 과정을 거쳤는데 여기에 skip connection을 적용하게 되면 residual block을 구성할 수 있다.
  * 이 때 신경망 a2 = g(z2+a)가 되며 이를 전개하면 a2 = g(a1W+b) + a 이다.
  * 그런데 만약, L2같은 규제로 인해 W와 b가 0이 된다면 a2 = g(a) = a 로 항등식이 된다.
  * 이는 신경망에 두 개의 층을 추가한다 하더라도 이와 같은 skip connection 덕분에 더 간단한 항등식을 학습할 수 있게 되며, 
  * 두 층 없이도 더 좋은 성능을 낼 수 있게 만드는 것을 의미한다. 
 
[Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385)

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
