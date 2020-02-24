# **Competition**
* semantic segmentation problem
* 구름을 촬영한 위성 영상을 segmentation하는 문제. 영상에서 구름을 모양 별로 분류해야 한다.
* 바다위에서 형성되는 특정 구름들은 우주로 많은 햇빛을 반사시켜 지구를 냉각시키는 동시에 온실 효과에 미미한 영향을 주기 때문에 기온이 따뜻해짐에 따라 이 구름들이 어떻게 변할지를 알아내는 것은 기상학에서 매우 중요한 문제라고 함.
  * https://towardsdatascience.com/sugar-flower-fish-or-gravel-now-a-kaggle-competition-8d2b6b3b118
* 연구자들이 구름 영상을 분석한 결과 Sugar, Flower, Fish, Gravel 이렇게 네 가지 유형으로 구름의 유형을 구분.
* 위성 영상에서 각 유형에 해당하는 구름이 있는지와 그 영역의 위치를 픽셀 단위로 예측해야 한다.
* https://www.kaggle.com/c/understanding_cloud_organization


# **References**
* RLE to MASK, MASK to RLE and visualization code.
  * https://www.kaggle.com/aleksandradeis/understanding-clouds-eda
* Keras U-net base-line
  * https://www.kaggle.com/xhlulu/satellite-clouds-u-net-with-resnet-encoder
  * https://www.kaggle.com/dimitreoliveira/understanding-clouds-eda-and-keras-u-net/output#EDA


# **U-Net?**
- FCN(Fully convolutional neural network) 기반의 segmentation 모델로, 생체 이미지를 분석하기 위한 목적으로 개발됨.
- 네트워크 구조가 U 자 형태여서 U-Net이라고 함.
- 구조는 크게 2가지로 구성되어 있는데, convoultion-pooling 과정을 통해 영상에서 특징을 추출하는 인코더 파트(downsampling)와 줄어든 영상을 원래 크기로 복원하여 localization을 수행하는 디코더 파트(up-sampling)로 구성.
* 위성영상 segmentation 연구 논문에 자주 등장함.
* U-Net 관련 참고 자료
  * https://gaussian37.github.io/vision-segmentation-unet/


# STEP
**1. EDA & pre-processing**
- 데이터 정보 가져오기, 클래스 분포 분석.
- 영상에 대한 정보가 담긴 테이블에는 영상 파일명과 각 영상에 클래스가 존재할 경우 RLE(Run-Length Encoding, 클래스에 해당하는 픽셀의 위치와 밝기값 정보임) 값이 있으며, 존재하지 않을 경우 Null값으로 표시되어 있음. 데이터프레임 수정 작업 필요.

**2. Image generator**
- 입력 X는 이미지이며 출력 y는 클래스에 해당하는 픽셀 값들임.
- 클래스 정보는 RLE값으로 주어지므로 압축된 정보를 복원해야하는 작업이 필요함.
- 참고한 커널에서는 이를 위해 keras에서 제공하는 제너레이터 함수가 아닌 직접 튜닝한 제너레이터 사용.

**3. Modeling**
- keras 'sm: segmentation model'라이브러리 사용. 해당 라이브러리로 U-net 모델을 학습할 수 있음.
- ResNet-34를 backbone으로 사용함.(영상에서 특징을 추출하는 인코더 파트를 resnet을 사용한다는 의미같음)

**4. Learning**
- epoch 25 / batch size 32
- 본 대회에서는 Dice coefficient를 metric으로, BCE dice loss function을 loss function으로 사용함.(https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/)

**5. Prediction**
- batch size 500
- 예측 결과 영상 350x525로 rescale 및 RLE로 압축.

