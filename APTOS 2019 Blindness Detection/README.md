# **Competition**
* multi-classification problem
* 안구 사진을 분석하여 5가지 진단 유형(0~4)으로 예측 분류.
* 5가지 진단 유형은 '당뇨병성 망막증'의 진행 정도로, 상태가 심할수록 실명될 확률이 높음. 
* 당뇨성 망막증을 자동으로 식별, 질병 검출을 가속화하기 위한 기계 학습 모델을 구축하여 질병 이미지를 자동으로 검사하고 상태가 얼마나 심각한지에 대한 정보를 제공하는 것이 대회의 목표.


# **References**
* https://www.kaggle.com/xhlulu/aptos-2019-densenet-keras-starter
* https://www.kaggle.com/ratan123/aptos-2019-keras-baseline
  - Keras를 이용한 baseline 참고.
* https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
  - 이미지 전처리.
* https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa
  - Quadratic Weighted Kappa optimizer.
* https://www.kaggle.com/carlolepelaars/efficientnetb5-with-keras-aptos-2019
  - 모델링 및 예측 부분은 전반적으로 해당 커널을 참고함.

 
# STEP
**1. load data**
- 데이터 정보 가져오기, 클래스 분포 분석.

**2. Image load & pre-processing**
- 영상에서 안구 영역을 제외한 배경 잘라냄.
- 영상 크기를 456으로 resize.
- opencv를 이용해 원본이미지와 가우시안 블러를 적용한 이미지에 가중치 합을 적용해 엣지와 고주파 성분들(감염 정도에 따라 안구에 반점이 있음.)을 선명하게 보정함으로써 모델이 특징 추출을 이롭게 한다.
- 이미지 전처리 함수를 만들고 keras image generator의 preprocessing_function 옵션을 주어 데이터 증식 단계에서 전처리 되도록 함.

**3. Image generator**
- 데이터 scale을 0 ~ 1사이로 rescale.
- rotation_range, zoom_range, horizontal_flip, vertical_filp 등 적용.

**4. Modeling**
- pretrain model 중 하나인 EfficientNetB5 사용.
- 대회 discussion에서 본 대회에서 제공하는 데이터(current data set) 외에 이전 대회(APTOS 2015?)에서 사용했던 이미지 데이터셋을 얻을 수 있었음.
- colab환경에서 이전 대회 데이터셋(external data set)을 이용해 EfficientNetB5모델을 30 epoch 정도 훈련시켜 사전훈련 모델을 만듬.
- 사전훈련모델 데이터 구성
    * train set=train set of external data + train set of current data
    * valid set=train set of current data(not exist in train set)
    
- 캐글 커널에서 사전훈련 모델과 current data set을 이용해 learning & prediction.

**5. Learning**
- epoch 100 / batch size 4
- 매 epoch 마다 kappa score 모니터링.
- EarlyStopping, ReduceLROnPlateau는 mse를 기준으로 함.(discussion에서 다른 참가자들이 올려놓은 글을 봤을 때 kappa score로 조절할 경우 매우 불안정하다고 함.)

**6. Prediction & TTA(Test Time Augmentation)**
- 본 대회에서는 Quadratic Weighted Kappa score로 예측 결과를 평가함.
- 여러 실험을 통해 확인한 결과, kappa score를 model의 metric으로 사용할 경우 예측 성능이 생각보다 좋게 나오지 않았음.
- 따라서 mse를 metric으로 모델을 학습시킨 후, 예측 성능을 높이기 위해 예측 결과를 qwk를 이용해 교정(Quadratic Weighted Kappa optimizer)
    1. validation set을 이용해 각 class에 대한 kappa score계산
    2. kappa score를 이용해 각 class에 대한 threshold 설정(coefficients).
    3. 예측을 수행한 뒤 결과값을 coefficitent를 기준으로 교정한다.
    
- tta 7 적용.(7번의 예측 후 평균)


# **Quadratic Weighted Kappa?**
- The quadratic weighted kappa is calculated between the scores which are expected/known and the predicted scores. 
- True값과 예측값 간의 점수를 계산하는 방법 중 하나. 다음의 5단계를 거쳐 qwk score를 구할 수 있다.
- 자세한 내용은 https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps 참조. 

      <5 step create qwk metric>

      step 1. create a multi class confusion matrix O between predicted and actual ratings.
      - confusion matrix(란, 두 벡터(실제값, 예측값)의 상관성을 나타낸 행렬 같은 것.  
      ex)
      class = [0, 1, 2, 3, 4]
      N = 5(class num)
      true=[1,2,3,3,3,4,1]
      pred=[0,0,1,2,3,1,4]

      NxN matrix O is
      0, 0, 0, 0, 0
      1, 0, 0, 0, 1
      1, 0, 0, 0, 0
      0, 1, 1, 1, 0
      0, 1, 0, 0, 0

      matrix의 i,j 값은 인덱스 순서에 따라 true가 i이고 pred가 j인 개수임.


      step 2. Create Weight matrix 
      - NxN weight matrix 생성. 

      for i in range(len(w)):
          for j in range(len(w)):
              w[i][j] = float(((i-j)**2)/(N-1)) 

      >>
      array([[0.    , 0.0625, 0.25  , 0.5625, 1.    ],
             [0.0625, 0.    , 0.0625, 0.25  , 0.5625],
             [0.25  , 0.0625, 0.    , 0.0625, 0.25  ],
             [0.5625, 0.25  , 0.0625, 0.    , 0.0625],
             [1.    , 0.5625, 0.25  , 0.0625, 0.    ]])



      step 3. Generate Histogram 
      - 두 벡터의 histogram 생성.
      true: [0, 2, 1, 3, 1]
      pred: [2, 2, 1, 1, 1]



      step 4. Expected matrix is calculated as the outer product between the actual rating's histogram vector of ratings and the predicted rating's histogram vector of ratings.
      - Expected matrix 생성. Expected matrix는 두 histogram 벡터 간 외적으로 생성

      0, 0, 0, 0, 0
      4, 4, 2, 2, 2
      2, 2, 1, 1, 1
      6, 6, 3, 3, 3
      2, 2, 1, 1, 1

      Normalise E and O matrix
      E = E/E.sum()
      O = O/O.sum()


      step 5.  Calculate Weighted Kappa
      num=0
      den=0
      for i in range(len(w)):
          for j in range(len(w)):
              num+=w[i][j]*O[i][j]
              den+=w[i][j]*E[i][j]

      weighted_kappa = (1 - (num/den)); weighted_kappa

* QWK score를 이용한 model monitoring.

      def get_preds_and_labels(model, generator):
          """
          Get predictions and labels from the generator
          """
          preds = []
          labels = []
          for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
              x, y = next(generator)
              preds.append(model.predict(x))
              labels.append(y)
          # Flatten list of numpy arrays
          return np.concatenate(preds).ravel(), np.concatenate(labels).ravel().astype(np.uint8)

      class Metrics(Callback):
          def __init__(self, model_name):
              self.model_name=model_name

          def on_train_begin(self, logs={}):
              self.val_kappas=[]

          def on_epoch_end(self, epoch, logs={}):
              """
              Gets QWK score on the validation data
              """
              # Get predictions and convert to integers
              y_pred, labels = get_preds_and_labels(self.model, val_gen)
              y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)

              # We can use sklearns implementation of QWK straight out of the box
              # as long as we specify weights as 'quadratic'
              _val_kappa = cohen_kappa_score(labels, y_pred, weights='quadratic')
              self.val_kappas.append(_val_kappa)

              print(f"val_kappa: {round(_val_kappa, 4)}")

              if _val_kappa == max(self.val_kappas):
                  print("Validation Kappa has improved. Saving model.")
                  self.model.save(self.model_name)
              return


      model = build_model()
      qwk = Metrics(model_name)

      model.fit_generator(train_gen,
                          steps_per_epoch=len(X_train)/BATCH_SIZE,
                          epochs=100,
                          validation_data=val_gen, 
                          validation_steps=len(X_val)/BATCH_SIZE,
                          verbose=1,
                          callbacks=[qwk]
                          )

* QWK score를 Keras metric으로 사용하는 방법.

      import tensorflow as tf

      # custom metric with TF
      def cohens_kappa(y_true, y_pred):
          y_true_classes = tf.argmax(y_true, 1)
          y_pred_classes = tf.argmax(y_pred, 1)

          return tf.contrib.metrics.cohen_kappa(y_true_classes, y_pred_classes, class_num)[1]
      
      
      densenet = DenseNet201(weights='../input/densenet/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
                             include_top=False,
                             input_shape=(SIZE,SIZE,3)
                            )

      model = Sequential()
      model.add(densenet)

      model.add(layers.GlobalAveragePooling2D())
      model.add(layers.Dropout(0.25))
      model.add(layers.Dense(class_num, activation='softmax'))

      model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(lr=0.0001),
                    metrics=['acc', cohens_kappa]
                   )

      return model
      
* QWK 관련 기타 참고 자료들
  * http://digital-thinking.de/keras-three-ways-to-use-custom-validation-metrics-in-keras/
  * https://stackoverflow.com/questions/54831044/how-can-i-specify-a-loss-function-to-be-quadratic-weighted-kappa-in-keras
  * https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps
  * https://www.kaggle.com/christofhenkel/weighted-kappa-loss-for-keras-tensorflow
  * https://www.kaggle.com/maxmanko/quadratic-weighted-kappa-metric-in-keras
  * https://ngeorge.us/tensorflow-with-keras/

