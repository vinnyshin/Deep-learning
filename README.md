# Deep Learning Assignments

이 저장소는 두 개의 딥러닝 과제 솔루션을 포함합니다.

- **Section 1: Assignment 1 – 이미지 분류 (CIFAR-10)**
- **Section 2: Assignment 2 – GAN을 활용한 이미지 생성 (CelebA)**

------

## Section 1: Assignment 1 – 이미지 분류

### 개요

과제 1은 오직 Linear Layer(즉, MLP)만을 사용하여 CIFAR-10 데이터셋의 이미지를 분류하는 것이 목표입니다. 제공된 조건에 맞춰 CNN이나 Transformer는 사용하지 않고, **batch normalization**, **dropout** 및 **L2 정규화**를 활용하여 모델의 분산을 줄이고 성능을 향상시키는 다양한 실험(Trial 1~26)을 진행하였습니다. 최종적으로 약 58%의 테스트 정확도를 달성하였습니다.

### 주요 특징

- 모델 아키텍처
  - 총 7개의 Linear Layer로 구성된 모델
  - 각 레이어마다 Batch Normalization과 ReLU 활성화 함수 적용
  - 레이어별로 차등 적용된 Dropout (3개로 분할 적용)
- 학습 세부 사항
  - Optimizer: Adam (lr = 0.0007, betas=(0.9, 0.999))
  - Loss Function: CrossEntropyLoss
  - Batch Size: 256
  - 입력 데이터 전처리: CIFAR-10의 평균 및 표준편차를 사용한 Input Normalization
- **실험 결과**:
  다양한 하이퍼파라미터(λ, dropout, layer/neuron 조정 등)를 변경하며 성능을 비교, 최적의 설정을 도출함
  
  (자세한 내용은 [2017029870_assignment_1.pdf](https://github.com/vinnyshin/Deep-learning/blob/main/Assignment/assignment_1/2017029870_assignment_1.pdf) 및 [딥러닝및응용_과제1.pdf](https://github.com/vinnyshin/Deep-learning/blob/main/Assignment/assignment_1/%EB%94%A5%EB%9F%AC%EB%8B%9D%EB%B0%8F%EC%9D%91%EC%9A%A9_%EA%B3%BC%EC%A0%9C1.pdf) 참조)

------

## Section 2: Assignment 2 – GAN을 활용한 이미지 생성

### 개요

과제 2는 GAN 네트워크를 완성하여 CelebA 데이터셋을 기반으로 이미지 생성을 수행하는 과제입니다.

- **Generator**: 제공된 코드를 그대로 사용 (수정 불가)

- Discriminator

  : 직접 구현하며, 두 가지 모델을 실험함

  - **Model 1**: 기존의 CNN 및 Linear Layer를 혼합한 구조 (FID 점수 약 240, 생성 이미지 품질 낮음)
  - **Model 2**: Generator의 구조를 반영한 미러 아키텍처 (FID 점수 38, 실제와 유사한 이미지 생성)
    
    (세부 내용은 [2017029870_assignment_2.pdf](https://github.com/vinnyshin/Deep-learning/blob/main/Assignment/assignment_2/2017029870_assignment_2.pdf) 및 [딥러닝및응용_과제2-1.pdf](https://github.com/vinnyshin/Deep-learning/blob/main/Assignment/assignment_2/%EB%94%A5%EB%9F%AC%EB%8B%9D%EB%B0%8F%EC%9D%91%EC%9A%A9_%EA%B3%BC%EC%A0%9C2-1.pdf) 참조)

### 주요 특징

- 모델 아키텍처
  - Generator
    - ConvTranspose2d, Batch Normalization, ReLU, 마지막에 Tanh 적용
    - 입력: [batch size, 128, 1, 1]
  - Discriminator
    - *Model 1*: Convolution layer 2겹, 이후 Linear Layer 및 dropout 적용
    - *Model 2*: Generator의 구조를 반영한 연속적인 Convolution layer 구성, 최종 Sigmoid로 진위 판별
- 학습 세부 사항
  - Optimizer: Adam (lr = 0.0002)
  - Loss Function: Binary Cross Entropy Loss (BCE Loss)
  - Batch Size: 128
- 평가 지표
  - 생성 이미지의 품질을 FID (Fréchet Inception Distance) score로 평가
  - Model 2를 통해 FID score 38 달성
