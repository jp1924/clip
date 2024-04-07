# Learning Transferable Visual Models From Natural Language Supervision

## Abstract

### 기존 방식의 문제점

- Vision Pretrain은 고정된 class에 대한 분류 학습을 통해 이루어짐.
    하지만 고정된 class만 예측하다 보니 class 이외의 객체를 예측하는 Zero-Shot 성능이 약함.

- 분류 학습을 위해선 각 이미지 별로 다수결의 사람이 선정한 golden-label이 필요함
    하지만 각 이미지 별로 golden-label을 구축하는 데에는 많은 비용과 시간이 들어감.

### 논문이 제안하는 방식

- Contrastive Loss를 이용해 Image-Text 쌍의 데이터에 대해 학습을 진행.
    classification이 아니기 때문에 제한된 class에 국한되지 않고 zero shot 성능이 월등히 뛰어남

- golden label이 필요로 하지 않기 때문에 크롤링을 통해 데이터를 대량으로 구축할 수 있음.
    학습을 위한 마땅한 데이터가 없어서 4억 쌍의 Image-Caption 쌍으로 구성된 WIP 데이터를 구축

## Introduction and Motivating Work

- NLP에서 원시 텍스트로만 Pretrain하는 CLM, MLM 방식이 공개됨
    Pretrain을 위해 굳이 golden label로 구성된 데이터로 학습할 필요가 없다는 사실이 증명됨.

- 하지만 Vision에선 여전히 많은 자원을 들여서 만든 Golden label로만 Pretrain하는 방식이 사용되고 있음.
    이전부터 Vision Pretrain에 자연어 사용이 꾸준히 제안되어 왔지만 기존 방식 대비 낮은 성능 때문에 활발히 연구되질 못함.

    Golden label: 다수결의 사람에 의해 선택된 데이터의 라벨,
    예: 사진 보여주고 `이 사진 고양이 일까 아닐까` 했을 때 다수의 사람이 고양이라 하면 1, 아니라 하면 0 이런식으로 만드는 방식
    다수의 사람을 통해 만들어 지다 보니 품질은 좋은데 만들기갸 힘듬

- 현재 높은 품질을 가진 재한된 양의 label(Golden-label)과 낮은 품질을 가진 무한한 양의 label(원시 텍스트)를 이용한 Pretrain 기법이 존재.
    우린 이 두 Pretrain기법의 중간 지점에 있는 기법인 CLIP을 제안.

## Approach

### Natural Language Supervision

- 기존 Vision Pretrain은 Softmax를 이용한 clasification을 통해 시각적 정보를 학습해 왔음.
    하지만 학습한 class외의 다른 class에 대한 Zero-Shot 성능이 제한되는 문제가 있었음

- 자연어를 통한 Pretrain은 Vision clasification대비 데이터를 통한 확장이 용이함.
    CLM, MLM를 통해 Text를 `단순` 보는 것이 아닌 언어적인 표현을 이해하기 때문에 유연한 Zero-Shot이 가능.

### Creating a Sufficiently Large Dataset

- 기존 Vision Pretrain에서 사용하던 MS-COCO, Visual Genome, YFCC100M를 그대로 사용하기에는 다음과 같은 문제가 있음.
    1. MS-COCO, Visual Genome 데이터를 사용하기엔 양이 적음.
    2. YFCC100M는 품질이 일정하지 않아 사용할 수 있는 데이터가 적음.

- 문제 해결를 위해 인터넷에서 수집한 4억 장의 Image-Text 쌍으로 구성된 WebImageText(WIT)를 구축해 pretrain에 사용. (WIT는 공개 X)

### Selecting an Efficient Pre-Training Method

- Vision Pretrain을 하기 위해 많은 양의 자원이 필요로 함.
    EfficientNet-L2을 Pretrain하기 위해 1대의 TPU로 33년이 걸림.

- CNN과 Transformer를 합친 VirTex는 Image-Caption 쌍의 데이터의 Caption을 구성하는 Text를 생성하는 방식으로 학습을 진행함.
    하지만 ResNet-50에 비해 3배 이상 느리면서 2배 많은 연산량을 필요로 하게 됨.
    Text를 생성하는 방식으로 학습을 진행하면 Text상에 존재하는 노이즈도 같이 예측되어서 속도도 더 느려짐

- Vision Classification조차 많은 자원이 들어가는 상황에서 Vision-Text는 부담스러울 정도의 자원이 필요
    Pretrain에서의 학습 효율성은 모델 개발에 중요한 요소

- Text를 생성하는 대신 Image-Text 쌍을 올바르게 매칭하는 것을 목표로 하는 Contrastive Loss를 사용
    N개의 Image-Text 쌍의 데이터에서 N개의 Positive(올바르게 매칭된 이미지와 텍스트 쌍)의 유사도는 최대화, (N x 2) - N개의 negative(잘못 매칭된 이미지와 텍스트 쌍)의 유사도를 최소화 시키는 방식으로 학습을 진행
    이후 N x N의 행렬에서 대각선상의 존재하는 N개의 값에 대해서 CrossEntropy를 진행해 loss를 계산.

- 새로 구축된 WIT의 성능 비교를 위해 CLIP은 Scratch부터 Pretrain 함.
    데이터가 크기 때문에 학습 시 과적합은 크게 고려하지 않았음
    Vision, language Encoder에서 출력된 각 신호의 크기를 맞추기 위해 Linear Projection을 사용
    Non-Linear Proejction과 Linear Projection간의 성능적 차이는 없었다고 함.(특이하네)

### Choosing and Scaling a Model

// 왜 굳이 ResNet-50과 VIT를 사용해 테스트를 했는지 의문, 어떤 부분을 볼려 한거지?

- image encoder로 ResNet-50과 VIT를 각각 사용해 테스트를 진행 함.
    단 ResNet-50에서 사용되고 있던 `rect-2 blur` 풀링 대신 `Attention` 풀링 사용

    Image encoder의 width, height

- Text Encoder로 63M의 파라메터를 가진 GPT-2를 사용
    단 계산 효율성을 위해 text encoder엗 들어가는 문장의 최대길이는 76토큰으로 제한 함.

- Tan & Le(2019)의 연구를 참고해 width, height를 동시에 늘림.
    width가 증가된 비율에 따라 Text Encoder의 크기도 같이 비례해서 늘림.

### Training

- 동일한 Text Encoder를 사용한 상태에서 5개의 ResNet과 3개의 VIT를 이용해 학습을 진행
  - ResNet
    ResNet-50: 50개의 layer
    ResNet-101: 101개의 layer
    RN50x4: EfficientNet 스타일의 스케일링을 따름.
    RN50x16: EfficientNet 스타일의 스케일링을 따름.
    RN50x64: EfficientNet 스타일의 스케일링을 따름.

  - VIT
    ViT-B/32
    ViT-B/16
    ViT-L/14

- Training Hyperparameter
  - epoch: 32
  - batch: 32,768
  - optimizer: adam decay
  - τ: 초기 값 0.07로 초기화, max 값으로 100 이상은 넘지 않도록 재한

- Training Detail
  - fp16
  - gpu: ResNet은 V100 592개, Transformers 256개
  - well time: ResNet 18일 Transformers 12일
  - Touvron et al.(2019)방법론을 통해 ViT-L/14는 해상도를 336로 높혀서 추가로 사전학습을 더 진행.
    Touvron et al.(2019): 서로 다른 해상도의 이미지를 2 stage 기법으로 학습시키는 방법론이라 보면 됨.
    CLIP에서 이 방식으로 학습시킨 모델이 가장 성능이 좋았다고 언급함.

## Experiments

### Zero-Shot Transfer

#### MOTIVATION

- Vision의 Zero-Shot은 학습하지 못한 입력을 본인이 학습 했던 category의 값으로 일반화 시키는 것을 말함.
    CLIP에선 학습하지 않은 데이터에 집합에 대한 일반화를 연구하는 것으로 확장함.
