# Learning Transferable Visual Models From Natural Language Supervision

## Abstract

### 기존 방식의 문제점

- Vision에서의 사전학습은 고정된 class에 대한 분류 학습을 통해 이루어짐.
하지만 고정된 class만 예측하다 보니 class 이외의 객체를 예측하는 Zero-Shot 성능이 약함.

- 분류 학습을 위해선 각 이미지 별로 다수결의 사람이 선정한 golden-label이 필요함
하지만 각 이미지 별로 golden-label을 구축하는 데에는 많은 비용과 시간이 들어감.

### 논문이 제안하는 방식

- 웹에서 크롤링한 4억개의 text-image 데이터를 이용해 vision encoder를 학습시킴.
    classification이 아니기 때문에 제한된 class에 국한되지 않고 zero shot 성능이 월등히 뛰어남

- text model, vision model를 같이 학습 시켜 text model의 `인식력`을 vision model에 포함할 수 잇음.
  
- classification이 아니기에 NLP와 같이 웹에서 크롤링한 대량의 image-text를 통해 학습을 하는 것이 가능함.
  그리고 이런 방식을 통해 기존 vision model보다 여러 task에서 SOTA를 달성할 수 있었음.

- 이전에 제시된 ConVIRT의 구조를 따와, CLIP만의 방식으로 학습을 시도 함.

- 왜 Contrastive Learning를 사용?
  - `vision model의 성능을 높이기 위해 text model을 사용하는 방식은 이전부터 있어 왔었음.
  주로 text model이 image와 매칭된 txt를 예측하는 방식으로 vision model의 성능을 높혀 왔음.
  - 다만 text model의 큰 용량과 txt 상의 노이즈(설명, 주석) 등으로 인하여 기존 방식 대비 3배 느린 단점이 존재함.
  - `그래서 CLIP은 기존 방식 대신 contrastive learning을 통해 문제를 해결 함

## Introduction and Motivating Work

- NLP에선 웹에서 크롤링된 대량의 데이터로 CLM, MLM 등의 비지도 학습을 진행해 각 분야의 SOTA를 달성함.
이는 저품질의 데이터로도 SOTA를 달성할 수 있는 방법이 있음을 시사함.

- Vision에선 크라우드 소싱을 통한 고품질의 데이터를 통해 사전학습을 진행함.
NLP와 같이 웹에서 크롤링이 된 대량의 image-text 쌍의 데이터로 사전학습하는 방법을 연구함.

- 이전에도 image-text 쌍의 데이터를 이용해 vision 모델을 학습시키는 방법이 연구되어 왔음.
  1. 1999년, [Y. Mori](https://www.semanticscholar.org/paper/Image-to-word-transformation-based-on-dividing-and-Mori-Takahashi/8b29ffb4207435540ddecf4b14a8a32106b33830)와 같은 연구자 들이 이미지 검색을 위해 이미지-문서 쌍의 데이터에서 명사, 형용사를 예측하는 학습법을 제안 함.
  2. 2007년, [Quattoni, A](https://www.cs.upc.edu/~aquattoni/AllMyPapers/cvpr_07.pdf)와 같은 연구자 들이 이미지-캡션 쌍의 데이터에서 부분적으로 캡션의 단어를 예측하도록 만들어 분류기를 통해 이미지에 대한 표현을 학습할 수 있다는 것을 증명 함.
  3. 2012년, [Srivastava, N.](https://papers.nips.cc/paper_files/paper/2012/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)는 이미지-태그 쌍의 데이터에서 저수준의 이미지를 태그와 함께 Deep Boltzmann 모델을 사용해 이미지속에 포함된 특징을 학습하는 방법을 제안 함.
  여기서 말하는 저수준은 저수준의 정보로 조도, RGB와 같은 데이터를 뜻함. 반대로 고수준은 이미지속의 객체와 같은 추상적인 정보들을 뜻함.
  4. 2016년, [Joulin](*)는 이미지-캡션 쌍의 데이터에서 부분적으로 캡션의 단어를 에측하는 방식을 통해 CNN이 이미지에 대한 표현을 학습할 수 있다는 것을 증명 함.
  5. 2017년, [Li](*)는 이미지-캡션 쌍의 데이터에서 개별적인 단어 이외의 N-Gram을 통한 구문 예측을 통해 Zero-Shot 성능을 입증 함.
  6. 2020년, [Desai & Johnson](*)의 VirTex, ICMLM(Bulent Sariyildiz 외, 2020), ConVIRT(Zhang 외, 2020)는 CLM과 MLM을 통해 이미지의 표현을 학습시킬 수 있다는 것을 입증 함.

- 이미지의 표현을 학습하는데 NLP을 사용하고자 하는 방법이 꾸준이 제안되어 왔지만 기존 방식 대비 낮은 성능 때문에 연구가 많이 진척되지 않음.
그리고 사전학습에 적은 양의 고품질 데이터, 많은 양의 저품질 데이터를 이용해 학습시켜도 Softmax를 이용한 분류를 통해 모델을 학습하다 보니
학습한 class 이외의 다른 class에 대한 Zero-Shot 성능이 떨어짐.

## Approach

### Natural Language Supervision

- 기존에 자연어를 Vision 모델 훈련에 사용하기 위해 N-Gram, Topic 모델링을 통해 학습시켜 옴.
하지만 최근 들어 CLM, MLM 방식이 제안 되면서 웹에서 크롤링이 된 저품질의 데이터로 부터 표현을 학습할 수 있는 방법이 제안됨.

- 우리의 접근 방식은 자연어를 Vision 모델의 라벨로 사용하는 것을 제안함.
자연어를 이용해 Vision 모델을 학습시키면 다음과 같은 장점이 있음
  1. 데이터를 수집하기가 쉬워짐
  2. 자연어와

### Creating a Sufficiently Large Dataset

- 기존 Vision 모델을 사전학습 하기 위해 MS-COCO(10만장), Visual-Genome(10만장), YFCC100M(35억장)등을 사용해 왔음
하지만 각 데이터 별로 품질이 일정하지 않으며, 3개의 데이터에서 학습에 사용할 수 있는 데이터만 필터링 하면 1,500만 장만 남게 됨.

- 자연어를 통한 Vision 모델 사전학습은 데이터의 품질에 많은 제약을 받지 않는 장점이 있음.
그래서 다양한 도메인의 웹에서 크롤링된 4억장으로 구성된 이미지-캡션 쌍의 데이터를 구축해 사전학습에 사용 함.
다양한 도메인의 웹: 대략 50만개의 검색어를 이용해 웹에서 검색해 텍스트를 구축 함.

### Selecting an Efficient Pre-Training Method

### Choosing and Scaling a Model

### Training
