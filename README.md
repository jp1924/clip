# CLIP

README에서 설명하고 있는 경로들은 프로젝트 폴더 아래쪽에서 실행하는 걸 상정하고 작성하고 있음.

## CLIP의 Contribution

image encoder pretrain하는 방식

- 기존 방식의 문제점
  - Vision Encoder를 supervision image classification을 통해 pretrain.  
    그러다 보니 학습한 class 이외의 다른 class에 대한 zero shot 성능이 떨어지는 문제 발생.

  - image classification으로 pretrain 하다 보니 image-class 데이터를 이용해 학습.  
    하지만 공개되어 있는 image-class 데이터가 얼마 없고 만드는데도 많은 비용이 발생.

- CLIP이 제안하는 방법
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

## Experiments

해당 절에선 CLIP으로 학습 시킨 vision model의 zero shot성능르 측정함.

zero shot: model이 전혀 보지 못한 입력 값에 대해 일반화 하는 것을 말함.
classification에서의 zero shot은 전혀 보지 못한 input값의
