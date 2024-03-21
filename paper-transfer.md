2.1. Natural Language Supervision
At the core of our approach is the idea of learning perception from supervision contained in natural language. As discussed in the introduction, this is not at all a new idea, however terminology used to describe work in this space is varied, even seemingly contradictory, and stated motivations are diverse. Zhang et al. (2020), Gomez et al. (2017), Joulin et al. (2016), and Desai & Johnson (2020) all introduce methods which learn visual representations from text paired with images but describe their approaches as unsupervised, self-supervised, weakly supervised, and supervised respectively.

We emphasize that what is common across this line of work is not any of the details of the particular methods used but the appreciation of natural language as a training signal. All these approaches are learning from natural language supervision. Although early work wrestled with the complexity of natural language when using topic model and n-gram representations, improvements in deep contextual representation learning suggest we now have the tools to effectively leverage this abundant source of supervision (McCann et al., 2017).

Learning from natural language has several potential strengths over other training methods. It’s much easier to scale natural language supervision compared to standard crowd-sourced labeling for image classification since it does not require annotations to be in a classic “machine learning compatible format” such as the canonical 1-of-N majority vote “gold label”. Instead, methods which work on natural language can learn passively from the supervision contained in the vast amount of text on the internet. Learning from natural language also has an important advantage over most unsupervised or self-supervised learning approaches in that it doesn’t “just” learn a representation but also connects that representation to language which enables flexible zero-shot transfer. In the following subsections, we detail the specific approach we settled on.

저희 접근 방식의 핵심은 자연어에 포함된 슈퍼비전을 통해 인식을 학습한다는 아이디어입니다. 서론에서 설명했듯이 이것은 전혀 새로운 아이디어는 아니지만, 이 분야의 작업을 설명하는 데 사용되는 용어는 다양하고 심지어 모순적으로 보이며, 명시된 동기도 다양합니다. Zhang 등(2020), Gomez 등(2017), Joulin 등(2016), Desai & Johnson(2020)은 모두 이미지와 짝을 이룬 텍스트로부터 시각적 표현을 학습하는 방법을 소개하지만 그 접근 방식을 각각 무감독, 자기 감독, 약한 감독, 감독으로 설명합니다.

이러한 일련의 작업에서 공통적으로 강조하는 것은 사용된 특정 방법의 세부 사항이 아니라 자연어를 훈련 신호로 인식한다는 점입니다. 이러한 모든 접근 방식은 자연어 감독을 통해 학습합니다. 초기 연구에서는 토픽 모델과 n-그램 표현을 사용할 때 자연어의 복잡성과 씨름했지만, 심층 문맥 표현 학습의 개선으로 이제 이 풍부한 감독 소스를 효과적으로 활용할 수 있는 도구를 갖추게 되었습니다(McCann et al., 2017).

자연어를 통한 학습은 다른 학습 방법에 비해 몇 가지 잠재적인 강점이 있습니다. 이미지 분류를 위한 표준 크라우드 소스 라벨링에 비해 자연어 감독을 확장하기가 훨씬 쉬우며, 주석이 일반적인 '머신 러닝 호환 형식'인 N분의 1 다수결 '골드 라벨'이 아니어도 되기 때문입니다. 대신 자연어에서 작동하는 방법은 인터넷의 방대한 양의 텍스트에 포함된 감독을 통해 수동적으로 학습할 수 있습니다. 또한 자연어 학습은 대부분의 비지도 또는 자가 지도 학습 접근 방식에 비해 표현을 '단순히' 학습하는 것이 아니라 그 표현을 언어와 연결하여 유연한 제로 샷 전송을 가능하게 한다는 점에서 중요한 이점이 있습니다. 다음 하위 섹션에서는 우리가 결정한 구체적인 접근 방식에 대해 자세히 설명합니다.

2.2. Creating a Sufficiently Large Dataset

Existing work has mainly used three datasets, MS-COCO (Lin et al., 2014), Visual Genome (Krishna et al., 2017), and YFCC100M (Thomee et al., 2016). While MS-COCO and Visual Genome are high quality crowd-labeled datasets, they are small by modern standards with approximately 100,000 training photos each. By comparison, other computer vision systems are trained on up to 3.5 billion Instagram photos (Mahajan et al., 2018). YFCC100M, at 100 million photos, is a possible alternative, but the metadata for each image is sparse and of varying quality. Many images use automatically generated filenames like 20160716 113957.JPG as “titles” or contain “descriptions” of camera exposure settings. After filtering to keep only images with natural language titles and/or descriptions in English, the dataset shrunk by a factor of 6 to only 15 million photos. This is approximately the same size as ImageNet.

A major motivation for natural language supervision is the large quantities of data of this form available publicly on the internet. Since existing datasets do not adequately reflect this possibility, considering results only on them would underestimate the potential of this line of research. To address this, we constructed a new dataset of 400 million (image, text) pairs collected form a variety of publicly available sources on the Internet. To attempt to cover as broad a set of visual concepts as possible, we search for (image, text) pairs as part of the construction process whose text includes one of a set of 500,000 queries.1 We approximately class balance the results by including up to 20,000 (image, text) pairs per query. The resulting dataset has a similar total word count as the WebText dataset used to train GPT-2. We refer to this dataset as WIT for WebImageText.

기존 연구에서는 주로 MS-COCO(Lin et al., 2014), Visual Genome(Krishna et al., 2017), YFCC100M(Thomee et al., 2016)의 세 가지 데이터 세트를 사용했습니다. MS-COCO와 Visual Genome은 고품질의 크라우드 라벨링 데이터 세트이지만, 각각 약 10만 장의 훈련 사진으로 최신 기준으로는 규모가 작습니다. 이에 비해 다른 컴퓨터 비전 시스템은 최대 35억 장의 Instagram 사진으로 학습합니다(Mahajan et al., 2018). 1억 장의 사진으로 구성된 YFCC100M도 대안이 될 수 있지만, 각 이미지의 메타데이터가 드물고 품질이 다양합니다. 많은 이미지가 20160716 113957.jpg와 같이 자동으로 생성된 파일명을 '제목'으로 사용하거나 카메라 노출 설정에 대한 '설명'을 포함하고 있습니다. 자연어 제목 및/또는 설명이 영어로 된 이미지만 유지하도록 필터링한 후, 데이터 세트는 6배나 줄어든 1,500만 장의 사진으로 축소되었습니다. 이는 ImageNet과 거의 같은 크기입니다.

자연어 수퍼비전의 주요 동기는 이러한 형태의 데이터가 인터넷에서 공개적으로 사용 가능하다는 점입니다. 기존 데이터 세트는 이러한 가능성을 적절히 반영하지 못하기 때문에, 그 결과만을 고려하면 이 연구 분야의 잠재력을 과소평가할 수 있습니다. 이를 해결하기 위해 인터넷에서 공개적으로 이용 가능한 다양한 출처에서 수집한 4억 쌍(이미지, 텍스트)의 새로운 데이터 세트를 구축했습니다. 가능한 한 광범위한 시각적 개념을 포괄하기 위해 구축 과정의 일부로 텍스트가 50만 개의 쿼리 세트 중 하나를 포함하는 (이미지, 텍스트) 쌍을 검색했습니다.1 쿼리당 최대 2만 개의 (이미지, 텍스트) 쌍을 포함하여 결과의 클래스 밸런스를 대략적으로 맞췄습니다. 결과 데이터 세트의 총 단어 수는 GPT-2 훈련에 사용된 웹 텍스트 데이터 세트와 비슷합니다. 우리는 이 데이터 세트를 WebImageText용 WIT라고 부릅니다.

2.3. Selecting an Efficient Pre-Training Method

State-of-the-art computer vision systems use very large amounts of compute. Mahajan et al. (2018) required 19 GPU years to train their ResNeXt101-32x48d and Xie et al. (2020) required 33 TPUv3 core-years to train their Noisy Student EfficientNet-L2. When considering that both these systems were trained to predict only 1000 ImageNet classes, the task of learning an open set of visual concepts from natural language seems daunting. In the course of our efforts, we found training efficiency was key to successfully scaling natural language supervision and we selected our final pre-training method based on this metric.

Our initial approach, similar to VirTex, jointly trained an image CNN and text transformer from scratch to predict the caption of an image. However, we encountered difficulties efficiently scaling this method. In Figure 2 we show that a 63 million parameter transformer language model, which already uses twice the compute of its ResNet-50 image encoder, learns to recognize ImageNet classes three times slower than a much simpler baseline that predicts a bag-ofwords encoding of the same text.

Both these approaches share a key similarity. They try to predict the exact words of the text accompanying each image. This is a difficult task due to the wide variety of descriptions, comments, and related text that co-occur with images. Recent work in contrastive representation learning for images has found that contrastive objectives can learn better representations than their equivalent predictive objective (Tian et al., 2019). Other work has found that although generative models of images can learn high quality image representations, they require over an order of magnitude more compute than contrastive models with the same performance (Chen et al., 2020a). Noting these findings, we explored training a system to solve the potentially easier proxy task of predicting only which text as a whole is paired with which image and not the exact words of that text. Starting with the same bag-of-words encoding baseline, we swapped the predictive objective for a contrastive objective in Figure 2 and observed a further 4x efficiency improvement in the rate of zero-shot transfer to ImageNet.

Given a batch of N (image, text) pairs, CLIP is trained to predict which of the N × N possible (image, text) pairings across a batch actually occurred. To do this, CLIP learns a multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of the N real pairs in the batch while minimizing the cosine similarity of the embeddings of the N2 − N incorrect pairings. We optimize a symmetric cross entropy loss over these similarity scores. In Figure 3 we include pseudocode of the core of an implementation of CLIP. To our knowledge this batch construction technique and objective was first introduced in the area of deep metric learning as the multi-class N-pair loss Sohn (2016), was popularized for contrastive representation learning by Oord et al. (2018) as the InfoNCE loss, and was recently adapted for contrastive (text, image) representation learning in the domain of medical imaging by Zhang et al. (2020).

Due to the large size of our pre-training dataset, over-fitting is not a major concern and the details of training CLIP are simplified compared to the implementation of Zhang et al. (2020). We train CLIP from scratch without initializing the image encoder with ImageNet weights or the text encoder with pre-trained weights. We do not use the non-linear projection between the representation and the contrastive embedding space, a change which was introduced by Bachman et al. (2019) and popularized by Chen et al. (2020b). We instead use only a linear projection to map from each encoder’s representation to the multi-modal embedding space. We did not notice a difference in training efficiency between the two versions and speculate that non-linear projections may be co-adapted with details of current image only in self-supervised representation learning methods. We also remove the text transformation function tu from Zhang et al. (2020) which samples a single sentence at uniform from the text since many of the (image, text) pairs in CLIP’s pretraining dataset are only a single sentence. We also simplify the image transformation function tv. A random square crop from resized images is the only data augmentation used during training. Finally, the temperature parameter which controls the range of the logits in the softmax, τ , is directly optimized during training as a log-parameterized multiplicative scalar to avoid turning as a hyper-parameter.

최첨단 컴퓨터 비전 시스템은 매우 많은 양의 컴퓨팅을 사용합니다. Mahajan 등(2018)은 ResNeXt101-32x48d를 훈련하는 데 19 GPU 년이 필요했고, Xie 등(2020)은 노이즈가 많은 학생 EfficientNet-L2를 훈련하는 데 33 TPUv3 코어 년이 필요했습니다. 이 두 시스템이 1000개의 ImageNet 클래스만 예측하도록 훈련되었다는 점을 고려하면, 자연어에서 개방형 시각 개념 세트를 학습하는 작업은 벅차 보입니다. 노력하는 과정에서 훈련 효율성이 자연어 감독을 성공적으로 확장하는 데 핵심이라는 사실을 알게 되었고, 이 지표를 기반으로 최종 사전 훈련 방법을 선택했습니다.

VirTex와 유사한 초기 접근 방식은 이미지의 캡션을 예측하기 위해 이미지 CNN과 텍스트 트랜스포머를 처음부터 공동으로 학습시켰습니다. 하지만 이 방법을 효율적으로 확장하는 데 어려움을 겪었습니다. 그림 2에서는 이미 ResNet-50 이미지 인코더의 두 배에 달하는 연산량을 사용하는 6,300만 개의 파라미터 트랜스포머 언어 모델이 동일한 텍스트의 단어 모음 인코딩을 예측하는 훨씬 간단한 기준선보다 3배 느리게 ImageNet 클래스를 인식하는 법을 학습하는 것을 보여줍니다.

이 두 가지 접근 방식은 중요한 유사점을 공유합니다. 이들은 각 이미지와 함께 제공되는 텍스트의 정확한 단어를 예측하려고 합니다. 이미지와 함께 제공되는 설명, 주석, 관련 텍스트가 매우 다양하기 때문에 이는 어려운 작업입니다. 이미지에 대한 대조적 표현 학습에 대한 최근 연구에 따르면 대조적 목표가 동등한 예측 목표보다 더 나은 표현을 학습할 수 있다는 사실이 밝혀졌습니다(Tian et al., 2019). 다른 연구에서는 이미지의 생성 모델이 고품질 이미지 표현을 학습할 수 있지만, 동일한 성능을 가진 대조 모델보다 훨씬 더 많은 컴퓨팅을 필요로 한다는 사실을 발견했습니다(Chen et al., 2020a). 이러한 결과에 주목하여, 우리는 텍스트의 정확한 단어가 아닌 전체 텍스트가 어떤 이미지와 짝을 이루는지만 예측하는 잠재적으로 더 쉬운 대리 작업을 해결하기 위해 시스템을 훈련시키는 방법을 모색했습니다. 동일한 단어 묶음 인코딩 기준선에서 시작하여 그림 2의 예측 목표를 대조 목표로 바꾼 결과, 이미지넷으로의 제로 샷 전송률이 4배 이상 향상되는 것을 관찰했습니다.

N(이미지, 텍스트) 쌍의 배치가 주어지면 CLIP은 배치에서 가능한 N × N(이미지, 텍스트) 쌍 중 어떤 것이 실제로 발생했는지 예측하도록 훈련됩니다. 이를 위해 CLIP은 이미지 인코더와 텍스트 인코더를 공동으로 훈련하여 배치에 포함된 N개의 실제 쌍의 이미지와 텍스트 임베딩의 코사인 유사도를 최대화하는 동시에 N2 - N개의 잘못된 쌍의 임베딩의 코사인 유사도를 최소화함으로써 멀티 모달 임베딩 공간을 학습합니다. 이러한 유사성 점수에 대해 대칭 교차 엔트로피 손실을 최적화합니다. 그림 3에는 CLIP 구현의 핵심에 대한 의사 코드가 포함되어 있습니다. 우리가 아는 바로는 이 배치 구성 기법과 목적은 딥 메트릭 학습 영역에서 멀티 클래스 N-쌍 손실 Sohn(2016)에 의해 처음 소개되었고, Oord 등(2018)에 의해 대조적 표현 학습에 InfoNCE 손실로 대중화되었으며, 최근에는 Zhang 등(2020)에 의해 의료 영상 영역에서 대조적(텍스트, 이미지) 표현 학습에 적용되었습니다.

사전 훈련 데이터 세트의 크기가 크기 때문에 과적합은 큰 문제가 되지 않으며, Zhang 등(2020)의 구현에 비해 CLIP 훈련의 세부 사항이 간소화되었습니다. 이미지 인코더를 이미지넷 가중치로 초기화하거나 텍스트 인코더를 사전 훈련된 가중치로 초기화하지 않고 처음부터 CLIP을 훈련합니다. 우리는 표현과 대비 임베딩 공간 사이에 비선형 투영을 사용하지 않는데, 이는 Bachman 등(2019)이 도입하고 Chen 등(2020b)이 대중화시킨 변경 사항입니다. 대신 각 인코더의 표현에서 멀티 모달 임베딩 공간으로 매핑할 때 선형 투영만 사용합니다. 두 버전 간의 훈련 효율성에 차이가 없었으며, 비선형 투영은 자기 지도 표현 학습 방법에서만 현재 이미지의 세부 사항과 함께 적용될 수 있을 것으로 추측합니다. 또한 CLIP의 사전 훈련 데이터 세트의 많은 (이미지, 텍스트) 쌍이 단일 문장으로만 구성되어 있기 때문에 텍스트에서 단일 문장을 균일하게 샘플링하는 Zhang 등(2020)의 텍스트 변환 함수 tu를 제거했습니다. 또한 이미지 변환 기능인 tv도 단순화했습니다. 크기가 조정된 이미지에서 무작위 정사각형 자르기가 훈련 중에 사용되는 유일한 데이터 증강입니다. 마지막으로, 소프트맥스에서 로그의 범위를 제어하는 온도 파라미터인 τ는 하이퍼 파라미터로 변하는 것을 방지하기 위해 로그 파라미터화된 곱셈 스칼라로 훈련 중에 직접 최적화됩니다.

2.4. Choosing and Scaling a Model

We consider two different architectures for the image encoder. For the first, we use ResNet-50 (He et al., 2016a) as the base architecture for the image encoder due to its widespread adoption and proven performance. We make several modifications to the original version using the ResNetD improvements from He et al. (2019) and the antialiased rect-2 blur pooling from Zhang (2019). We also replace the global average pooling layer with an attention pooling mechanism. The attention pooling is implemented as a single layer of “transformer-style” multi-head QKV attention where the query is conditioned on the global average-pooled representation of the image. For the second architecture, we experiment with the recently introduced Vision Transformer (ViT) (Dosovitskiy et al., 2020). We closely follow their implementation with only the minor modification of adding an additional layer normalization to the combined patch and position embeddings before the transformer and use a slightly different initialization scheme.

The text encoder is a Transformer (Vaswani et al., 2017) with the architecture modifications described in Radford et al. (2019). As a base size we use a 63M-parameter 12- layer 512-wide model with 8 attention heads. The transformer operates on a lower-cased byte pair encoding (BPE) representation of the text with a 49,152 vocab size (Sennrich et al., 2015). For computational efficiency, the max sequence length was capped at 76. The text sequence is bracketed with [SOS] and [EOS] tokens and the activations of the highest layer of the transformer at the [EOS] token are treated as the feature representation of the text which is layer normalized and then linearly projected into the multi-modal embedding space. Masked self-attention was used in the text encoder to preserve the ability to initialize with a pre-trained language model or add language modeling as an auxiliary objective, though exploration of this is left as future work.

While previous computer vision research has often scaled models by increasing the width (Mahajan et al., 2018) or depth (He et al., 2016a) in isolation, for the ResNet image encoders we adapt the approach of Tan & Le (2019) which found that allocating additional compute across all of width, depth, and resolution outperforms only allocating it to only one dimension of the model. While Tan & Le (2019) tune the ratio of compute allocated to each dimension for their EfficientNet architecture, we use a simple baseline of allocating additional compute equally to increasing the width, depth, and resolution of the model. For the text encoder, we only scale the width of the model to be proportional to the calculated increase in width of the ResNet and do not scale the depth at all, as we found CLIP’s performance to be less sensitive to the capacity of the text encoder.

이미지 인코더에는 두 가지 아키텍처를 고려합니다. 첫 번째는 널리 채택되고 성능이 입증된 ResNet-50(He et al., 2016a)을 이미지 인코더의 기본 아키텍처로 사용합니다. He et al.(2019)의 ResNetD 개선 사항과 Zhang(2019)의 앤티앨리어싱 렉트-2 블러 풀링을 사용하여 원본 버전을 몇 가지 수정했습니다. 또한 글로벌 평균 풀링 레이어를 관심 풀링 메커니즘으로 대체했습니다. 주의 풀링은 쿼리가 이미지의 글로벌 평균 풀링 표현에 따라 조절되는 "트랜스포머 스타일" 다중 헤드 QKV 주의의 단일 계층으로 구현됩니다. 두 번째 아키텍처의 경우, 최근에 도입된 비전 트랜스포머(ViT)를 실험합니다(Dosovitskiy et al., 2020). 우리는 트랜스포머 앞에 결합된 패치와 위치 임베딩에 추가 레이어 정규화를 추가하고 약간 다른 초기화 체계를 사용하는 약간의 수정만으로 그들의 구현을 면밀히 따릅니다.

텍스트 인코더는 Radford 외(2019)에서 설명한 아키텍처를 수정한 트랜스포머(Vaswani 외., 2017)입니다. 기본 크기로는 8개의 주의 헤드를 가진 63M 파라미터 12-레이어 512-폭 모델을 사용합니다. 트랜스포머는 49,152개의 어휘 크기를 가진 텍스트의 소문자 바이트 쌍 인코딩(BPE) 표현에서 작동합니다(Sennrich et al., 2015). 계산 효율성을 위해 최대 시퀀스 길이는 76으로 제한되었습니다. 텍스트 시퀀스는 [SOS] 및 [EOS] 토큰으로 괄호로 묶여 있으며, [EOS] 토큰에서 가장 높은 변환기 계층의 활성화는 텍스트의 특징 표현으로 처리되어 계층 정규화된 다음 다중 모드 임베딩 공간에 선형적으로 투영됩니다. 텍스트 인코더에는 사전 학습된 언어 모델로 초기화하거나 보조 목적으로 언어 모델링을 추가하는 기능을 유지하기 위해 마스크된 자기 주의가 사용되었지만, 이에 대한 탐색은 향후 작업으로 남겨져 있습니다.

이전의 컴퓨터 비전 연구에서는 종종 너비(Mahajan et al., 2018) 또는 깊이(He et al., 2016a)를 개별적으로 증가시켜 모델을 확장했지만, ResNet 이미지 인코더에서는 폭, 깊이, 해상도 모두에 추가 컴퓨팅을 할당하는 것이 모델의 한 차원에만 할당하는 것보다 성능이 더 우수하다는 Tan & Le(2019)의 접근 방식을 채택했습니다. Tan & Le(2019)는 각 차원에 할당되는 컴퓨팅 비율을 EfficientNet 아키텍처에 맞게 조정하지만, 저희는 추가 컴퓨팅을 모델의 폭, 깊이, 해상도 증가에 동일하게 할당하는 간단한 기준을 사용합니다. 텍스트 인코더의 경우, CLIP의 성능이 텍스트 인코더의 용량에 덜 민감하다는 것을 발견했기 때문에 모델의 너비만 계산된 ResNet의 너비 증가에 비례하도록 스케일링하고 깊이는 전혀 스케일링하지 않습니다.

2.5. Training
We train a series of 5 ResNets and 3 Vision Transformers. For the ResNets we train a ResNet-50, a ResNet-101, and then 3 more which follow EfficientNet-style model scaling and use approximately 4x, 16x, and 64x the compute of a ResNet-50. They are denoted as RN50x4, RN50x16, and RN50x64 respectively. For the Vision Transformers we train a ViT-B/32, a ViT-B/16, and a ViT-L/14. We train all models for 32 epochs. We use the Adam optimizer (Kingma & Ba, 2014) with decoupled weight decay regularization (Loshchilov & Hutter, 2017) applied to all weights that are not gains or biases, and decay the learning rate using a cosine schedule (Loshchilov & Hutter, 2016). Initial hyperparameters were set using a combination of grid searches, random search, and manual tuning on the baseline ResNet50 model when trained for 1 epoch. Hyper-parameters were then adapted heuristically for larger models due to computational constraints. The learnable temperature parameter τ was initialized to the equivalent of 0.07 from (Wu et al., 2018) and clipped to prevent scaling the logits by more than 100 which we found necessary to prevent training instability. We use a very large minibatch size of 32,768. Mixed-precision (Micikevicius et al., 2017) was used to accelerate training and save memory. To save additional memory, gradient checkpointing (Griewank & Walther, 2000; Chen et al., 2016), half-precision Adam statistics (Dhariwal et al., 2020), and half-precision stochastically rounded text encoder weights were used. The calculation of embedding similarities was also sharded with individual GPUs computing only the subset of the pairwise similarities necessary for their local batch of embeddings. The largest ResNet model, RN50x64, took 18 days to train on 592 V100 GPUs while the largest Vision Transformer took 12 days on 256 V100 GPUs. For the ViT-L/14 we also pre-train at a higher 336 pixel resolution for one additional epoch to boost performance similar to FixRes (Touvron et al., 2019). We denote this model as ViT-L/14@336px. Unless otherwise specified, all results reported in this paper as “CLIP” use this model which we found to perform best.

5개의 ResNet과 3개의 비전 트랜스포머를 차례로 훈련합니다. 레스넷의 경우 레스넷-50, 레스넷-101, 그리고 EfficientNet 스타일의 모델 스케일링을 따르고 레스넷-50의 약 4배, 16배, 64배의 연산량을 사용하는 3개의 레스넷을 추가로 훈련합니다. 각각 RN50x4, RN50x16, RN50x64로 표시됩니다. 비전 트랜스포머의 경우 ViT-B/32, ViT-B/16, ViT-L/14를 훈련합니다. 모든 모델을 32개의 에포크로 훈련합니다. 이득이나 편향이 아닌 모든 가중치에 디커플링 가중치 감쇠 정규화(Loshchilov & Hutter, 2017)를 적용한 Adam 최적화 도구(Kingma & Ba, 2014)를 사용하고 코사인 스케줄을 사용하여 학습 속도를 감쇠시킵니다(Loshchilov & Hutter, 2016). 초기 하이퍼파라미터는 그리드 검색, 무작위 검색, 수동 튜닝을 조합하여 기준 ResNet50 모델에 대해 1회에 걸쳐 훈련한 후 설정했습니다. 그런 다음 계산상의 제약으로 인해 더 큰 모델에 대해 하이퍼파라미터를 휴리스틱 방식으로 조정했습니다. 학습 가능한 온도 파라미터 τ는 (Wu et al., 2018)의 0.07에 해당하는 값으로 초기화되었고, 훈련 불안정성을 방지하기 위해 필요한 100 이상의 로짓 스케일링을 방지하기 위해 클리핑되었습니다. 32,768개의 매우 큰 미니배치 크기를 사용했습니다. 훈련 속도를 높이고 메모리를 절약하기 위해 혼합 정밀도(Micikevicius et al., 2017)를 사용했습니다. 5개의 ResNet과 3개의 비전 트랜스포머를 차례로 훈련합니다. 추가 메모리를 절약하기 위해 그라디언트 체크포인트(Griewank & Walther, 2000; Chen et al., 2016), 반정밀 아담 통계(Dhariwal et al., 2020), 반정밀 확률적으로 둥근 텍스트 인코더 가중치를 사용했습니다. 임베딩 유사도 계산도 개별 GPU가 로컬 임베딩 배치에 필요한 쌍별 유사도의 하위 집합만 계산하는 방식으로 샤딩되었습니다. 가장 큰 ResNet 모델인 RN50x64는 592개의 V100 GPU에서 훈련하는 데 18일이 걸렸고, 가장 큰 Vision Transformer는 256개의 V100 GPU에서 12일이 걸렸습니다. 또한 ViT-L/14의 경우, FixRes와 유사한 성능을 향상시키기 위해 336픽셀의 더 높은 해상도로 한 번 더 사전 훈련했습니다(Touvron et al., 2019). 이 모델을 ViT-L/14@336px로 표시합니다. 달리 명시되지 않는 한, 이 백서에서 "CLIP"으로 보고된 모든 결과는 이 모델이 가장 성능이 좋은 것으로 확인된 것을 사용합니다.

3. Experiments
3.1.1. MOTIVATION
In computer vision, zero-shot learning usually refers to thestudy of generalizing to unseen object categories in imageclassification (Lampert et al., 2009). We instead use theterm in a broader sense and study generalization to unseendatasets. We motivate this as a proxy for performing unseen tasks, as aspired to in the zero-data learningpaper of Larochelle et al. (2008). While much research in the field ofunsupervised learning focuses on the representation learning capabilities of machine learning systems, wemotivate studying zero-shot transfer as a way of measuring the tasklearning capabilities of machine learning systems.In this view, a dataset evaluates performance on a task on a specific distribution. However, many popular computervision datasets were created by the research community primarilyas benchmarks to guide the development of generic imageclassification methods rather than measuring performanceon a specific task. While it is reasonable to say that theSVHN dataset measures the task of street number transcription on the distribution of Google Street Viewphotos, it is unclear what “real” task the CIFAR-10 dataset measures.It is clear, however, what distribution CIFAR-10 is drawnfrom - TinyImages (Torralba et al., 2008). On these kinds ofdatasets, zero-shot transfer is more an evaluation of CLIP’srobustness to distribution shift and domain generalizationrather than task generalization. Please see Section 3.3 foranalysis focused on this.

To our knowledge, Visual N-Grams (Li et al., 2017) first studied zero-shot transfer to existing image classification datasets in the manner described above. It is also the only other work we are aware of that has studied zero-shot transfer to standard image classification datasets using a generically pre-trained model and serves as the best reference point for contextualizing CLIP. Their approach learns the parameters of a dictionary of 142,806 visual n-grams (spanning 1- to 5- grams) and optimizes these n-grams using a differential version of Jelinek-Mercer smoothing to maximize the probability of all text n-grams for a given image. In order to perform zero-shot transfer, they first convert the text of each of the dataset’s class names into its n-gram representation and then compute its probability according to their model, predicting the one with the highest score.

Our focus on studying zero-shot transfer as an evaluation of task learning is inspired by work demonstrating task learning in the field of NLP. To our knowledge Liu et al. (2018) first identified task learning as an “unexpected side-effect” when a language model trained to generate Wikipedia articles learned to reliably transliterate names between languages. While GPT-1 (Radford et al., 2018) focused on pretraining as a transfer learning method to improve supervised fine-tuning, it also included an ablation study demonstrating that the performance of four heuristic zero-shot transfer methods improved steadily over the course of pre-training, without any supervised adaption. This analysis served as the basis for GPT-2 (Radford et al., 2019) which focused exclusively on studying the task-learning capabilities of language models via zero-shot transfer.

컴퓨터 비전에서 제로 샷 학습은 일반적으로 이미지 분류에서 보이지 않는 객체 범주에 일반화하는 연구를 의미합니다(Lampert et al., 2009). 저희는 이 용어를 더 넓은 의미로 사용하며 보이지 않는 데이터 집합에 대한 일반화를 연구합니다. 이는 Larochelle 등(2008)의 제로 데이터 학습 논문에서 지향하는 것처럼, 보이지 않는 작업을 수행하기 위한 프록시로 동기를 부여합니다. 비지도 학습 분야의 많은 연구는 머신러닝 시스템의 표현 학습 능력에 초점을 맞추고 있지만, 우리는 머신러닝 시스템의 작업 학습 능력을 측정하는 방법으로 제로 샷 전이를 연구합니다. 이 관점에서 데이터 세트는 특정 분포에서 작업에 대한 성능을 평가합니다. 그러나 연구 커뮤니티에서 널리 사용되는 많은 컴퓨터 비전 데이터 세트는 주로 특정 작업에 대한 성능을 측정하기보다는 일반적인 이미지 분류 방법의 개발을 안내하기 위한 벤치마크로 만들어졌습니다. SVHN 데이터 세트가 Google 스트리트 뷰 사진의 분포에서 도로 번호 전사 작업을 측정한다고 말하는 것이 합리적이지만, CIFAR-10 데이터 세트가 어떤 "실제" 작업을 측정하는지는 불분명합니다. 그러나 CIFAR-10이 어떤 분포에서 가져온 것인지는 분명합니다 - TinyImages(Torralba et al., 2008). 이러한 종류의 데이터 세트에서 제로 샷 전송은 작업 일반화보다는 분포 이동 및 도메인 일반화에 대한 CLIP의 견고성을 평가하는 데 더 적합합니다. 이에 초점을 맞춘 분석은 섹션 3.3을 참조하세요.

우리가 아는 한, Visual N-Grams(Li et al., 2017)는 위에서 설명한 방식으로 기존 이미지 분류 데이터 세트에 대한 제로 샷 전이를 처음으로 연구했습니다. 이 연구는 일반적으로 사전 학습된 모델을 사용하여 표준 이미지 분류 데이터 세트에 대한 제로 샷 전이를 연구한 유일한 연구이며 CLIP의 컨텍스트화를 위한 가장 좋은 기준점이 됩니다. 이 접근 방식은 142,806개의 시각적 n-그램(1~5-그램에 걸쳐 있음)으로 구성된 사전의 파라미터를 학습하고 차등 버전의 Jelinek-Mercer 평활화를 사용하여 이러한 n-그램을 최적화하여 주어진 이미지에 대한 모든 텍스트 n-그램의 확률을 최대화합니다. 제로 샷 전송을 수행하기 위해 먼저 각 데이터 세트의 클래스 이름의 텍스트를 n-그램 표현으로 변환한 다음 모델에 따라 확률을 계산하여 가장 높은 점수를 얻은 클래스를 예측합니다.

작업 학습의 평가로서 제로 샷 전이를 연구하는 데 초점을 맞춘 것은 NLP 분야에서 작업 학습을 입증한 연구에서 영감을 얻었습니다. 우리가 아는 바로는 Liu 등(2018)은 위키백과 문서를 생성하도록 훈련된 언어 모델이 언어 간 이름을 안정적으로 음역하는 방법을 학습했을 때 '예상치 못한 부작용'으로 과제 학습을 처음 확인했습니다. GPT-1(Radford 외, 2018)은 감독 미세 조정을 개선하기 위한 전이 학습 방법으로서 사전 훈련에 중점을 두었지만, 여기에는 네 가지 휴리스틱 제로 샷 전이 방법의 성능이 감독 적응 없이 사전 훈련 과정에서 꾸준히 향상되었음을 보여주는 제거 연구도 포함되었습니다. 이 분석은 제로 샷 전이를 통한 언어 모델의 과제 학습 능력 연구에만 초점을 맞춘 GPT-2(Radford et al., 2019)의 기초가 되었습니다.

3.1.2. USING CLIP FOR ZERO-SHOT TRANSFER
CLIP is pre-trained to predict if an image and a text snippet are paired together in its dataset. To perform zero-shot classification, we reuse this capability. For each dataset, we use the names of all the classes in the dataset as the set of potential text pairings and predict the most probable (image, text) pair according to CLIP. In a bit more detail, we first compute the feature embedding of the image and the feature embedding of the set of possible texts by their respective encoders. The cosine similarity of these embeddings is then calculated, scaled by a temperature parameter τ , and normalized into a probability distribution via a softmax. Note that this prediction layer is a multinomial logistic regression classifier with L2-normalized inputs, L2-normalized weights, no bias, and temperature scaling. When interpreted this way, the image encoder is the computer vision backbone which computes a feature representation for the image and the text encoder is a hypernetwork (Ha et al., 2016) which generates the weights of a linear classifier based on the text specifying the visual concepts that the classes represent. Lei Ba et al. (2015) first introduced a zero-shot image classifier of this form while the idea of generating a classifier from natural language dates back to at least Elhoseiny et al. (2013). Continuing with this interpretation, every step of CLIP pre-training can be viewed as optimizing the performance of a randomly created proxy to a computer vision dataset which contains 1 example per class and has 32,768 total classes defined via natural language descriptions. For zero-shot evaluation, we cache the zero-shot classifier once it has been computed by the text encoder and reuse it for all subsequent predictions. This allows the cost of generating it to be amortized across all the predictions in a dataset.

CLIP은 데이터 세트에서 이미지와 텍스트 스니펫이 함께 짝을 이루는지 예측하도록 사전 학습됩니다. 제로 샷 분류를 수행하기 위해 이 기능을 재사용합니다. 각 데이터 세트에 대해 데이터 세트의 모든 클래스 이름을 잠재적인 텍스트 쌍의 집합으로 사용하고 CLIP에 따라 가장 가능성이 높은 (이미지, 텍스트) 쌍을 예측합니다. 좀 더 자세히 설명하면, 먼저 이미지의 특징 임베딩과 각 인코더에 의한 가능한 텍스트 세트의 특징 임베딩을 계산합니다. 그런 다음 이러한 임베딩의 코사인 유사성을 계산하고 온도 매개변수 τ 로 스케일링한 다음 소프트맥스를 통해 확률 분포로 정규화합니다. 이 예측 레이어는 L2 정규화된 입력, L2 정규화된 가중치, 편향 없음, 온도 스케일링을 사용하는 다항 로지스틱 회귀 분류기라는 점에 유의하세요. 이렇게 해석하면 이미지 인코더는 이미지의 특징 표현을 계산하는 컴퓨터 비전 백본이고 텍스트 인코더는 클래스가 나타내는 시각적 개념을 지정하는 텍스트를 기반으로 선형 분류기의 가중치를 생성하는 하이퍼네트워크(Ha et al., 2016)입니다. Lei Ba 등(2015)이 이러한 형태의 제로 샷 이미지 분류기를 처음 도입했지만, 자연어에서 분류기를 생성하는 아이디어는 적어도 Elhoseiny 등(2013)으로 거슬러 올라갑니다. 이러한 해석을 계속 이어가면, CLIP 사전 학습의 모든 단계는 클래스당 1개의 예제를 포함하고 자연어 설명을 통해 정의된 총 32,768개의 클래스가 있는 컴퓨터 비전 데이터 세트에 무작위로 생성된 프록시의 성능을 최적화하는 것으로 볼 수 있습니다. 제로 샷 평가의 경우, 텍스트 인코더에서 제로 샷 분류기를 계산한 후 이를 캐시하여 이후의 모든 예측에 재사용합니다. 이를 통해 데이터 세트의 모든 예측에 걸쳐 이를 생성하는 비용을 상각할 수 있습니다.

3.1.3. INITIAL COMPARISON TO VISUAL N-GRAMS
In Table 1 we compare Visual N-Grams to CLIP. The best CLIP model improves accuracy on ImageNet from a proof of concept 11.5% to 76.2% and matches the performance of the original ResNet-50 despite using none of the 1.28 million crowd-labeled training examples available for this dataset. Additionally, the top-5 accuracy of CLIP models are noticeably higher than their top-1, and this model has a 95% top-5 accuracy, matching Inception-V4 (Szegedy et al., 2016). The ability to match the performance of a strong, fully supervised baselines in a zero-shot setting suggests CLIP is a significant step towards flexible and practical zero-shot computer vision classifiers. As mentioned above, the comparison to Visual N-Grams is meant for contextualizing the performance of CLIP and should not be interpreted as a direct methods comparison between CLIP and Visual N-Grams as many performance relevant differences between the two systems were not controlled for. For instance, we train on a dataset that is 10x larger, use a vision model that requires nearly 100x more compute per prediction, likely used over 1000x their training compute, and use a transformer-based model which did not exist when Visual N-Grams was published. As a closer comparison, we trained a CLIP ResNet-50 on the same YFCC100M dataset that Visual N-Grams was trained on and found it matched their reported ImageNet performance within a V100 GPU day. This baseline was also trained from scratch instead of being initialized from pre-trained ImageNet weights as in Visual N-Grams.

CLIP also outperforms Visual N-Grams on the other 2 reported datasets. On aYahoo, CLIP achieves a 95% reduction in the number of errors, and on SUN, CLIP more than doubles the accuracy of Visual N-Grams. To conduct a more comprehensive analysis and stress test, we implement a much larger evaluation suite detailed in Appendix A. In total we expand from the 3 datasets reported in Visual NGrams to include over 30 datasets and compare to over 50 existing computer vision systems to contextualize results.

표 1에서는 Visual N-Gram과 CLIP을 비교합니다. 최고의 CLIP 모델은 이 데이터 세트에 사용할 수 있는 128만 개의 크라우드 라벨링 학습 예제 중 어느 것도 사용하지 않았음에도 불구하고 개념 증명 수준에서 이미지넷의 정확도를 11.5%에서 76.2%로 향상시켰으며 원래 ResNet-50의 성능과 일치합니다. 또한 CLIP 모델의 상위 5위 정확도는 상위 1위보다 눈에 띄게 높으며, 이 모델의 상위 5위 정확도는 95%에 달해 Inception-V4와 일치합니다(Szegedy et al., 2016). 제로 샷 설정에서 강력하고 완전히 감독된 기준선의 성능과 일치할 수 있다는 것은 CLIP이 유연하고 실용적인 제로 샷 컴퓨터 비전 분류기를 향한 중요한 단계임을 시사합니다. 위에서 언급했듯이 Visual N-Gram과의 비교는 CLIP의 성능을 맥락화하기 위한 것으로, 두 시스템 간의 많은 성능 관련 차이가 통제되지 않았으므로 CLIP과 Visual N-Gram 간의 직접적인 방법 비교로 해석해서는 안 됩니다. 예를 들어, 10배 더 큰 데이터 세트에서 학습하고, 예측당 거의 100배 더 많은 컴퓨팅을 필요로 하는 비전 모델을 사용하며, 학습 컴퓨팅을 1000배 이상 사용했을 가능성이 있는 트랜스포머 기반 모델을 사용하고, Visual N-Grams가 출시될 당시에는 존재하지 않았던 트랜스포머 기반 모델을 사용했습니다. 더 자세히 비교하기 위해 Visual N-Grams가 훈련된 것과 동일한 YFCC100M 데이터 세트에서 CLIP ResNet-50을 훈련한 결과, V100 GPU 하루 만에 보고된 ImageNet 성능과 일치하는 것으로 나타났습니다. 이 기준선 역시 Visual N-Grams에서처럼 사전 훈련된 이미지넷 가중치에서 초기화하지 않고 처음부터 훈련했습니다.

CLIP은 보고된 다른 2개의 데이터 세트에서도 Visual N-Gram을 능가하는 성능을 보였습니다. aYahoo에서 CLIP은 오류 수를 95% 감소시켰으며, SUN에서는 Visual N-Gram의 정확도를 두 배 이상 높였습니다. 보다 포괄적인 분석과 스트레스 테스트를 수행하기 위해 부록 A에 자세히 설명된 훨씬 더 큰 규모의 평가 세트를 구현했습니다. 총 3개의 데이터 세트에서 30개 이상의 데이터 세트를 포함하도록 확장하고 50개 이상의 기존 컴퓨터 비전 시스템과 비교하여 결과를 맥락화합니다.

3.1.4. PROMPT ENGINEERING AND ENSEMBLING
Most standard image classification datasets treat the information naming or describing classes which enables natural language based zero-shot transfer as an afterthought. The vast majority of datasets annotate images with just a numeric id of the label and contain a file mapping these ids back to their names in English. Some datasets, such as Flowers102 and GTSRB, don’t appear to include this mapping at all in their released versions preventing zero-shot transfer entirely.2 For many datasets, we observed these labels may be chosen somewhat haphazardly and do not anticipate issues related to zero-shot transfer which relies on task description in order to transfer successfully.

A common issue is polysemy. When the name of a class is the only information provided to CLIP’s text encoder it is unable to differentiate which word sense is meant due to the lack of context. In some cases multiple meanings of the same word might be included as different classes in the same dataset! This happens in ImageNet which contains both construction cranes and cranes that fly. Another example is found in classes of the Oxford-IIIT Pet dataset where the word boxer is, from context, clearly referring to a breed of dog, but to a text encoder lacking context could just as likely refer to a type of athlete.

Another issue we encountered is that it’s relatively rare in our pre-training dataset for the text paired with the image to be just a single word. Usually the text is a full sentence describing the image in some way. To help bridge this distribution gap, we found that using the prompt template “A photo of a {label}.” to be a good default that helps specify the text is about the content of the image. This often improves performance over the baseline of using only the label text. For instance, just using this prompt improves accuracy on ImageNet by 1.3%. Similar to the “prompt engineering” discussion around GPT3 (Brown et al., 2020; Gao et al., 2020), we have also observed that zero-shot performance can be significantly improved by customizing the prompt text to each task. A few, non exhaustive, examples follow. We found on several fine-grained image classification datasets that it helped to specify the category. For example on Oxford-IIIT Pets, using “A photo of a {label}, a type of pet.” to help provide context worked well. Likewise, on Food101 specifying a type of food and on FGVC Aircraft a type of aircraft helped too. For OCR datasets, we found that putting quotes around the text or number to be recognized improved performance. Finally, we found that on satellite image classification datasets it helped to specify that the images were of this form and we use variants of “a satellite photo of a {label}.”.

대부분의 표준 이미지 분류 데이터 세트는 자연어 기반 제로 샷 전송을 가능하게 하는 클래스 이름 지정 또는 설명 정보를 사후 처리합니다. 대부분의 데이터 세트는 라벨의 숫자 ID로만 이미지에 주석을 달고 이 ID를 다시 영어로 된 이름에 매핑하는 파일을 포함합니다. Flowers102 및 GTSRB와 같은 일부 데이터 세트는 릴리즈된 버전에 이 매핑이 전혀 포함되어 있지 않아 제로 샷 전송을 완전히 방지하는 것으로 보입니다.2 많은 데이터 세트에서 이러한 라벨이 다소 우연적으로 선택되어 성공적인 전송을 위해 작업 설명에 의존하는 제로 샷 전송과 관련된 문제를 예상하지 못하는 것으로 나타났습니다.

일반적인 문제는 다의어입니다. 클래스 이름만이 CLIP의 텍스트 인코더에 제공되는 유일한 정보인 경우, 문맥이 부족하기 때문에 어떤 단어의 의미를 의미하는지 구분할 수 없습니다. 경우에 따라 같은 단어의 여러 의미가 동일한 데이터 세트에 다른 클래스로 포함될 수 있습니다! 건설 크레인과 날아다니는 크레인이 모두 포함된 ImageNet에서 이런 일이 발생합니다. 또 다른 예는 Oxford-IIIT 애완동물 데이터 세트의 클래스에서 찾을 수 있는데, 문맥상 복서라는 단어는 분명히 개 품종을 가리키지만 문맥이 부족한 텍스트 인코더에게는 운동선수의 한 종류를 가리킬 가능성이 높습니다.

또 다른 문제는 사전 학습 데이터 세트에서 이미지와 짝을 이루는 텍스트가 단 한 단어인 경우가 비교적 드물다는 점입니다. 일반적으로 텍스트는 어떤 식으로든 이미지를 설명하는 전체 문장으로 구성됩니다. 이러한 분포 격차를 해소하기 위해 '{라벨}의 사진'이라는 프롬프트 템플릿을 사용하면 이미지의 내용에 대한 텍스트를 지정하는 데 도움이 되는 좋은 기본값이 된다는 사실을 발견했습니다. 이렇게 하면 라벨 텍스트만 사용하는 기본값보다 성능이 향상되는 경우가 많습니다. 예를 들어, 이 프롬프트를 사용하는 것만으로도 ImageNet의 정확도가 1.3% 향상됩니다. GPT3에 대한 "프롬프트 엔지니어링" 논의와 유사하게, 각 작업에 맞게 프롬프트 텍스트를 사용자 지정함으로써 제로 샷 성능을 크게 향상시킬 수 있다는 사실도 관찰되었습니다(Brown et al., 2020; Gao et al., 2020). 다음은 몇 가지 예시입니다. 여러 세분화된 이미지 분류 데이터 세트에서 카테고리를 지정하는 것이 도움이 된다는 것을 발견했습니다. 예를 들어 Oxford-IIIT Pets에서는 "{레이블}의 사진, 애완동물의 한 종류."를 사용하여 컨텍스트를 제공하는 것이 효과적이었습니다. 마찬가지로 Food101에서는 음식의 종류를, FGVC Aircraft에서는 항공기의 종류를 지정하는 것도 도움이 되었습니다. OCR 데이터 세트의 경우, 인식할 텍스트나 숫자 주위에 따옴표를 넣으면 성능이 향상되는 것을 발견했습니다. 마지막으로 위성 이미지 분류 데이터세트에서는 이미지가 이러한 형식임을 명시하고 "{라벨}의 위성 사진"과 같은 변형을 사용하는 것이 도움이 된다는 것을 발견했습니다.