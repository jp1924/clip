# CLIP
README에서 설명하고 있는 경로들은 프로젝트 폴더 아래쪽에서 실행하는 걸 상정하고 작성하고 있음.



## CLIP의 Contribution
- CLIP 이전의 Vision Pretrain 방식    
    1. supervision Image classification을 통해 Vision Encoder를 pretrain.     
    그러다 보니 학습한 class 재외한 다른 class에 대한 zero shot 성능이 떨어짐

- CLIP의 제안
    1. 웹에서 대량으로 크롤링한 4억개의 text-image쌍 데이터를 이용해 Vision Encoder를 학습시키자
    classification이 아니기 때문에 zero shot 성능이 월등히 뛰어남

