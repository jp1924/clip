from transformers.models.clip import CLIPPreTrainedModel, CLIPConfig, CLIPModel
from transformers import PreTrainedModel, AutoModel
from .configuration_custom_clip import CustomCLIPConfig
import torch.nn as nn
import torch


class CustomCLIPModel(CLIPModel):
    config_class = CustomCLIPConfig

    def __init__(self, config: CustomCLIPConfig) -> None:
        CLIPPreTrainedModel.__init__(self, config)
        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        print(config)
        self.text_model = AutoModel.from_config(text_config)
        self.vision_model = AutoModel.from_config(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # CLIPPreTrainedModel에 있는 inti_weight는 사용하지 않음. 어떤 모델이 들어올지 알 수 없기 때문에
        # 불러들인 각자 모델들에서 post_init을 진행함.
        # 이렇게 되면 CLIPPreTrainedModel에 있는 initialize_weight와 checkpoint 메서드는 사용하지 않게 됨. 단순 meta_data만 사용하게 됨.
        self.post_init()

    def post_init(self) -> None:
        self.text_model.post_init()
        self.vision_model.post_init()
