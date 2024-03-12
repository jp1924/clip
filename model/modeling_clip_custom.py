from transformers.models.clip import CLIPPreTrainedModel, CLIPModel
from transformers import AutoModel
from .configuration_clip_custom import CustomCLIPConfig
import torch.nn as nn
import torch


class CLIPCustomModel(CLIPModel):
    config_class = CustomCLIPConfig

    def __init__(self, config: CustomCLIPConfig) -> None:
        CLIPPreTrainedModel.__init__(self, config)
        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = AutoModel.from_config(text_config)
        self.vision_model = AutoModel.from_config(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
