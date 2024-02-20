from transformers import PretrainedConfig, AutoConfig, CLIPConfig
from typing import Union, Dict, Any, Optional
import transformers


class CustomCLIPConfig(PretrainedConfig):
    model_type = "clip"

    def __init__(
        self,
        text_config: Union[str, PretrainedConfig, dict] = None,
        vision_config: Union[str, PretrainedConfig, dict] = None,
        text_cofnig_kwagrs: Dict[str, Any] = {},
        vision_config_kwagrs: Dict[str, Any] = {},
        projection_dim: int = 512,
        logit_scale_init_value: int = 2.6592,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(text_config, str):
            text_config = AutoConfig.from_pretrained(text_config, **text_cofnig_kwagrs)
            text_name = text_config.__class__.__name__
            setattr(text_config, "class_name", text_name)

        elif isinstance(text_config, dict):
            # save_pretrained된 checkpoint된 모델을 불러올 때 config값이 dict 형태로 들어오기 때문에 dict를 처리하는 구간이 필요로 함.
            # text_config = PretrainedConfig.from_dict(text_config)
            text_config = getattr(transformers, text_config["class_name"])(**text_config)

        if isinstance(vision_config, str):
            vision_config = AutoConfig.from_pretrained(vision_config, **vision_config_kwagrs)
            vision_name = vision_config.__class__.__name__
            setattr(vision_config, "class_name", vision_name)
        elif isinstance(vision_config, dict):
            # vision_config = PretrainedConfig.from_dict(vision_config)
            vision_config = getattr(transformers, vision_config["class_name"])(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
