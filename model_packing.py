from dataclasses import dataclass, field

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    HfArgumentParser,
    VisionTextDualEncoderConfig,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
)


@dataclass
class PackingArgument:
    img_encoder: str = field(default="google/vit-base-patch16-224")
    txt_encoder: str = field(default="klue/roberta-base")
    projection_dim: int = field(default=512)

    save_dir: str = field(default="./dual_vision_encoder_model")


def main(packing_args: PackingArgument) -> None:
    img_config = AutoConfig.from_pretrained(packing_args.img_encoder)
    txt_config = AutoConfig.from_pretrained(packing_args.txt_encoder)

    img_extractor = AutoFeatureExtractor.from_pretrained(packing_args.img_encoder)
    txt_tokenizer = AutoTokenizer.from_pretrained(packing_args.txt_encoder)

    config = VisionTextDualEncoderConfig.from_vision_text_configs(
        vision_config=img_config,
        text_config=txt_config,
        projection_dim=packing_args.projection_dim,
    )

    model = VisionTextDualEncoderModel(config=config)
    processor = VisionTextDualEncoderProcessor(
        image_processor=img_extractor,
        tokenizer=txt_tokenizer,
    )

    model.save_pretrained(packing_args.save_dir)
    config.save_pretrained(packing_args.save_dir)
    processor.save_pretrained(packing_args.save_dir)


if "__main__" in __name__:
    packing_args, _ = HfArgumentParser([PackingArgument]).parse_args_into_dataclasses(return_remaining_strings=True)

    main(packing_args)
