from typing import Any, Optional

from datasets import Dataset
from torch.utils.data import Sampler
from transformers import Trainer

from .weighted_sampler import WeightedSampler


class ContrastiveTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[Sampler]:
        return WeightedSampler(
            batch_size=self.args.train_batch_size,
            dataset=self.train_dataset,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Any | None:
        return WeightedSampler(
            batch_size=self.args.per_device_eval_batch_size,
            dataset=self.eval_dataset,
        )
