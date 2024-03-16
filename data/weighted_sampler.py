from collections import Counter
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class WeightedSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size: int = 2,
    ) -> None:
        self.dataset = dataset

        self.main_obj = self.dataset["MAINOBJECT"]
        self.obj_idx_table = {obj_name: idx for idx, obj_name in enumerate(Counter(self.main_obj).keys())}

        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.num_replicas = dist.get_world_size()

    def __iter__(self) -> Iterator:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx_obj_ls = torch.tensor([self.obj_idx_table[x] for x in self.main_obj], dtype=torch.float)

        batch_ls = list()
        while len(idx_obj_ls):
            try:
                current_sampling_size = self.batch_size if self.batch_size < len(idx_obj_ls) else len(idx_obj_ls)
                selected_idx = torch.multinomial(
                    input=idx_obj_ls,
                    num_samples=current_sampling_size,
                    generator=g,
                )
                idx_obj_ls[selected_idx] = -100
                idx_obj_ls = idx_obj_ls[idx_obj_ls != -100]

                batch_ls.append(selected_idx.tolist())
            except:
                break

        batch_ls = [x for x in batch_ls if self.batch_size == len(x)]
        torch_batch_ls = torch.tensor(batch_ls)

        rank_considered_batch_ls = [
            torch_batch_ls[rank : len(batch_ls) : self.num_replicas].tolist() for rank in range(self.num_replicas)
        ]
        min_batch_len = min([len(x) for x in rank_considered_batch_ls])

        rank_considered_batch_ls = [x[:min_batch_len] for x in rank_considered_batch_ls]
        rank_considered_batch_ls = sum(rank_considered_batch_ls, [])

        return iter(sum(rank_considered_batch_ls, []))
        # for y in x:
        #     yield y

    def __len__(self) -> int:
        return len(list(self.__iter__()))
