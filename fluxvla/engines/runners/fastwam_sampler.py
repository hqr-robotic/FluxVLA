# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Iterator, List, Sized

import torch
from torch.utils.data import Dataset, Sampler


class GlobalIndexDatasetView(Dataset):
    """Expose a global-index view of a wrapped iterable dataset.

    ``DistributedRepeatingDataset`` owns the concatenation and normalization
    logic needed by the training recipes, but its iterator performs its own
    rank/worker sharding. FastWAM instead samples global indices from a
    map-style dataset before Accelerate shards whole batches. This adapter
    keeps the former's item construction while bypassing its iterator.

    Args:
        dataset: Dataset with ``__len__`` and either map-style ``__getitem__``
            or ``_get_item_from_global_idx`` support.
    """

    def __init__(self, dataset: Sized) -> None:
        if not hasattr(dataset, '__len__'):
            raise TypeError('FastWAM sampling requires a finite dataset.')
        if not (hasattr(dataset, '_get_item_from_global_idx')
                or hasattr(dataset, '__getitem__')):
            raise TypeError(
                'FastWAM sampling requires global-index item access.')
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the global dataset length."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        """Return the item at a global dataset index.

        Args:
            index: Global sample index.

        Returns:
            The wrapped dataset item.
        """
        get_global_item = getattr(self.dataset, '_get_item_from_global_idx',
                                  None)
        if get_global_item is not None:
            return get_global_item(index)
        return self.dataset[index]


class FastWAMBatchSampler(Sampler[List[int]]):
    """Reproduce FastWAM and Accelerate's per-rank batch ordering.

    The reference pipeline first applies ``torch.randperm`` using
    ``seed + epoch + epoch_offset``. PyTorch's ``BatchSampler`` then creates
    per-device batches, and Accelerate 1.12 ``BatchSamplerShard`` assigns
    whole batches round-robin across ranks with ``even_batches=True``. Any
    incomplete final rank group is padded from the first global batches.

    Args:
        dataset: Finite global dataset.
        seed: Base shuffle seed.
        batch_size: Per-device micro-batch size.
        num_processes: Distributed world size.
        process_index: Global rank in ``[0, num_processes)``.
    """

    def __init__(self, dataset: Sized, seed: int, batch_size: int,
                 num_processes: int, process_index: int) -> None:
        if batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}')
        if num_processes <= 0:
            raise ValueError(
                f'num_processes must be positive, got {num_processes}')
        if not 0 <= process_index < num_processes:
            raise ValueError('process_index must satisfy '
                             f'0 <= rank < {num_processes}, got '
                             f'{process_index}')

        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.process_index = int(process_index)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the reference shuffle epoch.

        Args:
            epoch: Zero-based epoch index.
        """
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int) -> None:
        """Set the epoch base used by FastWAM resume.

        Args:
            epoch_offset: Restored epoch offset.
        """
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int) -> None:
        """Skip completed global micro-batches on the first epoch.

        Args:
            batch_in_epoch: Number of completed per-rank micro-batches.
        """
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self) -> None:
        """Clear a previously configured resume offset."""
        self.resume_batch_offset = 0

    def __len__(self) -> int:
        """Return Accelerate's even per-rank batch count."""
        dataset_size = len(self.dataset)
        global_batches = (dataset_size + self.batch_size -
                          1) // self.batch_size
        return ((global_batches + self.num_processes - 1) //
                self.num_processes)

    def _build_global_batches(self) -> List[List[int]]:
        """Build the FastWAM sampler and PyTorch BatchSampler output."""
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(
            len(self.dataset), generator=generator).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = (
                self.resume_batch_offset * self.batch_size *
                self.num_processes)
            indices = indices[sample_offset:]
        return [
            indices[start:start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]

    def __iter__(self) -> Iterator[List[int]]:
        """Yield this rank's exact Accelerate-style index batches."""
        global_batches = self._build_global_batches()
        if not global_batches:
            return

        initial_data: List[int] = []
        batch_to_yield: List[int] = []
        batch: List[int] = []
        batch_index = -1

        for batch_index, batch in enumerate(global_batches):
            if batch_index < self.num_processes:
                initial_data.extend(batch)
            if batch_index % self.num_processes == self.process_index:
                batch_to_yield = list(batch)
            if (batch_index % self.num_processes == self.num_processes - 1
                    and len(batch) == self.batch_size):
                yield batch_to_yield
                batch_to_yield = []

        if len(batch_to_yield) == self.batch_size:
            yield batch_to_yield

        while len(initial_data) < self.num_processes * self.batch_size:
            initial_data.extend(initial_data)

        if len(batch) == self.batch_size:
            batch = []
            batch_index += 1

        cycle_index = 0
        while batch_index % self.num_processes != 0 or batch:
            end_index = cycle_index + self.batch_size - len(batch)
            batch.extend(initial_data[cycle_index:end_index])
            if batch_index % self.num_processes == self.process_index:
                yield list(batch)
            cycle_index = end_index
            batch = []
            batch_index += 1
