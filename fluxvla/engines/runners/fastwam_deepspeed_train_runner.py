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

import copy
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Sized, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from packaging.version import Version
from safetensors.torch import save_file
from torch.optim.lr_scheduler import (ConstantLR, CosineAnnealingLR, LinearLR,
                                      SequentialLR)
from torch.utils.data import DataLoader, Dataset, Sampler

from ..utils import initialize_overwatch
from ..utils.root import RUNNERS

overwatch = initialize_overwatch(__name__)


class _GlobalIndexDatasetView(Dataset):
    """Expose Flux's wrapped iterable dataset through global indices.

    Args:
        dataset: Finite dataset with global-index item access.
    """

    def __init__(self, dataset: Sized) -> None:
        if not hasattr(dataset, '__len__'):
            raise TypeError('FastWAM requires a finite training dataset.')
        if not (hasattr(dataset, '_get_item_from_global_idx')
                or hasattr(dataset, '__getitem__')):
            raise TypeError('FastWAM requires `_get_item_from_global_idx` or '
                            '`__getitem__`.')
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the global dataset length."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        """Read a sample by global index.

        Args:
            index: Global sample index.

        Returns:
            Dataset sample.
        """
        get_global_item = getattr(self.dataset, '_get_item_from_global_idx',
                                  None)
        if callable(get_global_item):
            return get_global_item(index)
        return self.dataset[index]


class _ResumableEpochSampler(Sampler[int]):
    """Exact copy of FastWAM's global pre-Accelerate sampler.

    Accelerate wraps the resulting PyTorch ``BatchSampler`` and assigns whole
    batches to ranks. Pre-sharding here would change both tail padding and
    rank-local order.

    Args:
        dataset: Finite map-style dataset.
        seed: Global shuffle seed.
        batch_size: Per-device micro-batch size.
        num_processes: Distributed process count.
    """

    def __init__(self, dataset: Sized, seed: int, batch_size: int,
                 num_processes: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the shuffle epoch.

        Args:
            epoch: Zero-based epoch index.
        """
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int) -> None:
        """Set the restored epoch offset.

        Args:
            epoch_offset: Epoch offset restored from a checkpoint.
        """
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int) -> None:
        """Skip already consumed global batches after resume.

        Args:
            batch_in_epoch: Completed per-rank micro-batches.
        """
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self) -> None:
        """Clear the restored batch offset."""
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        """Yield the source ``torch.randperm`` index stream."""
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(
            len(self.dataset), generator=generator).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = (
                self.resume_batch_offset * self.batch_size *
                self.num_processes)
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        """Return the unsharded dataset length."""
        return len(self.dataset)


class _MetricProxy:
    """Expose the small metric interface expected by Flux utilities."""

    def __init__(self, run_dir: str, log_fn: Callable[[int, Dict[str, float]],
                                                      None]) -> None:
        self.run_dir = run_dir
        self.global_step = 0
        self.epoch = 0
        self._log_fn = log_fn

    def log(self, global_step: int, metrics: Dict[str, float]) -> None:
        """Write metrics through the runner's source-style sink.

        Args:
            global_step: Optimizer step.
            metrics: Scalar metrics.
        """
        self._log_fn(global_step, metrics)


@RUNNERS.register_module()
class FastWAMDeepSpeedTrainRunner:
    """Run Flux FastWAM with its pinned source Accelerate/ZeRO-1 recipe.

    This runner intentionally does not inherit :class:`BaseTrainRunner`.
    FastWAM's released trajectory depends on several coupled runtime details
    that the shared DDP/FSDP loop does not implement:

    * a global ``torch.randperm`` sampler is batch-sharded by Accelerate;
    * DeepSpeed owns optimizer stepping and scheduler stepping;
    * the source calls a method delegated to the wrapped module instead of
      ``DeepSpeedEngine.forward``;
    * the final incomplete accumulation window steps at end-of-dataloader;
    * logged loss is the last micro-batch, averaged only across ranks.

    Args:
        cfg: Resolved Flux experiment config.
        args: Parsed training command-line arguments.
        optimizer: AdamW configuration.
        collator: Flux collator configuration.
        max_epochs: Number of source epochs.
        max_steps: Optional explicit optimizer-step limit.
        grad_accumulation_steps: DeepSpeed accumulation size.
        mixed_precision_dtype: Must be ``'bf16'`` for released parity.
        seed: Global source seed.
    """

    _PINNED_ACCELERATE = Version('1.12.0')
    _PINNED_DEEPSPEED = Version('0.18.5')
    _PINNED_TORCH = Version('2.6.0')
    _PINNED_CUDA = '12.4'

    def __init__(
        self,
        cfg: Dict,
        args: Any,
        max_grad_norm: float = 1.0,
        collator: Optional[Dict] = None,
        sampler: Optional[str] = None,
        metric: Optional[Dict] = None,
        optimizer: Optional[Dict] = None,
        max_epochs: Optional[int] = 10,
        max_steps: Optional[int] = None,
        save_epoch_interval: int = 1,
        save_iter_interval: int = 2000,
        max_keep_ckpts: int = 2,
        lr_scheduler: Optional[Dict] = None,
        enable_gradient_checkpointing: bool = False,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = True,
        mixed_precision_dtype: str = 'bf16',
        grad_accumulation_steps: int = 1,
        evaluator: Optional[Dict] = None,
        tokenizer: Optional[Dict] = None,
        resume_from: Optional[str] = None,
        seed: Optional[int] = None,
        log_every: int = 10,
        save_every: Optional[int] = None,
        eval_every: Optional[int] = None,
        eval_num_inference_steps: Optional[int] = None,
        strict_runtime_versions: bool = True,
        save_at_end: bool = True,
        **kwargs: Any,
    ) -> None:
        del save_epoch_interval, max_keep_ckpts, reduce_in_full_precision
        del tokenizer
        if kwargs:
            fields = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected runner config field(s): {fields}')
        if sampler not in (None, 'source'):
            raise ValueError(
                'FastWAMDeepSpeedTrainRunner owns source sampling; '
                f'got sampler={sampler!r}.')
        if not enable_mixed_precision_training:
            raise ValueError('Released FastWAM parity requires BF16 training.')
        if str(mixed_precision_dtype).lower() != 'bf16':
            raise ValueError('Released FastWAM parity requires `bf16`.')
        if enable_gradient_checkpointing:
            raise ValueError(
                'Runner-level checkpoint wrapping is not source-exact.')
        if max_epochs is None and max_steps is None:
            raise ValueError('`max_epochs` or `max_steps` must be provided.')

        from ..utils.builder import build_collator_from_cfg, build_vla_from_cfg

        self.cfg = cfg
        self.args = args
        self.model_build_device = self._resolve_source_train_device()
        model_cfg = copy.deepcopy(cfg.model)
        model_cfg['device'] = self.model_build_device
        self.vla = build_vla_from_cfg(model_cfg)
        self.collator = build_collator_from_cfg(collator)
        self.optimizer_cfg = self._validate_optimizer_cfg(optimizer)
        self.scheduler_cfg = dict(lr_scheduler or {})
        self.max_grad_norm = float(max_grad_norm)
        self.max_epochs = int(max_epochs or 1)
        self.requested_max_steps = (None
                                    if max_steps is None else int(max_steps))
        self.grad_accumulation_steps = int(grad_accumulation_steps)
        if self.grad_accumulation_steps < 1:
            raise ValueError('grad_accumulation_steps must be positive.')
        self.per_device_batch_size = int(
            cfg.train_dataloader.per_device_batch_size)
        self.per_device_num_workers = int(
            getattr(cfg.train_dataloader, 'per_device_num_workers', 0))
        self.seed = self._resolve_seed(seed)
        self.log_every = int(log_every)
        self.save_every = int(
            save_iter_interval if save_every is None else save_every)

        evaluator_cfg = dict(evaluator or {})
        evaluator_cfg.pop('type', None)
        inferred_eval_every = evaluator_cfg.pop('eval_every', 0)
        self.eval_every = int(
            inferred_eval_every if eval_every is None else eval_every)
        inferred_inference_steps = evaluator_cfg.pop('num_inference_steps', 10)
        self.eval_num_inference_steps = int(
            inferred_inference_steps
            if eval_num_inference_steps is None else eval_num_inference_steps)
        self.eval_seed = int(evaluator_cfg.pop('seed', 42))
        self.save_eval_video = bool(evaluator_cfg.pop('save_video', True))
        self.eval_video_fps = int(evaluator_cfg.pop('video_fps', 8))
        if evaluator_cfg:
            fields = ', '.join(sorted(evaluator_cfg))
            raise TypeError(f'Unsupported evaluator field(s): {fields}')

        metric_cfg = dict(metric or {})
        metric_cfg.pop('type', None)
        self.active_trackers = tuple(
            metric_cfg.pop('active_trackers', ('jsonl', )))
        metric_cfg.pop('window_size', None)
        configured_run_dir = metric_cfg.pop('run_dir', None)
        if metric_cfg:
            fields = ', '.join(sorted(metric_cfg))
            raise TypeError(f'Unsupported metric field(s): {fields}')
        args_work_dir = getattr(args, 'work_dir', None)
        self.output_dir = str(args_work_dir or configured_run_dir
                              or 'work_dirs')
        self.metric = _MetricProxy(self.output_dir, self._log_metrics)

        self.strict_runtime_versions = bool(strict_runtime_versions)
        self.save_at_end = bool(save_at_end)
        self.resume_from = resume_from
        self.accelerator: Optional[Accelerator] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_loader = None
        self.train_sampler: Optional[_ResumableEpochSampler] = None
        self.max_steps: Optional[int] = None
        self.epoch = 0
        self.batch_in_epoch = 0
        self._n_train_examples: Optional[int] = None
        self._wandb_run = None
        self._last_weights_path: Optional[str] = None
        self._last_eval_weights_path: Optional[str] = None
        self.last_train_metrics: Dict[str, float] = {}
        self.step_history: list[Dict[str, float]] = []
        self.optimizer_audit: Dict[str, Any] = {}

        self.checkpoint_root = os.path.join(self.output_dir, 'checkpoints')
        self.weights_dir = os.path.join(self.checkpoint_root, 'weights')
        self.state_dir = os.path.join(self.checkpoint_root, 'state')
        self.eval_dir = os.path.join(self.output_dir, 'eval')
        for path in (self.output_dir, self.checkpoint_root, self.weights_dir,
                     self.state_dir, self.eval_dir):
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def _validate_optimizer_cfg(optimizer: Optional[Dict]) -> Dict:
        """Validate the released AdamW recipe.

        Args:
            optimizer: Flux optimizer configuration.

        Returns:
            Normalized optimizer configuration.
        """
        if optimizer is None:
            raise ValueError('runner.optimizer must be provided.')
        result = dict(optimizer)
        optimizer_type = result.pop('type', 'AdamW')
        if optimizer_type != 'AdamW':
            raise ValueError('FastWAM source parity requires AdamW.')
        if 'lr' not in result:
            raise ValueError('optimizer.lr must be provided.')
        betas = tuple(result.pop('betas', (0.9, 0.95)))
        if betas != (0.9, 0.95):
            raise ValueError('FastWAM source betas are exactly (0.9, 0.95).')
        allowed = {'lr', 'weight_decay', 'eps'}
        unexpected = sorted(set(result) - allowed)
        if unexpected:
            raise TypeError('Unsupported AdamW field(s): '
                            f'{", ".join(unexpected)}')
        return {
            'lr': float(result['lr']),
            'weight_decay': float(result.get('weight_decay', 0.0)),
            'eps': float(result.get('eps', 1e-8)),
            'betas': betas,
        }

    @staticmethod
    def _resolve_source_train_device() -> str:
        """Resolve the pre-Accelerator build device exactly as FastWAM.

        Returns:
            ``cpu`` without CUDA, otherwise the valid local-rank CUDA device.
        """
        if not torch.cuda.is_available():
            return 'cpu'
        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return 'cuda:0'
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if local_rank < 0 or local_rank >= device_count:
            return 'cuda:0'
        return f'cuda:{local_rank}'

    def _resolve_seed(self, seed: Optional[int]) -> int:
        """Resolve the source seed from runner or top-level config."""
        if seed is not None:
            return int(seed)
        cfg_seed = getattr(self.cfg, 'seed', None)
        if cfg_seed is not None:
            return int(cfg_seed)
        raise ValueError('FastWAM source parity requires an explicit seed.')

    @classmethod
    def _validate_runtime_versions(cls, strict: bool = True) -> Dict[str, str]:
        """Validate the pinned parity runtime.

        Args:
            strict: Raise when any version differs from the released stack.

        Returns:
            Detected runtime versions.
        """
        try:
            import accelerate
            import deepspeed
        except ImportError as exc:
            raise RuntimeError(
                'FastWAM ZeRO-1 requires accelerate==1.12.0 and '
                'deepspeed==0.18.5.') from exc

        versions = {
            'accelerate': accelerate.__version__,
            'deepspeed': deepspeed.__version__,
            'torch': torch.__version__,
            'cuda': str(torch.version.cuda),
        }
        actual = {
            'accelerate': Version(versions['accelerate'].split('+')[0]),
            'deepspeed': Version(versions['deepspeed'].split('+')[0]),
            'torch': Version(versions['torch'].split('+')[0]),
        }
        expected = {
            'accelerate': cls._PINNED_ACCELERATE,
            'deepspeed': cls._PINNED_DEEPSPEED,
            'torch': cls._PINNED_TORCH,
        }
        mismatches = [
            f'{name}={versions[name]} (expected {expected[name]})'
            for name in expected if actual[name] != expected[name]
        ]
        if versions['cuda'] != cls._PINNED_CUDA:
            mismatches.append(
                f'cuda={versions["cuda"]} (expected {cls._PINNED_CUDA})')
        if strict and mismatches:
            raise RuntimeError('FastWAM parity runtime mismatch: '
                               f'{"; ".join(mismatches)}')
        return versions

    @staticmethod
    def _rank_seed(seed: int, rank: int) -> Callable[[int], None]:
        """Apply FastWAM's process-specific RNG seed.

        Args:
            seed: Global experiment seed.
            rank: Global distributed rank.

        Returns:
            Source-compatible DataLoader worker initializer.
        """
        process_seed = int(seed) + int(rank)
        os.environ['EXPERIMENT_GLOBAL_SEED'] = str(seed)
        random.seed(process_seed)
        np.random.seed(process_seed)
        torch.manual_seed(process_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(process_seed)
        return _source_worker_init_fn

    def _assert_deepspeed_config(self) -> None:
        """Assert the released ZeRO-1 launcher configuration."""
        if self.accelerator is None:
            raise RuntimeError('Accelerator is not initialized.')
        if str(self.accelerator.distributed_type) != \
                'DistributedType.DEEPSPEED':
            raise RuntimeError(
                'Launch this runner with FastWAM accelerate_zero1_ds.yaml.')
        plugin = self.accelerator.state.deepspeed_plugin
        config = plugin.deepspeed_config
        zero = config.get('zero_optimization', {})
        required = {
            'stage': 1,
            'overlap_comm': False,
            'contiguous_gradients': False,
            'reduce_bucket_size': 200000000.0,
            'allgather_bucket_size': 200000000.0,
        }
        mismatches = {
            key: (zero.get(key), value)
            for key, value in required.items() if zero.get(key) != value
        }
        offload_device = zero.get('offload_optimizer', {}).get('device')
        if str(offload_device).lower() != 'none':
            mismatches['offload_optimizer.device'] = (offload_device, 'none')
        param_offload_device = zero.get('offload_param', {}).get('device')
        if str(param_offload_device).lower() != 'none':
            mismatches['offload_param.device'] = (param_offload_device, 'none')
        if mismatches:
            raise RuntimeError('DeepSpeed config is not source-exact: '
                               f'{mismatches}')

    def _apply_source_trainable_policy(self) -> list[torch.nn.Parameter]:
        """Freeze everything except configured FastWAM training modules."""
        model = self.vla
        model.eval()
        model.requires_grad_(False)

        head = getattr(model, 'vla_head', None)
        dit = getattr(head, 'mot', None) if head is not None else None
        if dit is None:
            dit = getattr(model, 'dit', None)
        if dit is None:
            raise AttributeError(
                'FastWAM model must expose `vla_head.mot` or `dit`.')
        dit.train()
        dit.requires_grad_(True)
        trainable_params = list(dit.parameters())

        proprio = (
            getattr(head, 'proprio_encoder', None)
            if head is not None else getattr(model, 'proprio_encoder', None))
        if proprio is not None:
            proprio.train()
            proprio.requires_grad_(True)
            trainable_params.extend(list(proprio.parameters()))
        track_decoder = (
            getattr(head, 'track_decoder', None) if head is not None else None)
        if track_decoder is not None:
            track_decoder.train()
            track_decoder.requires_grad_(True)
            trainable_params.extend(list(track_decoder.parameters()))
        trainable_ids = {id(parameter) for parameter in trainable_params}
        actual_ids = {
            id(parameter)
            for parameter in model.parameters() if parameter.requires_grad
        }
        if trainable_ids != actual_ids:
            raise RuntimeError(
                'FastWAM trainable-policy mismatch: '
                f'listed={len(trainable_ids)}, actual={len(actual_ids)}')
        return trainable_params

    def _build_scheduler(self, total_steps: int):
        """Build FastWAM's PyTorch warmup and cosine scheduler."""
        if self.optimizer is None:
            raise RuntimeError('Optimizer must exist before the scheduler.')
        scheduler_type = str(self.scheduler_cfg.get('type', 'cosine')).lower()
        if scheduler_type == 'linear-warmup+cosine-decay-min-lr':
            scheduler_type = 'cosine'
        warmup_ratio = float(self.scheduler_cfg.get('warmup_ratio', 0.05))
        min_lr_ratio = float(self.scheduler_cfg.get('min_lr_ratio', 0.01))
        warmup_steps = int(total_steps * warmup_ratio)
        warmup_steps = min(max(warmup_steps, 0), total_steps - 1)
        remaining_steps = max(total_steps - warmup_steps, 1)

        if scheduler_type == 'cosine':
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.optimizer_cfg['lr'] * min_lr_ratio,
            )
        elif scheduler_type == 'constant':
            main_scheduler = ConstantLR(
                self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f'Unsupported FastWAM scheduler: {scheduler_type}.')
        if warmup_steps <= 0:
            return main_scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def run_setup(self, n_train_examples: int) -> None:
        """Build the source model/optimizer/scheduler before data prepare.

        Args:
            n_train_examples: Global dataset size.
        """
        versions = self._validate_runtime_versions(
            strict=self.strict_runtime_versions)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.grad_accumulation_steps,
            mixed_precision='bf16',
            step_scheduler_with_optimizer=False,
        )
        self._assert_deepspeed_config()

        self.vla.freeze_backbones()
        self.vla.from_pretrained()
        self.vla.to(
            device=self.accelerator.device,
            dtype=torch.bfloat16,
        )
        trainable_params = self._apply_source_trainable_policy()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.optimizer_cfg['lr'],
            weight_decay=self.optimizer_cfg['weight_decay'],
            betas=self.optimizer_cfg['betas'],
            eps=self.optimizer_cfg['eps'],
        )
        optimizer_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group['params']
        }
        trainable_ids = {
            id(parameter)
            for parameter in self.vla.parameters() if parameter.requires_grad
        }
        if optimizer_ids != trainable_ids:
            raise RuntimeError(
                'Optimizer parameters do not exactly match trainable model '
                f'parameters: optimizer={len(optimizer_ids)}, '
                f'trainable={len(trainable_ids)}')

        self._n_train_examples = int(n_train_examples)
        if self.requested_max_steps is not None:
            self.max_steps = max(self.requested_max_steps, 1)
        else:
            global_micro_batch = (
                self.per_device_batch_size * self.accelerator.num_processes)
            micro_steps = max(
                math.ceil(self._n_train_examples / global_micro_batch), 1)
            optimizer_steps = max(
                math.ceil(micro_steps / self.grad_accumulation_steps), 1)
            self.max_steps = max(optimizer_steps * self.max_epochs, 1)
        self.lr_scheduler = self._build_scheduler(self.max_steps)

        if self.accelerator.is_main_process:
            overwatch.info('FastWAM source runner initialized: '
                           f'versions={versions}, world_size='
                           f'{self.accelerator.num_processes}, batch='
                           f'{self.per_device_batch_size}, gas='
                           f'{self.grad_accumulation_steps}, max_steps='
                           f'{self.max_steps}.')

    def _build_train_loader(self, dataset: Sized) -> DataLoader:
        """Build the unsharded loader that Accelerate will batch-shard."""
        if self.accelerator is None:
            raise RuntimeError('run_setup must be called before run.')
        dataset_view = _GlobalIndexDatasetView(dataset)
        self.train_sampler = _ResumableEpochSampler(
            dataset=dataset_view,
            seed=self.seed,
            batch_size=self.per_device_batch_size,
            num_processes=self.accelerator.num_processes,
        )
        worker_init_fn = self._rank_seed(self.seed,
                                         self.accelerator.process_index)
        return DataLoader(
            dataset_view,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.per_device_num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            collate_fn=self.collator,
        )

    def _assert_dataset_length_consistent(self, dataset: Sized) -> None:
        """Fail before collectives diverge on rank-local dataset lengths."""
        if self.accelerator is None:
            raise RuntimeError('Accelerator is not initialized.')
        local_length = len(dataset)
        lengths = self.accelerator.gather(
            torch.tensor(
                [local_length],
                device=self.accelerator.device,
                dtype=torch.int64,
            )).reshape(-1)
        if not torch.all(lengths == lengths[0]):
            raise RuntimeError('Training dataset length differs across ranks: '
                               f'{lengths.cpu().tolist()}')

    def _audit_zero1_optimizer(self, require_moments: bool) -> Dict[str, Any]:
        """Verify ZeRO-1 FP32 masters and AdamW moments."""
        if self.accelerator is None:
            raise RuntimeError('Accelerator is not initialized.')
        if bool(self.accelerator.native_amp):
            raise RuntimeError(
                'FastWAM DeepSpeed parity must not use torch autocast.')
        engine = self.vla
        if float(engine.gradient_clipping()) != 0.0:
            raise RuntimeError(
                'Source DeepSpeed has gradient_clipping=0; trainer-side '
                'clip_grad_norm_ only reads the cached norm after step.')
        if getattr(engine, 'lr_scheduler', None) is not None:
            raise RuntimeError(
                'Source scheduler must remain outside DeepSpeedEngine.')
        expected_global_batch = (
            self.per_device_batch_size * self.grad_accumulation_steps *
            self.accelerator.num_processes)
        batch_checks = {
            'train_micro_batch_size_per_gpu': (
                engine.train_micro_batch_size_per_gpu(),
                self.per_device_batch_size,
            ),
            'gradient_accumulation_steps': (
                engine.gradient_accumulation_steps(),
                self.grad_accumulation_steps,
            ),
            'train_batch_size': (
                engine.train_batch_size(),
                expected_global_batch,
            ),
        }
        batch_mismatches = {
            key: values
            for key, values in batch_checks.items() if values[0] != values[1]
        }
        if batch_mismatches:
            raise RuntimeError('DeepSpeed batch configuration mismatch: '
                               f'{batch_mismatches}')
        engine_optimizer = getattr(self.vla, 'optimizer', None)
        if engine_optimizer is None:
            raise RuntimeError('Prepared DeepSpeed engine has no optimizer.')
        zero_stage = getattr(engine_optimizer, 'zero_stage_string', None)
        fp32_groups = getattr(engine_optimizer,
                              'single_partition_of_fp32_groups', None)
        bit16_groups = getattr(engine_optimizer, 'bit16_groups_flat', None)
        if zero_stage != 'ZeRO-1' or not fp32_groups or not bit16_groups:
            raise RuntimeError('Prepared optimizer is not DeepSpeed ZeRO-1.')
        if any(group.dtype != torch.float32 for group in fp32_groups):
            raise RuntimeError('ZeRO-1 master weights are not FP32.')
        if any(group.dtype != torch.bfloat16 for group in bit16_groups):
            raise RuntimeError('ZeRO-1 model partitions are not BF16.')
        optimizer_checks = {
            'partition_gradients': False,
            'cpu_offload': False,
            'reduce_scatter': True,
            'check_grad_overflow': False,
        }
        optimizer_mismatches = {
            key: (getattr(engine_optimizer, key, None), expected)
            for key, expected in optimizer_checks.items()
            if getattr(engine_optimizer, key, None) != expected
        }
        if optimizer_mismatches:
            raise RuntimeError('ZeRO-1 runtime mismatch: '
                               f'{optimizer_mismatches}')
        dtype_checks = {
            'dtype': torch.bfloat16,
            'gradient_accumulation_dtype': torch.bfloat16,
            'communication_data_type': torch.bfloat16,
            'master_weights_and_grads_dtype': torch.float32,
        }
        dtype_mismatches = {
            key: (getattr(engine_optimizer, key, None), expected)
            for key, expected in dtype_checks.items()
            if getattr(engine_optimizer, key, None) != expected
        }
        if dtype_mismatches:
            raise RuntimeError('ZeRO-1 dtype mismatch: '
                               f'{dtype_mismatches}')

        inner_optimizer = getattr(engine_optimizer, 'optimizer', None)
        states = list(getattr(inner_optimizer, 'state', {}).values())
        moment_dtypes = sorted({
            str(value.dtype)
            for state in states for key, value in state.items()
            if key in {'exp_avg', 'exp_avg_sq'}
            and isinstance(value, torch.Tensor)
        })
        if require_moments and moment_dtypes != ['torch.float32']:
            raise RuntimeError('AdamW moments are not exclusively FP32: '
                               f'{moment_dtypes}')
        audit = {
            'zero_stage': zero_stage,
            'master_dtype': str(fp32_groups[0].dtype),
            'model_partition_dtype': str(bit16_groups[0].dtype),
            'moment_dtypes': moment_dtypes,
            'master_numel': int(sum(item.numel() for item in fp32_groups)),
            'gradient_clipping': float(engine.gradient_clipping()),
            'native_amp': bool(self.accelerator.native_amp),
        }
        self.optimizer_audit = audit
        return audit

    def _prepare_training(self, dataset: Sized) -> None:
        """Prepare model, optimizer, loader and scheduler together."""
        if self.accelerator is None or self.optimizer is None \
                or self.lr_scheduler is None:
            raise RuntimeError('run_setup must complete before run.')
        if self._n_train_examples != len(dataset):
            raise ValueError('Dataset length changed between setup and run: '
                             f'{self._n_train_examples} != {len(dataset)}')
        self._assert_dataset_length_consistent(dataset)
        train_loader = self._build_train_loader(dataset)
        prepared = self.accelerator.prepare(self.vla, self.optimizer,
                                            train_loader, self.lr_scheduler)
        self.vla, self.optimizer, self.train_loader, self.lr_scheduler = \
            prepared
        self.optimizer.zero_grad(set_to_none=True)
        self._audit_zero1_optimizer(require_moments=False)
        self._init_wandb()
        self._resume_or_load_checkpoint()
        if self.metric.global_step > 0:
            self._audit_zero1_optimizer(require_moments=True)

    def _compute_training_loss(
            self, sample: Dict[str,
                               Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Call the raw module, deliberately bypassing engine ``forward``.

        This is a source behavior, not an optimization. DeepSpeed 0.18.5
        installs its GAS scaling output hook only in ``DeepSpeedEngine``
        ``forward``. FastWAM calls a delegated module method instead, so its
        gradients are not divided by GAS. A runner that calls
        ``self.vla(**sample)`` produces a different trajectory.
        """
        if self.accelerator is None:
            raise RuntimeError('Accelerator is not initialized.')
        raw_model = self.accelerator.unwrap_model(self.vla)
        output = raw_model(**sample)
        if not isinstance(output, dict) or 'loss' not in output:
            raise TypeError(
                'Flux FastWAM forward must return a dict containing `loss`.')
        loss = output['loss']
        metrics = {}
        for key, value in output.items():
            if key == 'loss':
                continue
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[key] = float(value.detach().float().item())
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
        return loss, metrics

    def _gather_step_metrics(
        self,
        loss: torch.Tensor,
        loss_metrics: Dict[str, float],
        grad_norm: Any,
    ) -> Dict[str, float]:
        """Gather source last-microbatch metrics across ranks."""
        if self.accelerator is None:
            raise RuntimeError('Accelerator is not initialized.')
        global_loss = float(
            self.accelerator.gather(
                loss.detach().float().reshape(1)).mean().item())
        result = {'loss': global_loss}
        for key, value in loss_metrics.items():
            metric_tensor = torch.tensor(
                float(value),
                device=loss.device,
                dtype=torch.float32,
            ).reshape(1)
            result[key] = float(
                self.accelerator.gather(metric_tensor).mean().item())
        grad_tensor = torch.as_tensor(
            grad_norm, device=loss.device, dtype=torch.float32).reshape(1)
        result['grad_norm'] = float(
            self.accelerator.gather(grad_tensor).mean().item())
        result['lr'] = float(self.optimizer.param_groups[0]['lr'])
        return result

    def _log_metrics(self, global_step: int, metrics: Dict[str,
                                                           float]) -> None:
        """Write exact scalar payloads to JSONL and optional W&B."""
        if self.accelerator is not None \
                and not self.accelerator.is_main_process:
            return
        payload = {'global_step': int(global_step), **metrics}
        if 'jsonl' in self.active_trackers:
            path = os.path.join(self.output_dir,
                                'fastwam-source-metrics.jsonl')
            with open(path, 'a', encoding='utf-8') as file_handle:
                file_handle.write(json.dumps(payload, sort_keys=True) + '\n')
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=int(global_step))

    def _init_wandb(self) -> None:
        """Initialize W&B only on the main process when requested."""
        if self.accelerator is None or not self.accelerator.is_main_process:
            return
        if 'wandb' not in self.active_trackers:
            return
        mode = os.environ.get('WANDB_MODE', 'online')
        if mode == 'disabled':
            return
        import wandb
        run_name = Path(self.output_dir).name
        self._wandb_run = wandb.init(
            entity=os.environ.get('WANDB_ENTITY'),
            project=os.environ.get('WANDB_PROJECT', 'fluxvla'),
            name=run_name,
            mode=mode,
            dir=self.output_dir,
            config=self.cfg,
        )

    def _estimate_eta(self, start_time: float,
                      start_step: int) -> Tuple[str, float]:
        """Return source ETA text and optimizer steps per second."""
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        done_steps = max(self.metric.global_step - start_step, 1)
        steps_per_second = done_steps / elapsed
        remaining = max(int(self.max_steps) - self.metric.global_step, 0)
        eta_seconds = int(remaining / max(steps_per_second, 1e-9))
        eta_hours, eta_remainder = divmod(eta_seconds, 3600)
        eta_minutes, eta_seconds = divmod(eta_remainder, 60)
        return (f'{eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}',
                steps_per_second)

    def _set_source_train_mode(self) -> None:
        """Restore source DiT-only train/eval flags after evaluation."""
        if self.accelerator is None:
            return
        raw_model = self.accelerator.unwrap_model(self.vla)
        raw_model.eval()
        head = getattr(raw_model, 'vla_head', None)
        dit = getattr(head, 'mot', None) if head is not None else None
        if dit is None:
            dit = getattr(raw_model, 'dit')
        dit.train()
        proprio = (
            getattr(head, 'proprio_encoder', None) if head is not None else
            getattr(raw_model, 'proprio_encoder', None))
        if proprio is not None:
            proprio.train()
        track_decoder = (
            getattr(head, 'track_decoder', None) if head is not None else None)
        if track_decoder is not None:
            track_decoder.train()

    def _get_eval_sample(self, dataset: Sized, index: int) -> Dict[str, Any]:
        """Collate one deterministic global-index evaluation sample."""
        view = _GlobalIndexDatasetView(dataset)
        return self.collator([view[index]])

    @torch.no_grad()
    def evaluate(self, dataset: Sized) -> Optional[Dict[str, float]]:
        """Run the Flux adapter of FastWAM's one-sample-per-rank eval."""
        if self.accelerator is None:
            raise RuntimeError('Accelerator is not initialized.')
        raw_model = self.accelerator.unwrap_model(self.vla)
        eval_fn = getattr(raw_model, 'compute_training_eval', None)
        if not callable(eval_fn):
            if self.accelerator.is_main_process:
                overwatch.warning(
                    'Model has no `compute_training_eval`; skipping eval.')
            return None
        was_training = self.vla.training
        self.vla.eval()
        generator = torch.Generator(
            device='cpu').manual_seed(self.metric.global_step +
                                      self.accelerator.process_index)
        index = int(
            torch.randint(0, len(dataset), (1, ), generator=generator).item())
        batch = self._get_eval_sample(dataset, index)
        from accelerate.utils import send_to_device
        batch = send_to_device(
            batch, self.accelerator.device, non_blocking=True)
        with self.accelerator.autocast():
            output = eval_fn(
                batch,
                num_inference_steps=self.eval_num_inference_steps,
                seed=self.eval_seed,
            )
        output = output or {}
        metrics = output.get('metrics') or {}
        keys = sorted(
            key for key, value in metrics.items()
            if isinstance(value, (int, float))
            or isinstance(value, torch.Tensor) and value.numel() == 1)
        if not keys:
            return None
        local_values = torch.tensor(
            [float(metrics[key]) for key in keys],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered = self.accelerator.gather_for_metrics(local_values)
        mean_values = gathered.mean(dim=0).cpu().tolist()
        result = {key: float(value) for key, value in zip(keys, mean_values)}

        frames = output.get('video_frames')
        if self.save_eval_video and frames:
            from ..utils.video_metrics import save_video_frames
            video_path = os.path.join(
                self.eval_dir,
                f'step_{self.metric.global_step:06d}_rank_'
                f'{self.accelerator.process_index:03d}.mp4',
            )
            save_video_frames(frames, video_path, fps=self.eval_video_fps)
        if was_training:
            self._set_source_train_mode()
        return result

    def _source_weight_payload(self) -> Dict[str, Any]:
        """Build the released trainable-only FastWAM weight payload."""
        if self.accelerator is None:
            raise RuntimeError('Accelerator is not initialized.')
        model = self.accelerator.unwrap_model(self.vla)
        head = getattr(model, 'vla_head', None)
        dit = getattr(head, 'mot', None) if head is not None else None
        if dit is None:
            dit = getattr(model, 'dit', None)
        if dit is None:
            raise AttributeError('Model does not expose trainable DiT.')
        payload = {
            'mot': dit.state_dict(),
            'step': self.metric.global_step,
            'torch_dtype': str(getattr(model, 'torch_dtype', torch.bfloat16)),
        }
        proprio = (
            getattr(head, 'proprio_encoder', None)
            if head is not None else getattr(model, 'proprio_encoder', None))
        if proprio is not None:
            payload['proprio_encoder'] = proprio.state_dict()
        track_decoder = (
            getattr(head, 'track_decoder', None) if head is not None else None)
        if track_decoder is not None:
            payload['track_decoder'] = track_decoder.state_dict()
        return payload

    @staticmethod
    def _full_eval_state_dict(
            model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Materialize a complete CPU state dict for standalone eval.

        Args:
            model: Unwrapped replicated ZeRO-1 module.

        Returns:
            Flat, contiguous CPU tensors accepted by ``safetensors``.
        """
        return {
            key: value.detach().to(device='cpu', copy=True).contiguous()
            for key, value in model.state_dict().items()
            if isinstance(value, torch.Tensor)
        }

    def _copy_eval_dataset_statistics(self) -> None:
        """Copy statistics to the path inferred from ``weights/*.pt``."""
        source = Path(self.output_dir) / 'dataset_statistics.json'
        if not source.is_file():
            return
        destination = Path(self.checkpoint_root) / source.name
        shutil.copy2(source, destination)

    def save_final_eval_checkpoint(self) -> Optional[str]:
        """Export the final complete VLA as a sibling safetensors file.

        Periodic ``.pt`` payloads remain source-exact and trainable-only.
        This one final full checkpoint lets ``scripts/train.py`` and
        ``LiberoEvalRunner`` evaluate without reconstructing a base-plus-
        overlay checkpoint.

        Returns:
            Final safetensors path on the main rank, otherwise ``None``.
        """
        if self.accelerator is None:
            raise RuntimeError('Accelerator is not initialized.')
        self.accelerator.wait_for_everyone()
        eval_weights_path = None
        if self.accelerator.is_main_process:
            step_tag = f'step_{self.metric.global_step:06d}'
            eval_weights_path = os.path.join(self.weights_dir,
                                             f'{step_tag}.safetensors')
            raw_model = self.accelerator.unwrap_model(self.vla)
            state_dict = self._full_eval_state_dict(raw_model)
            save_file(state_dict, eval_weights_path)
            del state_dict
            self._copy_eval_dataset_statistics()
            self._last_eval_weights_path = eval_weights_path
        self.accelerator.wait_for_everyone()
        return eval_weights_path

    def _save_trainer_state(self, state_path: str) -> None:
        """Save source resume offsets beside the Accelerate state."""
        payload = {
            'global_step': int(self.metric.global_step),
            'epoch': int(self.epoch),
            'batch_in_epoch': int(self.batch_in_epoch),
        }
        state_file = os.path.join(state_path, 'trainer_state.json')
        with open(state_file, 'w', encoding='utf-8') as file_handle:
            json.dump(payload, file_handle, indent=2, sort_keys=True)
            file_handle.write('\n')

    def save_checkpoint(self) -> Dict[str, Optional[str]]:
        """Save source trainable weights plus full Accelerate/ZeRO state."""
        if self.accelerator is None:
            raise RuntimeError('Accelerator is not initialized.')
        step_tag = f'step_{self.metric.global_step:06d}'
        self.accelerator.wait_for_everyone()
        weights_path = None
        if self.accelerator.is_main_process:
            weights_path = os.path.join(self.weights_dir, f'{step_tag}.pt')
            torch.save(self._source_weight_payload(), weights_path)
            self._last_weights_path = weights_path
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        os.makedirs(state_path, exist_ok=True)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()
        return {'weights_path': weights_path, 'state_path': state_path}

    def _load_weight_payload(self, path: str) -> None:
        """Load a released trainable-only FastWAM checkpoint."""
        if self.accelerator is None:
            raise RuntimeError('Accelerator is not initialized.')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        model = self.accelerator.unwrap_model(self.vla)
        head = getattr(model, 'vla_head', None)
        dit = getattr(head, 'mot', None) if head is not None else None
        if dit is None:
            dit = getattr(model, 'dit', None)
        if 'mot' not in payload:
            raise ValueError(f'Checkpoint has no `mot` payload: {path}')
        dit.load_state_dict(payload['mot'], strict=True)
        proprio = (
            getattr(head, 'proprio_encoder', None)
            if head is not None else getattr(model, 'proprio_encoder', None))
        if proprio is not None and 'proprio_encoder' in payload:
            proprio.load_state_dict(payload['proprio_encoder'], strict=True)
        track_decoder = (
            getattr(head, 'track_decoder', None) if head is not None else None)
        if track_decoder is not None:
            if 'track_decoder' not in payload:
                raise ValueError(
                    'Track-enabled checkpoint has no `track_decoder` payload: '
                    f'{path}')
            track_decoder.load_state_dict(
                payload['track_decoder'], strict=True)
        elif 'track_decoder' in payload:
            raise ValueError(
                'Checkpoint contains `track_decoder`, but the model does not '
                f'enable track supervision: {path}')

    def _resume_or_load_checkpoint(self) -> None:
        """Restore source full-state directory or trainable weights."""
        if not self.resume_from:
            return
        if self.accelerator is None or self.train_sampler is None:
            raise RuntimeError('Training must be prepared before resume.')
        resume_path = Path(str(self.resume_from))
        if resume_path.is_file():
            self._load_weight_payload(str(resume_path))
            return
        if not resume_path.is_dir():
            raise FileNotFoundError(
                f'Resume checkpoint not found: {resume_path}')
        self.accelerator.load_state(input_dir=str(resume_path))
        state_file = resume_path / 'trainer_state.json'
        if state_file.exists():
            with state_file.open('r', encoding='utf-8') as file_handle:
                payload = json.load(file_handle)
            self.metric.global_step = int(payload['global_step'])
            self.epoch = int(payload['epoch'])
            self.batch_in_epoch = int(payload['batch_in_epoch'])
            self.train_sampler.set_epoch_offset(self.epoch)
            self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
            return
        match = re.search(r'step[_-](\d+)$', str(resume_path))
        self.metric.global_step = int(match.group(1)) if match else 0

    def run(self,
            vla_dataset: Sized,
            eval_dataset: Optional[Sized] = None) -> str:
        """Train through the source step loop.

        Args:
            vla_dataset: Flux FastWAM training dataset.
            eval_dataset: Optional validation dataset; training data is used
                when omitted, matching FastWAM.

        Returns:
            Last source-format weight checkpoint path.
        """
        self._prepare_training(vla_dataset)
        if self.accelerator is None or self.train_loader is None:
            raise RuntimeError('Training preparation failed.')
        validation_dataset = vla_dataset if eval_dataset is None \
            else eval_dataset
        self._set_source_train_mode()
        data_iterator = iter(self.train_loader)
        run_start_step = self.metric.global_step
        run_start_time = time.perf_counter()

        while self.metric.global_step < int(self.max_steps):
            try:
                sample = next(data_iterator)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.metric.epoch = self.epoch
                self.batch_in_epoch = 0
                if self.train_sampler is not None:
                    self.train_sampler.clear_resume_batch_offset()
                data_iterator = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.vla):
                with self.accelerator.autocast():
                    loss, loss_metrics = self._compute_training_loss(sample)
                self.accelerator.backward(loss)
                if not self.accelerator.sync_gradients:
                    continue

                grad_norm = self.accelerator.clip_grad_norm_(
                    self.vla.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if not self.accelerator.optimizer_step_was_skipped:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.metric.global_step += 1
                step_metrics = self._gather_step_metrics(
                    loss, loss_metrics, grad_norm)
                self.last_train_metrics = step_metrics
                self.step_history.append({
                    'global_step':
                    float(self.metric.global_step),
                    **step_metrics,
                })
                if self.metric.global_step == 1:
                    self._audit_zero1_optimizer(require_moments=True)

                if (self.log_every > 0
                        and self.metric.global_step % self.log_every == 0):
                    eta, steps_per_second = self._estimate_eta(
                        run_start_time, run_start_step)
                    payload = {
                        'train/loss':
                        step_metrics['loss'],
                        'train/grad_norm':
                        step_metrics['grad_norm'],
                        'train/lr':
                        step_metrics['lr'],
                        'performance/steps_per_sec':
                        steps_per_second,
                        'performance/samples_per_sec':
                        (steps_per_second * self.per_device_batch_size *
                         self.accelerator.num_processes),
                    }
                    for key, value in step_metrics.items():
                        if key not in {'loss', 'grad_norm', 'lr'}:
                            payload[f'train/{key}'] = value
                    self.metric.log(self.metric.global_step, payload)
                    if self.accelerator.is_main_process:
                        details = ' '.join(
                            f'{key}={value:.4f}'
                            for key, value in sorted(step_metrics.items())
                            if key not in {'loss', 'grad_norm', 'lr'})
                        overwatch.info(
                            f'[train] epoch={self.epoch} step='
                            f'{self.metric.global_step}/{self.max_steps} '
                            f'loss={step_metrics["loss"]:.4f} {details} '
                            f'lr={step_metrics["lr"]:.2e} eta={eta}')

                if (self.eval_every > 0
                        and self.metric.global_step % self.eval_every == 0):
                    eval_metrics = self.evaluate(validation_dataset)
                    self.accelerator.wait_for_everyone()
                    if eval_metrics and self.accelerator.is_main_process:
                        payload = {
                            f'eval/{key}': value
                            for key, value in eval_metrics.items()
                        }
                        self.metric.log(self.metric.global_step, payload)

                if (self.save_every > 0
                        and self.metric.global_step % self.save_every == 0):
                    self.save_checkpoint()
                if self.metric.global_step >= int(self.max_steps):
                    if self.save_at_end:
                        self.save_checkpoint()
                        self.save_final_eval_checkpoint()
                    return os.path.join(
                        self.weights_dir,
                        f'step_{self.metric.global_step:06d}.pt',
                    )

        if self.save_at_end:
            self.save_checkpoint()
            self.save_final_eval_checkpoint()
        return os.path.join(
            self.weights_dir,
            f'step_{self.metric.global_step:06d}.pt',
        )

    def cleanup(self) -> None:
        """Release runner-owned resources before post-training evaluation."""
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
        self.train_loader = None
        self.train_sampler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.vla = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _source_worker_init_fn(worker_id: int) -> None:
    """Copy FastWAM's rank-aware DataLoader worker seeding.

    Args:
        worker_id: DataLoader worker index.
    """
    process_seed = torch.initial_seed()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        global_rank = int(torch.distributed.get_rank())
    else:
        global_rank = int(
            os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    base_seed = process_seed - worker_id
    seed_sequence = np.random.SeedSequence([base_seed, worker_id, global_rank])
    np.random.seed(seed_sequence.generate_state(4))
    torch_seed, random_seed, _ = seed_sequence.spawn(3)
    torch.manual_seed(torch_seed.generate_state(1, dtype=np.uint64)[0])
    random_value = (
        random_seed.generate_state(2, dtype=np.uint64).astype(list) *
        [1 << 64, 1]).sum()
    random.seed(random_value)
