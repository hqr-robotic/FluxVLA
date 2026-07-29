import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch
import yaml
from mmengine import Config
from safetensors.torch import load_file

from fluxvla.engines.runners.fastwam_deepspeed_train_runner import (
    FastWAMDeepSpeedTrainRunner, _ResumableEpochSampler)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CONFIGS = (
    'fastwam_libero_full_finetune.py',
    'fastwam_idm_libero_full_finetune.py',
    'fastwam_joint_libero_full_finetune.py',
)


class _SizedDataset:
    """Minimal finite dataset for sampler tests."""

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __len__(self) -> int:
        """Return the configured size."""
        return self.size


class _SingleProcessAccelerator:
    """Minimal main-rank accelerator for final export tests."""

    is_main_process = True

    @staticmethod
    def wait_for_everyone() -> None:
        """Represent a completed single-process barrier."""

    @staticmethod
    def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        """Return the already unwrapped model."""
        return model


class _NoneMetadataDataset:
    """Dataset whose metadata cannot be handled by default collate."""

    def __len__(self) -> int:
        """Return two samples for one physical batch."""
        return 2

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return a tensor plus an intentionally optional metadata field."""
        return {
            'value': torch.tensor(index),
            'metadata': None,
        }


class _RecordingCollator:
    """Record calls while selecting the tensor field from each sample."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, batch: list[Dict[str, Any]]) -> torch.Tensor:
        """Stack values without attempting to collate optional metadata."""
        self.calls += 1
        return torch.stack([sample['value'] for sample in batch])


def test_resumable_sampler_matches_source_randperm() -> None:
    """The unsharded sampler must exactly match FastWAM's randperm."""
    dataset = _SizedDataset(23)
    sampler = _ResumableEpochSampler(
        dataset=dataset,
        seed=42,
        batch_size=2,
        num_processes=4,
    )
    sampler.set_epoch(3)

    generator = torch.Generator(device='cpu').manual_seed(45)
    expected = torch.randperm(23, generator=generator).tolist()
    assert list(sampler) == expected


def test_train_loader_uses_configured_collator(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Production loading must not fall back to PyTorch default collate."""
    runner = FastWAMDeepSpeedTrainRunner.__new__(FastWAMDeepSpeedTrainRunner)
    runner.accelerator = SimpleNamespace(
        num_processes=1,
        process_index=0,
    )
    runner.seed = 42
    runner.per_device_batch_size = 2
    runner.per_device_num_workers = 0
    runner.collator = _RecordingCollator()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    loader = runner._build_train_loader(_NoneMetadataDataset())
    batch = next(iter(loader))

    assert runner.collator.calls == 1
    assert sorted(batch.tolist()) == [0, 1]


def test_resumable_sampler_skips_global_resume_batches() -> None:
    """Resume offsets skip batch_size times world_size per local batch."""
    dataset = _SizedDataset(31)
    sampler = _ResumableEpochSampler(
        dataset=dataset,
        seed=7,
        batch_size=2,
        num_processes=4,
    )
    sampler.set_epoch_offset(5)
    sampler.set_resume_batch_offset(2)

    generator = torch.Generator(device='cpu').manual_seed(12)
    expected = torch.randperm(31, generator=generator).tolist()[16:]
    assert list(sampler) == expected


def test_optimizer_recipe_rejects_non_source_betas() -> None:
    """A different AdamW beta pair must fail before training."""
    with pytest.raises(ValueError, match='betas'):
        FastWAMDeepSpeedTrainRunner._validate_optimizer_cfg({
            'type':
            'AdamW',
            'lr':
            1e-4,
            'betas': (0.9, 0.999),
        })


@pytest.mark.parametrize(
    ('cuda_available', 'device_count', 'local_rank', 'expected'),
    (
        (False, 0, '7', 'cpu'),
        (True, 1, '7', 'cuda:0'),
        (True, 8, '3', 'cuda:3'),
        (True, 8, '-1', 'cuda:0'),
        (True, 8, '8', 'cuda:0'),
    ),
)
def test_source_build_device_resolution(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    device_count: int,
    local_rank: str,
    expected: str,
) -> None:
    """Pre-Accelerator construction must use FastWAM's rank device rule."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: device_count)
    monkeypatch.setenv('LOCAL_RANK', local_rank)

    actual = FastWAMDeepSpeedTrainRunner._resolve_source_train_device()

    assert actual == expected


def test_runner_injects_build_device_without_mutating_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Production model construction must not allocate a CPU copy per rank."""
    from fluxvla.engines.utils import builder

    captured: Dict[str, Any] = {}

    def _build_model(model_cfg: Dict[str, Any]) -> object:
        captured.update(model_cfg)
        return object()

    monkeypatch.setattr(builder, 'build_vla_from_cfg', _build_model)
    monkeypatch.setattr(builder, 'build_collator_from_cfg', lambda _: object())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 8)
    monkeypatch.setenv('LOCAL_RANK', '5')
    model_cfg = {'type': 'FakeFastWAM', 'device': 'cpu'}
    config = SimpleNamespace(
        model=model_cfg,
        seed=42,
        train_dataloader=SimpleNamespace(
            per_device_batch_size=4,
            per_device_num_workers=8,
        ),
    )

    runner = FastWAMDeepSpeedTrainRunner(
        cfg=config,
        args=SimpleNamespace(work_dir=str(tmp_path)),
        optimizer={
            'type': 'AdamW',
            'lr': 1e-4,
            'weight_decay': 1e-2,
            'betas': (0.9, 0.95),
            'eps': 1e-8,
        },
        collator={},
        max_steps=1,
        grad_accumulation_steps=4,
    )

    assert runner.model_build_device == 'cuda:5'
    assert captured['device'] == 'cuda:5'
    assert model_cfg['device'] == 'cpu'
    assert runner.save_at_end is True


def test_final_eval_checkpoint_is_full_and_stats_path_resolves(
        tmp_path: Path) -> None:
    """Final eval export must be loadable with checkpoint-relative stats."""
    runner = FastWAMDeepSpeedTrainRunner.__new__(FastWAMDeepSpeedTrainRunner)
    runner.accelerator = _SingleProcessAccelerator()
    runner.vla = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.LayerNorm(3),
    )
    runner.metric = SimpleNamespace(global_step=7)
    runner.output_dir = str(tmp_path)
    runner.checkpoint_root = str(tmp_path / 'checkpoints')
    runner.weights_dir = str(tmp_path / 'checkpoints/weights')
    runner._last_eval_weights_path = None
    Path(runner.weights_dir).mkdir(parents=True)
    statistics = {'libero_all_no_noops': {'action': {'mean': [0.0]}}}
    statistics_path = tmp_path / 'dataset_statistics.json'
    statistics_path.write_text(json.dumps(statistics) + '\n', encoding='utf-8')

    eval_path = runner.save_final_eval_checkpoint()

    assert eval_path is not None
    assert eval_path.endswith('/checkpoints/weights/step_000007.safetensors')
    source_weights_path = Path(eval_path).with_suffix('.pt')
    source_weights_path.touch()
    from scripts.train import _resolve_eval_ckpt_path
    assert _resolve_eval_ckpt_path(str(source_weights_path)) == eval_path
    loaded = load_file(eval_path, device='cpu')
    expected = runner.vla.state_dict()
    assert loaded.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(loaded[key], value)
    inferred_statistics_path = (
        Path(eval_path).resolve().parent.parent / 'dataset_statistics.json')
    assert json.loads(inferred_statistics_path.read_text()) == statistics
    assert runner._last_eval_weights_path == eval_path


def test_source_scheduler_values_for_gold_recipe() -> None:
    """The released 21,700-step LR trajectory must stay exact."""
    runner = FastWAMDeepSpeedTrainRunner.__new__(FastWAMDeepSpeedTrainRunner)
    runner.optimizer_cfg = {
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'eps': 1e-8,
        'betas': (0.9, 0.95),
    }
    runner.scheduler_cfg = {
        'type': 'linear-warmup+cosine-decay-min-lr',
        'warmup_ratio': 0.05,
        'min_lr_ratio': 0.01,
    }
    parameter = torch.nn.Parameter(torch.zeros(()))
    runner.optimizer = torch.optim.AdamW(
        [parameter],
        lr=runner.optimizer_cfg['lr'],
        betas=runner.optimizer_cfg['betas'],
        weight_decay=runner.optimizer_cfg['weight_decay'],
    )

    scheduler = runner._build_scheduler(total_steps=21700)
    assert runner.optimizer.param_groups[0]['lr'] == pytest.approx(
        9.216589861751152e-8, rel=0.0, abs=1e-20)
    runner.optimizer.step()
    scheduler.step()
    assert runner.optimizer.param_groups[0]['lr'] == pytest.approx(
        1.842468517063433e-7, rel=0.0, abs=1e-20)

    for _ in range(1084):
        runner.optimizer.step()
        scheduler.step()
    assert runner.optimizer.param_groups[0]['lr'] == pytest.approx(
        1e-4, rel=0.0, abs=1e-18)


def test_version_report_can_be_non_strict(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-strict reports support diagnostics outside the parity env."""
    fake_deepspeed = SimpleNamespace(__version__='0.0.0')
    monkeypatch.setitem(__import__('sys').modules, 'deepspeed', fake_deepspeed)
    versions = FastWAMDeepSpeedTrainRunner._validate_runtime_versions(
        strict=False)
    assert versions['deepspeed'] == '0.0.0'
    assert versions['cuda'] == str(torch.version.cuda)


@pytest.mark.parametrize('config_name', PRODUCTION_CONFIGS)
def test_production_config_uses_source_zero1_recipe(config_name: str) -> None:
    """Every production variant must retain the source training recipe."""
    config = Config.fromfile(REPO_ROOT / 'configs/fastwam' / config_name)

    assert config.runner.type == 'FastWAMDeepSpeedTrainRunner'
    assert config.runner.sampler == 'source'
    assert config.train_dataloader.per_device_batch_size == 4
    assert config.train_dataloader.per_device_num_workers == 8
    assert config.runner.grad_accumulation_steps == 4
    assert config.runner.max_epochs == 10
    assert config.runner.optimizer == {
        'type': 'AdamW',
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'betas': (0.9, 0.95),
        'eps': 1e-8,
    }
    assert config.runner.lr_scheduler.type == (
        'linear-warmup+cosine-decay-min-lr')
    assert config.runner.lr_scheduler.warmup_ratio == 0.05
    assert config.runner.lr_scheduler.min_lr_ratio == 0.01
    assert config.runner.enable_gradient_checkpointing is False
    assert config.runner.enable_mixed_precision_training is True
    assert config.runner.mixed_precision_dtype == 'bf16'
    assert config.runner.log_every == 10
    assert config.runner.save_every == 2000
    assert config.runner.evaluator.eval_every == 200
    assert config.runner.evaluator.num_inference_steps == 10
    assert tuple(config.runner.metric.active_trackers) == ('jsonl', )


def test_zero1_launcher_matches_source_configuration() -> None:
    """The checked-in launcher must select the exact ZeRO-1 JSON."""
    deepspeed_path = (REPO_ROOT / 'scripts/ds_configs/fastwam_zero1.json')
    accelerate_path = (
        REPO_ROOT / 'scripts/accelerate_configs/fastwam_zero1.yaml')
    launcher_path = REPO_ROOT / 'scripts/train_fastwam_zero1.sh'

    with deepspeed_path.open('r', encoding='utf-8') as file_handle:
        deepspeed_config = json.load(file_handle)
    with accelerate_path.open('r', encoding='utf-8') as file_handle:
        accelerate_config = yaml.safe_load(file_handle)

    assert deepspeed_config == {
        'train_batch_size': 'auto',
        'train_micro_batch_size_per_gpu': 'auto',
        'gradient_accumulation_steps': 'auto',
        'zero_optimization': {
            'stage': 1,
            'offload_optimizer': {
                'device': 'none',
            },
            'offload_param': {
                'device': 'none',
            },
            'overlap_comm': False,
            'contiguous_gradients': False,
            'reduce_bucket_size': 200000000,
            'allgather_bucket_size': 200000000,
        },
    }
    assert accelerate_config['distributed_type'] == 'DEEPSPEED'
    assert accelerate_config['mixed_precision'] is None
    assert accelerate_config['deepspeed_config'] == {
        'deepspeed_config_file': ('scripts/ds_configs/fastwam_zero1.json'),
        'zero3_init_flag': False,
    }

    launcher = launcher_path.read_text(encoding='utf-8')
    assert launcher_path.stat().st_mode & stat.S_IXUSR
    assert 'accelerate launch' in launcher
    assert '--config_file scripts/accelerate_configs/fastwam_zero1.yaml' \
        in launcher
    assert 'FASTWAM_TEXT_CACHE_DIR' in launcher
    assert 'export FASTWAM_TEXT_CACHE_DIR' in launcher
    assert 'FASTWAM_HF_DATASETS_CACHE' in launcher
    assert 'export HF_DATASETS_CACHE' in launcher
    assert 'conda run --no-capture-output' in launcher
