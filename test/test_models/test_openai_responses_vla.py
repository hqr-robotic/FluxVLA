# Copyright 2026 Limx Dynamics

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine import Config

from fluxvla.datasets.openai_eval_dataset import OpenAIAlohaInferenceDataset
from fluxvla.engines.runners.aloha_inference_runner import AlohaInferenceRunner
from fluxvla.engines.runners.base_inference_runner import BaseInferenceRunner
from fluxvla.models.vlas.openai_responses_vla import OpenAIResponsesVLA
from fluxvla.transforms.normalize import IdentityRobotAction

ROOT = Path(__file__).resolve().parents[2]


def _joint_state() -> np.ndarray:
    state = np.zeros(14, dtype=np.float32)
    state[[6, 13]] = 0.08
    return state


def test_aloha_joint_targets_are_bounded_and_interpolated() -> None:
    model = OpenAIResponsesVLA(
        control_mode='aloha_joint_delta',
        action_horizon=4,
        aloha_max_joint_delta=0.1,
        aloha_joint_limits=[[-0.05, 0.05]] * 12,
    )

    actions = model._aloha_actions_from_targets(
        {
            'left_joint_delta': [0.2] * 6,
            'left_gripper': 'close',
            'right_gripper': 'hold',
        }, _joint_state())

    assert actions.shape == (4, 14)
    assert actions.dtype == torch.float32
    np.testing.assert_allclose(actions[-1, :6], 0.05)
    assert actions[-1, 6].item() == pytest.approx(-0.01)
    np.testing.assert_allclose(actions[:, 7:13], 0.0)
    np.testing.assert_allclose(actions[:, 13], 0.08)
    assert torch.all(actions[1:, :6] >= actions[:-1, :6])


def test_default_libero_control_contract_is_preserved() -> None:
    model = OpenAIResponsesVLA(
        action_horizon=3,
        max_speed_fraction=0.5,
        position_action_scale=0.01,
    )

    actions = model._actions_from_targets(
        {
            'x': 0.01,
            'gripper': 1.0,
        },
        eef_position=np.zeros(3, dtype=np.float32),
        joint_position=None,
    )

    assert model.tool_name == 'move_to'
    assert actions.shape == (1, 3, 7)
    assert actions[0, -1, 0].item() == pytest.approx(1.0 / 3.0)
    assert actions[0, -1, 6].item() == pytest.approx(-1.0)


def test_aloha_predict_action_uses_move_joints_tool(
        monkeypatch: pytest.MonkeyPatch) -> None:
    model = OpenAIResponsesVLA(
        control_mode='aloha_joint_delta',
        action_horizon=2,
        aloha_max_joint_delta=0.1,
    )
    arguments = {
        'targets': {
            'right_joint_delta': [0.01, 0, 0, 0, 0, 0],
            'right_gripper': 'open',
        },
        'note': 'Small reversible correction.',
    }
    response = {
        'id':
        'response-test',
        'model':
        'gpt-6-astra',
        'output': [{
            'type': 'function_call',
            'call_id': 'call-test',
            'name': 'move_joints',
            'arguments': json.dumps(arguments),
        }],
        'usage': {
            'input_tokens': 10,
            'output_tokens': 5,
        },
    }
    captured = {}

    def fake_post_json(body: dict) -> dict:
        captured.update(body)
        return response

    monkeypatch.setattr(model, '_post_json', fake_post_json)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    actions = model.predict_action(
        images=[image, image, image],
        image_names=['front', 'left_wrist', 'right_wrist'],
        task_description='pick up the toy with right arm',
        joint_position=_joint_state(),
        reset_history=True,
    )

    assert captured['tools'][0]['name'] == 'move_joints'
    assert actions.shape == (2, 14)
    assert actions[-1, 7].item() == pytest.approx(0.01)
    assert actions[-1, 13].item() == pytest.approx(0.08)
    assert model.last_note == 'Small reversible correction.'


def test_aloha_dataset_preserves_rgb_order_and_joint_state() -> None:
    dataset = OpenAIAlohaInferenceDataset(
        img_keys=['front', 'left', 'right'], resize_size=None)
    front = np.zeros((2, 3, 3), dtype=np.uint8)
    front[0, 0] = [10, 20, 30]
    inputs = {
        'front': front,
        'left': np.ones_like(front),
        'right': np.full_like(front, 2),
        'qpos': _joint_state(),
        'task_description': 'test task',
    }

    batch = dataset(inputs)

    np.testing.assert_array_equal(batch['images'][0], front)
    np.testing.assert_array_equal(batch['joint_position'], _joint_state())
    assert batch['image_names'] == ['front', 'left', 'right']
    assert batch['reset_history'] is False


def test_identity_robot_action_removes_singleton_batch() -> None:
    transform = IdentityRobotAction(action_dim=14)
    action = np.zeros((1, 3, 14), dtype=np.float32)

    result = transform({'action': action})

    assert result.shape == (3, 14)


def test_gpt6_aloha_config_is_checkpoint_free_and_dry_run(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LITELLM_VIRTUAL_KEY', 'test-key')
    monkeypatch.delenv('LITELLM_BASE_URL', raising=False)
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY_ENV', raising=False)
    cfg = Config.fromfile(ROOT /
                          'configs/openai/gpt6_astra_aloha_inference.py')

    assert cfg.inference_model.model == 'gpt-6-astra'
    assert cfg.inference_model.base_url == 'https://litellm.limx.cn/v1'
    assert cfg.inference_model.api_key_env == 'LITELLM_VIRTUAL_KEY'
    assert cfg.inference_model.control_mode == 'aloha_joint_delta'
    assert cfg.inference.requires_checkpoint is False
    assert cfg.inference.model_build_device == 'cpu'
    assert cfg.inference.disable_puppet_arm is True
    assert cfg.inference.dataset.type == 'OpenAIAlohaInferenceDataset'
    assert cfg.inference.denormalize_action.type == 'IdentityRobotAction'


def test_base_runner_builds_checkpoint_free_cpu_policy(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import fluxvla.engines as engines
    import fluxvla.engines.runners.base_inference_runner as runner_module

    class DummyVLA:

        def __init__(self) -> None:
            self.device = None
            self.evaluating = False

        def eval(self) -> None:
            self.evaluating = True

        def to(self, device: str) -> None:
            self.device = device

    def fake_build_dataset(cfg: dict) -> str:
        return 'dataset'

    def fake_build_transform(cfg: dict) -> str:
        return 'transform'

    def fake_build_vla(cfg: dict) -> DummyVLA:
        return dummy_vla

    def fake_build_operator(cfg: dict) -> str:
        return 'operator'

    dummy_vla = DummyVLA()
    monkeypatch.setattr(engines, 'build_dataset_from_cfg', fake_build_dataset)
    monkeypatch.setattr(engines, 'build_transform_from_cfg',
                        fake_build_transform)
    monkeypatch.setattr(engines, 'build_vla_from_cfg', fake_build_vla)
    monkeypatch.setattr(runner_module, 'build_operator_from_cfg',
                        fake_build_operator)

    runner = BaseInferenceRunner(
        cfg={'inference_model': {
            'type': 'DummyVLA'
        }},
        dataset={'type': 'DummyDataset'},
        denormalize_action={'type': 'DummyTransform'},
        operator={'type': 'DummyOperator'},
        requires_checkpoint=False,
        model_build_device='cpu',
    )
    runner.run_setup()

    assert runner.dataset == 'dataset'
    assert runner.denormalize_action == 'transform'
    assert runner.ros_operator == 'operator'
    assert dummy_vla.evaluating is True
    assert dummy_vla.device == 'cpu'


def test_aloha_dry_run_skips_prepare_pose() -> None:

    class DummyOperator:

        def __init__(self) -> None:
            self.move_calls = 0

        def move_to_joints(self, left_pose: list, right_pose: list) -> None:
            self.move_calls += 1

    runner = AlohaInferenceRunner.__new__(AlohaInferenceRunner)
    runner.disable_puppet_arm = True
    runner.prepare_pose = ([0.0] * 7, [0.0] * 7)
    runner.ros_operator = DummyOperator()

    runner._move_to_prepare_pose()

    assert runner.ros_operator.move_calls == 0
