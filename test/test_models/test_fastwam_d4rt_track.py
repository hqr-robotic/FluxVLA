# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from fluxvla.engines.runners.fastwam_deepspeed_train_runner import \
    FastWAMDeepSpeedTrainRunner
from fluxvla.models.heads.fastwam_head import FastWAMHead, FastWAMTrackDecoder
from fluxvla.models.vlas.fastwam_vla import FastWAMVLA


class _DummyVideoExpert(nn.Module):

    def __init__(self, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim


class _DummyActionExpert(nn.Module):
    pass


class _DummyMoT(nn.Module):

    def __init__(self, video_expert, action_expert):
        super().__init__()
        self.mixtures = nn.ModuleDict({
            'video': video_expert,
            'action': action_expert,
        })


def test_track_decoder_samples_tiled_video_tokens_and_backpropagates():
    decoder = FastWAMTrackDecoder(
        hidden_dim=8,
        decoder_dim=4,
        num_views=2,
        view_index=0,
        tile_direction='horizontal')
    tokens = torch.randn(2, 3 * 2 * 4, 8, requires_grad=True)
    query_uv = torch.tensor([[[0.2, 0.3], [0.8, 0.7]], [[0.1, 0.2], [0.9,
                                                                     0.8]]])

    output = decoder(tokens, (3, 2, 4), query_uv)

    assert output['delta_uv'].shape == (2, 3, 2, 2)
    assert output['visibility_logit'].shape == (2, 3, 2)
    sum(value.sum() for value in output.values()).backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_track_decoder_selects_one_camera_without_boundary_leakage():
    horizontal = FastWAMTrackDecoder(
        hidden_dim=1,
        decoder_dim=1,
        num_views=2,
        view_index=0,
        tile_direction='horizontal')
    feature_map = torch.cat([
        torch.ones(1, 1, 1, 2, 3),
        torch.full((1, 1, 1, 2, 3), 9.0),
    ],
                            dim=-1)
    selected = horizontal._select_view_features(feature_map, 2, 6)
    assert selected.shape == (1, 1, 1, 2, 3)
    torch.testing.assert_close(selected, torch.ones_like(selected))

    vertical = FastWAMTrackDecoder(
        hidden_dim=1,
        decoder_dim=1,
        num_views=2,
        view_index=1,
        tile_direction='vertical')
    feature_map = torch.cat([
        torch.ones(1, 1, 1, 2, 3),
        torch.full((1, 1, 1, 2, 3), 9.0),
    ],
                            dim=-2)
    selected = vertical._select_view_features(feature_map, 4, 3)
    torch.testing.assert_close(selected, torch.full_like(selected, 9.0))


def test_track_auxiliary_loss_is_finite_and_differentiable():
    video_expert = _DummyVideoExpert()
    action_expert = _DummyActionExpert()
    head = FastWAMHead(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=_DummyMoT(video_expert, action_expert),
        text_dim=8,
        temporal_downsample_factor=4,
        loss_lambda_track_uv=0.1,
        loss_lambda_track_visibility=0.01,
        track_decoder_dim=4,
        track_num_views=2,
        track_view_index=0,
        track_tile_direction='horizontal',
    )
    batch_size, num_queries = 2, 4
    video_tokens = torch.randn(batch_size, 3 * 2 * 4, 8, requires_grad=True)
    query_uv = torch.rand(batch_size, num_queries, 2)
    target_uv = query_uv[:, None].repeat(1, 9, 1, 1)
    target_uv[..., 0] += torch.linspace(0.0, 0.08, 9)[None, :, None]

    losses = head._compute_track_losses(
        video_tokens=video_tokens,
        video_meta={'grid_size': (3, 2, 4)},
        track_gt_query_uv=query_uv,
        track_gt_uv=target_uv,
        track_gt_visibility=torch.ones(batch_size, 9, num_queries),
        track_gt_confidence=torch.ones(batch_size, 9, num_queries),
        track_gt_weight=torch.ones(batch_size, 9, num_queries),
        track_gt_mask=torch.ones(batch_size, 9, num_queries, dtype=torch.bool),
        track_gt_frame_valid_mask=torch.ones(batch_size, 9, dtype=torch.bool),
    )

    assert set(losses) == {
        'loss_track', 'loss_track_uv', 'loss_track_visibility'
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses['loss_track'].backward()
    assert video_tokens.grad is not None
    assert any(parameter.grad is not None
               for parameter in head.track_decoder.parameters())


def test_track_loss_uses_last_valid_frame_in_each_causal_group():
    video_expert = _DummyVideoExpert()
    action_expert = _DummyActionExpert()
    head = FastWAMHead(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=_DummyMoT(video_expert, action_expert),
        text_dim=8,
        temporal_downsample_factor=4,
        loss_lambda_track_uv=1.0,
        track_decoder_dim=4,
    )

    class FixedDecoder(nn.Module):

        def forward(self, video_tokens, grid_size, query_uv):
            delta = torch.tensor(
                [[[[0.0, 0.0]], [[0.02, 0.0]], [[99.0, 0.0]]]],
                device=video_tokens.device)
            return {
                'delta_uv':
                delta,
                'visibility_logit':
                torch.zeros(1, 3, 1, device=video_tokens.device),
            }

    head.track_decoder = FixedDecoder()
    query_uv = torch.zeros(1, 1, 2)
    target_uv = query_uv[:, None].repeat(1, 9, 1, 1)
    target_uv[..., 0] = torch.arange(9)[None, :, None] * 0.01
    frame_valid = torch.tensor(
        [[True, True, True, False, False, False, False, False, False]])
    losses = head._compute_track_losses(
        video_tokens=torch.zeros(1, 3, 8),
        video_meta={'grid_size': (3, 1, 1)},
        track_gt_query_uv=query_uv,
        track_gt_uv=target_uv,
        track_gt_visibility=torch.ones(1, 9, 1),
        track_gt_confidence=torch.ones(1, 9, 1),
        track_gt_weight=torch.ones(1, 9, 1),
        track_gt_mask=torch.ones(1, 9, 1, dtype=torch.bool),
        track_gt_frame_valid_mask=frame_valid,
    )
    torch.testing.assert_close(losses['loss_track_uv'], torch.tensor(0.0))


def test_source_runner_optimizes_and_checkpoints_track_decoder(tmp_path):
    video_expert = _DummyVideoExpert()
    action_expert = _DummyActionExpert()
    head = FastWAMHead(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=_DummyMoT(video_expert, action_expert),
        text_dim=8,
        proprio_dim=2,
        loss_lambda_track_uv=0.1,
        track_decoder_dim=4,
    )

    class DummyModel(nn.Module):

        def __init__(self, model_head):
            super().__init__()
            self.vla_head = model_head
            self.torch_dtype = torch.float32

    model = DummyModel(head)
    runner = FastWAMDeepSpeedTrainRunner.__new__(FastWAMDeepSpeedTrainRunner)
    runner.vla = model
    trainable = runner._apply_source_trainable_policy()
    decoder_ids = {
        id(parameter)
        for parameter in head.track_decoder.parameters()
    }
    assert decoder_ids <= {id(parameter) for parameter in trainable}
    assert all(parameter.requires_grad
               for parameter in head.track_decoder.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    before = [
        parameter.detach().clone()
        for parameter in head.track_decoder.parameters()
    ]
    sum(parameter.sum() for parameter in trainable).backward()
    optimizer.step()
    assert any(not torch.equal(old, new)
               for old, new in zip(before, head.track_decoder.parameters()))

    class DummyAccelerator:

        @staticmethod
        def unwrap_model(value):
            return value

    runner.accelerator = DummyAccelerator()
    runner.metric = type('Metric', (), {'global_step': 3})()
    payload = runner._source_weight_payload()
    assert 'track_decoder' in payload
    path = tmp_path / 'weights.pt'
    torch.save(payload, path)
    expected = {
        key: value.detach().clone()
        for key, value in head.track_decoder.state_dict().items()
    }
    with torch.no_grad():
        for parameter in head.track_decoder.parameters():
            parameter.zero_()
    runner._load_weight_payload(str(path))
    for key, value in head.track_decoder.state_dict().items():
        torch.testing.assert_close(value, expected[key])


def _minimal_fastwam_vla_for_track_checkpoint() -> FastWAMVLA:
    model = FastWAMVLA.__new__(FastWAMVLA)
    nn.Module.__init__(model)
    model.vla_head = nn.Module()
    model.vla_head.track_decoder = nn.Linear(2, 3)
    model.pretrained_name_or_path = None
    return model


def test_track_safetensors_allows_base_or_complete_and_rejects_partial(
        tmp_path):
    model = _minimal_fastwam_vla_for_track_checkpoint()
    complete = tmp_path / 'complete.safetensors'
    save_file(model.state_dict(), complete)
    model.pretrained_name_or_path = str(complete)
    model.from_pretrained()

    base = tmp_path / 'base.safetensors'
    save_file({}, base)
    model.pretrained_name_or_path = str(base)
    model.from_pretrained()

    partial = tmp_path / 'partial.safetensors'
    state = model.state_dict()
    save_file(
        {
            'vla_head.track_decoder.weight':
            state['vla_head.track_decoder.weight']
        }, partial)
    model.pretrained_name_or_path = str(partial)
    with pytest.raises(RuntimeError, match='Partial FastWAM track decoder'):
        model.from_pretrained()


def test_decoder_topology_can_be_enabled_without_track_loss():
    video_expert = _DummyVideoExpert()
    action_expert = _DummyActionExpert()
    head = FastWAMHead(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=_DummyMoT(video_expert, action_expert),
        text_dim=8,
        track_decoder_enabled=True,
        loss_lambda_track_uv=0.0,
        loss_lambda_track_visibility=0.0,
        track_decoder_dim=4,
    )
    assert head.track_decoder is not None
    assert head._compute_track_losses(torch.empty(1, 1, 8), {}) == {}
