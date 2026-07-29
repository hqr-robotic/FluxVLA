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

import json
from pathlib import Path

import numpy as np
import pytest

from fluxvla.transforms.libero_d4rt import LoadLiberoD4RTTracks


def _write_sidecar(root: Path) -> None:
    num_frames = 48
    num_queries = 4
    query_uv = np.array([[0.2, 0.2], [0.4, 0.2], [0.2, 0.4], [0.4, 0.4]],
                        dtype=np.float32)
    uv = np.repeat(query_uv[None], num_frames, axis=0)
    uv[..., 0] += np.arange(num_frames, dtype=np.float32)[:, None] * 0.001
    xyz_relative = np.zeros((num_frames, num_queries, 3), dtype=np.float32)
    xyz_relative[..., 0] = np.arange(
        num_frames, dtype=np.float32)[:, None] * 0.01
    relative_path = Path('episode_000000/clip_000000.npz')
    path = root / relative_path
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        frame_index=np.arange(num_frames, dtype=np.int32),
        track_uv_norm=uv,
        track_xyz_relative=xyz_relative,
        visibility_logit=np.full((num_frames, num_queries),
                                 4.0,
                                 dtype=np.float32),
        confidence_logit=np.full((num_frames, num_queries),
                                 3.0,
                                 dtype=np.float32),
        valid_mask=np.ones((num_frames, num_queries), dtype=np.bool_),
        supervision_weight=np.ones((num_frames, num_queries),
                                   dtype=np.float32),
        point_group=np.zeros((num_queries, ), dtype=np.uint8),
        scene_scale=np.asarray(4.0, dtype=np.float32),
    )
    record = {
        'schema_version': 'libero-d4rt-v1.1',
        'label_source': 'opend4rt_pseudo_gt',
        'build_signature': '0' * 64,
        'complete': True,
        'suite': 'spatial',
        'episode_index': 0,
        'camera_key': 'observation.images.image',
        'clip_start': 0,
        'clip_end': num_frames,
        'length': num_frames,
        'sidecar_path': relative_path.as_posix(),
    }
    (root / 'manifest.jsonl').write_text(
        json.dumps(record) + '\n', encoding='utf-8')


def test_load_libero_d4rt_tracks_aligns_and_masks_padding(tmp_path):
    _write_sidecar(tmp_path)
    transform = LoadLiberoD4RTTracks(sidecar_root=str(tmp_path), cache_size=1)
    data = {
        'data_root':
        '/tmp/libero_spatial_no_noops_lerobotv2.1',
        'episode_index':
        0,
        'frame_indices':
        np.array([1, 5, 9, 13, 17, 21, 25, 29, 29], dtype=np.int32),
        'frame_masks':
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.float32),
    }
    output = transform(data)

    assert output['track_gt_uv'].shape == (9, 4, 2)
    assert output['track_gt_xyz_relative'].shape == (9, 4, 3)
    np.testing.assert_allclose(output['track_gt_xyz_relative'][0], 0.0)
    np.testing.assert_allclose(output['track_gt_query_uv'],
                               output['track_gt_uv'][0])
    assert not output['track_gt_mask'][-1].any()
    np.testing.assert_array_equal(output['track_gt_weight'][-1], 0.0)


def test_load_libero_d4rt_tracks_fails_closed_by_default(tmp_path):
    _write_sidecar(tmp_path)
    transform = LoadLiberoD4RTTracks(sidecar_root=str(tmp_path))
    data = {
        'data_root': '/tmp/libero_object_no_noops_lerobotv2.1',
        'episode_index': 0,
        'frame_indices': np.array([0], dtype=np.int32),
        'frame_masks': np.array([1], dtype=np.float32),
    }
    with pytest.raises(KeyError, match='No LIBERO-D4RT records'):
        transform(data)
