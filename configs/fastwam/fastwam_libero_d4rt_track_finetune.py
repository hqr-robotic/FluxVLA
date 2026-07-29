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
#
# FastWAM full finetuning with LIBERO-D4RT v1.1 2D-track supervision.

import os

_base_ = ['./fastwam_libero_full_finetune.py']

_d4rt_sidecar_root = os.environ.get(
    'LIBERO_D4RT_SIDECAR_ROOT',
    'datasets/libero_d4rt_v1_1_main_local256',
)

model = dict(
    vla_head=dict(
        loss=dict(
            lambda_track_uv=0.1,
            lambda_track_visibility=0.01,
            track_decoder_dim=256,
            track_num_views=2,
            track_view_index=0,
            track_tile_direction='horizontal',
        ), ), )

# The final full checkpoint contains the decoder. Evaluation keeps the same
# module topology so strict state loading succeeds; action inference does not
# execute the decoder.
inference_model = dict(
    vla_head=dict(
        loss=dict(
            lambda_track_uv=0.0,
            lambda_track_visibility=0.0,
            track_decoder_enabled=True,
            track_decoder_dim=256,
            track_num_views=2,
            track_view_index=0,
            track_tile_direction='horizontal',
        ), ), )

train_dataloader = dict(
    dataset=dict(
        datasets=dict(
            pre_transforms=[
                dict(
                    type='LoadLiberoD4RTTracks',
                    sidecar_root=_d4rt_sidecar_root,
                    camera_key='observation.images.image',
                    cache_size=2,
                    required=True,
                ),
            ], ), ), )

runner = dict(
    collator=dict(
        keys=[
            'states',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'embodiment_ids',
            'frame_masks',
            'lang_tokens',
            'lang_masks',
            'track_gt_frame_index',
            'track_gt_frame_valid_mask',
            'track_gt_query_uv',
            'track_gt_uv',
            'track_gt_xyz_relative',
            'track_gt_visibility',
            'track_gt_confidence',
            'track_gt_weight',
            'track_gt_mask',
            'track_gt_point_group',
            'track_gt_scene_scale',
            'track_gt_anchor_frame_index',
        ],
        meta_keys=['task_description', 'info', 'stats', 'timestamp'],
    ), )
