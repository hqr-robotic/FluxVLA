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
# First-10%-per-suite LIBERO-D4RT auxiliary-loss ablation.

import os

_base_ = ['./fastwam_libero_d4rt_track_finetune.py']

_d4rt_sidecar_root = os.environ.get(
    'LIBERO_D4RT_SIDECAR_ROOT',
    'datasets/libero_d4rt_v1_1_10pct_local256',
)

train_dataloader = dict(
    dataset=dict(
        datasets=dict(
            train_episode_fraction=0.1,
            repeat_to_full_length=False,
            pre_transforms=[
                dict(
                    type='LoadLiberoD4RTTracks',
                    sidecar_root=_d4rt_sidecar_root,
                    camera_key='observation.images.image',
                    cache_size=2,
                    required=True,
                ),
            ],
        ), ), )
