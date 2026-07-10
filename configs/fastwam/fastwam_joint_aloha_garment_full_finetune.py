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
# FastWAM world-action model (joint) finetuned on the real ALOHA garment
# folding dataset.

_base_ = './fastwam_aloha_garment_full_finetune.py'

model = dict(vla_head=dict(type='FastWAMJointHead'))
inference_model = dict(vla_head=dict(type='FastWAMJointHead'))
