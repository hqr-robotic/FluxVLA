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
"""Checkpoint-free GPT-6 Astra inference for an ALOHA dual-arm robot.

The runner starts in observation-only dry-run mode. After validating camera
ordering, joint-state ordering, API output, and the configured joint limits,
set ``inference.disable_puppet_arm=False`` explicitly to enable motion.

Connection settings come only from environment variables. For LiteLLM::

    export LITELLM_VIRTUAL_KEY='...'
The project gateway is the default when that key exists. ``LITELLM_BASE_URL``
can select another gateway. ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY_ENV``
take precedence when set. Secrets must not be placed in this config.
"""

from os import environ as _environ

_DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1'
_DEFAULT_LITELLM_BASE_URL = 'https://litellm.limx.cn/v1'
_HAS_LITELLM_KEY = bool(_environ.get('LITELLM_VIRTUAL_KEY'))
_OPENAI_BASE_URL = (_environ.get('OPENAI_BASE_URL')
                    or _environ.get('LITELLM_BASE_URL')
                    or (_DEFAULT_LITELLM_BASE_URL if _HAS_LITELLM_KEY else
                        _DEFAULT_OPENAI_BASE_URL)).rstrip('/')
_OPENAI_API_KEY_ENV = _environ.get('OPENAI_API_KEY_ENV')
if not _OPENAI_API_KEY_ENV:
    _OPENAI_API_KEY_ENV = ('LITELLM_VIRTUAL_KEY'
                           if _environ.get('LITELLM_VIRTUAL_KEY') else
                           'OPENAI_API_KEY')

inference_model = dict(
    type='OpenAIResponsesVLA',
    model='gpt-6-astra',
    base_url=_OPENAI_BASE_URL,
    api_key_env=_OPENAI_API_KEY_ENV,
    control_mode='aloha_joint_delta',
    reasoning_effort='medium',
    max_output_tokens=None,
    request_timeout=120.0,
    max_retries=2,
    retry_backoff=2.0,
    image_detail='high',
    image_format='JPEG',
    jpeg_quality=90,
    image_horizon=2,
    max_llm_calls=100,
    action_horizon=20,
    # Each API call can change a revolute joint by at most about 6.9 degrees.
    aloha_max_joint_delta=0.12,
    # Six revolute joints per arm. Replace these conservative generic bounds
    # with limits from the deployed ALOHA calibration before enabling motion.
    aloha_joint_limits=[[-3.14, 3.14]] * 12,
    aloha_gripper_open=0.08,
    aloha_gripper_closed=-0.01,
)

# Keep environment helpers out of MMEngine's serialized config namespace.
del _environ, _DEFAULT_OPENAI_BASE_URL, _DEFAULT_LITELLM_BASE_URL
del _HAS_LITELLM_KEY, _OPENAI_BASE_URL, _OPENAI_API_KEY_ENV

inference = dict(
    type='AlohaInferenceRunner',
    requires_checkpoint=False,
    model_build_device='cpu',
    enable_mixed_precision=False,
    keep_params_fp32=True,
    seed=7,
    state_dim=14,
    action_chunk=20,
    execute_horizon=20,
    publish_rate=30,
    max_publish_step=10000,
    # Safety default: API calls and logs run, but no arm command is published.
    disable_puppet_arm=True,
    use_robot_base=False,
    task_descriptions={
        '1': 'pick up the brown bird toy with left arm',
        '2': 'pick up the brown bird toy with right arm',
        '3': 'pick up the purple knitted teddy bear toy with left arm',
        '4': 'pick up the purple knitted teddy bear toy with right arm',
        '5': 'pick up the white racing car toy with left arm',
        '6': 'pick up the white racing car toy with right arm',
        '7': 'pick up the purple caterpillar toy with left arm',
        '8': 'pick up the purple caterpillar toy with right arm',
        '9': 'place it in the brown flat cardboard box with left arm',
        '10': 'place it in the brown flat cardboard box with right arm',
    },
    dataset=dict(
        type='OpenAIAlohaInferenceDataset',
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        resize_size=512,
    ),
    denormalize_action=dict(
        type='IdentityRobotAction',
        action_dim=14,
        squeeze_batch=True,
    ),
    # Disable threshold snapping because GPT emits native ALOHA gripper
    # positions and the policy already constrains them to open/closed values.
    gripper_threshold=-0.011,
    gripper_closed_value=-0.01,
    operator=dict(
        type='AlohaOperator',
        image_encoding='rgb8',
        img_front_topic='/camera_h/color/image_raw',
        img_left_topic='/camera_l/color/image_raw',
        img_right_topic='/camera_r/color/image_raw',
        img_front_depth_topic='/camera_h/depth/image_raw',
        img_left_depth_topic='/camera_l/depth/image_raw',
        img_right_depth_topic='/camera_r/depth/image_raw',
        puppet_arm_left_cmd_topic='/master/joint_left',
        puppet_arm_right_cmd_topic='/master/joint_right',
        puppet_arm_left_topic='/puppet/joint_left',
        puppet_arm_right_topic='/puppet/joint_right',
        robot_base_topic='/odom_raw',
        robot_base_cmd_topic='/cmd_vel',
    ),
)
