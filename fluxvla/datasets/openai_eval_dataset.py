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
"""Raw observation adapters for API-hosted policies."""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from fluxvla.engines import DATASETS


@DATASETS.register_module()
class OpenAILiberoEvalDataset:
    """Prepare raw images and proprioception for ``OpenAIResponsesVLA``.

    Unlike local VLA datasets, this adapter performs no tokenization,
    normalization, tensor conversion, or CUDA transfer. The API policy owns
    image encoding and returns native LIBERO control actions.
    """

    def __init__(self,
                 norm_stats: Any = None,
                 task_suite_name: str = None,
                 norm_stats_key: str = None,
                 img_keys: List[str] = None,
                 resize_size: int = 256) -> None:
        del norm_stats, task_suite_name, norm_stats_key
        self.img_keys = img_keys or [
            'agentview_image', 'robot0_eye_in_hand_image'
        ]
        self.resize_size = resize_size

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)[::-1, ::-1].copy()
        if self.resize_size is not None:
            if isinstance(self.resize_size, int):
                size = (self.resize_size, self.resize_size)
            else:
                height, width = self.resize_size
                size = (width, height)
            image = np.asarray(
                Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
        return image

    def __call__(self, inputs: Dict[str, Any]):
        images = []
        for image_key in self.img_keys:
            if image_key not in inputs:
                raise KeyError(f'Missing image key: {image_key!r}')
            images.append(self._prepare_image(inputs[image_key]))

        batch = {
            'images': images,
            'image_names': list(self.img_keys),
            'task_description': inputs['task_description'],
            'eef_position': np.asarray(inputs['robot0_eef_pos']).copy(),
            'eef_quaternion': np.asarray(inputs['robot0_eef_quat']).copy(),
            'gripper_position':
            np.asarray(inputs['robot0_gripper_qpos']).copy(),
            'joint_position': np.asarray(inputs['robot0_joint_pos']).copy(),
            'reset_history': bool(inputs.get('is_new_episode', False)),
        }
        return batch, images[0]


@DATASETS.register_module()
class OpenAIAlohaInferenceDataset:
    """Prepare ALOHA RGB images and joint state for an API policy.

    This adapter intentionally avoids the normalization and tokenization used
    by trained local VLAs. ALOHA camera frames are already requested as RGB by
    the inference config, and joint positions remain in the robot controller's
    native units.

    Args:
        img_keys: Ordered camera keys to send to the API policy.
        resize_size: Optional integer or ``(height, width)`` image size.
    """

    def __init__(self,
                 norm_stats: Any = None,
                 model_path: Optional[str] = None,
                 img_keys: Optional[List[str]] = None,
                 resize_size: Union[int, Sequence[int], None] = 512) -> None:
        del norm_stats, model_path
        self.img_keys = img_keys or [
            'cam_high', 'cam_left_wrist', 'cam_right_wrist'
        ]
        self.resize_size = resize_size

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Convert one RGB frame to the configured API image size."""
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f'Expected an RGB image with shape (H, W, 3), got '
                f'{image.shape}')
        if self.resize_size is None:
            return image.copy()
        if isinstance(self.resize_size, int):
            size = (self.resize_size, self.resize_size)
        else:
            height, width = self.resize_size
            size = (width, height)
        return np.asarray(
            Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Build an ``OpenAIResponsesVLA.predict_action`` input dictionary."""
        images = []
        for image_key in self.img_keys:
            if image_key not in inputs:
                raise KeyError(f'Missing image key: {image_key!r}')
            images.append(self._prepare_image(inputs[image_key]))

        joint_position = np.asarray(
            inputs['qpos'], dtype=np.float32).reshape(-1)
        if joint_position.size != 14:
            raise ValueError(f'ALOHA qpos must contain 14 values, got '
                             f'{joint_position.size}')
        return {
            'images': images,
            'image_names': list(self.img_keys),
            'task_description': inputs['task_description'],
            'joint_position': joint_position.copy(),
            'reset_history': bool(inputs.get('is_new_episode', False)),
        }
