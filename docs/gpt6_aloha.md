# GPT-6 ALOHA real-robot inference

This integration runs `gpt-6-astra` through the OpenAI Responses API (or a
compatible LiteLLM gateway) without a local checkpoint. GPT-6 receives the
front and two wrist RGB images plus the current 14-dimensional ALOHA joint
state. It emits one bounded `move_joints` tool call per observation.

The policy does not send model-provided values directly to ROS. The local
controller:

- clips each revolute-joint delta to `aloha_max_joint_delta`;
- clamps every target to `aloha_joint_limits`;
- holds any omitted arm at its measured position;
- maps gripper commands to configured open/closed positions; and
- linearly interpolates the result into absolute 14-dimensional joint targets.

## Configure credentials

Keep the key outside the repository:

```bash
export LITELLM_VIRTUAL_KEY='...'
```

When that variable is present, the config selects `LITELLM_VIRTUAL_KEY` and
the project gateway `https://litellm.limx.cn/v1`. Set `LITELLM_BASE_URL` to
use another gateway. For a different key variable, set
`OPENAI_API_KEY_ENV`; `OPENAI_BASE_URL` takes precedence over all LiteLLM
settings.

## Validate in dry-run mode

The checked-in config has `disable_puppet_arm=True`, so it reads ROS sensors,
calls GPT-6, validates the action, and logs the proposed trajectory without
publishing arm commands:

```bash
python scripts/inference_real_robot.py \
  --config configs/openai/gpt6_astra_aloha_inference.py
```

Before enabling motion, verify all of the following:

1. `/camera_h/color/image_raw` is the front RGB camera.
2. `/camera_l/color/image_raw` and `/camera_r/color/image_raw` correspond to
   the left and right wrist cameras.
3. `/puppet/joint_left` and `/puppet/joint_right` each publish six revolute
   joints followed by one gripper position.
4. The logged left/right state agrees with the physical arm ordering.
5. Replace the generic `[-3.14, 3.14]` joint limits in the config with the
   calibrated limits of the deployed robot.
6. The hardware emergency stop and a human safety operator are available.

## Enable arm commands

Only after the dry-run and calibration checks pass, explicitly override the
safety default:

```bash
python scripts/inference_real_robot.py \
  --config configs/openai/gpt6_astra_aloha_inference.py \
  --cfg-options inference.disable_puppet_arm=False
```

Start with an empty workspace and a small `aloha_max_joint_delta`, then test
one arm and one joint direction at a time. API latency makes this a
stop-and-observe controller rather than a high-rate reactive policy; the
robot holds its last target while the next API response is pending.
