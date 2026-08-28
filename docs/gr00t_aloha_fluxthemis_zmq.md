# GR00T ALOHA real-robot inference over FluxThemis ZMQ

This workflow keeps ROS and robot control on the ALOHA computer. FluxThemis
reads the real observations and sends them to the FluxVLA ZMQ server on the
Alibaba Cloud GPU machine. FluxVLA performs preprocessing, model inference,
and action denormalization.

## Validated revisions and assets

- FluxVLA worktree: `/root/projects/FluxVLA-test-hqr-fluxthemis`
- FluxVLA branch: `test/hqr/fluxthemis`, based on `main@1e5589b`
- FluxThemis branch: `test/hqr/fluxthemis`, based on
  `origin/feat/hqr/enhance-real-robot`
- FluxThemis revision validated here: `9985c4e`
- Checkpoint:
  `/root/projects/ryanhu/checkpoints/gr00t_flod_cloth_aloha_only/checkpoints/step-047720-epoch-20-loss=0.0209.safetensors`
- Statistics:
  `/root/projects/ryanhu/checkpoints/gr00t_flod_cloth_aloha_only/dataset_statistics.json`
- Instruction: `fold cloth`
- Observation keys: `cam_high`, `cam_left_wrist`, `cam_right_wrist`, `qpos`
- Returned action shape: `[T, 14]`

The original `dataset_statistics_aloha.json` is copied to the standard
`dataset_statistics.json` filename beside the `checkpoints/` directory. This
matches FluxVLA's existing lookup rule and avoids modifying the shared ZMQ
server implementation.

## Data flow and protocol

```text
ALOHA ROS1 topics
  -> FluxThemis AlohaEnvironment
  -> FluxVLAZMQModelClient (msgpack, compress=False)
  -> SSH tunnel or private TCP network
  -> Alibaba Cloud FluxVLA ZMQ server :3333
  -> denormalized float32 action [32, 14]
  -> FluxThemis safety checks and ALOHA command topics
```

The shared FluxVLA config defines `RealRobotBenchmarkRunner`,
`AlohaEnvironment`, and `FluxVLAZMQModelClient`. ZMQ does not support the
FluxVLA evaluation reporter; benchmark results are saved locally by
FluxThemis. `parallel_workers` must remain `1`.

## Prepare the local checkpoint layout

FluxVLA expects `dataset_statistics.json` two levels above the checkpoint.
Create that standard layout without changing the original OSS files:

```bash
mkdir -p \
  /root/projects/ryanhu/checkpoints/gr00t_flod_cloth_aloha_only/checkpoints
cp --reflink=auto \
  /mnt/data/oss/users/Mayer/checkpoints/gr00t_flod_cloth_aloha_only/checkpoints/step-047720-epoch-20-loss=0.0209.safetensors \
  /root/projects/ryanhu/checkpoints/gr00t_flod_cloth_aloha_only/checkpoints/
cp \
  /mnt/data/oss/users/Mayer/checkpoints/gr00t_flod_cloth_aloha_only/dataset_statistics_aloha.json \
  /root/projects/ryanhu/checkpoints/gr00t_flod_cloth_aloha_only/dataset_statistics.json
cp -a \
  /mnt/data/oss/users/Mayer/checkpoints/gr00t_flod_cloth_aloha_only/tokenizer \
  /root/projects/ryanhu/checkpoints/gr00t_flod_cloth_aloha_only/
```

The tokenizer directory is required because the unchanged server injects the
checkpoint root into `PrivateInferenceDataset`, which resolves its tokenizer
as `<checkpoint_root>/tokenizer`.

## Alibaba Cloud: start FluxVLA

Activate the existing environment and run the path/config preflight. It does
not load the model:

```bash
conda activate fluxvla
cd /root/projects/FluxVLA-test-hqr-fluxthemis
pip install -r requirements-real.txt
MAX_JOBS=8 python setup.py build_ext --inplace
bash scripts/serve_gr00t_aloha_fluxthemis.sh --dry-run
```

If the prompt shows both `(fluxvla)` and `(.venv)`, the virtualenv may shadow
the Conda interpreter. The launcher prioritizes `$CONDA_PREFIX/bin/python`.
It can also be selected explicitly:

```bash
FLUXVLA_PYTHON=/root/miniconda3/envs/fluxvla/bin/python \
  bash scripts/serve_gr00t_aloha_fluxthemis.sh
```

The in-place CUDA extension must be built once in each new worktree because
the binaries are ignored by Git. Start the server with the secure loopback
default when using an SSH tunnel:

```bash
bash scripts/serve_gr00t_aloha_fluxthemis.sh
```

In another cloud terminal, validate ping and one full synthetic inference:

```bash
conda activate fluxvla
cd /root/projects/FluxVLA-test-hqr-fluxthemis
python -m scripts.zmq_aloha_smoke_test
python -m scripts.zmq_aloha_smoke_test --predict
```

For a private VPN/VPC only, bind a reachable interface and restrict TCP port
3333 with the cloud firewall:

```bash
FLUXVLA_ZMQ_HOST=0.0.0.0 \
  bash scripts/serve_gr00t_aloha_fluxthemis.sh
```

The ZMQ service has no authentication or encryption. Never expose this port
to the public Internet.

## ALOHA: use the FluxThemis integration branch

Fetch the integration branch into a separate worktree so an existing
FluxThemis checkout is not disturbed:

```bash
cd /home/agilex/projects/FluxThemis
git fetch origin test/hqr/fluxthemis
git worktree add --track -b test/hqr/fluxthemis \
  /home/agilex/projects/FluxThemis-test-hqr-fluxthemis \
  origin/test/hqr/fluxthemis
cd /home/agilex/projects/FluxThemis-test-hqr-fluxthemis
git rev-parse HEAD
python -m pip install -e '.[real]'
```

The integration branch is based on the real-robot revision `9985c4e`.

FluxThemis requires the shared config to remain below a FluxVLA `configs/`
directory. On ALOHA, use an existing lightweight FluxVLA checkout only as the
config container; FluxVLA does not need to be installed there and no model
weights are required. Copy the deployment config from the development machine:

```bash
scp \
  /root/projects/FluxVLA-test-hqr-fluxthemis/configs/gr00t/gr00t_eagle_3b_aloha_fold_cloth_zmq.py \
  <ALOHA_USER>@<ALOHA_HOST>:/home/agilex/projects/FluxVLA/configs/gr00t/
```

The target checkout must already contain the inherited
`gr00t_eagle_3b_aloha_full_finetune.py`. Then set:

```bash
export FLUXVLA_ROOT=/home/agilex/projects/FluxVLA
export FLUXVLA_CONFIG="$FLUXVLA_ROOT/configs/gr00t/gr00t_eagle_3b_aloha_fold_cloth_zmq.py"
```

When the cloud server is bound to loopback, create the tunnel on ALOHA. The
config already uses `127.0.0.1:3333`:

```bash
ssh -N -L 3333:127.0.0.1:3333 \
  -p <ALIYUN_SSH_PORT> <ALIYUN_USER>@<ALIYUN_HOST>
```

Keep this process running in its own terminal.

## Safe validation before actuation

First validate component construction. This does not connect to ROS or the
ZMQ server and does not publish commands:

```bash
conda activate fluxthemis
cd /home/agilex/projects/FluxThemis-test-hqr-fluxthemis
fluxthemis-eval --config "$FLUXVLA_CONFIG" --dry-run
```

Then use the smoke script supplied by `feat/hqr/enhance-real-robot`. It sends
synthetic observations using the exact `msgpack`, `compress=False` protocol:

```bash
python scripts/fluxvla_zmq_smoke.py \
  --server-host 127.0.0.1 \
  --server-port 3333 \
  --timeout-s 30 \
  --unnorm-key private \
  --task-description 'fold cloth'
```

Check the real ROS inputs without writing commands:

```bash
rostopic list | grep -E 'camera_[fhlr]|puppet|master'
rostopic hz /camera_f/color/image_raw
rostopic hz /camera_l/color/image_raw
rostopic hz /camera_r/color/image_raw
rostopic echo -n1 /puppet/joint_left
rostopic echo -n1 /puppet/joint_right
```

For a private network without a tunnel, replace `127.0.0.1` with the cloud
server's private IP in the smoke command and override the benchmark client:

```bash
--cfg-options \
  themis.runner.model_client.server_host=<FLUXVLA_PRIVATE_IP>
```

## Hardware values that must be verified

The config intentionally has `allow_actuation=False`. It also sets
`require_action_bounds=False` and leaves `action_low` and `action_high` unset
to match the source ALOHA inference runner. This disables FluxThemis' software
action-bound check. Before enabling actuation, verify the target ALOHA's own
joint limits, watchdog, workspace clearance, and independent emergency stop.

The configured `image_encoding` is `rgb8`, matching the encoding validated by
the ALOHA observation probe. FluxThemis rejects a ROS image whose declared
encoding differs instead of silently converting it.

The FluxThemis environment reads the front image from
`/camera_f/color/image_raw`, publishes at 30 Hz, and executes the complete
32-step action chunk before requesting a new one.

Confirm left/right qpos ordering, radians versus other units, command topic
semantics, control rate, workspace clearance, stale-command watchdog, and an
independent reachable hardware emergency stop.

## Attended one-episode benchmark

Only after verifying the safety requirements, explicitly enable actuation for
one task and one episode:

```bash
conda activate fluxthemis
cd /home/agilex/projects/FluxThemis-test-hqr-fluxthemis

fluxthemis-eval \
  --config "$FLUXVLA_CONFIG" \
  --output-dir fluxthemis/aloha_benchmarks \
  --run-name gr00t-fold-cloth-real-001 \
  --task-ids 1 \
  --episodes-per-task 1 \
  --max-rollout-seconds 120 \
  --cfg-options \
    themis.runner.environment.allow_actuation=True
```

The benchmark is interactive and requires a graphical desktop. Follow the
FluxThemis prompts to capture a reset reference, confirm the physical reset,
stop the rollout, and record success or failure. Software stop paths do not
replace the ALOHA hardware emergency stop.
