# LIBERO-D4RT auxiliary training

## Scope

The integration consumes `libero-d4rt-v1.1` OpenD4RT pseudo-label sidecars for
LIBERO main-camera clips. The sidecars are external, read-only data; original
LeRobot parquet and videos are not modified.

The initial ablation adds two losses to FastWAM:

- query-wise normalized 2D track displacement, weighted by teacher visibility
  and confidence;
- visibility probability, weighted by teacher confidence.

Relative pseudo-3D is loaded and aligned for future ablations, but it is not
used by the initial loss.

## Data contract

`LoadLiberoD4RTTracks` runs before `ProcessParquetInputs`. It resolves a sidecar
from `(suite, episode_index, camera_key, raw frame indices)`, selects one local
anchor clip that covers the complete nine-frame FastWAM window, and re-anchors
labels to the first returned frame.

The transform returns:

- `track_gt_query_uv`: `[N, 2]`;
- `track_gt_uv`: `[T, N, 2]`;
- `track_gt_xyz_relative`: `[T, N, 3]`;
- `track_gt_visibility`, `track_gt_confidence`, `track_gt_weight`,
  `track_gt_mask`: `[T, N]`;
- point groups and provenance frame indices.

The current FastWAM recipe uses `T=9`, raw-frame stride 4, and `N=256`. Because
training only resizes images without crop or flip, normalized UV coordinates do
not require a geometric transform. Padding timesteps are explicitly masked.

## Configurations

- Full dataset: `configs/fastwam/fastwam_libero_d4rt_track_finetune.py`
- First 10% of every suite:
  `configs/fastwam/fastwam_libero_d4rt_track_10pct_finetune.py`

The initial weights are:

- `lambda_track_uv = 0.1`
- `lambda_track_visibility = 0.01`

The decoder adds approximately 0.79 million parameters. Existing FastWAM
configs leave both weights at zero and do not instantiate the decoder, so source
parity remains unchanged.

## Preflight

Set `LIBERO_D4RT_SIDECAR_ROOT` when using a non-default sidecar location. Before
an expensive run, verify:

1. the sidecar dataset passes OpenD4RT validation with source, parquet,
   checkpoint, and hash checks;
2. a real transformed sample has image shape `[3, 9, 224, 448]` and track shape
   `[9, 256, 2]`;
3. `track_gt_query_uv == track_gt_uv[0]`;
4. a full-checkpoint forward produces finite `loss_track_uv`,
   `loss_track_visibility`, and decoder gradients.

The 10% configuration intentionally fails closed when an episode has no
sidecar. It must only be paired with the matching leading-10%-per-suite build.

## Verified 10% asset

The first ablation dataset is expected at
`datasets/libero_d4rt_v1_1_10pct_local256`, or at the path specified by
`LIBERO_D4RT_SIDECAR_ROOT`:

- spatial: 43 episodes;
- object: 45 episodes;
- goal: 43 episodes;
- LIBERO-10: 38 episodes;
- total: 169 episodes, 1,435 sidecars, and 68,880 overlapping label frames;
- sidecar payload size: about 435.14 MiB;
- manifest SHA-256:
  `e28865496ee6f4b9603888f97e39f5b5dd6d4fa3f3c1ecade2e53b3730071d7f`;
- build signature:
  `0de860383fc71d5f67e000c41a425bd487601ade30ab24608810861aa1ce10c1`.

All source-video, parquet, sidecar, checkpoint, and build-signature checks pass.
The supervision-weight p05/p50/p95 values are approximately
0.881/0.936/0.976; maximum source-frame reprojection p99 is approximately
0.0246 normalized UV.

## Production safeguards

- The decoder samples only the selected camera's token subgrid, so boundary
  queries cannot interpolate into an adjacent tiled camera.
- Each causal VAE temporal group uses its last valid raw frame; a group is
  masked only when every frame in that group is padding.
- `FastWAMDeepSpeedTrainRunner` includes the decoder in the trainable policy,
  optimizer, source-format checkpoints, and post-eval train mode.
- Existing FastWAM configurations instantiate no decoder and preserve source
  parity.
- Base safetensors may omit every decoder key, and complete track checkpoints
  may contain every decoder key. Partial decoder checkpoints fail closed.
- `save_at_end=False` is available only for short smoke runs that should not
  write large ZeRO and full-model checkpoints; the production default remains
  `True`.
