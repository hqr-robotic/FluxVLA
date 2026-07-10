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
# FastWAM world-action model (uncond) finetuned on the real ALOHA garment
# folding dataset.

_ckpt_root = '/root/projects/ryanhu/checkpoints'

_frame_window_size = 9
_action_window_size = 32
_frame_sample_stride = 4

_data_root_path = [
    '/root/projects/RealRobot_AgileX_aloha_lerobot/'
    '20260704_20260704_01_4090_e2e_02',
    '/root/projects/RealRobot_AgileX_aloha_lerobot/'
    '20260706_20260706_01_4090_e2e_02',
]
_task_description = 'Fold the garment neatly into a compact rectangle.'
_statistic_name = 'aloha_garment_fold'
_text_embed_cache_dir = (
    '/root/projects/ryanhu/data/text_embeds_cache/aloha_garment_fold')

model = dict(
    type='FastWAMVLA',
    pretrained_name_or_path=None,
    num_views=3,
    frame_window_size=_frame_window_size,
    proprio_dim=14,
    action_horizon=_action_window_size,
    mot_checkpoint_mixed_attn=True,
    vlm_backbone=dict(
        type='Wan22Backbone',
        model_id='Wan-AI/Wan2.2-TI2V-5B',
        tokenizer_model_id='Wan-AI/Wan2.1-T2V-1.3B',
        tokenizer_max_len=128,
        load_text_encoder=False,
        redirect_common_files=True,
    ),
    vla_head=dict(
        type='FastWAMHead',
        video_dit_config=dict(
            has_image_input=False,
            patch_size=[1, 2, 2],
            in_dim=48,
            hidden_dim=3072,
            ffn_dim=14336,
            freq_dim=256,
            text_dim=4096,
            out_dim=48,
            num_heads=24,
            attn_head_dim=128,
            num_layers=30,
            eps=1.0e-06,
            seperated_timestep=True,
            require_clip_embedding=False,
            require_vae_embedding=False,
            fuse_vae_embedding_in_latents=True,
            video_attention_mask_mode='first_frame_causal',
            action_conditioned=False,
            action_dim=14,
            action_group_causal_mask_mode='group_diagonal',
            use_gradient_checkpointing=True,
        ),
        action_dit_config=dict(
            action_dim=14,
            hidden_dim=1024,
            ffn_dim=4096,
            num_heads=24,
            attn_head_dim=128,
            num_layers=30,
            text_dim=4096,
            freq_dim=256,
            eps=1.0e-06,
            use_gradient_checkpointing=True,
        ),
        action_dit_pretrained_path=(
            _ckpt_root +
            '/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt'),
        skip_dit_load_from_pretrain=False,
        video_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        action_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        loss=dict(lambda_video=1.0, lambda_action=1.0),
    ),
)

inference_model = model.copy()

train_dataloader = dict(
    per_device_batch_size=16,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_statistic_name,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_data_root_path,
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state',
                        'timestamp',
                        'actions',
                        'info',
                        'stats',
                        'action_masks',
                    ],
                    video_keys=[
                        'observation.images.cam_high',
                        'observation.images.cam_left_wrist',
                        'observation.images.cam_right_wrist',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                    embodiment_id=0,
                ),
                dict(
                    type='OverrideTaskDescription',
                    task_description=_task_description,
                    original_task_key='raw_task_description',
                ),
                dict(
                    type='ResizeImages',
                    height=224,
                    width=224,
                    backend='torchvision',
                    scale_to_unit_interval=True,
                ),
                dict(
                    type='NormalizeImages',
                    means=[0.5, 0.5, 0.5],
                    stds=[0.5, 0.5, 0.5],
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=14,
                    state_dim=14,
                    state_key='proprio',
                    action_key='action',
                    norm_type='min_max',
                ),
                dict(
                    type='PrepareVideo',
                    num_views=3,
                    frame_window_size=_frame_window_size,
                    tile_direction='horizontal',
                ),
                dict(
                    type='LoadCachedTextEmbedding',
                    cache_dir=_text_embed_cache_dir,
                    context_len=128,
                    enc_id='wan22ti2v5b',
                ),
            ],
            action_window_size=_action_window_size,
            action_key='action',
            use_delta=False,
            statistic_name=_statistic_name,
            window_start_idx=0,
            frame_window_size=_frame_window_size,
            frame_sample_stride=_frame_sample_stride,
        ),
    ),
)

inference = dict(
    type='AlohaInferenceRunner',
    seed=7,
    dataset=dict(
        type='FastWAMPrivateInferenceDataset',
        statistic_name=_statistic_name,
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
            dict(
                type='ResizeImages',
                height=224,
                width=224,
                backend='torchvision',
                scale_to_unit_interval=True,
            ),
            dict(
                type='NormalizeImages',
                means=[0.5, 0.5, 0.5],
                stds=[0.5, 0.5, 0.5],
            ),
            dict(
                type='NormalizeStatesAndActions',
                action_dim=14,
                state_dim=14,
                state_key='proprio',
                action_key=None,
                norm_type='min_max',
            ),
            dict(
                type='PrepareVideo',
                num_views=3,
                frame_window_size=1,
                tile_direction='horizontal',
            ),
            dict(
                type='LoadCachedTextEmbedding',
                cache_dir=_text_embed_cache_dir,
                context_len=128,
                enc_id='wan22ti2v5b',
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        statistic_name=_statistic_name,
        norm_type='min_max',
        action_dim=14,
    ),
    task_suite_name=_statistic_name,
    task_descriptions={'1': _task_description},
    action_chunk=_action_window_size,
    state_dim=14,
    publish_rate=30,
    max_publish_step=10000,
)

val_dataloader = None
eval_dataset = None

runner = dict(
    type='DDPTrainRunner',
    max_epochs=10,
    max_keep_ckpts=10,
    optimizer=dict(lr=1e-4, type='AdamW', weight_decay=1e-2),
    max_grad_norm=1.0,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'embodiment_ids',
            'frame_masks',
            'context',
            'context_mask',
        ],
        meta_keys=[
            'task_description',
            'raw_task_description',
            'prompt',
            'info',
            'stats',
            'timestamp',
        ],
    ),
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay-min-lr',
        warmup_ratio=0.05,
        min_lr_ratio=0.01,
        betas=(0.9, 0.95),
        weight_decay_style='uniform',
    ),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    grad_accumulation_steps=1,
    mixed_precision_dtype='bf16',
    evaluator=dict(
        type='training-eval',
        eval_every=1000,
        num_inference_steps=10,
        seed=42,
        save_video=True,
        video_fps=8,
    ),
)
