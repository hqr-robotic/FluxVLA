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
# FastWAM world-action model (uncond) trained on RoboCasa GR1 24-task data.

_ckpt_root = './checkpoints'

_frame_window_size = 9
_action_window_size = 32
_frame_sample_stride = 4

_statistic_name = 'robocasa_gr1_24tasks_30ep'
_robocasa_data_root = './datasets/robocasa_gr1_24tasks_first30ep'
_official_gr1_stats_path = (
    f'{_robocasa_data_root}/official_groot_gr1_dataset_statistics.json')
_text_embed_cache_dir = (
    '/root/projects/ryanhu/data/text_embeds_cache/robocasa_gr1_24tasks_30ep')
_eval_output_root = (
    '/root/projects/ryanhu/FluxVLA/FastWAM/evaluate_results/robocas/'
    'fastwam_robocasa_full_finetune_ddp_stride4_bs32')

_robocasa_task_dirs = [
    'PnPBottleToCabinetClose',
    'PnPCanToDrawerClose',
    'PnPCupToDrawerClose',
    'PnPMilkToMicrowaveClose',
    'PnPPotatoToMicrowaveClose',
    'PnPWineToCabinetClose',
    'PosttrainPnPNovelFromCuttingboardToBasketSplitA',
    'PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA',
    'PosttrainPnPNovelFromCuttingboardToPanSplitA',
    'PosttrainPnPNovelFromCuttingboardToPotSplitA',
    'PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA',
    'PosttrainPnPNovelFromPlacematToBasketSplitA',
    'PosttrainPnPNovelFromPlacematToBowlSplitA',
    'PosttrainPnPNovelFromPlacematToPlateSplitA',
    'PosttrainPnPNovelFromPlacematToTieredshelfSplitA',
    'PosttrainPnPNovelFromPlateToBowlSplitA',
    'PosttrainPnPNovelFromPlateToCardboardboxSplitA',
    'PosttrainPnPNovelFromPlateToPanSplitA',
    'PosttrainPnPNovelFromPlateToPlateSplitA',
    'PosttrainPnPNovelFromTrayToCardboardboxSplitA',
    'PosttrainPnPNovelFromTrayToPlateSplitA',
    'PosttrainPnPNovelFromTrayToPotSplitA',
    'PosttrainPnPNovelFromTrayToTieredbasketSplitA',
    'PosttrainPnPNovelFromTrayToTieredshelfSplitA',
]

model = dict(
    type='FastWAMVLA',
    pretrained_name_or_path=None,
    num_views=1,
    frame_window_size=_frame_window_size,
    proprio_dim=64,
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
            action_dim=29,
            action_group_causal_mask_mode='group_diagonal',
            use_gradient_checkpointing=True,
        ),
        action_dit_config=dict(
            action_dim=29,
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
    per_device_batch_size=32,
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_statistic_name,
        dataset_statistics_path=_official_gr1_stats_path,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=[
                f'{_robocasa_data_root}/{task_dir}'
                for task_dir in _robocasa_task_dirs
            ],
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
                    video_keys=['observation.images.ego_view'],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                    embodiment_id=24,
                ),
                dict(type='RobocasaGR1N15Bridge'),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='NormalizeImages',
                    means=[0.5, 0.5, 0.5],
                    stds=[0.5, 0.5, 0.5],
                    scale_to_unit_interval=True,
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=29,
                    state_dim=64,
                    state_key='proprio',
                    action_key='action',
                    norm_type='min_max',
                    normalize_states=False,
                ),
                dict(
                    type='PrepareVideo',
                    num_views=1,
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

val_dataloader = None
eval_dataset = None

runner = dict(
    type='DDPTrainRunner',
    max_epochs=10,
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
        meta_keys=['task_description', 'prompt', 'info', 'stats', 'timestamp'],
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
        eval_every=200,
        num_inference_steps=10,
        seed=42,
        save_video=True,
        video_fps=8,
    ),
)

eval = dict(
    type='RobocasaEvalRunner',
    model_family='fastwam',
    task_list=[
        f'gr1_unified/{task_dir}_GR1ArmsAndWaistFourierHands_Env'
        for task_dir in _robocasa_task_dirs
    ],
    eval_chunk_size=10,
    max_episode_steps=720,
    num_trials_per_task=50,
    num_inference_steps=10,
    seed=7,
    unnorm_key=_statistic_name,
    action_order='n15',
    save_video=True,
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_statistic_name,
        transforms=[
            dict(
                type='ProcessRobocasaEvalInputs',
                img_key='video.ego_view_pad_res256_freq20',
                resize_size=224,
                normalize=True,
                embodiment_id=24,
            ),
            dict(type='RobocasaGR1N15Bridge'),
            dict(
                type='NormalizeImages',
                means=[0.5, 0.5, 0.5],
                stds=[0.5, 0.5, 0.5],
            ),
            dict(
                type='NormalizeStatesAndActions',
                action_dim=29,
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='min_max',
                normalize_states=False,
            ),
            dict(
                type='PrepareVideo',
                num_views=1,
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
        type='DenormalizeRobocasaAction',
        norm_type='min_max',
        action_dim=29,
        clip_actions=False,
        stats_order='fluxvla',
    ),
    manager=dict(
        output_dir=_eval_output_root,
        num_gpus=8,
        max_tasks_per_gpu=2,
    ),
)
