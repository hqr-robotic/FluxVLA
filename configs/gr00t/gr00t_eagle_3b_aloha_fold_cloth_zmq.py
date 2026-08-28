"""Serve the ALOHA-only fold-cloth checkpoint through FluxThemis ZMQ."""

import os

_base_ = ['./gr00t_eagle_3b_aloha_full_finetune_fold_cloth.py']

inference_model = dict(
    pretrained_name_or_path=os.environ.get(
        'GR00T_PRETRAINED_MODEL_PATH',
        '/mnt/data/oss/users/liyinhao/projects/GR00T-N1.5-3B',
    ))

# This checkpoint was trained with a single instruction. Keep the server-side
# dataset and denormalization pipeline identical to its saved config.
inference = dict(
    task_descriptions={
        '_delete_': True,
        '1': 'fold cloth',
    },
    task_suite_name='private',
    state_dim=14,
    action_chunk=32,
)

# FluxThemis reads only this section. This schema targets
# feat/hqr/enhance-real-robot at commit 9985c4e.
themis = dict(
    runner=dict(
        type='RealRobotBenchmarkRunner',
        environment=dict(
            type='AlohaEnvironment',
            allow_actuation=False,
            action_low=None,
            action_high=None,
            require_action_bounds=False,
            tasks={'1': 'fold cloth'},
            image_encoding='rgb8',
            # Match AlohaInferenceRunner/BaseInferenceRunner defaults used by
            # the source real-robot config.
            publish_rate=30,
            observation_timeout_s=5.0,
            img_front_topic='/camera_f/color/image_raw',
            img_left_topic='/camera_l/color/image_raw',
            img_right_topic='/camera_r/color/image_raw',
            puppet_arm_left_topic='/puppet/joint_left',
            puppet_arm_right_topic='/puppet/joint_right',
            puppet_arm_left_cmd_topic='/master/joint_left',
            puppet_arm_right_cmd_topic='/master/joint_right',
        ),
        model_client=dict(
            type='FluxVLAZMQModelClient',
            server_host='127.0.0.1',
            server_port=3333,
            timeout_s=30.0,
            unnorm_key='private',
            serializer='msgpack',
            compress=False,
            enable_profiling=True,
        ),
        evaluator=dict(type='SuccessRateEvaluator'),
        seed=7,
        episodes_per_task=1,
        max_rollout_seconds=120.0,
        max_episode_steps=10000,
        # None executes the complete 32-step action chunk, matching the source
        # AlohaInferenceRunner default.
        execute_horizon=None,
        stop_on_success=False,
        parallel_workers=1,
        task_ids=['1'],
        work_dir='fluxthemis/aloha_benchmarks',
        run_name='gr00t-fold-cloth-manual-001',
        resume=False,
        overwrite=False,
        overlay_alpha=0.5,
    ), )
