from pathlib import Path

from mmengine import Config


def test_gr00t_aloha_fluxthemis_config_matches_checkpoint_contract() -> None:
    """The deployment config preserves the checkpoint observation contract."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root / 'configs' / 'gr00t' /
        'gr00t_eagle_3b_aloha_fold_cloth_zmq.py')

    cfg = Config.fromfile(config_path)

    assert cfg.inference.task_descriptions == {'1': 'fold cloth'}
    assert cfg.inference.task_suite_name == 'private'
    assert cfg.inference.state_dim == 14
    assert cfg.inference.action_chunk == 32
    assert cfg.inference.dataset.img_keys == [
        'cam_high', 'cam_left_wrist', 'cam_right_wrist'
    ]
    assert cfg.inference.denormalize_action.type == (
        'DenormalizePrivateAction')
    assert cfg.inference.denormalize_action.action_dim == 14


def test_gr00t_aloha_fluxthemis_config_matches_real_robot_branch() -> None:
    """The Themis section matches the real-robot ZMQ client contract."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root / 'configs' / 'gr00t' /
        'gr00t_eagle_3b_aloha_fold_cloth_zmq.py')

    cfg = Config.fromfile(config_path)
    runner = cfg.themis.runner
    environment = runner.environment
    model_client = runner.model_client

    assert runner.type == 'RealRobotBenchmarkRunner'
    assert runner.execute_horizon is None
    assert runner.parallel_workers == 1
    assert runner.task_ids == ['1']
    assert environment.type == 'AlohaEnvironment'
    assert environment.allow_actuation is False
    assert environment.tasks == {'1': 'fold cloth'}
    assert environment.action_low is None
    assert environment.action_high is None
    assert environment.require_action_bounds is False
    assert environment.image_encoding == 'rgb8'
    assert environment.publish_rate == 30
    assert environment.img_front_topic == '/camera_f/color/image_raw'
    assert model_client == {
        'type': 'FluxVLAZMQModelClient',
        'server_host': '127.0.0.1',
        'server_port': 3333,
        'timeout_s': 30.0,
        'unnorm_key': 'private',
        'serializer': 'msgpack',
        'compress': False,
        'enable_profiling': True,
    }
