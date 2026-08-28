"""Exercise the FluxVLA ZMQ endpoint without commanding an ALOHA robot."""

import argparse
import io
import json
from pathlib import Path
from typing import Any, Dict

import msgpack
import numpy as np
import zmq

from fluxvla.engines.runners.serving.serializers import (
    decode_predict_response, encode_predict_request)

DEFAULT_STATS_PATH = Path(
    '/root/projects/ryanhu/checkpoints/'
    'gr00t_flod_cloth_aloha_only/dataset_statistics.json')


def parse_args() -> argparse.Namespace:
    """Parse smoke-test command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Check an ALOHA FluxVLA ZMQ inference server.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=3333)
    parser.add_argument('--timeout-s', type=float, default=120.0)
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Also send a synthetic observation through the full model.')
    parser.add_argument(
        '--norm-stats-path', type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument('--image-height', type=int, default=480)
    parser.add_argument('--image-width', type=int, default=640)
    return parser.parse_args()


def build_synthetic_observation(norm_stats_path: Path, image_height: int,
                                image_width: int) -> Dict[str, Any]:
    """Build a shape-valid observation without reading robot sensors.

    Args:
        norm_stats_path: Statistics used to choose an in-distribution qpos.
        image_height: Synthetic camera image height.
        image_width: Synthetic camera image width.

    Returns:
        Raw ALOHA observation accepted by the server dataset pipeline.
    """
    with norm_stats_path.open('r', encoding='utf-8') as stats_file:
        stats = json.load(stats_file)
    qpos = np.asarray(stats['private']['proprio']['mean'], dtype=np.float32)
    if qpos.shape != (14, ):
        raise ValueError(f'Expected 14-D ALOHA qpos, got {qpos.shape}')
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    return {
        'cam_high': image,
        'cam_left_wrist': image,
        'cam_right_wrist': image,
        'qpos': qpos,
        'task_description': 'fold cloth',
    }


def main() -> None:
    """Ping the server and optionally run one synthetic prediction."""
    args = parse_args()
    if args.timeout_s <= 0:
        raise ValueError('--timeout-s must be positive')

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    timeout_ms = int(args.timeout_s * 1000)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(f'tcp://{args.host}:{args.port}')
    try:
        socket.send(msgpack.packb({'endpoint': 'ping'}))
        ping = msgpack.unpackb(socket.recv(), raw=False)
        if ping.get('status') != 'ok':
            raise RuntimeError(f'Unexpected ping response: {ping}')
        print(f'[smoke-test] ping OK: tcp://{args.host}:{args.port}')

        if not args.predict:
            return
        observation = build_synthetic_observation(
            args.norm_stats_path,
            args.image_height,
            args.image_width,
        )
        request = encode_predict_request(
            observation,
            unnorm_key='private',
            fmt='msgpack',
            # Match FluxThemis feat/hqr/enhance-real-robot exactly.
            compress=False,
        )
        socket.send(request)
        response = decode_predict_response(socket.recv())
        if 'error' in response:
            error = response['error']
            raise RuntimeError(f'ZMQ server error: {error}')
        actions = np.load(
            io.BytesIO(response['action_data']), allow_pickle=False)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(
                f'Expected actions with shape [T, 14], got {actions.shape}')
        if not np.isfinite(actions).all():
            raise ValueError('Server returned NaN or infinite actions')
        infer_time = float(response['infer_time'])
        print(f'[smoke-test] prediction OK: shape={actions.shape}, '
              f'dtype={actions.dtype}, '
              f'infer={infer_time:.3f}s')
    finally:
        socket.close(linger=0)
        context.term()


if __name__ == '__main__':
    main()
