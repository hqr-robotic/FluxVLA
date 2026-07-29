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

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from fluxvla.engines import TRANSFORMS


class _LiberoD4RTIndex:
    """Process-local index and LRU cache for v1.1 sidecars."""

    SCHEMA_VERSION = 'libero-d4rt-v1.1'
    LABEL_SOURCE = 'opend4rt_pseudo_gt'

    def __init__(self,
                 root: str,
                 manifest_name: str = 'manifest.jsonl',
                 cache_size: int = 2) -> None:
        self.root = Path(root).expanduser().resolve()
        self.cache_size = max(0, int(cache_size))
        self.cache: OrderedDict[Path, Dict[str, np.ndarray]] = OrderedDict()
        self.records = self._load_manifest(self.root / manifest_name)

    @staticmethod
    def _suite_name(data_root: str) -> str:
        name = Path(data_root).name.lower()
        for suite in ('spatial', 'object', 'goal', '10', '90'):
            if name == f'libero_{suite}' or name.startswith(
                    f'libero_{suite}_'):
                return suite
        raise ValueError(f'Cannot infer LIBERO suite from {data_root}')

    def _load_manifest(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f'LIBERO-D4RT manifest not found: {path}')
        records = {}
        build_signature = None
        identities = set()
        with path.open('r', encoding='utf-8') as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                if record.get('schema_version') != self.SCHEMA_VERSION:
                    raise ValueError(
                        f'Unsupported LIBERO-D4RT schema at {path}:'
                        f'{line_number}: {record.get("schema_version")!r}')
                if record.get('label_source') != self.LABEL_SOURCE:
                    raise ValueError(
                        f'Unexpected label source at {path}:{line_number}')
                if not bool(record.get('complete', False)):
                    raise ValueError(
                        f'Incomplete sidecar record at {path}:{line_number}')
                signature = str(record.get('build_signature', ''))
                if len(signature) != 64:
                    raise ValueError(
                        f'Missing build signature at {path}:{line_number}')
                if build_signature is None:
                    build_signature = signature
                elif signature != build_signature:
                    raise ValueError(
                        f'Mixed build signatures in {path}:{line_number}')
                key = (str(record['suite']), int(record['episode_index']),
                       str(record['camera_key']))
                clip_start = int(record.get('clip_start', 0))
                clip_end = int(record.get('clip_end', record['length']))
                if clip_end - clip_start != int(record['length']):
                    raise ValueError(
                        f'Invalid clip coverage at {path}:{line_number}')
                identity = (*key, clip_start)
                if identity in identities:
                    raise ValueError(
                        f'Duplicate sidecar identity at {path}:{line_number}')
                identities.add(identity)
                records.setdefault(key, []).append(record)
        if not records:
            raise ValueError(f'LIBERO-D4RT manifest is empty: {path}')
        for values in records.values():
            values.sort(key=lambda record: int(record.get('clip_start', 0)))
        return records

    def _select_record(self, data_root: str, episode_index: int,
                       camera_key: str, frame_indices: np.ndarray) -> Dict:
        key = (self._suite_name(data_root), int(episode_index), camera_key)
        records = self.records.get(key)
        if records is None:
            raise KeyError(f'No LIBERO-D4RT records for {key}')
        first_frame = int(frame_indices.min())
        last_frame = int(frame_indices.max())
        covering = [
            record for record in records
            if int(record.get('clip_start', 0)) <= first_frame
            and int(record.get('clip_end', record['length'])) > last_frame
        ]
        if not covering:
            ranges = [(int(record.get('clip_start', 0)),
                       int(record.get('clip_end', record['length'])))
                      for record in records]
            raise ValueError(f'No LIBERO-D4RT clip for {key} covers frames '
                             f'{frame_indices.tolist()}; ranges={ranges}')
        return max(
            covering, key=lambda record: int(record.get('clip_start', 0)))

    def _load_arrays(self, relative_path: str) -> Dict[str, np.ndarray]:
        path = self.root / relative_path
        arrays = self.cache.pop(path, None)
        if arrays is not None:
            self.cache[path] = arrays
            return arrays
        with np.load(path, allow_pickle=False) as pack:
            arrays = {key: np.asarray(pack[key]) for key in pack.files}
        required = {
            'frame_index', 'track_uv_norm', 'track_xyz_relative',
            'visibility_logit', 'confidence_logit', 'valid_mask',
            'supervision_weight', 'point_group', 'scene_scale'
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f'Missing sidecar fields in {path}: {missing}')
        frame_index = np.asarray(arrays['frame_index'])
        uv = np.asarray(arrays['track_uv_norm'])
        xyz = np.asarray(arrays['track_xyz_relative'])
        valid = np.asarray(arrays['valid_mask'])
        weight = np.asarray(arrays['supervision_weight'])
        if frame_index.ndim != 1 or not np.issubdtype(frame_index.dtype,
                                                      np.integer):
            raise ValueError(f'Invalid frame_index in {path}')
        if frame_index.size > 1 and not np.all(np.diff(frame_index) == 1):
            raise ValueError(f'Non-contiguous frame_index in {path}')
        expected_shape = (frame_index.shape[0], uv.shape[1])
        if uv.shape != (*expected_shape, 2) or xyz.shape != (*expected_shape,
                                                             3):
            raise ValueError(f'Invalid dense track shape in {path}')
        if valid.shape != expected_shape or valid.dtype != np.bool_:
            raise ValueError(f'Invalid valid_mask in {path}')
        if weight.shape != expected_shape or not np.isfinite(weight).all():
            raise ValueError(f'Invalid supervision_weight in {path}')
        if not np.isfinite(uv).all() or not np.isfinite(xyz).all():
            raise ValueError(f'Non-finite dense track values in {path}')
        if self.cache_size > 0:
            self.cache[path] = arrays
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return arrays

    def load_window(self, data_root: str, episode_index: int, camera_key: str,
                    frame_indices: Iterable[int],
                    frame_valid_mask: Iterable[bool]) -> Dict[str, np.ndarray]:
        requested = np.asarray(list(frame_indices), dtype=np.int32)
        frame_valid = np.asarray(list(frame_valid_mask), dtype=np.bool_)
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError(
                'frame_indices must be one-dimensional and non-empty')
        if frame_valid.shape != requested.shape:
            raise ValueError('frame_valid_mask must match frame_indices')
        record = self._select_record(data_root, episode_index, camera_key,
                                     requested)
        arrays = self._load_arrays(str(record['sidecar_path']))
        stored_frames = np.asarray(arrays['frame_index'], dtype=np.int32)
        positions = np.searchsorted(stored_frames, requested)
        if (np.any(positions >= stored_frames.shape[0])
                or not np.array_equal(stored_frames[positions], requested)):
            raise ValueError(
                f'Frame identity mismatch in {record["sidecar_path"]}')

        anchor_relative = np.asarray(
            arrays['track_xyz_relative'], dtype=np.float32)
        xyz_relative = (
            anchor_relative[positions] - anchor_relative[positions[0]][None])
        valid = np.asarray(arrays['valid_mask'][positions], dtype=np.bool_)
        source_valid = valid[0].copy()
        valid &= source_valid[None]
        valid &= frame_valid[:, None]
        weight = np.asarray(
            arrays['supervision_weight'][positions], dtype=np.float32)
        weight *= source_valid[None].astype(np.float32)
        weight *= frame_valid[:, None].astype(np.float32)
        query_uv = np.asarray(
            arrays['track_uv_norm'][positions[0]], dtype=np.float32)
        visibility = 1.0 / (1.0 + np.exp(-np.clip(
            np.asarray(
                arrays['visibility_logit'][positions], dtype=np.float32),
            -30.0, 30.0)))
        confidence = 1.0 / (1.0 + np.exp(-np.clip(
            np.asarray(
                arrays['confidence_logit'][positions], dtype=np.float32),
            -30.0, 30.0)))
        return {
            'track_gt_frame_index':
            requested,
            'track_gt_frame_valid_mask':
            frame_valid,
            'track_gt_query_uv':
            query_uv,
            'track_gt_uv':
            np.asarray(arrays['track_uv_norm'][positions], dtype=np.float32),
            'track_gt_xyz_relative':
            xyz_relative.astype(np.float32),
            'track_gt_visibility':
            visibility.astype(np.float32),
            'track_gt_confidence':
            confidence.astype(np.float32),
            'track_gt_weight':
            weight,
            'track_gt_mask':
            valid,
            'track_gt_point_group':
            np.asarray(arrays['point_group'], dtype=np.uint8),
            'track_gt_scene_scale':
            np.asarray(arrays['scene_scale'], dtype=np.float32),
            'track_gt_anchor_frame_index':
            np.asarray(int(record.get('clip_start', 0)), dtype=np.int32),
        }


@TRANSFORMS.register_module()
class LoadLiberoD4RTTracks:
    """Attach aligned OpenD4RT pseudo-labels to a LIBERO sample.

    This transform must run before :class:`ProcessParquetInputs`. Normalized UV
    coordinates are invariant under the later resize-only image transform.
    """

    def __init__(self,
                 sidecar_root: str,
                 camera_key: str = 'observation.images.image',
                 manifest_name: str = 'manifest.jsonl',
                 cache_size: int = 2,
                 required: bool = True) -> None:
        self.sidecar_root = sidecar_root
        self.camera_key = camera_key
        self.manifest_name = manifest_name
        self.cache_size = int(cache_size)
        self.required = bool(required)
        self._index: Optional[_LiberoD4RTIndex] = None

    def _get_index(self) -> _LiberoD4RTIndex:
        if self._index is None:
            self._index = _LiberoD4RTIndex(
                self.sidecar_root,
                manifest_name=self.manifest_name,
                cache_size=self.cache_size)
        return self._index

    def __call__(self, data: Dict) -> Dict:
        frame_indices = data.get('frame_indices')
        frame_masks = data.get('frame_masks')
        if frame_indices is None or frame_masks is None:
            raise KeyError(
                'LoadLiberoD4RTTracks requires frame_indices and frame_masks')
        try:
            labels = self._get_index().load_window(
                data_root=data['data_root'],
                episode_index=int(data['episode_index']),
                camera_key=self.camera_key,
                frame_indices=frame_indices,
                frame_valid_mask=np.asarray(frame_masks, dtype=np.bool_),
            )
        except KeyError:
            if self.required:
                raise
            return data
        data.update(labels)
        return data
