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

import fcntl
import hashlib
import inspect
import json
import os
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image

from fluxvla.engines import VLM_BACKBONES
from .wan_backbone import WanBaseBackbone

__all__ = ['Wan22Backbone']


@VLM_BACKBONES.register_module()
class Wan22Backbone(WanBaseBackbone):
    """Wan2.2 encoding frontend for FastWAM: VAE + optional T5 encoder.

    This backbone owns the frozen encoders used by FastWAM:

        ``vae`` encodes observation videos or conditioning images into latents
        and decodes predicted latents back to RGB frames. ``text_encoder`` is
        the optional umt5-xxl T5 encoder used when eval supplies tokenized
        prompts. Training usually consumes pre-computed ``context`` embeddings.

    The encoders are always frozen. Encoding helpers mirror the upstream
    ``fastwam.models.wan22.fastwam.FastWAM`` implementation verbatim so the
    split ``backbone`` + ``head`` pipeline stays numerically identical to the
    monolithic model.
    """

    frozen_module_names = ('text_encoder', 'vae')
    DEFAULT_TEXT_PROMPT = (
        "A video recorded from a robot's point of view executing the "
        'following instruction: {task}')
    HISTORICAL_TEXT_BATCH_SIZE = 5
    TEXT_CACHE_FORMAT_VERSION = 1

    def __init__(
        self,
        vae: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        text_embed_cache_dir: Optional[str] = None,
        text_embed_cache_context_len: int = 128,
        text_embed_cache_enc_id: str = 'wan22ti2v5b',
        text_embed_cache_size: int = 256,
        text_embed_cache_device: str = 'cpu',
        text_embed_prompt_template: Optional[str] = None,
        text_embed_cache_required: bool = False,
        text_encoder_checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
        freeze: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        if vae is None:
            raise ValueError('`Wan22Backbone` requires a `vae` module.')
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_embed_cache_dir = (
            None if text_embed_cache_dir is None else str(
                Path(text_embed_cache_dir).expanduser()))
        self.text_embed_cache_context_len = int(text_embed_cache_context_len)
        self.text_embed_cache_enc_id = str(text_embed_cache_enc_id)
        self.text_embed_cache_size = int(text_embed_cache_size)
        self.text_embed_cache_device = str(text_embed_cache_device).lower()
        self.text_embed_prompt_template = (
            text_embed_prompt_template or self.DEFAULT_TEXT_PROMPT)
        self.text_embed_cache_required = bool(text_embed_cache_required)
        self.text_encoder_checkpoint_path = (
            None if text_encoder_checkpoint_path is None else str(
                Path(text_encoder_checkpoint_path).expanduser()))
        self.tokenizer_path = (None if tokenizer_path is None else str(
            Path(tokenizer_path).expanduser()))
        if self.text_embed_cache_context_len <= 0:
            raise ValueError('`text_embed_cache_context_len` must be > 0.')
        if self.text_embed_cache_size < 0:
            raise ValueError('`text_embed_cache_size` must be >= 0.')
        if self.text_embed_cache_device not in {'cpu', 'model'}:
            raise ValueError(
                '`text_embed_cache_device` must be "cpu" or "model".')
        if (self.text_embed_cache_required
                and self.text_embed_cache_dir is None):
            raise ValueError(
                '`text_embed_cache_required=True` requires a cache '
                'directory.')
        self._text_embed_cache = OrderedDict()
        self._text_cache_token_fingerprints: Dict[str, str] = {}

        if freeze:
            self.freeze_encoder_modules()

    @property
    def temporal_downsample_factor(self) -> int:
        return int(self.vae.temporal_downsample_factor)

    # ------------------------------------------------------------------
    # Prompt token encoding (training usually uses cached ``context``)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tokenized prompts into ``(context, context_mask)``.

        Tokenization belongs to the data transform layer, matching
        :class:`Wan21Backbone`. This method only runs the frozen T5 encoder on
        ``input_ids`` / ``attention_mask`` and applies FastWAM's padded-token
        post-processing.
        """
        ids, mask = self._prepare_prompt_inputs(input_ids, attention_mask)
        return self._encode_prompt_fixed_batch(ids, mask)

    def _encode_prompt_fixed_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts with FastWAM's historical physical batch size.

        The original LIBERO cache was generated by eight ranks, each of
        which encoded exactly five prompts. T5 BF16 kernels are sensitive to
        physical batch shape, so every online call uses BS=5 as well. A short
        final chunk is deterministically padded by repeating its last real
        prompt; padded outputs are discarded.

        Args:
            input_ids: Token IDs, shape ``(B, L)``.
            attention_mask: Valid-token mask, shape ``(B, L)``.

        Returns:
            Context and context mask for the ``B`` real prompts.
        """
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError('Prompt token IDs and masks must both be 2D.')
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                'Prompt token IDs and masks must have identical shapes: '
                f'{tuple(input_ids.shape)} != '
                f'{tuple(attention_mask.shape)}.')
        batch_size = int(input_ids.shape[0])
        if batch_size == 0:
            raise ValueError('Cannot encode an empty prompt batch.')

        contexts = []
        context_masks = []
        physical_batch = self.HISTORICAL_TEXT_BATCH_SIZE
        for start in range(0, batch_size, physical_batch):
            end = min(start + physical_batch, batch_size)
            chunk_ids = input_ids[start:end]
            chunk_mask = attention_mask[start:end]
            real_size = end - start
            pad_size = physical_batch - real_size
            if pad_size > 0:
                repeat_shape = (pad_size, 1)
                chunk_ids = torch.cat(
                    [chunk_ids, chunk_ids[-1:].repeat(repeat_shape)], dim=0)
                chunk_mask = torch.cat(
                    [chunk_mask, chunk_mask[-1:].repeat(repeat_shape)], dim=0)
            chunk_context, chunk_context_mask = (
                self.encode_prompt_context(chunk_ids, chunk_mask))
            contexts.append(chunk_context[:real_size])
            context_masks.append(chunk_context_mask[:real_size])
        return torch.cat(contexts, dim=0), torch.cat(context_masks, dim=0)

    def _text_cache_path(self, cache_key: str) -> Optional[Path]:
        if self.text_embed_cache_dir is None:
            return None
        filename = (f'{cache_key}.t5_len{self.text_embed_cache_context_len}.'
                    f'{self.text_embed_cache_enc_id}.pt')
        return Path(self.text_embed_cache_dir) / filename

    @staticmethod
    def _file_sha256(path: Path) -> str:
        """Hash a file without materializing it in memory."""
        digest = hashlib.sha256()
        with path.open('rb') as file_handle:
            for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _path_identity(path: Path) -> Dict[str, Any]:
        """Return cheap file identity fields used to reuse a known digest."""
        stat = path.stat()
        return {
            'path': str(path.resolve()),
            'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns),
        }

    @classmethod
    def _fingerprint_file(
        cls,
        path_value: Optional[str],
        previous: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fingerprint one provenance file, reusing a matching digest."""
        if path_value is None:
            return {
                'available': False,
                'algorithm': 'sha256',
                'sha256': None,
                'path': None,
            }
        path = Path(path_value).expanduser()
        if not path.is_file():
            return {
                'available': False,
                'algorithm': 'sha256',
                'sha256': None,
                'path': str(path),
            }
        identity = cls._path_identity(path)
        if previous is not None:
            same_identity = all(
                previous.get(key) == value for key, value in identity.items())
            if same_identity and previous.get('sha256'):
                return dict(previous)
        return {
            **identity,
            'available': True,
            'algorithm': 'sha256',
            'sha256': cls._file_sha256(path),
        }

    @classmethod
    def _fingerprint_directory(
        cls,
        path_value: Optional[str],
        previous: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fingerprint every tokenizer artifact under a directory."""
        if path_value is None:
            return {
                'available': False,
                'algorithm': 'sha256',
                'sha256': None,
                'path': None,
                'files': [],
            }
        root = Path(path_value).expanduser()
        if not root.is_dir():
            return {
                'available': False,
                'algorithm': 'sha256',
                'sha256': None,
                'path': str(root),
                'files': [],
            }
        files = sorted(path for path in root.rglob('*') if path.is_file())
        identities = []
        for path in files:
            stat = path.stat()
            identities.append({
                'path': path.relative_to(root).as_posix(),
                'size': int(stat.st_size),
                'mtime_ns': int(stat.st_mtime_ns),
            })
        resolved_root = str(root.resolve())
        if previous is not None:
            same_identity = (
                previous.get('path') == resolved_root
                and previous.get('files') == identities)
            if same_identity and previous.get('sha256'):
                return dict(previous)

        digest = hashlib.sha256()
        for path, identity in zip(files, identities):
            digest.update(identity['path'].encode('utf-8'))
            digest.update(b'\0')
            with path.open('rb') as file_handle:
                for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024),
                                  b''):
                    digest.update(chunk)
            digest.update(b'\0')
        return {
            'available': True,
            'algorithm': 'sha256',
            'sha256': digest.hexdigest(),
            'path': resolved_root,
            'files': identities,
        }

    def _implementation_path(self) -> Optional[str]:
        """Resolve the concrete T5 implementation source file."""
        if self.text_encoder is None:
            return None
        source_path = inspect.getsourcefile(type(self.text_encoder))
        return None if source_path is None else str(source_path)

    def _build_text_cache_provenance(
        self,
        previous_manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build reproducibility fingerprints for the text cache."""
        previous_manifest = previous_manifest or {}
        checkpoint = self._fingerprint_file(
            self.text_encoder_checkpoint_path,
            previous_manifest.get('encoder_checkpoint'),
        )
        checkpoint['scope'] = 'checkpoint_containing_text_encoder'
        tokenizer = self._fingerprint_directory(
            self.tokenizer_path,
            previous_manifest.get('tokenizer'),
        )
        tokenizer['scope'] = 'recursive_tokenizer_artifacts'
        implementation = self._fingerprint_file(
            self._implementation_path(),
            previous_manifest.get('implementation'),
        )
        implementation['scope'] = 'text_encoder_module_source'
        return {
            'encoder_checkpoint': checkpoint,
            'tokenizer': tokenizer,
            'implementation': implementation,
        }

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor) -> str:
        """Hash a tensor's shape, dtype, and contiguous raw bytes."""
        value = tensor.detach().to(device='cpu').contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def _token_input_sha256(
        cls,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> str:
        """Bind a manifest entry to exact token IDs and mask bytes."""
        digest = hashlib.sha256()
        digest.update(cls._tensor_sha256(input_ids).encode('ascii'))
        digest.update(
            cls._tensor_sha256(attention_mask.bool()).encode('ascii'))
        return digest.hexdigest()

    def _validate_text_cache_manifest(
        self,
        manifest: Dict[str, Any],
        provenance: Dict[str, Any],
    ) -> None:
        """Reject attempts to mix incompatible prompt-cache artifacts."""
        expected = {
            'format_version': self.TEXT_CACHE_FORMAT_VERSION,
            'prompt_template': self.text_embed_prompt_template,
            'context_length': self.text_embed_cache_context_len,
            'physical_batch_size': self.HISTORICAL_TEXT_BATCH_SIZE,
            'padding_policy': 'repeat_last',
            'encoder_compute_dtype':
            str(self.torch_dtype).removeprefix('torch.'),
            'context_dtype': 'bfloat16',
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    'Text-cache manifest mismatch for '
                    f'{key}: {manifest.get(key)!r} != {value!r}.')
        for key in ('encoder_checkpoint', 'tokenizer', 'implementation'):
            recorded = manifest.get(key, {})
            current = provenance[key]
            if (recorded.get('available') and current.get('available')
                    and recorded.get('sha256') != current.get('sha256')):
                raise RuntimeError(
                    f'Text-cache manifest mismatch for {key} fingerprint.')

    def _record_text_cache_manifest(
        self,
        cache_key: str,
        prompt: Optional[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_path: Path,
    ) -> None:
        """Atomically add one prompt-to-context record to the manifest."""
        cache_dir = cache_path.parent
        manifest_path = cache_dir / 'manifest.json'
        lock_path = cache_dir / '.manifest.lock'
        with lock_path.open('a+', encoding='utf-8') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if manifest_path.exists():
                with manifest_path.open('r', encoding='utf-8') as file_handle:
                    manifest = json.load(file_handle)
            else:
                manifest = {}
            provenance = self._build_text_cache_provenance(manifest)
            if manifest:
                self._validate_text_cache_manifest(manifest, provenance)
            else:
                manifest = {
                    'format_version':
                    self.TEXT_CACHE_FORMAT_VERSION,
                    'cache_type':
                    'fastwam_t5_prompt_context',
                    'prompt_template':
                    self.text_embed_prompt_template,
                    'context_length':
                    self.text_embed_cache_context_len,
                    'physical_batch_size':
                    self.HISTORICAL_TEXT_BATCH_SIZE,
                    'padding_policy':
                    'repeat_last',
                    'encoder_compute_dtype':
                    str(self.torch_dtype).removeprefix('torch.'),
                    'context_dtype':
                    'bfloat16',
                    'entries': [],
                }
            manifest.update(provenance)
            entry = {
                'cache_key':
                cache_key,
                'prompt':
                prompt,
                'prompt_sha256': (None if prompt is None else hashlib.sha256(
                    prompt.encode('utf-8')).hexdigest()),
                'token_sha256':
                self._token_input_sha256(input_ids, attention_mask),
                'file':
                cache_path.name,
                'file_sha256':
                self._file_sha256(cache_path),
            }
            entries = {item['cache_key']: item for item in manifest['entries']}
            entries[cache_key] = entry
            manifest['entries'] = [entries[key] for key in sorted(entries)]
            temp_path = cache_dir / (
                f'.manifest.json.tmp.{os.getpid()}.{uuid.uuid4().hex}')
            try:
                with temp_path.open('w', encoding='utf-8') as file_handle:
                    json.dump(
                        manifest,
                        file_handle,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    file_handle.write('\n')
                os.replace(temp_path, manifest_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _verify_text_cache_manifest_entry(
        self,
        cache_key: str,
        prompt: Optional[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_path: Path,
    ) -> None:
        """Verify a persisted payload against its manifest when present."""
        manifest_path = cache_path.parent / 'manifest.json'
        if not manifest_path.exists():
            if self.text_embed_cache_required:
                raise RuntimeError(f'Required text-cache manifest is missing: '
                                   f'{manifest_path}.')
            return
        with manifest_path.open('r', encoding='utf-8') as file_handle:
            manifest = json.load(file_handle)
        provenance = self._build_text_cache_provenance(manifest)
        self._validate_text_cache_manifest(manifest, provenance)
        entries = {
            item['cache_key']: item
            for item in manifest.get('entries', [])
        }
        entry = entries.get(cache_key)
        if entry is None:
            raise RuntimeError(
                f'Text-cache manifest has no entry for {cache_key}.')
        expected_prompt = None if prompt is None else str(prompt)
        if entry.get('prompt') != expected_prompt:
            raise ValueError(f'Text-cache prompt mismatch for {cache_key}.')
        token_sha256 = self._token_input_sha256(input_ids, attention_mask)
        if entry.get('token_sha256') != token_sha256:
            raise ValueError(
                f'Text-cache token fingerprint mismatch for {cache_key}.')
        if entry.get('file') != cache_path.name:
            raise ValueError(f'Text-cache filename mismatch for {cache_key}.')
        if entry.get('file_sha256') != self._file_sha256(cache_path):
            raise ValueError(
                f'Text-cache payload fingerprint mismatch for {cache_key}.')

    @staticmethod
    def _token_cache_key(input_ids: torch.Tensor,
                         attention_mask: torch.Tensor) -> str:
        valid_ids = input_ids[attention_mask].detach().to(
            device='cpu', dtype=torch.int64).contiguous()
        return hashlib.sha256(valid_ids.numpy().tobytes()).hexdigest()

    def _resolve_text_cache_keys(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompts: Optional[Sequence[str]],
    ) -> list[str]:
        batch_size = int(input_ids.shape[0])
        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            prompts = list(prompts)
            if len(prompts) != batch_size:
                raise ValueError(
                    '`prompts` length must match the token batch size: '
                    f'{len(prompts)} != {batch_size}.')
            return [
                hashlib.sha256(str(prompt).encode('utf-8')).hexdigest()
                for prompt in prompts
            ]
        return [
            self._token_cache_key(input_ids[index], attention_mask[index])
            for index in range(batch_size)
        ]

    def _get_memory_cached_text(self, cache_key: str):
        cached = self._text_embed_cache.get(cache_key)
        if cached is None:
            return None
        self._text_embed_cache.move_to_end(cache_key)
        context, context_mask = cached
        return (
            context.to(device=self.device, dtype=self.torch_dtype),
            context_mask.to(device=self.device, dtype=torch.bool),
        )

    def _put_memory_cached_text(self, cache_key: str, context: torch.Tensor,
                                context_mask: torch.Tensor) -> None:
        if self.text_embed_cache_size == 0:
            return
        cache_device = (
            self.device if self.text_embed_cache_device == 'model' else
            torch.device('cpu'))
        self._text_embed_cache[cache_key] = (
            context.detach().to(device=cache_device).clone(),
            context_mask.detach().to(device=cache_device).clone(),
        )
        self._text_embed_cache.move_to_end(cache_key)
        while len(self._text_embed_cache) > self.text_embed_cache_size:
            self._text_embed_cache.popitem(last=False)

    def _load_disk_cached_text(
        self,
        cache_key: str,
        prompt: Optional[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        cache_path = self._text_cache_path(cache_key)
        if cache_path is None or not cache_path.exists():
            return None
        self._verify_text_cache_manifest_entry(
            cache_key=cache_key,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_path=cache_path,
        )
        payload = torch.load(cache_path, map_location='cpu', weights_only=True)
        context = payload['context']
        source_mask = payload['mask'].bool()
        expected_len = self.text_embed_cache_context_len
        if context.ndim != 2 or context.shape[0] != expected_len:
            raise ValueError(
                'Cached `context` must have shape [context_len, D], got '
                f'{tuple(context.shape)} in {cache_path}.')
        if source_mask.ndim != 1 or source_mask.shape[0] != expected_len:
            raise ValueError(
                'Cached `mask` must have shape [context_len], got '
                f'{tuple(source_mask.shape)} in {cache_path}.')
        context = context.to(device=self.device, dtype=self.torch_dtype)
        source_mask = source_mask.to(device=self.device, dtype=torch.bool)
        context = context.clone()
        context[~source_mask] = 0
        return context, torch.ones_like(source_mask)

    def _save_disk_cached_text(self, cache_key: str, context: torch.Tensor,
                               source_mask: torch.Tensor) -> None:
        cache_path = self._text_cache_path(cache_key)
        if cache_path is None or cache_path.exists():
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.parent / (
            f'.{cache_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}')
        payload = {
            'context':
            context.detach().to(device='cpu',
                                dtype=torch.bfloat16).contiguous(),
            'mask':
            source_mask.detach().to(device='cpu',
                                    dtype=torch.bool).contiguous(),
        }
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, cache_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @torch.no_grad()
    def encode_prompt_cached(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompts: Optional[Sequence[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode each unique prompt and reuse verified cache hits.

        Args:
            input_ids: Token IDs, shape ``(B, L)``.
            attention_mask: Valid-token mask, shape ``(B, L)``.
            prompts: Formatted prompts. Required for persistent caching.

        Returns:
            Context and all-ones context mask, shapes ``(B, L, D)`` and
            ``(B, L)``.

        Raises:
            FileNotFoundError: A strict cache is missing a requested prompt.
            ValueError: Prompt, token, or payload fingerprints do not match.
        """
        ids, mask = self._prepare_prompt_inputs(input_ids, attention_mask)
        if ids.shape[1] != self.text_embed_cache_context_len:
            raise ValueError(
                'Token sequence length must match '
                f'`text_embed_cache_context_len`: {ids.shape[1]} != '
                f'{self.text_embed_cache_context_len}.')
        caching_enabled = (
            self.text_embed_cache_size > 0
            or self.text_embed_cache_dir is not None)
        if not caching_enabled:
            return self._encode_prompt_fixed_batch(ids, mask)

        prompt_values = None
        if prompts is not None:
            prompt_values = [prompts] if isinstance(prompts,
                                                    str) else list(prompts)
        if self.text_embed_cache_dir is not None and prompt_values is None:
            raise ValueError(
                '`prompts` are required when reading or writing the '
                'persistent text cache.')
        cache_keys = self._resolve_text_cache_keys(ids, mask, prompt_values)
        request_token_fingerprints = [
            self._token_input_sha256(ids[index], mask[index])
            for index in range(len(cache_keys))
        ]
        request_by_key = {}
        for cache_key, token_fingerprint in zip(cache_keys,
                                                request_token_fingerprints):
            previous = request_by_key.get(cache_key)
            if previous is not None and previous != token_fingerprint:
                raise ValueError(
                    'One prompt maps to conflicting tokenizations in the '
                    f'same batch: {cache_key}.')
            known = self._text_cache_token_fingerprints.get(cache_key)
            if known is not None and known != token_fingerprint:
                raise ValueError('A cached prompt was requested with changed '
                                 f'tokenization: {cache_key}.')
            request_by_key[cache_key] = token_fingerprint
        resolved = [None] * len(cache_keys)
        missing_by_key = OrderedDict()
        newly_encoded = {}
        for index, cache_key in enumerate(cache_keys):
            cached = self._get_memory_cached_text(cache_key)
            if cached is None:
                prompt = (None if prompt_values is None else str(
                    prompt_values[index]))
                cached = self._load_disk_cached_text(
                    cache_key=cache_key,
                    prompt=prompt,
                    input_ids=ids[index],
                    attention_mask=mask[index],
                )
                if cached is not None:
                    self._put_memory_cached_text(cache_key, *cached)
            if cached is not None:
                resolved[index] = cached
                self._text_cache_token_fingerprints[cache_key] = (
                    request_token_fingerprints[index])
            elif cache_key not in missing_by_key:
                missing_by_key[cache_key] = index

        if missing_by_key:
            if self.text_embed_cache_required:
                missing_keys = ', '.join(missing_by_key)
                raise FileNotFoundError(
                    'Required text-cache payloads are missing for: '
                    f'{missing_keys}.')
            missing_keys = list(missing_by_key)
            missing_indices = [missing_by_key[key] for key in missing_keys]
            index_tensor = torch.tensor(
                missing_indices, device=ids.device, dtype=torch.long)
            missing_ids = ids.index_select(0, index_tensor)
            missing_mask = mask.index_select(0, index_tensor)
            contexts, context_masks = self._encode_prompt_fixed_batch(
                missing_ids, missing_mask)
            for offset, cache_key in enumerate(missing_keys):
                context = contexts[offset]
                context_mask = context_masks[offset]
                source_index = missing_indices[offset]
                newly_encoded[cache_key] = (context, context_mask)
                self._text_cache_token_fingerprints[cache_key] = (
                    request_token_fingerprints[source_index])
                self._put_memory_cached_text(cache_key, context, context_mask)
                self._save_disk_cached_text(cache_key, context,
                                            missing_mask[offset])
                cache_path = self._text_cache_path(cache_key)
                if cache_path is not None:
                    prompt = (None if prompt_values is None else str(
                        prompt_values[source_index]))
                    self._record_text_cache_manifest(
                        cache_key=cache_key,
                        prompt=prompt,
                        input_ids=missing_ids[offset],
                        attention_mask=missing_mask[offset],
                        cache_path=cache_path,
                    )

        for index, cache_key in enumerate(cache_keys):
            if resolved[index] is None:
                resolved[index] = newly_encoded.get(cache_key)
            if resolved[index] is None:
                resolved[index] = self._get_memory_cached_text(cache_key)
            if resolved[index] is None:
                raise RuntimeError(
                    f'Failed to resolve cached text embedding {cache_key}.')
        contexts, context_masks = zip(*resolved)
        return torch.stack(contexts), torch.stack(context_masks)

    @torch.no_grad()
    def encode_prompt_tokens(self, lang_tokens, lang_masks):
        """Backward-compatible alias for tokenized eval batches."""
        return self.encode_prompt(lang_tokens, lang_masks)

    # ------------------------------------------------------------------
    # Video / image latent encoding (deterministic; returns ``mu``)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_video_latents(
            self,
            video_tensor,
            tiled: bool = False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
    ):
        return self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @torch.no_grad()
    def encode_input_image_latents(
            self,
            input_image: torch.Tensor,
            tiled: bool = False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
    ):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (input_image.ndim != 4 or input_image.shape[0] != 1
                or input_image.shape[1] != 3):
            raise ValueError(
                '`input_image` must have shape [1,3,H,W] or [3,H,W], got '
                f'{tuple(input_image.shape)}')
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode(
            [image],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def prepare_context(self, context, context_mask):
        if context is None or context_mask is None:
            raise ValueError(
                '`context` and `context_mask` must be provided together.')
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True)
        return context, context_mask

    def decode_latents(
            self,
            latents,
            tiled: bool = False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
    ):
        video_tensor = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def forward(
            self,
            video: Optional[torch.Tensor] = None,
            input_image: Optional[torch.Tensor] = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            context: Optional[torch.Tensor] = None,
            context_mask: Optional[torch.Tensor] = None,
            latents: Optional[torch.Tensor] = None,
            tiled: bool = False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
    ):
        """Encode Wan2.2 inputs into FastWAM-ready tensors.

        Returns a dictionary keyed by the encoded products requested by the
        supplied inputs: ``input_latents`` for training videos,
        ``first_frame_latents`` for single-frame inference, ``context`` /
        ``context_mask`` for text conditioning, and ``video`` for decoded
        latents.
        """
        self.set_frozen_modules_to_eval_mode()
        outputs = {}

        if video is not None:
            outputs['input_latents'] = self.encode_video_latents(
                video,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        if input_image is not None:
            outputs['first_frame_latents'] = self.encode_input_image_latents(
                input_image,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )

        has_tokens = input_ids is not None or attention_mask is not None
        has_context = context is not None or context_mask is not None
        if has_tokens and has_context:
            raise ValueError(
                '`input_ids/attention_mask` and `context/context_mask` are '
                'mutually exclusive.')
        if has_tokens:
            if input_ids is None or attention_mask is None:
                raise ValueError(
                    '`input_ids` and `attention_mask` must be provided '
                    'together.')
            outputs['context'], outputs['context_mask'] = self.encode_prompt(
                input_ids, attention_mask)
        elif has_context:
            prepared_context, prepared_mask = self.prepare_context(
                context, context_mask)
            outputs['context'] = prepared_context
            outputs['context_mask'] = prepared_mask

        if latents is not None:
            outputs['video'] = self.decode_latents(
                latents,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        return outputs
