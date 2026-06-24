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

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from PIL import Image

from fluxvla.engines import VLM_BACKBONES

__all__ = ['Wan22Backbone']


@VLM_BACKBONES.register_module()
class Wan22Backbone(nn.Module):
    """Wan2.2 encoding frontend for FastWAM: VAE + (optional) T5.

    This backbone owns the frozen encoders used by FastWAM:

    * ``vae`` – ``WanVideoVAE38`` (48-dim latent space), used to encode the
      observation video / conditioning image into latents and to decode
      predicted latents back to RGB frames.
    * ``text_encoder`` / ``tokenizer`` – the umt5-xxl T5 stack. During
      training FastWAM consumes pre-computed ``context`` embeddings, so the
      text encoder is optional and only required for prompt-based inference.

    The encoders are always frozen. Encoding helpers mirror the upstream
    ``fastwam.models.wan22.fastwam.FastWAM`` implementation verbatim so the
    split ``backbone`` + ``head`` pipeline stays numerically identical to the
    monolithic model.
    """

    def __init__(
        self,
        vae: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        tokenizer: Optional[object] = None,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
        freeze: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        if vae is None:
            raise ValueError('`Wan22Backbone` requires a `vae` module.')
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self._device = torch.device(device)
        self.torch_dtype = torch_dtype

        if freeze:
            self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        # Infer the live device from the (frozen) parameters/buffers so that a
        # later ``module.to(...)`` is reflected here. Falls back to the build
        # device when the backbone happens to hold no tensors.
        for tensor in self.parameters():
            return tensor.device
        for tensor in self.buffers():
            return tensor.device
        return self._device

    @property
    def temporal_downsample_factor(self) -> int:
        return int(self.vae.temporal_downsample_factor)

    def set_frozen_modules_to_eval_mode(self) -> None:
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()

    # ------------------------------------------------------------------
    # Prompt encoding (inference only; training uses cached ``context``)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                'Prompt encoding requires loaded text encoder/tokenizer. '
                'Set `load_text_encoder=True` or provide precomputed '
                '`context/context_mask`.')
        ids, mask = self.tokenizer(
            prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    @torch.no_grad()
    def encode_prompt_tokens(self, lang_tokens, lang_masks):
        """Encode pre-tokenized prompt ids into ``(context, context_mask)``.

        Mirrors :meth:`encode_prompt` but consumes ``lang_tokens`` /
        ``lang_masks`` produced upstream by a tokenizer transform (e.g. the
        shared ``LiberoPromptFromInputs``), so FastWAM can reuse the standard
        ``LiberoParquetEvalDataset`` eval batch. The padded-embedding zeroing
        and all-ones mask post-processing match :meth:`encode_prompt` exactly.
        """
        if self.text_encoder is None:
            raise ValueError(
                'Token encoding requires a loaded text encoder. Set '
                '`load_text_encoder=True`.')
        ids = lang_tokens.to(self.device)
        mask = lang_masks.to(self.device, dtype=torch.bool)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

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
