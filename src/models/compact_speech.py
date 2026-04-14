from typing import Any

import torch

from grad_tts_diffusion import CompactSpeechSynth


class CompactSpeechSynthModel(CompactSpeechSynth):
    """Model wrapper with practical helpers for training/inference scripts."""

    @classmethod
    def build(cls, vocab_size: int, config, device: str | None = None):
        model = cls(vocab_size, config)
        if device is not None:
            model = model.to(device)
        return model

    def make_noise_levels(self, device: str | None = None):
        target_device = device if device is not None else next(self.parameters()).device
        return torch.arange(self.config.diffusion_steps, device=target_device)

    def compute_losses_for_batch(self, batch: dict, noise_levels=None):
        if noise_levels is None:
            noise_levels = self.make_noise_levels()
        return self.compute_losses(batch, noise_levels)

    def load_compatible_state(self, checkpoint: dict[str, Any] | None):
        if not isinstance(checkpoint, dict):
            return False, None, "checkpoint is not dict"
        candidate_keys = ("ema_model_state", "raw_model_state", "model_state")
        errors = []
        for key in candidate_keys:
            state = checkpoint.get(key)
            if not isinstance(state, dict):
                continue
            try:
                self.load_state_dict(state, strict=True)
                return True, key, None
            except Exception as err:
                errors.append(f"{key}: {str(err).splitlines()[0]}")
        if not errors:
            return False, None, "no model state found"
        return False, None, " | ".join(errors)
