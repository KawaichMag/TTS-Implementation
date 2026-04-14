from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    project_dir: Path = Path(".")
    metadata_path: Path | None = None
    wav_dir: Path | None = None
    artifacts_dir: Path | None = None
    aligned_cache_path: Path | None = None

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: int = 0
    fmax: int = 8000

    hidden_size: int = 512
    encoder_dropout: float = 0.1
    attention_dropout: float = 0.1
    encoder_num_layers: int = 6
    encoder_num_heads: int = 8
    encoder_prenet_layers: int = 3
    encoder_kernel_size: int = 5
    encoder_ffn_kernel_size: int = 3
    encoder_ffn_multiplier: int = 4
    encoder_max_relative_position: int = 4
    duration_predictor_layers: int = 2
    duration_predictor_kernel_size: int = 3
    duration_predictor_filter_size: int = 256
    decoder_layers: int = 6
    decoder_base_channels: int = 64
    decoder_dropout: float = 0.1
    prior_layers: int = 3
    prior_kernel_size: int = 5
    detach_duration_predictor_input: bool = True
    diffusion_beta_min: float = 0.05
    diffusion_beta_max: float = 20.0
    diffusion_pe_scale: int = 1000
    diffusion_dim_mults: tuple[int, ...] = (1, 2, 4)
    diffusion_groups: int = 8
    diffusion_stochastic: bool = False
    diffusion_temperature_mode: str = "inverse"
    diffusion_steps: int = 80
    diffusion_x0_loss_weight: float = 0.05
    diffusion_sample_clamp_value: float = 4.0
    decoder_train_segment_frames: int = 192
    inference_temperature: float = 2.0
    length_scale: float = 1.0
    max_predicted_duration: int = 30

    batch_size: int = 2
    grad_accum_steps: int = 4
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    lr_warmup_epochs: int = 5
    ema_decay: float = 0.999
    use_ema: bool = True
    max_grad_norm: float = 1.0
    epochs: int = 20
    duration_loss_weight: float = 0.05
    prior_loss_weight: float = 1.0
    loader_workers: int = 0
    batch_log_every: int = 25
    max_train_batches: int = 0
    use_amp: bool = True

    val_ratio: float = 0.2
    max_items: int = 0
    max_infer_items: int = 12
    griffin_lim_iters: int = 48

    checkpoint_best_path: Path | None = None
    checkpoint_last_path: Path | None = None
    warm_start_checkpoint_path: Path | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
