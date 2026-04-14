import copy
import math
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from models import CompactSpeechSynthModel, Config
from utils.checkpointing import (
    apply_checkpoint_config,
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from utils.data_pipeline import build_loaders, prepare_items
from utils.io_paths import resolve_runtime_paths
from utils.logging_utils import log_epoch_summary, set_seed


def _build_optimizer_and_scheduler(model, config: Config, steps_per_epoch: int):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=float(getattr(config, "weight_decay", 0.0)),
    )

    total_optimizer_steps = max(steps_per_epoch * max(int(config.epochs), 1), 1)
    warmup_steps = (
        min(total_optimizer_steps - 1, steps_per_epoch * int(getattr(config, "lr_warmup_epochs", 0)))
        if total_optimizer_steps > 1
        else 0
    )
    min_lr = float(getattr(config, "min_learning_rate", config.learning_rate))
    min_lr_scale = min_lr / max(float(config.learning_rate), 1e-8)

    def lr_lambda(step):
        if total_optimizer_steps <= 1:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, min_lr_scale)
        progress_denom = max(total_optimizer_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / progress_denom, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if warmup_steps > 0:
        warmup_lr = float(config.learning_rate) * lr_lambda(0)
        for group in optimizer.param_groups:
            group["lr"] = warmup_lr
    return optimizer, scheduler


def _run_epoch(
    *,
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    ema_model,
    ema_decay: float,
    grad_accum_steps: int,
    max_train_batches: int,
    max_grad_norm: float,
    amp_enabled: bool,
    noise_levels,
    training: bool,
    desc: str,
):
    model.train(training)
    total_steps = len(loader)
    if training and max_train_batches > 0:
        total_steps = min(total_steps, max_train_batches)
    if total_steps <= 0:
        return {
            "total_loss": float("nan"),
            "l1": float("nan"),
            "diffusion": float("nan"),
            "duration": float("nan"),
            "prior": float("nan"),
            "updates": 0,
        }

    metrics_sum = {"total_loss": 0.0, "l1": 0.0, "diffusion": 0.0, "duration": 0.0, "prior": 0.0}
    metrics_count = 0
    optimizer_updates = 0
    iterator = iter(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(range(1, total_steps + 1), total=total_steps, desc=desc)
    for step in pbar:
        batch = next(iterator)
        autocast_device = "cuda" if noise_levels.device.type == "cuda" else "cpu"
        autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.bfloat16

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=amp_enabled):
                total_loss, l1_loss, diff_loss, dur_loss, prior_loss = model.compute_losses(batch, noise_levels)
                scaled_loss = total_loss / grad_accum_steps if training else total_loss

            if training:
                if amp_enabled:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = step % grad_accum_steps == 0 or step == total_steps
                if should_step:
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    optimizer_updates += 1
                    if ema_model is not None:
                        with torch.no_grad():
                            ema_params = dict(ema_model.named_parameters())
                            model_params = dict(model.named_parameters())
                            for name, param in model_params.items():
                                ema_params[name].mul_(ema_decay).add_(param.detach(), alpha=1.0 - ema_decay)
                            ema_buffers = dict(ema_model.named_buffers())
                            for name, buffer in model.named_buffers():
                                ema_buffers[name].copy_(buffer.detach())

        metrics_sum["total_loss"] += float(total_loss.item())
        metrics_sum["l1"] += float(l1_loss.item())
        metrics_sum["diffusion"] += float(diff_loss.item())
        metrics_sum["duration"] += float(dur_loss.item())
        metrics_sum["prior"] += float(prior_loss.item())
        metrics_count += 1

        pbar.set_postfix(
            total=f"{metrics_sum['total_loss']/metrics_count:.4f}",
            diff=f"{metrics_sum['diffusion']/metrics_count:.4f}",
            dur=f"{metrics_sum['duration']/metrics_count:.4f}",
            prior=f"{metrics_sum['prior']/metrics_count:.4f}",
            upd=optimizer_updates,
        )

    out = {key: value / max(metrics_count, 1) for key, value in metrics_sum.items()}
    out["updates"] = int(optimizer_updates)
    return out


def run_training(config: Config):
    set_seed(42)
    config = resolve_runtime_paths(config)

    warm_start_checkpoint = load_checkpoint(config.warm_start_checkpoint_path)
    if isinstance(warm_start_checkpoint, dict):
        apply_checkpoint_config(config, warm_start_checkpoint.get("config", {}))
    config = resolve_runtime_paths(config)

    tokenizer_state = warm_start_checkpoint.get("tokenizer") if isinstance(warm_start_checkpoint, dict) else None
    prepared = prepare_items(config, preset_symbol_to_id=tokenizer_state)
    train_loader, val_loader, _, _ = build_loaders(config, prepared.items, prepared.tokenizer.pad_id)

    model = CompactSpeechSynthModel.build(prepared.tokenizer.vocab_size, config, device=config.device)
    loaded, key, error = model.load_compatible_state(warm_start_checkpoint)
    if loaded:
        print(f"Loaded warm-start model state: {key}")
    elif error:
        print(f"Warm-start skipped: {error}")

    grad_accum_steps = max(int(getattr(config, "grad_accum_steps", 1)), 1)
    max_train_batches = max(int(getattr(config, "max_train_batches", 0) or 0), 0)
    effective_batches = min(len(train_loader), max_train_batches) if max_train_batches > 0 else len(train_loader)
    steps_per_epoch = max(math.ceil(effective_batches / grad_accum_steps), 1)

    optimizer, scheduler = _build_optimizer_and_scheduler(model, config, steps_per_epoch)
    amp_enabled = bool(getattr(config, "use_amp", False) and config.device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    use_ema = bool(getattr(config, "use_ema", True))
    ema_decay = float(getattr(config, "ema_decay", 0.999))
    ema_model = copy.deepcopy(model).eval() if use_ema else None
    if ema_model is not None:
        for param in ema_model.parameters():
            param.requires_grad_(False)

    noise_levels = model.make_noise_levels(device=config.device)
    history: list[dict] = []
    best_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema_model=ema_model,
            ema_decay=ema_decay,
            grad_accum_steps=grad_accum_steps,
            max_train_batches=max_train_batches,
            max_grad_norm=float(getattr(config, "max_grad_norm", 1.0)),
            amp_enabled=amp_enabled,
            noise_levels=noise_levels,
            training=True,
            desc=f"train epoch {epoch}/{config.epochs}",
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema_model=None,
            ema_decay=ema_decay,
            grad_accum_steps=grad_accum_steps,
            max_train_batches=0,
            max_grad_norm=float(getattr(config, "max_grad_norm", 1.0)),
            amp_enabled=amp_enabled,
            noise_levels=noise_levels,
            training=False,
            desc=f"val epoch {epoch}/{config.epochs}",
        )

        row = {
            "epoch": epoch,
            "train_total": train_metrics["total_loss"],
            "train_l1": train_metrics["l1"],
            "train_diffusion": train_metrics["diffusion"],
            "train_duration": train_metrics["duration"],
            "train_prior": train_metrics["prior"],
            "val_total": val_metrics["total_loss"],
            "val_l1": val_metrics["l1"],
            "val_diffusion": val_metrics["diffusion"],
            "val_duration": val_metrics["duration"],
            "val_prior": val_metrics["prior"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "updates": int(train_metrics["updates"]),
        }
        history.append(row)
        log_epoch_summary(epoch, {"total_loss": row["train_total"], "diffusion": row["train_diffusion"], "duration": row["train_duration"], "prior": row["train_prior"], "lr": row["lr"]}, prefix="train")
        log_epoch_summary(epoch, {"total_loss": row["val_total"], "diffusion": row["val_diffusion"], "duration": row["val_duration"], "prior": row["val_prior"], "lr": row["lr"]}, prefix="val")

        export_model = ema_model if ema_model is not None else model
        improved = row["val_total"] < best_loss
        if improved:
            best_loss = row["val_total"]
            best_payload = build_checkpoint_payload(
                config=config,
                epoch=epoch,
                best_loss=best_loss,
                history=history,
                model_state=export_model.state_dict(),
                tokenizer_symbol_to_id=prepared.tokenizer.symbol_to_id,
                mel_mean=prepared.mel_mean,
                mel_std=prepared.mel_std,
            )
            save_checkpoint(Path(config.checkpoint_best_path), best_payload)

        last_payload = build_checkpoint_payload(
            config=config,
            epoch=epoch,
            best_loss=best_loss,
            history=history,
            model_state=export_model.state_dict(),
            tokenizer_symbol_to_id=prepared.tokenizer.symbol_to_id,
            mel_mean=prepared.mel_mean,
            mel_std=prepared.mel_std,
            raw_model_state=model.state_dict(),
            ema_model_state=ema_model.state_dict() if ema_model is not None else None,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            scaler_state=scaler.state_dict() if amp_enabled else None,
        )
        save_checkpoint(Path(config.checkpoint_last_path), last_payload)

    history_df = pd.DataFrame(history)
    history_path = Path(config.artifacts_dir) / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved history: {history_path}")
    print(f"Best checkpoint: {config.checkpoint_best_path}")
    print(f"Last checkpoint: {config.checkpoint_last_path}")
    return history_df
