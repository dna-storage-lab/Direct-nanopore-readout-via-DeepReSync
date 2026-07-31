from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_WATERMARK_PATH = Path("configureFiles") / "watermark_sequence_length_235"

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover - exercised only without a working torch install
    raise SystemExit(
        "PyTorch is required for training, but importing torch failed. "
        "Use a clean environment and install the CPU build, for example:\n"
        "  python -m venv .venv\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  python -m pip install --upgrade pip\n"
        "  python -m pip install numpy tqdm\n"
        "  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        f"Original error: {exc}"
    ) from exc

from training.utils.data import (
    OnlineHalfWatermarkDataset,
)
from training.utils.coding import SPARSE_ID_TO_5_BITS
from training.utils.simulator import SIMULATION_PROFILE_NAME
from training.model import MODEL_NAME, DeepResync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepResync.")
    parser.set_defaults(profile=SIMULATION_PROFILE_NAME)
    parser.add_argument("--watermark-path", default=str(DEFAULT_WATERMARK_PATH))
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument(
        "--resample-each-epoch",
        action="store_true",
        help="Draw a fresh deterministic simulated dataset each epoch.",
    )
    parser.add_argument("--max-read-len", type=int, default=280)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="AdamW beta1.",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.999,
        help="AdamW beta2.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine"],
        default="cosine",
        help="Learning-rate scheduler.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine annealing.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="DataLoader prefetch factor when --num-workers > 0.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/conv on Ampere+ NVIDIA GPUs.")
    parser.add_argument(
        "--amp",
        choices=["none", "fp16", "bf16"],
        default="none",
        help="Use automatic mixed precision on CUDA. fp16 uses GradScaler; bf16 does not.",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument(
        "--watermark-consistency-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for half-watermark consistency regularization: sparse-symbol "
            "posterior XOR watermark should agree with the transmitted upper-bit layer."
        ),
    )
    parser.add_argument(
        "--watermark-consistency-mode",
        choices=["base_upper", "symbol_upper", "symbol_watermark", "symbol_and_base"],
        default="symbol_upper",
        help=(
            "base_upper constrains the base-head upper-bit marginal against upper235. "
            "symbol_upper constrains symbol posterior XOR watermark against upper235. "
            "symbol_watermark constrains symbol posterior XOR upper235 against the known watermark. "
            "symbol_and_base also aligns the base-head upper marginal to that posterior."
        ),
    )
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--base-loss-weight",
        type=float,
        default=1.0,
        help="Weight for 235-position A/T/G/C base CE. Set to 0 for pure 423 information-posterior training.",
    )
    parser.add_argument("--aux-bit-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--lower-loss-weight",
        type=float,
        default=0.2,
        help="Weight for direct lower-bit auxiliary BCE.",
    )
    parser.add_argument(
        "--symbol-loss-weight",
        type=float,
        default=0.2,
        help="Weight for 47x16 sparse-symbol auxiliary CE.",
    )
    parser.add_argument(
        "--upper-info-loss-weight",
        type=float,
        default=0.2,
        help="Weight for upper188 information-bit BCE computed from the 47x16 symbol posterior.",
    )
    parser.add_argument("--symbol-hidden-size", type=int, default=0)
    parser.add_argument("--symbol-layers", type=int, default=1)
    parser.add_argument(
        "--log-detail",
        choices=["compact", "full"],
        default="compact",
        help="compact prints only key metrics; full prints all auxiliary diagnostics.",
    )

    parser.add_argument("--save-dir", default="runs/deepresync")
    return parser.parse_args()


def make_datasets(args: argparse.Namespace):
    if not args.watermark_path:
        raise ValueError("watermark path must not be empty.")
    train_ds = OnlineHalfWatermarkDataset(
        args.train_samples,
        args.watermark_path,
        args.max_read_len,
        seed=args.seed,
        jitter=args.jitter,
    )
    val_ds = OnlineHalfWatermarkDataset(
        args.val_samples,
        args.watermark_path,
        args.max_read_len,
        seed=args.seed + 10_000_000,
        jitter=args.jitter,
    )
    return train_ds, val_ds


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def bit_logits_from_base_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    upper_one = torch.logsumexp(logits[..., [2, 3]], dim=-1)
    upper_zero = torch.logsumexp(logits[..., [0, 1]], dim=-1)
    lower_one = torch.logsumexp(logits[..., [1, 3]], dim=-1)
    lower_zero = torch.logsumexp(logits[..., [0, 2]], dim=-1)
    return upper_one - upper_zero, lower_one - lower_zero


def unpack_model_output(
    model_output: torch.Tensor | dict[str, torch.Tensor],
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    if isinstance(model_output, dict):
        return model_output.get("base_logits"), model_output
    return model_output, {}


def sparse_codebook_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(SPARSE_ID_TO_5_BITS, device=device, dtype=dtype)


def symbol_id_bits(symbol_ids: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(3, -1, -1, device=symbol_ids.device, dtype=symbol_ids.dtype)
    return ((symbol_ids.unsqueeze(-1) >> shifts) & 1).float()


def symbol_logits_to_info_bit_probs(symbol_logits: torch.Tensor) -> torch.Tensor:
    symbol_probs = F.softmax(symbol_logits.float(), dim=-1)
    symbol_ids = torch.arange(16, device=symbol_logits.device, dtype=torch.long)
    info_bits = symbol_id_bits(symbol_ids).to(dtype=symbol_probs.dtype)
    bit_probs = torch.einsum("bgs,sk->bgk", symbol_probs, info_bits)
    return bit_probs.reshape(symbol_logits.shape[0], -1).clamp(1e-6, 1.0 - 1e-6)


def symbol_logits_to_sparse_bit_probs(symbol_logits: torch.Tensor) -> torch.Tensor:
    symbol_probs = F.softmax(symbol_logits.float(), dim=-1)
    codebook = sparse_codebook_tensor(symbol_logits.device, symbol_probs.dtype)
    sparse_probs = torch.einsum("bgs,sj->bgj", symbol_probs, codebook)
    return sparse_probs.reshape(symbol_logits.shape[0], -1).clamp(1e-6, 1.0 - 1e-6)


def watermark_upper_probs_from_symbol(
    symbol_logits: torch.Tensor,
    watermark_bits: torch.Tensor,
) -> torch.Tensor:
    sparse_probs = symbol_logits_to_sparse_bit_probs(symbol_logits)
    watermark = watermark_bits.to(dtype=sparse_probs.dtype)
    return (sparse_probs * (1.0 - watermark) + (1.0 - sparse_probs) * watermark).clamp(1e-6, 1.0 - 1e-6)


def watermark_probs_from_symbol_and_upper(
    symbol_logits: torch.Tensor,
    upper_bits: torch.Tensor,
) -> torch.Tensor:
    sparse_probs = symbol_logits_to_sparse_bit_probs(symbol_logits)
    upper = upper_bits.to(dtype=sparse_probs.dtype)
    return (sparse_probs * (1.0 - upper) + (1.0 - sparse_probs) * upper).clamp(1e-6, 1.0 - 1e-6)


def xor_bits(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.to(dtype=torch.long) ^ right.to(dtype=torch.long)).float()


def nearest_sparse_symbols(sparse_bits: torch.Tensor) -> torch.Tensor:
    """Hard de-sparsify 235 bits by nearest valid 5-bit sparse codeword."""

    grouped = sparse_bits.float().reshape(sparse_bits.shape[0], -1, 5)
    codebook = sparse_codebook_tensor(sparse_bits.device, grouped.dtype)
    distances = (grouped.unsqueeze(2) - codebook.view(1, 1, 16, 5)).abs().sum(dim=-1)
    return distances.argmin(dim=-1)


def info_metrics_from_predictions(
    pred_symbols: torch.Tensor,
    lower_pred_bits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    prefix: str = "",
) -> dict[str, float]:
    codebook = sparse_codebook_tensor(pred_symbols.device, lower_pred_bits.dtype)
    sparse_pred = codebook[pred_symbols].reshape_as(batch["upper_bits"])
    upper_tx_pred = xor_bits(sparse_pred, batch["watermark_bits"]).to(dtype=lower_pred_bits.dtype)
    base_pred = (2.0 * upper_tx_pred + lower_pred_bits).long()
    upper_info_pred = symbol_id_bits(pred_symbols).reshape_as(batch["upper_info_bits"])

    upper_acc = ((upper_info_pred >= 0.5).float() == batch["upper_info_bits"]).float().mean()
    lower_acc = (lower_pred_bits.float() == batch["lower_bits"]).float().mean()
    base_acc = (base_pred == batch["labels"]).float().mean()

    key = f"{prefix}_" if prefix else ""
    stats = {
        f"{key}upper188_acc": float(upper_acc.detach().cpu()),
        f"{key}upper188_err": float((1.0 - upper_acc).detach().cpu()),
        f"{key}lower235_acc": float(lower_acc.detach().cpu()),
        f"{key}lower235_err": float((1.0 - lower_acc).detach().cpu()),
        f"{key}info_base_acc": float(base_acc.detach().cpu()),
        f"{key}info_base_err": float((1.0 - base_acc).detach().cpu()),
    }
    return stats


def add_information_metrics(
    stats: dict[str, float],
    logits: torch.Tensor | None,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> None:
    """Add common upper188/lower235/re-encoded-base hard-decision metrics."""

    if "symbol_logits" in outputs and "lower_bit_logits" in outputs:
        pred_symbols = outputs["symbol_logits"].float().argmax(dim=-1)
        lower_pred = (outputs["lower_bit_logits"].float() > 0).float()
        stats.update(info_metrics_from_predictions(pred_symbols, lower_pred, batch))

    if logits is not None:
        upper_logit, lower_logit = bit_logits_from_base_logits(logits)
        upper_tx_pred = (upper_logit > 0).float()
        lower_pred = (lower_logit > 0).float()
        sparse_pred = xor_bits(upper_tx_pred, batch["watermark_bits"])
        pred_symbols = nearest_sparse_symbols(sparse_pred)
        base_stats = info_metrics_from_predictions(pred_symbols, lower_pred, batch, prefix="basepath")
        stats.update(base_stats)

        if "upper188_acc" not in stats:
            stats["upper188_acc"] = base_stats["basepath_upper188_acc"]
            stats["upper188_err"] = base_stats["basepath_upper188_err"]
        if "lower235_acc" not in stats:
            stats["lower235_acc"] = base_stats["basepath_lower235_acc"]
            stats["lower235_err"] = base_stats["basepath_lower235_err"]
        if "info_base_acc" not in stats:
            stats["info_base_acc"] = base_stats["basepath_info_base_acc"]
            stats["info_base_err"] = base_stats["basepath_info_base_err"]


def watermark_consistency_loss(
    logits: torch.Tensor | None,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if mode not in {"base_upper", "symbol_upper", "symbol_watermark", "symbol_and_base"}:
        raise ValueError(f"Unknown watermark consistency mode: {mode}")

    if mode == "base_upper":
        if logits is None:
            raise ValueError("watermark consistency mode base_upper requires base logits.")
        base_upper_logit, _ = bit_logits_from_base_logits(logits)
        base_upper_loss = F.binary_cross_entropy_with_logits(
            base_upper_logit,
            batch["upper_bits"],
        )
        return base_upper_loss, {
            "watermark_consistency_bce": float(base_upper_loss.detach().cpu()),
            "watermark_base_upper_bce": float(base_upper_loss.detach().cpu()),
            "watermark_upper_acc": float(
                ((base_upper_logit > 0).float() == batch["upper_bits"]).float().mean().detach().cpu()
            ),
        }

    if "symbol_logits" not in outputs:
        raise ValueError(
            "--watermark-consistency-loss-weight with symbol mode requires a model with symbol_logits."
        )

    if mode == "symbol_watermark":
        watermark_from_symbol = watermark_probs_from_symbol_and_upper(outputs["symbol_logits"], batch["upper_bits"])
        watermark_from_symbol_logits = torch.logit(watermark_from_symbol, eps=1e-6)
        watermark_targets = batch["watermark_bits"].to(
            device=watermark_from_symbol_logits.device,
            dtype=watermark_from_symbol_logits.dtype,
        )
        symbol_to_watermark_loss = F.binary_cross_entropy_with_logits(
            watermark_from_symbol_logits,
            watermark_targets,
        )
        return symbol_to_watermark_loss, {
            "watermark_consistency_bce": float(symbol_to_watermark_loss.detach().cpu()),
            "watermark_symbol_watermark_bce": float(symbol_to_watermark_loss.detach().cpu()),
            "watermark_upper_acc": float(
                ((watermark_from_symbol >= 0.5).float() == watermark_targets).float().mean().detach().cpu()
            ),
        }

    upper_from_symbol = watermark_upper_probs_from_symbol(outputs["symbol_logits"], batch["watermark_bits"])
    upper_from_symbol_logits = torch.logit(upper_from_symbol, eps=1e-6)
    symbol_to_upper_loss = F.binary_cross_entropy_with_logits(
        upper_from_symbol_logits,
        batch["upper_bits"],
    )

    loss = symbol_to_upper_loss
    stats = {
        "watermark_consistency_bce": float(loss.detach().cpu()),
        "watermark_symbol_upper_bce": float(symbol_to_upper_loss.detach().cpu()),
        "watermark_upper_acc": float(
            ((upper_from_symbol >= 0.5).float() == batch["upper_bits"]).float().mean().detach().cpu()
        ),
    }
    if mode == "symbol_and_base":
        if logits is None:
            raise ValueError("watermark consistency mode symbol_and_base requires base logits.")
        base_upper_logit, _ = bit_logits_from_base_logits(logits)
        base_consistency_loss = F.binary_cross_entropy_with_logits(
            base_upper_logit,
            upper_from_symbol.detach(),
        )
        loss = 0.5 * (symbol_to_upper_loss + base_consistency_loss)
        stats["watermark_consistency_bce"] = float(loss.detach().cpu())
        stats["watermark_base_consistency_bce"] = float(base_consistency_loss.detach().cpu())
    return loss, stats


def compute_loss(
    model_output: torch.Tensor | dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    base_weight: float,
    aux_weight: float,
    lower_weight: float,
    symbol_weight: float,
    upper_info_weight: float,
    watermark_consistency_weight: float,
    watermark_consistency_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, outputs = unpack_model_output(model_output)
    labels = batch["labels"]
    first_tensor = logits if logits is not None else next(iter(outputs.values()))
    loss = first_tensor.float().sum() * 0.0
    stats: dict[str, float] = {}

    needs_base_logits = (
        base_weight > 0
        or aux_weight > 0
        or (watermark_consistency_weight > 0 and watermark_consistency_mode in {"base_upper", "symbol_and_base"})
    )
    if logits is None and needs_base_logits:
        raise ValueError("This loss configuration requires base logits, but the model does not output a base head.")

    logits_for_loss = None
    if logits is not None:
        logits_for_loss = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        base_loss = F.cross_entropy(logits_for_loss.reshape(-1, 4), labels.reshape(-1))
        stats["base_ce"] = float(base_loss.detach().cpu())
        if base_weight > 0:
            loss = loss + base_weight * base_loss

    if aux_weight > 0:
        assert logits_for_loss is not None
        upper_logit, lower_logit = bit_logits_from_base_logits(logits_for_loss)
        upper_loss = F.binary_cross_entropy_with_logits(upper_logit, batch["upper_bits"])
        lower_loss = F.binary_cross_entropy_with_logits(lower_logit, batch["lower_bits"])
        bit_loss = 0.5 * (upper_loss + lower_loss)
        loss = loss + aux_weight * bit_loss
        stats["bit_bce"] = float(bit_loss.detach().cpu())

    if lower_weight > 0:
        if "lower_bit_logits" not in outputs:
            raise ValueError(
                "--lower-loss-weight requires a model with lower_bit_logits, "
                "such as v2_7_watermark_sparse_info_base."
            )
        lower_logits = outputs["lower_bit_logits"].float()
        lower_aux_loss = F.binary_cross_entropy_with_logits(lower_logits, batch["lower_bits"])
        loss = loss + lower_weight * lower_aux_loss
        stats["lower_bce"] = float(lower_aux_loss.detach().cpu())
        stats["lower_aux_acc"] = float(((lower_logits > 0).float() == batch["lower_bits"]).float().mean().detach().cpu())

    if symbol_weight > 0:
        if "symbol_logits" not in outputs:
            raise ValueError(
                "--symbol-loss-weight requires a model with symbol_logits, "
                "such as v2_7_watermark_sparse_info_base."
            )
        symbol_logits = outputs["symbol_logits"].float()
        symbol_ids = batch["symbol_ids"]
        symbol_loss = F.cross_entropy(symbol_logits.reshape(-1, 16), symbol_ids.reshape(-1))
        loss = loss + symbol_weight * symbol_loss
        stats["symbol_ce"] = float(symbol_loss.detach().cpu())

        pred_symbols = symbol_logits.argmax(dim=-1)
        stats["symbol_acc"] = float((pred_symbols == symbol_ids).float().mean().detach().cpu())

        codebook = sparse_codebook_tensor(symbol_logits.device, batch["upper_sparse_bits"].dtype)
        pred_sparse_bits = codebook[pred_symbols].reshape_as(batch["upper_sparse_bits"])
        stats["sparse_bit_acc"] = float(
            (pred_sparse_bits == batch["upper_sparse_bits"]).float().mean().detach().cpu()
        )
        stats["upper_info_bit_acc"] = float(
            (symbol_id_bits(pred_symbols) == symbol_id_bits(symbol_ids)).float().mean().detach().cpu()
        )

    if upper_info_weight > 0:
        if "symbol_logits" not in outputs:
            raise ValueError("--upper-info-loss-weight requires a model with symbol_logits.")
        upper_info_probs = symbol_logits_to_info_bit_probs(outputs["symbol_logits"])
        upper_info_logits = torch.logit(upper_info_probs, eps=1e-6)
        upper_info_loss = F.binary_cross_entropy_with_logits(upper_info_logits, batch["upper_info_bits"])
        loss = loss + upper_info_weight * upper_info_loss
        stats["upper_info_bce"] = float(upper_info_loss.detach().cpu())
        stats["upper_info_prob_acc"] = float(
            ((upper_info_probs >= 0.5).float() == batch["upper_info_bits"]).float().mean().detach().cpu()
        )

    if "symbol_upper_bit_logits" in outputs:
        symbol_upper_logits = outputs["symbol_upper_bit_logits"]
        stats["symbol_upper_acc"] = float(
            ((symbol_upper_logits > 0).float() == batch["upper_bits"]).float().mean().detach().cpu()
        )
    if "symbol_mix_weight" in outputs:
        stats["symbol_mix_weight"] = float(outputs["symbol_mix_weight"].float().mean().detach().cpu())

    if watermark_consistency_weight > 0:
        consistency_loss, consistency_stats = watermark_consistency_loss(
            logits_for_loss,
            outputs,
            batch,
            watermark_consistency_mode,
        )
        loss = loss + watermark_consistency_weight * consistency_loss
        stats.update(consistency_stats)

    if logits_for_loss is not None:
        pred = logits_for_loss.argmax(dim=-1)
        stats["base_acc"] = float((pred == labels).float().mean().detach().cpu())
    if "prior_base_logits" in outputs:
        prior_pred = outputs["prior_base_logits"].argmax(dim=-1)
        stats["prior_base_acc"] = float((prior_pred == labels).float().mean().detach().cpu())
    add_information_metrics(stats, logits_for_loss, outputs, batch)
    return loss, stats


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    base_weight: float,
    aux_weight: float,
    lower_weight: float,
    symbol_weight: float,
    upper_info_weight: float,
    watermark_consistency_weight: float,
    watermark_consistency_mode: str,
    amp: str,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    total_loss = 0.0
    total_batches = 0

    use_amp = device.startswith("cuda") and amp != "none"
    amp_dtype = torch.float16 if amp == "fp16" else torch.bfloat16

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            model_output = model(batch["read_tokens"], batch["watermark_bits"], batch["read_padding_mask"])
            loss, stats = compute_loss(
                model_output,
                batch,
                base_weight,
                aux_weight,
                lower_weight,
                symbol_weight,
                upper_info_weight,
                watermark_consistency_weight,
                watermark_consistency_mode,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_batches += 1
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value

    out = {"loss": total_loss / max(total_batches, 1)}
    out.update({key: value / max(total_batches, 1) for key, value in totals.items()})
    return out


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "metrics": metrics,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)


def count_parameters(model: torch.nn.Module) -> tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def main() -> None:
    args = parse_args()
    args.model_version = MODEL_NAME

    torch.manual_seed(args.seed)

    if args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.amp != "none" and not args.device.startswith("cuda"):
        print(f"AMP={args.amp} requested on device={args.device}; AMP will be disabled.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "config.json").open("w", encoding="utf-8") as fout:
        json.dump(vars(args), fout, indent=2)
    history_path = save_dir / "history.json"
    best_checkpoint_path = save_dir / "best.pt"
    last_checkpoint_path = save_dir / "last.pt"

    train_ds, val_ds = make_datasets(args)
    loader_kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model_kwargs = {
        "max_read_len": args.max_read_len,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    model_kwargs.update(symbol_layers=args.symbol_layers)
    if args.symbol_hidden_size > 0:
        model_kwargs.update(symbol_hidden_size=args.symbol_hidden_size)
    model = DeepResync(**model_kwargs).to(args.device)
    total_params, trainable_params, frozen_params = count_parameters(model)
    print(
        "\n"
        "================ DeepResync Model Size ================\n"
        f"Total parameters      : {total_params:,} ({total_params / 1_000_000:.3f} M)\n"
        f"Trainable parameters  : {trainable_params:,} ({trainable_params / 1_000_000:.3f} M)\n"
        f"Frozen parameters     : {frozen_params:,} ({frozen_params / 1_000_000:.3f} M)\n"
        "===================================================\n",
        flush=True,
    )
    print(
        f"Training precision: device={args.device}, tf32={'on' if args.tf32 else 'off'}, amp={args.amp}",
        flush=True,
    )
    print(f"Model version: {args.model_version}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    scheduler: Any | None = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )

    print(
        "Optimizer: "
        f"AdamW(lr={args.lr:.2e}, betas=({args.adam_beta1}, {args.adam_beta2}), "
        f"weight_decay={args.weight_decay:.2e}); "
        f"scheduler={args.lr_scheduler}, min_lr={args.min_lr:.2e}",
        flush=True,
    )
    use_amp = args.device.startswith("cuda") and args.amp != "none"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.amp == "fp16")

    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = float(optimizer.param_groups[0]["lr"])
        if args.resample_each_epoch:
            train_ds.set_epoch(epoch - 1)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            args.base_loss_weight,
            args.aux_bit_loss_weight,
            args.lower_loss_weight,
            args.symbol_loss_weight,
            args.upper_info_loss_weight,
            args.watermark_consistency_loss_weight,
            args.watermark_consistency_mode,
            args.amp,
            scaler,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            args.device,
            args.base_loss_weight,
            args.aux_bit_loss_weight,
            args.lower_loss_weight,
            args.symbol_loss_weight,
            args.upper_info_loss_weight,
            args.watermark_consistency_loss_weight,
            args.watermark_consistency_mode,
            args.amp,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        if args.log_detail == "compact":
            msg = (
                f"epoch {epoch:03d} "
                f"lr={current_lr:.2e} "
                f"time={epoch_seconds:.1f}s "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f}"
            )
            if "base_acc" in val_metrics:
                msg += f" val_acc={val_metrics['base_acc']:.4f}"
            print(msg, flush=True)
        else:
            watermark_msg = ""
            if args.watermark_consistency_loss_weight > 0:
                watermark_msg = (
                    f" train_wm={train_metrics.get('watermark_consistency_bce', 0.0):.4f}"
                    f" val_wm={val_metrics.get('watermark_consistency_bce', 0.0):.4f}"
                    f" val_wm_upper={val_metrics.get('watermark_upper_acc', 0.0):.4f}"
                )
            aux_msg = ""
            if args.symbol_loss_weight > 0:
                aux_msg += (
                    f" train_sym={train_metrics.get('symbol_acc', 0.0):.4f}"
                    f" val_sym={val_metrics.get('symbol_acc', 0.0):.4f}"
                    f" val_sparse={val_metrics.get('sparse_bit_acc', 0.0):.4f}"
                )
            if args.lower_loss_weight > 0:
                aux_msg += (
                    f" train_lower={train_metrics.get('lower_aux_acc', 0.0):.4f}"
                    f" val_lower={val_metrics.get('lower_aux_acc', 0.0):.4f}"
                )
            if args.upper_info_loss_weight > 0:
                aux_msg += (
                    f" train_upper188={train_metrics.get('upper_info_prob_acc', 0.0):.4f}"
                    f" val_upper188={val_metrics.get('upper_info_prob_acc', 0.0):.4f}"
                )
            if "upper188_err" in val_metrics:
                aux_msg += (
                    f" train_upper188_err={train_metrics.get('upper188_err', 0.0):.4f}"
                    f" val_upper188_err={val_metrics.get('upper188_err', 0.0):.4f}"
                )
            if "lower235_err" in val_metrics:
                aux_msg += (
                    f" train_lower235_err={train_metrics.get('lower235_err', 0.0):.4f}"
                    f" val_lower235_err={val_metrics.get('lower235_err', 0.0):.4f}"
                )
            if "info_base_err" in val_metrics:
                aux_msg += (
                    f" train_info_base_err={train_metrics.get('info_base_err', 0.0):.4f}"
                    f" val_info_base_err={val_metrics.get('info_base_err', 0.0):.4f}"
                )
            if "basepath_upper188_err" in val_metrics:
                aux_msg += (
                    f" val_basepath_upper188_err={val_metrics.get('basepath_upper188_err', 0.0):.4f}"
                    f" val_basepath_lower235_err={val_metrics.get('basepath_lower235_err', 0.0):.4f}"
                    f" val_basepath_info_base_err={val_metrics.get('basepath_info_base_err', 0.0):.4f}"
                )
            acc_msg = ""
            if "base_acc" in train_metrics and "base_acc" in val_metrics:
                acc_msg = f" train_acc={train_metrics['base_acc']:.4f} val_acc={val_metrics['base_acc']:.4f}"
            print(
                f"epoch {epoch:03d} "
                f"lr={current_lr:.2e} "
                f"time={epoch_seconds:.1f}s "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f}"
                f"{acc_msg}"
                f"{watermark_msg}"
                f"{aux_msg}",
                flush=True,
            )

        if scheduler is not None:
            scheduler.step()

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            args,
            epoch,
            row,
        )
        current_val_loss = float(val_metrics["loss"])
        if math.isfinite(current_val_loss) and (current_val_loss < best_val or not best_checkpoint_path.exists()):
            best_val = current_val_loss
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                args,
                epoch,
                row,
            )

        with history_path.open("w", encoding="utf-8") as fout:
            json.dump(history, fout, indent=2)


if __name__ == "__main__":
    main()
