from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self\.use_nested_tensor is False because encoder_layer\.norm_first was True",
    category=UserWarning,
    module=r"torch\.nn\.modules\.transformer",
)

BLOCK_LENGTH = 235
UPPER_INFO_LENGTH = 188
POSTERIOR_LEN_423 = UPPER_INFO_LENGTH + BLOCK_LENGTH
POSTERIOR_MAGIC_423 = b"RPST423\0"
POSTERIOR_VERSION = 1
OLIGO_NUMS = 153 * 196
PAYLOAD_START = 45

SPARSE_5_TO_4 = {
    "00000": "0000",
    "00001": "0001",
    "00010": "0010",
    "00011": "0011",
    "00100": "0100",
    "00101": "0101",
    "00110": "0110",
    "11000": "0111",
    "01000": "1000",
    "01001": "1001",
    "01010": "1010",
    "10100": "1011",
    "01100": "1100",
    "10010": "1101",
    "10001": "1110",
    "10000": "1111",
}
SPARSE_4_TO_5 = {v: k for k, v in SPARSE_5_TO_4.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepResync on read-wise payloads and write read_posteriors.bin."
    )
    parser.add_argument("--reads", required=True, help="Primer-trimmed valid_reads.fa.")
    parser.add_argument("--index-results", required=True, help="index_identification_results.txt.")
    parser.add_argument("--output", required=True, help="Output read_posteriors.bin path.")
    parser.add_argument("--checkpoint", required=True, help="DeepResync checkpoint, usually best.pt or last.pt.")
    parser.add_argument("--watermark-path", required=True, help="watermark_sequence_length_235.")
    parser.add_argument("--reference-fasta", default=None, help="Noiseless encoded_sequence.fasta for hard-decision stats.")
    parser.add_argument("--pool", type=int, default=1, help="Pool index in [1,6], used to offset reference labels.")
    parser.add_argument("--payload-start", type=int, default=PAYLOAD_START)
    parser.add_argument("--per-read-stats", default=None, help="Output per-read hard-decision hamming table.")
    parser.add_argument("--hard-base-fasta", default=None, help="Output hard-decided 235-base payload sequences.")
    parser.add_argument("--stats-json", default=None, help="Output aggregate hard-decision statistics JSON.")
    parser.add_argument(
        "--deepresync-root",
        default=None,
        help="Path to the DeepResync project root. Defaults to DEEPRESYNC_ROOT or the repository root.",
    )
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-read-len", type=int, default=0, help="0 means read from checkpoint args.")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Default logit temperature before posterior export. T>1 softens probabilities.",
    )
    parser.add_argument(
        "--symbol-temperature",
        type=float,
        default=None,
        help="Temperature for 16-state sparse-symbol logits. Defaults to --temperature.",
    )
    parser.add_argument(
        "--lower-temperature",
        type=float,
        default=None,
        help="Temperature for lower-bit logits. Defaults to --temperature.",
    )
    parser.add_argument("--summary", default=None, help="Optional JSON summary output path.")
    return parser.parse_args()


def _is_deepresync_package_root(path: Path) -> bool:
    return (path / "training").is_dir()


def _version_roots_under(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and _is_deepresync_package_root(child):
            out.append(child)
    return out


def _resolve_from_candidate(candidate: Path, checkpoint_path: Path | None = None) -> Path | None:
    candidate = candidate.resolve()
    if _is_deepresync_package_root(candidate):
        return candidate

    if checkpoint_path is not None:
        checkpoint = checkpoint_path.resolve()
        try:
            checkpoint.relative_to(candidate)
            for parent in checkpoint.parents:
                if _is_deepresync_package_root(parent):
                    return parent
                if parent == candidate:
                    break
        except ValueError:
            pass

    version_roots = _version_roots_under(candidate)
    if len(version_roots) == 1:
        return version_roots[0].resolve()
    return None


def resolve_deepresync_root(value: str | None, checkpoint_path: Path | None = None) -> Path:
    if value:
        candidate = Path(value).resolve()
        resolved = _resolve_from_candidate(candidate, checkpoint_path)
        return resolved if resolved is not None else candidate

    env_value = None
    try:
        import os

        env_value = os.environ.get("DEEPRESYNC_ROOT")
    except Exception:
        env_value = None
    if env_value:
        candidate = Path(env_value).resolve()
        resolved = _resolve_from_candidate(candidate, checkpoint_path)
        return resolved if resolved is not None else candidate

    script_dir = Path(__file__).resolve().parent
    embedded_root = script_dir.parents[1]
    candidates = [embedded_root]
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = _resolve_from_candidate(candidate, checkpoint_path)
        if resolved is not None:
            return resolved
    return embedded_root.resolve()


def add_deepresync_to_path(deepresync_root: Path) -> None:
    if not (deepresync_root / "training").is_dir():
        version_roots = _version_roots_under(deepresync_root)
        hint = ""
        if version_roots:
            hint = " Found version directories: " + ", ".join(str(path) for path in version_roots)
        raise FileNotFoundError(
            f"Cannot find the training package under {deepresync_root}. "
            "Set --deepresync-root/DEEPRESYNC_ROOT to a directory containing training/."
            f"{hint}"
        )
    sys.path.insert(0, str(deepresync_root))


def torch_load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_watermark_bits(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    bits = [1 if ch == "1" else 0 for ch in text if ch in "01"]
    if len(bits) < BLOCK_LENGTH:
        raise ValueError(f"{path} contains only {len(bits)} bits; expected {BLOCK_LENGTH}.")
    return bits[:BLOCK_LENGTH]


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(chunks)
                current_id = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        records[current_id] = "".join(chunks)
    return records


def read_fasta_ordered(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(chunks)))
                current_id = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        records.append((current_id, "".join(chunks)))
    return records


def iter_payload_records(index_path: Path, seq_map: dict[str, str]) -> Iterable[tuple[int, str]]:
    with index_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin)
        for row in reader:
            if len(row) < 3:
                continue
            read_id, index_len_text, index_text = row[0], row[1], row[2]
            if index_text == "-1":
                continue
            try:
                pos_in_codeword = int(index_text)
                index_len = int(index_len_text)
            except ValueError:
                continue
            if pos_in_codeword < 1 or pos_in_codeword > OLIGO_NUMS:
                continue
            full_seq = seq_map.get(read_id)
            if full_seq is None:
                continue
            if index_len < 0 or index_len > len(full_seq):
                continue
            yield pos_in_codeword, read_id, full_seq[index_len:]


def payload_base_to_bits(payload: str) -> tuple[str, str]:
    upper: list[str] = []
    lower: list[str] = []
    for base in payload:
        if base == "A":
            upper.append("0")
            lower.append("0")
        elif base == "T":
            upper.append("0")
            lower.append("1")
        elif base == "G":
            upper.append("1")
            lower.append("0")
        elif base == "C":
            upper.append("1")
            lower.append("1")
        else:
            upper.append("0")
            lower.append("0")
    return "".join(upper), "".join(lower)


def xor_bits(a: str, b: str) -> str:
    return "".join("1" if x != y else "0" for x, y in zip(a, b))


def sparse45_decode_hard(sparse_bits: str) -> str:
    out: list[str] = []
    for i in range(0, len(sparse_bits), 5):
        group = sparse_bits[i : i + 5]
        if group in SPARSE_5_TO_4:
            out.append(SPARSE_5_TO_4[group])
            continue
        # A hard-decided noisy group can be outside the legal sparse code.
        # Use nearest legal code so every read still has a 188-bit decision.
        nearest = min(
            SPARSE_5_TO_4,
            key=lambda code: (sum(x != y for x, y in zip(group, code)), code),
        )
        out.append(SPARSE_5_TO_4[nearest])
    return "".join(out)


def sparse45_decode_probs_to_hard_info_bits(sparse_p1: Any) -> str:
    import numpy as np

    probs = np.asarray(sparse_p1, dtype=float)
    out: list[str] = []
    codewords = list(SPARSE_5_TO_4)
    for start in range(0, BLOCK_LENGTH, 5):
        p_group = probs[start : start + 5]
        code_probs = []
        for code in codewords:
            p = 1.0
            for bit, p1 in zip(code, p_group):
                p = p * (p1 if bit == "1" else (1.0 - p1))
            code_probs.append(p)
        denom = sum(code_probs)
        if denom <= 0.0:
            code_probs = [1.0 / len(code_probs)] * len(code_probs)
        else:
            code_probs = [p / denom for p in code_probs]

        bit_probs = [0.0, 0.0, 0.0, 0.0]
        for code, p in zip(codewords, code_probs):
            info = SPARSE_5_TO_4[code]
            for idx, bit in enumerate(info):
                if bit == "1":
                    bit_probs[idx] += p
        out.extend("1" if p >= 0.5 else "0" for p in bit_probs)
    return "".join(out)


def load_reference_labels(reference_fasta: Path, watermark_bits: list[int], payload_start: int) -> dict[str, list[str]]:
    watermark = "".join("1" if bit else "0" for bit in watermark_bits)
    payloads: list[str] = []
    upper235: list[str] = []
    lower235: list[str] = []
    upper188: list[str] = []

    for _, full_seq in read_fasta_ordered(reference_fasta):
        payload = full_seq[payload_start : payload_start + BLOCK_LENGTH]
        if len(payload) != BLOCK_LENGTH:
            continue
        upper, lower = payload_base_to_bits(payload)
        upper_sparse = xor_bits(upper, watermark)
        payloads.append(payload)
        upper235.append(upper)
        lower235.append(lower)
        upper188.append(sparse45_decode_hard(upper_sparse))

    return {
        "payload": payloads,
        "upper235": upper235,
        "lower235": lower235,
        "upper188": upper188,
    }


def bits_to_payload_bases(upper_bits: str, lower_bits: str) -> str:
    bases: list[str] = []
    for upper, lower in zip(upper_bits, lower_bits):
        if upper == "0" and lower == "0":
            bases.append("A")
        elif upper == "0" and lower == "1":
            bases.append("T")
        elif upper == "1" and lower == "0":
            bases.append("G")
        else:
            bases.append("C")
    return "".join(bases)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: str):
    from training.model import MODEL_NAME, DeepResync

    ckpt_args = checkpoint.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    version = MODEL_NAME
    model_kwargs = {
        "max_read_len": int(ckpt_args.get("max_read_len", 256)),
        "d_model": int(ckpt_args.get("d_model", 128)),
        "nhead": int(ckpt_args.get("nhead", 8)),
        "encoder_layers": int(ckpt_args.get("encoder_layers", 3)),
        "decoder_layers": int(ckpt_args.get("decoder_layers", 3)),
        "dim_feedforward": int(ckpt_args.get("dim_feedforward", 512)),
        "dropout": float(ckpt_args.get("dropout", 0.1)),
    }
    symbol_hidden_size = int(ckpt_args.get("symbol_hidden_size", 0))
    model_kwargs.update(symbol_layers=int(ckpt_args.get("symbol_layers", 1)))
    if symbol_hidden_size > 0:
        model_kwargs.update(symbol_hidden_size=symbol_hidden_size)

    model = DeepResync(**model_kwargs)
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    if len(compatible) != len(source_state):
        target_state.update(compatible)
        model.load_state_dict(target_state)
    else:
        model.load_state_dict(source_state)
    model.to(device)
    model.eval()
    return model, version, model_kwargs


_READ_BYTE_LOOKUP = None


def _read_byte_lookup():
    global _READ_BYTE_LOOKUP
    if _READ_BYTE_LOOKUP is not None:
        return _READ_BYTE_LOOKUP

    import numpy as np
    from training.utils.data import PAD_TOKEN_ID, READ_TOKEN_TO_ID

    lookup = np.full(256, PAD_TOKEN_ID, dtype=np.int64)
    for base, token_id in READ_TOKEN_TO_ID.items():
        lookup[ord(base)] = int(token_id)
        lookup[ord(base.lower())] = int(token_id)
    _READ_BYTE_LOOKUP = lookup
    return lookup


def encode_reads(reads: list[str], max_read_len: int):
    import numpy as np
    import torch
    from training.utils.data import PAD_TOKEN_ID

    lookup = _read_byte_lookup()
    tokens_np = np.full((len(reads), max_read_len), PAD_TOKEN_ID, dtype=np.int64)
    lengths = np.zeros(len(reads), dtype=np.int32)
    for row, read in enumerate(reads):
        raw = read[:max_read_len].encode("ascii", "replace")
        if not raw:
            continue
        encoded = lookup[np.frombuffer(raw, dtype=np.uint8)]
        tokens_np[row, : encoded.size] = encoded
        lengths[row] = encoded.size

    positions = np.arange(max_read_len, dtype=np.int32)
    mask_np = positions[None, :] >= lengths[:, None]
    return torch.from_numpy(tokens_np), torch.from_numpy(mask_np)


def validate_temperature(name: str, value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    return value


def symbol_logits_to_upper188_probs(symbol_logits, temperature: float = 1.0) -> Any:
    import torch

    temperature = validate_temperature("symbol temperature", temperature)
    symbol_probs = torch.softmax(symbol_logits.float() / temperature, dim=-1)
    symbol_ids = torch.arange(16, device=symbol_logits.device, dtype=torch.long)
    shifts = torch.arange(3, -1, -1, device=symbol_logits.device, dtype=torch.long)
    info_bits = ((symbol_ids.unsqueeze(-1) >> shifts) & 1).to(dtype=symbol_probs.dtype)
    upper188 = torch.einsum("bgs,sk->bgk", symbol_probs, info_bits).reshape(symbol_logits.shape[0], -1)
    return upper188.clamp(1e-9, 1.0 - 1e-9)


def model_output_to_posteriors(
    model_output,
    *,
    symbol_temperature: float = 1.0,
    lower_temperature: float = 1.0,
) -> Any:
    import torch

    if not isinstance(model_output, dict) or "symbol_logits" not in model_output or "lower_bit_logits" not in model_output:
        raise ValueError("The model must output symbol_logits and lower_bit_logits for the 423-bit posterior.")
    upper188 = symbol_logits_to_upper188_probs(model_output["symbol_logits"], symbol_temperature)
    lower_temperature = validate_temperature("lower temperature", lower_temperature)
    lower = torch.sigmoid(model_output["lower_bit_logits"].float() / lower_temperature).clamp(1e-9, 1.0 - 1e-9)
    return torch.cat([upper188, lower], dim=-1).clamp(1e-9, 1.0 - 1e-9)


def write_posterior_file(path: Path, records: list[tuple[int, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fout:
        fout.write(POSTERIOR_MAGIC_423)
        fout.write(struct.pack("<I", POSTERIOR_VERSION))
        fout.write(struct.pack("<I", POSTERIOR_LEN_423))
        fout.write(struct.pack("<Q", len(records)))
        for pos, posterior in records:
            arr = posterior.astype("<f8", copy=False)
            if arr.size != POSTERIOR_LEN_423:
                raise ValueError(f"Expected posterior length {POSTERIOR_LEN_423}, got {arr.size}.")
            fout.write(struct.pack("<i", int(pos)))
            fout.write(arr.tobytes(order="C"))


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    if args.pool < 1:
        raise ValueError(f"--pool must be >= 1, got {args.pool}.")

    checkpoint_path = Path(args.checkpoint)
    deepresync_root = resolve_deepresync_root(args.deepresync_root, checkpoint_path)
    add_deepresync_to_path(deepresync_root)

    import torch

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    model, version, model_kwargs = build_model_from_checkpoint(checkpoint, device)
    max_read_len = args.max_read_len if args.max_read_len > 0 else int(model_kwargs["max_read_len"])
    default_temperature = validate_temperature("--temperature", args.temperature)
    symbol_temperature = validate_temperature(
        "--symbol-temperature",
        default_temperature if args.symbol_temperature is None else args.symbol_temperature,
    )
    lower_temperature = validate_temperature(
        "--lower-temperature",
        default_temperature if args.lower_temperature is None else args.lower_temperature,
    )
    watermark_bits = load_watermark_bits(Path(args.watermark_path))
    watermark_text = "".join("1" if bit else "0" for bit in watermark_bits)
    watermark = torch.tensor(watermark_bits, dtype=torch.long, device=device)

    seq_map = read_fasta(Path(args.reads))
    payload_records = list(iter_payload_records(Path(args.index_results), seq_map))
    reference_labels = None
    if args.reference_fasta:
        reference_labels = load_reference_labels(Path(args.reference_fasta), watermark_bits, args.payload_start)
    need_reference_stats = reference_labels is not None
    need_hard_base = bool(args.hard_base_fasta)
    need_hard_decisions = need_reference_stats or need_hard_base

    # print(
    #     "DeepResync inference setup: "
    #     f"version={version}, device={device}, amp={args.amp}, "
    #     f"temperature={default_temperature:g}, "
    #     f"symbol_temperature={symbol_temperature:g}, "
    #     f"lower_temperature={lower_temperature:g}, "
    #     f"batch_size={args.batch_size}, max_read_len={max_read_len}, "
    #     f"input_reads={len(seq_map)}, indexed_payload_reads={len(payload_records)}",
    #     flush=True,
    # )

    out_records: list[tuple[int, Any]] = []
    per_read_rows: list[dict[str, Any]] = []
    hard_base_records: list[tuple[str, str]] = []
    totals = {
        "upper235_errors": 0,
        "upper188_errors": 0,
        "lower235_errors": 0,
        "base235_errors": 0,
        "compared_reads": 0,
    }
    use_amp = device.startswith("cuda") and args.amp != "none"
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    timings = {
        "encode_seconds": 0.0,
        "transfer_seconds": 0.0,
        "forward_seconds": 0.0,
        "cpu_post_seconds": 0.0,
        "write_seconds": 0.0,
    }
    batch_count = 0

    with torch.inference_mode():
        for start_idx in range(0, len(payload_records), args.batch_size):
            batch_count += 1
            chunk = payload_records[start_idx : start_idx + args.batch_size]
            positions = [pos for pos, _, _ in chunk]
            read_ids = [read_id for _, read_id, _ in chunk]
            reads = [read for _, _, read in chunk]

            t0 = time.perf_counter()
            tokens, padding_mask = encode_reads(reads, max_read_len)
            timings["encode_seconds"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            tokens = tokens.to(device)
            padding_mask = padding_mask.to(device)
            wm = watermark.unsqueeze(0).expand(len(reads), -1)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings["transfer_seconds"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                model_output = model(tokens, wm, padding_mask)
                posteriors = model_output_to_posteriors(
                    model_output,
                    symbol_temperature=symbol_temperature,
                    lower_temperature=lower_temperature,
                )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings["forward_seconds"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            post_np = posteriors.cpu().numpy()
            for pos, read_id, posterior in zip(positions, read_ids, post_np):
                out_records.append((pos, posterior))

                if need_hard_decisions:
                    upper188_probs = posterior[:UPPER_INFO_LENGTH]
                    lower_probs = posterior[UPPER_INFO_LENGTH:]
                    pred_upper188 = "".join("1" if p >= 0.5 else "0" for p in upper188_probs)
                    pred_upper_sparse = "".join(
                        SPARSE_4_TO_5[pred_upper188[i : i + 4]]
                        for i in range(0, UPPER_INFO_LENGTH, 4)
                    )
                    pred_upper235 = xor_bits(pred_upper_sparse, watermark_text)
                    pred_lower235 = "".join("1" if p >= 0.5 else "0" for p in lower_probs)
                    pred_base235 = bits_to_payload_bases(pred_upper235, pred_lower235)
                    if need_hard_base:
                        hard_base_records.append((f"{read_id}|pos={pos}", pred_base235))

                if need_reference_stats:
                    ref_idx = pos - 1 + (args.pool - 1) * OLIGO_NUMS
                    if 0 <= ref_idx < len(reference_labels["payload"]):
                        err_upper235 = hamming(pred_upper235, reference_labels["upper235"][ref_idx])
                        err_upper188 = hamming(pred_upper188, reference_labels["upper188"][ref_idx])
                        err_lower235 = hamming(pred_lower235, reference_labels["lower235"][ref_idx])
                        err_base235 = hamming(pred_base235, reference_labels["payload"][ref_idx])

                        totals["upper235_errors"] += err_upper235
                        totals["upper188_errors"] += err_upper188
                        totals["lower235_errors"] += err_lower235
                        totals["base235_errors"] += err_base235
                        totals["compared_reads"] += 1

                        per_read_rows.append(
                            {
                                "pos_in_codeword": pos,
                                "read_id": read_id,
                                "upper235_errors": err_upper235,
                                "upper235_rate": err_upper235 / BLOCK_LENGTH,
                                "upper188_errors": err_upper188,
                                "upper188_rate": err_upper188 / UPPER_INFO_LENGTH,
                                "lower235_errors": err_lower235,
                                "lower235_rate": err_lower235 / BLOCK_LENGTH,
                                "base235_errors": err_base235,
                                "base235_rate": err_base235 / BLOCK_LENGTH,
                            }
                        )
            timings["cpu_post_seconds"] += time.perf_counter() - t0

            if len(out_records) % 5000 < len(chunk):
                elapsed = time.perf_counter() - start
                rate = len(out_records) / elapsed if elapsed > 0 else 0.0
                print(
                    f"DeepResync inferred {len(out_records)} reads "
                    f"({rate:.1f} reads/s, batches={batch_count})",
                    flush=True,
                )

    write_start = time.perf_counter()
    write_posterior_file(Path(args.output), out_records)
    timings["write_seconds"] = time.perf_counter() - write_start
    if args.per_read_stats and per_read_rows:
        stats_path = Path(args.per_read_stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "pos_in_codeword",
            "read_id",
            "upper235_errors",
            "upper235_rate",
            "upper188_errors",
            "upper188_rate",
            "lower235_errors",
            "lower235_rate",
            "base235_errors",
            "base235_rate",
        ]
        with stats_path.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(per_read_rows)

    if args.hard_base_fasta and hard_base_records:
        fasta_path = Path(args.hard_base_fasta)
        fasta_path.parent.mkdir(parents=True, exist_ok=True)
        with fasta_path.open("w", encoding="utf-8") as fout:
            for header, seq in hard_base_records:
                fout.write(f">{header}\n")
                for i in range(0, len(seq), 80):
                    fout.write(seq[i : i + 80] + "\n")

    seconds = time.perf_counter() - start

    compared = totals["compared_reads"]
    hard_stats = {
        "compared_reads": compared,
        "upper235_error_rate": (totals["upper235_errors"] / (compared * BLOCK_LENGTH)) if compared else None,
        "upper188_error_rate": (totals["upper188_errors"] / (compared * UPPER_INFO_LENGTH)) if compared else None,
        "lower235_error_rate": (totals["lower235_errors"] / (compared * BLOCK_LENGTH)) if compared else None,
        "base235_error_rate": (totals["base235_errors"] / (compared * BLOCK_LENGTH)) if compared else None,
        **totals,
    }
    if args.stats_json:
        stats_json_path = Path(args.stats_json)
        stats_json_path.parent.mkdir(parents=True, exist_ok=True)
        stats_json_path.write_text(json.dumps(hard_stats, indent=2), encoding="utf-8")

    summary = {
        "decoder": "DeepResync",
        "model_version": version,
        "posterior_format": "423",
        "posterior_length": POSTERIOR_LEN_423,
        "temperature": default_temperature,
        "symbol_temperature": symbol_temperature,
        "lower_temperature": lower_temperature,
        "checkpoint": str(checkpoint_path),
        "deepresync_root": str(deepresync_root),
        "device": device,
        "amp": args.amp,
        "pool": args.pool,
        "batch_size": args.batch_size,
        "max_read_len": max_read_len,
        "input_reads": len(seq_map),
        "indexed_payload_reads": len(payload_records),
        "written_records": len(out_records),
        "output": str(Path(args.output)),
        "seconds": seconds,
        "timings": timings,
        "hard_decision_stats": hard_stats,
    }
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # print(
    #     f"DeepResync wrote {len(out_records)} posterior records to {args.output} "
    #     f"in {seconds:.2f} s",
    #     flush=True,
    # )
    print(
        "DeepResync timing: "
        f"encode={timings['encode_seconds']:.2f}s, "
        f"transfer={timings['transfer_seconds']:.2f}s, "
        f"forward={timings['forward_seconds']:.2f}s, "
        f"cpu_post={timings['cpu_post_seconds']:.2f}s, "
        f"write={timings['write_seconds']:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
