from __future__ import annotations

import argparse
import contextlib
import csv
import json
import struct
import time
from pathlib import Path

import numpy as np


BLOCK_LENGTH = 235
UPPER_INFO_LENGTH = 188
POSTERIOR_LEN_423 = UPPER_INFO_LENGTH + BLOCK_LENGTH
POSTERIOR_MAGIC_423 = b"RPST423\0"
POSTERIOR_VERSION = 1
OLIGOS_PER_POOL = 153 * 196
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
SPARSE_4_TO_5 = {value: key for key, value in SPARSE_5_TO_4.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute hard-decision error statistics from read_posteriors.bin."
    )
    parser.add_argument("--posterior-bin", required=True)
    parser.add_argument("--watermark-path", required=True)
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--pool", type=int, required=True)
    parser.add_argument("--payload-start", type=int, default=PAYLOAD_START)
    parser.add_argument("--per-read-stats", required=True)
    parser.add_argument("--hard-base-fasta", required=True)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument("--stats-batch-size", type=int, default=20000)
    parser.add_argument("--skip-per-read-stats", action="store_true")
    parser.add_argument("--skip-hard-base-fasta", action="store_true")
    return parser.parse_args()


def load_watermark(path: Path) -> str:
    bits = [ch for ch in path.read_text(encoding="utf-8") if ch in "01"]
    if len(bits) < BLOCK_LENGTH:
        raise ValueError(f"{path} contains only {len(bits)} bits; expected {BLOCK_LENGTH}.")
    return "".join(bits[:BLOCK_LENGTH])


def read_fasta_ordered(path: Path):
    current_id = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, "".join(chunks)
                current_id = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        yield current_id, "".join(chunks)


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
    for start in range(0, len(sparse_bits), 5):
        group = sparse_bits[start : start + 5]
        if group in SPARSE_5_TO_4:
            out.append(SPARSE_5_TO_4[group])
            continue
        nearest = min(
            SPARSE_5_TO_4,
            key=lambda code: (sum(x != y for x, y in zip(group, code)), code),
        )
        out.append(SPARSE_5_TO_4[nearest])
    return "".join(out)


def sparse45_decode_probs_to_hard_info_bits(sparse_p1: list[float]) -> str:
    out: list[str] = []
    codewords = list(SPARSE_5_TO_4)
    for start in range(0, BLOCK_LENGTH, 5):
        p_group = sparse_p1[start : start + 5]
        code_probs: list[float] = []
        for code in codewords:
            p = 1.0
            for bit, p1 in zip(code, p_group):
                p *= p1 if bit == "1" else (1.0 - p1)
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


def load_reference_labels(reference_fasta: Path, watermark: str, payload_start: int) -> dict[str, list[str]]:
    labels = {
        "payload": [],
        "upper235": [],
        "lower235": [],
        "upper188": [],
    }
    for _, full_seq in read_fasta_ordered(reference_fasta):
        payload = full_seq[payload_start : payload_start + BLOCK_LENGTH]
        if len(payload) != BLOCK_LENGTH:
            continue
        upper, lower = payload_base_to_bits(payload)
        upper_sparse = xor_bits(upper, watermark)
        labels["payload"].append(payload)
        labels["upper235"].append(upper)
        labels["lower235"].append(lower)
        labels["upper188"].append(sparse45_decode_hard(upper_sparse))
    return labels


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


def read_posterior_records(path: Path):
    with path.open("rb") as fin:
        magic = fin.read(8)
        if magic != POSTERIOR_MAGIC_423:
            raise ValueError(f"Unsupported posterior format in {path}; expected RPST423, got {magic!r}")
        version, posterior_len = struct.unpack("<II", fin.read(8))
        (record_count,) = struct.unpack("<Q", fin.read(8))
        if version != POSTERIOR_VERSION or posterior_len != POSTERIOR_LEN_423:
            raise ValueError(
                f"Unsupported posterior file: version={version}, posterior_len={posterior_len}"
            )
        fmt = "<i" + ("d" * posterior_len)
        record_size = struct.calcsize(fmt)
        for record_index in range(record_count):
            raw = fin.read(record_size)
            if len(raw) != record_size:
                raise ValueError(f"Posterior file ended at record {record_index}.")
            unpacked = struct.unpack(fmt, raw)
            yield record_index, int(unpacked[0]), list(unpacked[1:])


def watermark_to_array(watermark: str) -> np.ndarray:
    return np.fromiter((ch == "1" for ch in watermark), dtype=np.bool_, count=BLOCK_LENGTH)


def sparse_code_arrays() -> tuple[np.ndarray, np.ndarray]:
    code_bits = np.array(
        [[1 if ch == "1" else 0 for ch in code] for code in SPARSE_5_TO_4],
        dtype=np.float64,
    )
    info_bits = np.array(
        [[1 if ch == "1" else 0 for ch in info] for info in SPARSE_5_TO_4.values()],
        dtype=np.float64,
    )
    return code_bits, info_bits


def sparse_hard_lut() -> np.ndarray:
    codes = list(SPARSE_5_TO_4)
    lut = np.zeros((32, 4), dtype=np.bool_)
    for value in range(32):
        bits = f"{value:05b}"
        nearest = min(codes, key=lambda code: (sum(x != y for x, y in zip(bits, code)), code))
        lut[value] = [ch == "1" for ch in SPARSE_5_TO_4[nearest]]
    return lut


def decode_sparse_hard_matrix(sparse_bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(sparse_bits, dtype=np.uint8).reshape(-1, BLOCK_LENGTH // 5, 5)
    weights = np.array([16, 8, 4, 2, 1], dtype=np.uint8)
    values = (bits * weights).sum(axis=-1)
    return sparse_hard_lut()[values].reshape(-1, UPPER_INFO_LENGTH)


def decode_sparse_probs_matrix(sparse_p1: np.ndarray) -> np.ndarray:
    probs = np.asarray(sparse_p1, dtype=np.float64).reshape(-1, BLOCK_LENGTH // 5, 5)
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    code_bits, info_bits = sparse_code_arrays()

    log_p1 = np.log(probs)[:, :, None, :]
    log_p0 = np.log1p(-probs)[:, :, None, :]
    code = code_bits[None, None, :, :]
    log_scores = (code * log_p1 + (1.0 - code) * log_p0).sum(axis=-1)
    log_scores -= log_scores.max(axis=-1, keepdims=True)
    symbol_probs = np.exp(log_scores)
    symbol_probs /= symbol_probs.sum(axis=-1, keepdims=True)

    bit_probs = symbol_probs @ info_bits
    return (bit_probs >= 0.5).reshape(-1, UPPER_INFO_LENGTH)


def encode_info_hard_matrix(info_bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(info_bits, dtype=np.bool_).reshape(-1, UPPER_INFO_LENGTH // 4, 4)
    weights = np.array([8, 4, 2, 1], dtype=np.uint8)
    values = (bits.astype(np.uint8) * weights).sum(axis=-1)
    code_lut = np.zeros((16, 5), dtype=np.bool_)
    for value in range(16):
        code = SPARSE_4_TO_5[f"{value:04b}"]
        code_lut[value] = [ch == "1" for ch in code]
    return code_lut[values].reshape(-1, BLOCK_LENGTH)


def load_reference_arrays(
    reference_fasta: Path,
    watermark_bits: np.ndarray,
    payload_start: int,
    pool: int,
) -> dict[str, np.ndarray]:
    base_lut = np.zeros(256, dtype=np.uint8)
    base_lut[ord("A")] = 0
    base_lut[ord("T")] = 1
    base_lut[ord("G")] = 2
    base_lut[ord("C")] = 3
    payload_rows: list[np.ndarray] = []
    start_record = (pool - 1) * OLIGOS_PER_POOL
    end_record = start_record + OLIGOS_PER_POOL
    for record_idx, (_, full_seq) in enumerate(read_fasta_ordered(reference_fasta)):
        if record_idx < start_record:
            continue
        if record_idx >= end_record:
            break
        payload = full_seq[payload_start : payload_start + BLOCK_LENGTH]
        if len(payload) != BLOCK_LENGTH:
            continue
        payload_rows.append(np.frombuffer(payload.encode("ascii", "replace"), dtype=np.uint8))

    if not payload_rows:
        empty_bool_235 = np.zeros((0, BLOCK_LENGTH), dtype=np.bool_)
        return {
            "payload": np.zeros((0, BLOCK_LENGTH), dtype=np.uint8),
            "upper235": empty_bool_235,
            "lower235": empty_bool_235.copy(),
            "upper188": np.zeros((0, UPPER_INFO_LENGTH), dtype=np.bool_),
        }

    payload_bytes = np.stack(payload_rows, axis=0)
    payload_states = base_lut[payload_bytes]
    upper235 = payload_states >= 2
    lower235 = (payload_states & 1).astype(np.bool_)
    upper_sparse = np.logical_xor(upper235, watermark_bits[None, :])
    upper188 = decode_sparse_hard_matrix(upper_sparse)
    return {
        "payload": payload_states,
        "upper235": upper235,
        "lower235": lower235,
        "upper188": upper188,
    }


def read_posterior_batches(path: Path, batch_size: int):
    with path.open("rb") as fin:
        magic = fin.read(8)
        if magic != POSTERIOR_MAGIC_423:
            raise ValueError(f"Unsupported posterior format in {path}; expected RPST423, got {magic!r}")
        version, posterior_len = struct.unpack("<II", fin.read(8))
        (record_count,) = struct.unpack("<Q", fin.read(8))
        if version != POSTERIOR_VERSION or posterior_len != POSTERIOR_LEN_423:
            raise ValueError(
                f"Unsupported posterior file: version={version}, posterior_len={posterior_len}"
            )
        record_dtype = np.dtype([("pos", "<i4"), ("posterior", "<f8", (posterior_len,))])
        offset = 0
        while offset < record_count:
            count = min(batch_size, record_count - offset)
            batch = np.fromfile(fin, dtype=record_dtype, count=count)
            if batch.size != count:
                raise ValueError(f"Posterior file ended at record {offset + batch.size}.")
            yield offset, batch["pos"].astype(np.int64), batch["posterior"], posterior_len
            offset += count


def base_states_to_strings(states: np.ndarray) -> list[str]:
    base_lut = np.frombuffer(b"ATGC", dtype=np.uint8)
    chars = base_lut[np.asarray(states, dtype=np.uint8)]
    return [row.tobytes().decode("ascii") for row in chars]


def load_reference_labels_for_pool(reference_fasta: Path, watermark: str, payload_start: int, pool: int) -> dict[str, list[str]]:
    labels = {
        "payload": [],
        "upper235": [],
        "lower235": [],
        "upper188": [],
    }
    start_record = (pool - 1) * OLIGOS_PER_POOL
    end_record = start_record + OLIGOS_PER_POOL
    for record_idx, (_, full_seq) in enumerate(read_fasta_ordered(reference_fasta)):
        if record_idx < start_record:
            continue
        if record_idx >= end_record:
            break
        payload = full_seq[payload_start : payload_start + BLOCK_LENGTH]
        if len(payload) != BLOCK_LENGTH:
            continue
        upper, lower = payload_base_to_bits(payload)
        labels["payload"].append(payload)
        labels["upper235"].append(upper)
        labels["lower235"].append(lower)
        labels["upper188"].append(sparse45_decode_hard(xor_bits(upper, watermark)))
    return labels


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    if args.pool < 1:
        raise ValueError(f"--pool must be >= 1, got {args.pool}.")
    if args.stats_batch_size < 1:
        raise ValueError(f"--stats-batch-size must be >= 1, got {args.stats_batch_size}.")

    watermark = load_watermark(Path(args.watermark_path))
    watermark_bits = watermark_to_array(watermark)
    labels = load_reference_arrays(Path(args.reference_fasta), watermark_bits, args.payload_start, args.pool)

    per_read_path = Path(args.per_read_stats)
    hard_base_path = Path(args.hard_base_fasta)
    stats_json_path = Path(args.stats_json)
    per_read_path.parent.mkdir(parents=True, exist_ok=True)
    hard_base_path.parent.mkdir(parents=True, exist_ok=True)
    stats_json_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "compared_reads": 0,
        "upper235_errors": 0,
        "upper188_errors": 0,
        "lower235_errors": 0,
        "base235_errors": 0,
        "skipped_records": 0,
    }
    timings = {
        "read_seconds": 0.0,
        "compute_seconds": 0.0,
        "per_read_write_seconds": 0.0,
        "hard_base_write_seconds": 0.0,
    }

    fieldnames = [
        "record_index",
        "pos_in_codeword",
        "upper235_errors",
        "upper235_rate",
        "upper188_errors",
        "upper188_rate",
        "lower235_errors",
        "lower235_rate",
        "base235_errors",
        "base235_rate",
    ]

    per_read_context = (
        per_read_path.open("w", encoding="utf-8", newline="")
        if not args.skip_per_read_stats
        else contextlib.nullcontext(None)
    )
    hard_base_context = (
        hard_base_path.open("w", encoding="utf-8")
        if not args.skip_hard_base_fasta
        else contextlib.nullcontext(None)
    )

    with per_read_context as per_read, hard_base_context as hard_base:
        if per_read is not None:
            per_read.write("\t".join(fieldnames) + "\n")

        posterior_len_seen: int | None = None
        for record_offset, positions, posterior, posterior_len in read_posterior_batches(
            Path(args.posterior_bin),
            args.stats_batch_size,
        ):
            if posterior_len_seen is None:
                posterior_len_seen = int(posterior_len)
            elif posterior_len_seen != int(posterior_len):
                raise ValueError("Posterior length changed within one file.")

            t0 = time.perf_counter()
            ref_idx = positions - 1
            valid = (ref_idx >= 0) & (ref_idx < labels["payload"].shape[0])
            timings["read_seconds"] += time.perf_counter() - t0

            invalid_count = int((~valid).sum())
            totals["skipped_records"] += invalid_count
            if not bool(valid.any()):
                continue

            t0 = time.perf_counter()
            valid_rows = np.nonzero(valid)[0]
            record_indices = record_offset + valid_rows
            positions_valid = positions[valid]
            ref_idx_valid = ref_idx[valid]
            posterior_valid = posterior[valid]

            upper188_probs = posterior_valid[:, :UPPER_INFO_LENGTH]
            lower_probs = posterior_valid[:, UPPER_INFO_LENGTH:]
            pred_lower235 = lower_probs >= 0.5
            pred_upper188 = upper188_probs >= 0.5
            pred_upper_sparse = encode_info_hard_matrix(pred_upper188)
            pred_upper235 = np.logical_xor(pred_upper_sparse, watermark_bits[None, :])
            pred_base235 = (pred_upper235.astype(np.uint8) << 1) | pred_lower235.astype(np.uint8)

            ref_upper235 = labels["upper235"][ref_idx_valid]
            ref_upper188 = labels["upper188"][ref_idx_valid]
            ref_lower235 = labels["lower235"][ref_idx_valid]
            ref_payload = labels["payload"][ref_idx_valid]

            err_upper235 = np.count_nonzero(pred_upper235 != ref_upper235, axis=1)
            err_upper188 = np.count_nonzero(pred_upper188 != ref_upper188, axis=1)
            err_lower235 = np.count_nonzero(pred_lower235 != ref_lower235, axis=1)
            err_base235 = np.count_nonzero(pred_base235 != ref_payload, axis=1)

            compared_now = int(ref_idx_valid.size)
            totals["compared_reads"] += compared_now
            totals["upper235_errors"] += int(err_upper235.sum())
            totals["upper188_errors"] += int(err_upper188.sum())
            totals["lower235_errors"] += int(err_lower235.sum())
            totals["base235_errors"] += int(err_base235.sum())
            timings["compute_seconds"] += time.perf_counter() - t0

            if per_read is not None:
                t0 = time.perf_counter()
                lines = []
                for row in range(compared_now):
                    lines.append(
                        "\t".join(
                            [
                                str(int(record_indices[row])),
                                str(int(positions_valid[row])),
                                str(int(err_upper235[row])),
                                f"{err_upper235[row] / BLOCK_LENGTH:.12g}",
                                str(int(err_upper188[row])),
                                f"{err_upper188[row] / UPPER_INFO_LENGTH:.12g}",
                                str(int(err_lower235[row])),
                                f"{err_lower235[row] / BLOCK_LENGTH:.12g}",
                                str(int(err_base235[row])),
                                f"{err_base235[row] / BLOCK_LENGTH:.12g}",
                            ]
                        )
                        + "\n"
                    )
                per_read.writelines(lines)
                timings["per_read_write_seconds"] += time.perf_counter() - t0

            if hard_base is not None:
                t0 = time.perf_counter()
                seqs = base_states_to_strings(pred_base235)
                lines = []
                for record_index, pos, seq in zip(record_indices, positions_valid, seqs):
                    lines.append(f">record={int(record_index)}|pos={int(pos)}\n")
                    lines.append(seq[:80] + "\n")
                    lines.append(seq[80:160] + "\n")
                    lines.append(seq[160:] + "\n")
                hard_base.writelines(lines)
                timings["hard_base_write_seconds"] += time.perf_counter() - t0

    compared = totals["compared_reads"]
    posterior_format = "423" if "posterior_len_seen" in locals() and posterior_len_seen is not None else None
    summary = {
        "posterior_bin": str(Path(args.posterior_bin)),
        "posterior_format": posterior_format,
        "posterior_length": posterior_len_seen if "posterior_len_seen" in locals() else None,
        "reference_fasta": str(Path(args.reference_fasta)),
        "watermark_path": str(Path(args.watermark_path)),
        "pool": args.pool,
        "payload_start": args.payload_start,
        "seconds": time.perf_counter() - start,
        "stats_batch_size": args.stats_batch_size,
        "skip_per_read_stats": args.skip_per_read_stats,
        "skip_hard_base_fasta": args.skip_hard_base_fasta,
        "timings": timings,
        "upper235_source": "reencoded_from_hard_upper188",
        "base235_source": "reencoded_upper188_plus_lower235",
        **totals,
        "upper235_error_rate": totals["upper235_errors"] / (compared * BLOCK_LENGTH) if compared else None,
        "upper188_error_rate": totals["upper188_errors"] / (compared * UPPER_INFO_LENGTH) if compared else None,
        "lower235_error_rate": totals["lower235_errors"] / (compared * BLOCK_LENGTH) if compared else None,
        "base235_error_rate": totals["base235_errors"] / (compared * BLOCK_LENGTH) if compared else None,
    }
    stats_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # print(
    #     f"Posterior hard-decision stats: compared={compared}, "
    #     f"format={summary['posterior_format']}, "
    #     f"upper188_error_rate={summary['upper188_error_rate']}, "
    #     f"lower235_error_rate={summary['lower235_error_rate']}, "
    #     f"base235_error_rate={summary['base235_error_rate']}",
    #     flush=True,
    # )
    # print(
    #     "Posterior stats timing: "
    #     f"read={timings['read_seconds']:.2f}s, "
    #     f"compute={timings['compute_seconds']:.2f}s, "
    #     f"per_read_write={timings['per_read_write_seconds']:.2f}s, "
    #     f"hard_base_write={timings['hard_base_write_seconds']:.2f}s",
    #     flush=True,
    # )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
