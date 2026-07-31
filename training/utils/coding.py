"""Half-watermark payload construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


BLOCK_LENGTH = 235
UPPER_INFO_LENGTH = 188
LOWER_INFO_LENGTH = 235
POSTERIOR_LENGTH = 2 * BLOCK_LENGTH
INFO_BITS_PER_SYMBOL = 4
SPARSE_BITS_PER_SYMBOL = 5
SPARSE_SYMBOL_COUNT = UPPER_INFO_LENGTH // INFO_BITS_PER_SYMBOL

BASES = ("A", "T", "G", "C")
BASE_TO_STATE = {"A": 0, "T": 1, "G": 2, "C": 3}
STATE_TO_BASE = {0: "A", 1: "T", 2: "G", 3: "C"}

SPARSE_4_TO_5 = {
    "0000": "00000",
    "0001": "00001",
    "0010": "00010",
    "0011": "00011",
    "0100": "00100",
    "0101": "00101",
    "0110": "00110",
    "0111": "11000",
    "1000": "01000",
    "1001": "01001",
    "1010": "01010",
    "1011": "10100",
    "1100": "01100",
    "1101": "10010",
    "1110": "10001",
    "1111": "10000",
}
SPARSE_ID_TO_5_BITS = tuple(
    tuple(1 if ch == "1" else 0 for ch in SPARSE_4_TO_5[f"{idx:04b}"]) for idx in range(16)
)
SPARSE_5_TO_ID = {"".join(str(bit) for bit in bits): idx for idx, bits in enumerate(SPARSE_ID_TO_5_BITS)}


@dataclass(frozen=True)
class HalfWatermarkPayload:
    """A simulated transmitted half-watermark payload before channel noise."""

    strand: str
    watermark_bits: np.ndarray
    upper_info_bits: np.ndarray
    upper_sparse_bits: np.ndarray
    upper_transmitted_bits: np.ndarray
    lower_bits: np.ndarray
    base_labels: np.ndarray


def _as_bit_array(bits: str | Iterable[int] | np.ndarray, expected_len: int | None = None) -> np.ndarray:
    if isinstance(bits, str):
        arr = np.fromiter((1 if ch == "1" else 0 for ch in bits.strip()), dtype=np.int64)
    else:
        arr = np.asarray(list(bits) if not isinstance(bits, np.ndarray) else bits, dtype=np.int64)

    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"Expected {expected_len} bits, got {arr.size}.")
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError("Bit arrays must contain only 0/1 values.")
    return arr


def bits_to_string(bits: Iterable[int] | np.ndarray) -> str:
    arr = np.asarray(bits, dtype=np.int64)
    return "".join("1" if int(v) else "0" for v in arr)


def read_watermark(path: str | Path, block_len: int = BLOCK_LENGTH) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8").strip()
    bits = "".join(ch for ch in text if ch in "01")
    return _as_bit_array(bits[:block_len], expected_len=block_len)


def random_bits(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 2, size=n, dtype=np.int64)


def sparse45_encode(upper_info_bits: str | Iterable[int] | np.ndarray) -> np.ndarray:
    bits = _as_bit_array(upper_info_bits, expected_len=UPPER_INFO_LENGTH)
    out: list[int] = []
    for i in range(0, UPPER_INFO_LENGTH, 4):
        key = bits_to_string(bits[i : i + 4])
        out.extend(1 if ch == "1" else 0 for ch in SPARSE_4_TO_5[key])
    arr = np.asarray(out, dtype=np.int64)
    if arr.size != BLOCK_LENGTH:
        raise RuntimeError(f"Sparse encoder produced {arr.size} bits, expected {BLOCK_LENGTH}.")
    return arr


def upper_info_to_symbol_ids(upper_info_bits: str | Iterable[int] | np.ndarray) -> np.ndarray:
    bits = _as_bit_array(upper_info_bits, expected_len=UPPER_INFO_LENGTH)
    symbols = []
    for i in range(0, UPPER_INFO_LENGTH, INFO_BITS_PER_SYMBOL):
        key = bits_to_string(bits[i : i + INFO_BITS_PER_SYMBOL])
        symbols.append(int(key, 2))
    return np.asarray(symbols, dtype=np.int64)


def sparse45_to_symbol_ids(upper_sparse_bits: str | Iterable[int] | np.ndarray) -> np.ndarray:
    bits = _as_bit_array(upper_sparse_bits, expected_len=BLOCK_LENGTH)
    symbols = []
    for i in range(0, BLOCK_LENGTH, SPARSE_BITS_PER_SYMBOL):
        key = bits_to_string(bits[i : i + SPARSE_BITS_PER_SYMBOL])
        try:
            symbols.append(SPARSE_5_TO_ID[key])
        except KeyError as exc:
            raise ValueError(f"Invalid sparse symbol '{key}' at sparse group {i // SPARSE_BITS_PER_SYMBOL}.") from exc
    return np.asarray(symbols, dtype=np.int64)


def base_states_to_strand(base_labels: Iterable[int] | np.ndarray) -> str:
    labels = np.asarray(base_labels, dtype=np.int64)
    return "".join(STATE_TO_BASE[int(v)] for v in labels)


def build_half_watermark_payload(
    rng: np.random.Generator,
    watermark_bits: str | Iterable[int] | np.ndarray,
    upper_info_bits: str | Iterable[int] | np.ndarray | None = None,
    lower_bits: str | Iterable[int] | np.ndarray | None = None,
) -> HalfWatermarkPayload:
    """Build one transmitted 235-base half-watermark payload.

    The upper layer is sparse-encoded from 188 to 235 bits and XORed with the
    known watermark. The lower layer is fully informational.
    """

    watermark = _as_bit_array(watermark_bits, expected_len=BLOCK_LENGTH)
    upper_info = random_bits(rng, UPPER_INFO_LENGTH) if upper_info_bits is None else _as_bit_array(
        upper_info_bits, expected_len=UPPER_INFO_LENGTH
    )
    lower = random_bits(rng, LOWER_INFO_LENGTH) if lower_bits is None else _as_bit_array(
        lower_bits, expected_len=LOWER_INFO_LENGTH
    )

    upper_sparse = sparse45_encode(upper_info)
    upper_transmitted = np.bitwise_xor(upper_sparse, watermark).astype(np.int64)

    base_labels = (2 * upper_transmitted + lower).astype(np.int64)
    strand = base_states_to_strand(base_labels)

    return HalfWatermarkPayload(
        strand=strand,
        watermark_bits=watermark,
        upper_info_bits=upper_info,
        upper_sparse_bits=upper_sparse,
        upper_transmitted_bits=upper_transmitted,
        lower_bits=lower,
        base_labels=base_labels,
    )
