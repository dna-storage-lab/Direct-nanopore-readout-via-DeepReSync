"""PyTorch datasets for DeepResync."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError("PyTorch is required for training.utils.data. Install torch to train DeepResync.") from exc

from .coding import (
    BASE_TO_STATE,
    BLOCK_LENGTH,
    UPPER_INFO_LENGTH,
    read_watermark,
    sparse45_to_symbol_ids,
    upper_info_to_symbol_ids,
)
from .simulator import HalfWatermarkSimulator, get_error_profile


READ_TOKEN_TO_ID = {"A": 0, "T": 1, "G": 2, "C": 3}
PAD_TOKEN_ID = 4
_READ_BYTE_LOOKUP = np.full(256, PAD_TOKEN_ID, dtype=np.int64)
for _base, _token_id in READ_TOKEN_TO_ID.items():
    _READ_BYTE_LOOKUP[ord(_base)] = _token_id
    _READ_BYTE_LOOKUP[ord(_base.lower())] = _token_id


def _bits_from_string(text: str, expected_len: int) -> np.ndarray:
    arr = np.fromiter((1 if ch == "1" else 0 for ch in text.strip()), dtype=np.int64)
    if arr.size != expected_len:
        raise ValueError(f"Expected {expected_len} bits, got {arr.size}.")
    return arr


def encode_read(read: str, max_read_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    trimmed = read[:max_read_len]
    tokens_np = np.full(max_read_len, PAD_TOKEN_ID, dtype=np.int64)
    padding_np = np.ones(max_read_len, dtype=np.bool_)
    if trimmed:
        raw = np.frombuffer(trimmed.encode("ascii"), dtype=np.uint8)
        n = min(raw.size, max_read_len)
        tokens_np[:n] = _READ_BYTE_LOOKUP[raw[:n]]
        padding_np[:n] = False
    return torch.from_numpy(tokens_np), torch.from_numpy(padding_np)


def record_to_tensors(record: dict[str, Any], max_read_len: int) -> dict[str, torch.Tensor]:
    read_tokens, read_padding_mask = encode_read(record["read"], max_read_len)
    watermark = torch.tensor(_bits_from_string(record["watermark"], BLOCK_LENGTH), dtype=torch.long)
    labels = torch.tensor(record["base_labels"], dtype=torch.long)
    upper_bits = torch.tensor(_bits_from_string(record["upper_transmitted_bits"], BLOCK_LENGTH), dtype=torch.float32)
    lower_bits = torch.tensor(_bits_from_string(record["lower_bits"], BLOCK_LENGTH), dtype=torch.float32)
    if "upper_info_bits" in record:
        upper_info_np = _bits_from_string(record["upper_info_bits"], UPPER_INFO_LENGTH)
    else:
        upper_info_np = None
    if "upper_sparse_bits" in record:
        upper_sparse_np = _bits_from_string(record["upper_sparse_bits"], BLOCK_LENGTH)
    else:
        upper_sparse_np = np.bitwise_xor(upper_bits.numpy().astype(np.int64), watermark.numpy().astype(np.int64))
    upper_sparse_bits = torch.tensor(upper_sparse_np, dtype=torch.float32)

    if upper_info_np is not None:
        symbol_ids_np = upper_info_to_symbol_ids(upper_info_np)
    else:
        symbol_ids_np = sparse45_to_symbol_ids(upper_sparse_np)
    symbol_ids = torch.tensor(symbol_ids_np, dtype=torch.long)
    if upper_info_np is None:
        symbol_shifts = np.asarray([3, 2, 1, 0], dtype=np.int64)
        upper_info_np = (((symbol_ids_np[:, None] >> symbol_shifts[None, :]) & 1).reshape(-1)).astype(np.int64)
    upper_info_bits = torch.tensor(upper_info_np, dtype=torch.float32)

    if labels.numel() != BLOCK_LENGTH:
        raise ValueError(f"Expected {BLOCK_LENGTH} base labels, got {labels.numel()}.")

    return {
        "read_tokens": read_tokens,
        "read_padding_mask": read_padding_mask,
        "watermark_bits": watermark,
        "labels": labels,
        "upper_bits": upper_bits,
        "upper_sparse_bits": upper_sparse_bits,
        "lower_bits": lower_bits,
        "upper_info_bits": upper_info_bits,
        "symbol_ids": symbol_ids,
    }


class OnlineHalfWatermarkDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        watermark_path: str | Path,
        max_read_len: int,
        seed: int = 1,
        jitter: float = 0.0,
    ) -> None:
        self.num_samples = num_samples
        self.max_read_len = max_read_len
        watermark = read_watermark(watermark_path)
        self.simulator = HalfWatermarkSimulator(watermark, get_error_profile(), jitter=jitter)
        self.seed = seed
        self.epoch = 0
        self._epoch_tensor = torch.zeros((), dtype=torch.long).share_memory_()

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._epoch_tensor.fill_(self.epoch)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        epoch = int(self._epoch_tensor.item())
        rng = np.random.default_rng(self.seed + idx + epoch * 1_000_003)
        record = self.simulator.sample(rng).to_json_record()
        return record_to_tensors(record, self.max_read_len)
