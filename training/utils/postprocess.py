"""Posterior serialization helpers for DeepResync."""

from __future__ import annotations

import numpy as np

POSTERIOR_LEN_423 = 423
POSTERIOR_MAGIC_423 = b"RPST423\0"


def write_read_posteriors_bin(
    records: list[tuple[int, np.ndarray]],
    path: str,
) -> None:
    """Write 423-value information posterior records to read_posteriors.bin."""

    import struct

    with open(path, "wb") as fout:
        fout.write(POSTERIOR_MAGIC_423)
        fout.write(struct.pack("<I", 1))
        fout.write(struct.pack("<I", POSTERIOR_LEN_423))
        fout.write(struct.pack("<Q", len(records)))
        for pos, posterior in records:
            arr = np.asarray(posterior, dtype="<f8")
            if arr.size != POSTERIOR_LEN_423:
                raise ValueError(f"Expected {POSTERIOR_LEN_423} posterior values, got {arr.size}.")
            fout.write(struct.pack("<i", int(pos)))
            fout.write(arr.tobytes(order="C"))
