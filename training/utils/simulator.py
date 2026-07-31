"""IDS simulator for DeepResync training.

The empirical error model follows the data-simulation approaches described in:

- O. Sabary, et al., SOLQC: Synthetic oligo library quality control tool.
  Bioinformatics 37, 720–722 (2021).
- D. Bar-Lev, I. Orr, O. Sabary, T. Etzion, E. Yaakobi, Scalable and robust
  DNA-based storage via coding theory and deep learning. Nat. Mach. Intell. 7,
  639–649 (2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np

from .coding import (
    BASES,
    BLOCK_LENGTH,
    HalfWatermarkPayload,
    build_half_watermark_payload,
    bits_to_string,
)

SIMULATION_PROFILE_NAME = "twist_nanopore_sup"


@dataclass(frozen=True)
class ErrorProfile:
    """Twist synthesis, Nanopore sequencing, and SUP basecalling error profile."""

    name: str
    substitution: float
    insertion: float
    deletion: float
    long_deletion: float = 0.0
    per_base: dict[str, dict[str, float]] | None = None
    deletion_length_rates: dict[int, float] = field(
        default_factory=lambda: {2: 2.8e-4, 3: 7.75e-5, 4: 3.25e-5, 5: 1.0e-6, 6: 5.5e-8}
    )

    def rates_for_base(self, base: str) -> dict[str, float]:
        if self.per_base and base in self.per_base:
            rates = self.per_base[base]
            return {
                "s": float(rates.get("s", self.substitution)),
                "i": float(rates.get("i", self.insertion)),
                "pi": float(rates.get("pi", rates.get("i", self.insertion))),
                "d": float(rates.get("d", self.deletion)),
                "ld": float(rates.get("ld", self.long_deletion)),
            }
        return {
            "s": self.substitution,
            "i": self.insertion,
            "pi": self.insertion,
            "d": self.deletion,
            "ld": self.long_deletion,
        }

    def total_error_rate(self) -> float:
        return self.substitution + self.insertion + self.deletion + self.long_deletion


def get_error_profile() -> ErrorProfile:
    """Return the fixed Twist synthesis + Nanopore sequencing + SUP basecalling profile."""

    base_sub = 1.56e-2
    base_ins = 1.24e-2
    base_del = 9.79e-3
    base_long_del = 2.67e-3
    target_sub = 0.005
    target_ins = 0.003
    target_del = 0.0025

    sub_mult = target_sub / base_sub
    ins_mult = target_ins / base_ins
    del_mult = target_del / base_del

    return ErrorProfile(
        name=SIMULATION_PROFILE_NAME,
        substitution=target_sub,
        insertion=target_ins,
        deletion=target_del,
        long_deletion=del_mult * base_long_del,
        per_base={
            "A": {
                "s": sub_mult * 1.700e-2,
                "i": ins_mult * 1.320e-2,
                "pi": ins_mult * 1.343e-2,
                "d": del_mult * 1.095e-2,
                "ld": del_mult * 0.294e-2,
            },
            "T": {
                "s": sub_mult * 1.358e-2,
                "i": ins_mult * 1.116e-2,
                "pi": ins_mult * 1.094e-2,
                "d": del_mult * 0.849e-2,
                "ld": del_mult * 0.224e-2,
            },
            "G": {
                "s": sub_mult * 1.593e-2,
                "i": ins_mult * 1.256e-2,
                "pi": ins_mult * 1.255e-2,
                "d": del_mult * 1.014e-2,
                "ld": del_mult * 0.273e-2,
            },
            "C": {
                "s": sub_mult * 1.606e-2,
                "i": ins_mult * 1.271e-2,
                "pi": ins_mult * 1.272e-2,
                "d": del_mult * 1.038e-2,
                "ld": del_mult * 0.277e-2,
            },
        },
    )


def _sample_other_base(base: str, rng: np.random.Generator) -> str:
    choices = [b for b in BASES if b != base]
    return choices[int(rng.integers(0, len(choices)))]


def _sample_inserted_base_empirical(profile: ErrorProfile, rng: np.random.Generator) -> str:
    bases = ("A", "T", "C", "G")
    weights = np.asarray([profile.rates_for_base(base)["i"] for base in bases], dtype=np.float64)
    weights = weights / weights.sum()
    return bases[int(rng.choice(len(bases), p=weights))]


def _sample_long_deletion_len(profile: ErrorProfile, rng: np.random.Generator) -> int:
    lengths = np.asarray(list(profile.deletion_length_rates.keys()), dtype=np.int64)
    weights = np.asarray(list(profile.deletion_length_rates.values()), dtype=np.float64)
    weights = weights / weights.sum()
    return int(rng.choice(lengths, p=weights))


def _jitter_profile_rates(profile: ErrorProfile, rates: dict[str, float], jitter: float, rng: np.random.Generator) -> dict[str, float]:
    if jitter <= 0:
        return rates
    return {key: max(0.0, float(rng.normal(value, jitter * value))) for key, value in rates.items()}


def _simulate_read_empirical(
    strand: str,
    profile: ErrorProfile,
    rng: np.random.Generator,
    jitter: float = 0.0,
) -> str:
    """Simulate empirical substitution, pre-insertion, and deletion events."""

    working = list(strand)
    idx = 0
    while idx < len(working):
        base = working[idx]
        total_rates = {
            "s": profile.substitution,
            "i": profile.insertion,
            "d": profile.deletion,
            "ld": profile.long_deletion,
        }
        total_rates = _jitter_profile_rates(profile, total_rates, jitter, rng)
        total_error_rate = min(sum(total_rates.values()), 0.95)

        if rng.random() < total_error_rate:
            base_rates = profile.rates_for_base(base)
            base_rates = _jitter_profile_rates(profile, base_rates, jitter, rng)
            event_names = ("s", "pi", "d", "ld")
            weights = np.asarray([base_rates[event] for event in event_names], dtype=np.float64)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                event = event_names[int(rng.choice(len(event_names), p=weights))]

                if event == "s":
                    working[idx] = _sample_other_base(base, rng)
                elif event == "pi":
                    working.insert(idx, _sample_inserted_base_empirical(profile, rng))
                    idx += 1
                elif event == "d":
                    del working[idx]
                    idx -= 1
                else:
                    deletion_len = _sample_long_deletion_len(profile, rng)
                    del working[idx : idx + deletion_len]
                    idx -= 1

        idx += 1

    return "".join(working)


def simulate_read(
    strand: str,
    profile: ErrorProfile,
    rng: np.random.Generator,
    jitter: float = 0.0,
) -> str:
    """Simulate a noisy read from one transmitted strand.

    At each original base, one of substitution, pre-insertion, deletion, or
    long deletion may occur. The implementation is stateless and deterministic
    under a NumPy random generator.
    """

    return _simulate_read_empirical(strand, profile, rng, jitter=jitter)


@dataclass(frozen=True)
class SimulatedSample:
    payload: HalfWatermarkPayload
    noisy_read: str

    def to_json_record(self) -> dict[str, Any]:
        return {
            "read": self.noisy_read,
            "transmitted": self.payload.strand,
            "watermark": bits_to_string(self.payload.watermark_bits),
            "base_labels": self.payload.base_labels.astype(int).tolist(),
            "upper_info_bits": bits_to_string(self.payload.upper_info_bits),
            "upper_sparse_bits": bits_to_string(self.payload.upper_sparse_bits),
            "upper_transmitted_bits": bits_to_string(self.payload.upper_transmitted_bits),
            "lower_bits": bits_to_string(self.payload.lower_bits),
        }


class HalfWatermarkSimulator:
    """Generate DeepResync supervised examples."""

    def __init__(
        self,
        watermark_bits: np.ndarray,
        profile: ErrorProfile,
        jitter: float = 0.0,
    ) -> None:
        if watermark_bits.size != BLOCK_LENGTH:
            raise ValueError(f"Expected watermark length {BLOCK_LENGTH}, got {watermark_bits.size}.")
        self.watermark_bits = watermark_bits.astype(np.int64)
        self.profile = profile
        self.jitter = jitter

    def sample(self, rng: np.random.Generator) -> SimulatedSample:
        payload = build_half_watermark_payload(rng, self.watermark_bits)
        noisy_read = simulate_read(payload.strand, self.profile, rng, jitter=self.jitter)
        return SimulatedSample(payload=payload, noisy_read=noisy_read)

    def write_jsonl(self, path: str | Path, num_samples: int, seed: int = 1) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        with path.open("w", encoding="utf-8") as fout:
            for _ in range(num_samples):
                fout.write(json.dumps(self.sample(rng).to_json_record(), separators=(",", ":")) + "\n")
