from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_WATERMARK_PATH = ROOT / "configureFiles" / "watermark_sequence_length_235"

from training.utils.coding import read_watermark
from training.utils.simulator import HalfWatermarkSimulator, get_error_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate DeepResync training data.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--watermark-path", default=str(DEFAULT_WATERMARK_PATH))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jitter", type=float, default=0.0, help="Relative per-read rate jitter, e.g. 0.2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    watermark = read_watermark(args.watermark_path)
    simulator = HalfWatermarkSimulator(watermark, get_error_profile(), jitter=args.jitter)
    simulator.write_jsonl(args.output, args.num_samples, seed=args.seed)
    print(f"Wrote {args.num_samples} samples to {args.output}")


if __name__ == "__main__":
    main()
