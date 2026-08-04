from __future__ import annotations

import argparse
import json
from pathlib import Path

from msc_cali.dataset import load_calibration_jsonl
from msc_cali.geometry import solve_rigid_transform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve camera-to-Franka-base calibration from 3D point pairs.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL correspondence file")
    parser.add_argument("--output", type=Path, required=True, help="Output transform JSON")
    parser.add_argument("--max-rmse-m", type=float, default=0.020)
    parser.add_argument("--max-error-m", type=float, default=0.040)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = load_calibration_jsonl(args.input)
    result = solve_rigid_transform(dataset.camera_points_m, dataset.robot_points_m)
    accepted = result.rmse_m <= args.max_rmse_m and result.max_m <= args.max_error_m
    output = result.to_dict()
    output["quality"]["accepted"] = accepted
    output["quality"]["thresholds_m"] = {
        "max_rmse_m": args.max_rmse_m,
        "max_error_m": args.max_error_m,
    }
    output["sample_ids"] = list(dataset.sample_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"samples={result.sample_count} rmse={result.rmse_m * 1000:.2f} mm "
        f"p95={result.p95_m * 1000:.2f} mm max={result.max_m * 1000:.2f} mm "
        f"accepted={accepted}"
    )
    print(args.output.resolve())
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

