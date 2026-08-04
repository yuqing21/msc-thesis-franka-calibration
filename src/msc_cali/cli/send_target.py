from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from msc_cali.geometry import RigidTransform
from msc_cali.protocol import make_preview_target
from msc_cali.transport import JsonLineClient


def _load_transform(path: Path) -> RigidTransform:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RigidTransform.from_dict(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a non-moving target preview to Herbert.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-point", nargs=3, type=float, metavar=("X", "Y", "Z"))
    source.add_argument("--camera-point", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--transform", type=Path, help="Required when --camera-point is used")
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--host", default="100.68.210.77")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.camera_point is not None:
        if args.transform is None:
            parser.error("--transform is required with --camera-point")
        point = _load_transform(args.transform).apply(np.asarray(args.camera_point, dtype=np.float64))
    else:
        point = np.asarray(args.base_point, dtype=np.float64)
    client = JsonLineClient(args.host, args.port)
    sync = client.estimate_clock_offset()
    message = make_preview_target(
        point.tolist(),
        args.confidence,
        sent_at_s=time.time() + sync.offset_s,
    )
    response = client.request(message)
    output = dict(response)
    output["estimated_herbert_clock_offset_s"] = sync.offset_s
    output["clock_sync_round_trip_s"] = sync.round_trip_s
    print(json.dumps(output, indent=2))
    return 0 if response.get("type") == "ack" else 2


if __name__ == "__main__":
    raise SystemExit(main())
