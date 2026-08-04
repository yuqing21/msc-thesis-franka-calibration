from __future__ import annotations

import argparse
import json

from msc_cali.transport import JsonLineClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Ping the Herbert preview bridge over Tailscale.")
    parser.add_argument("--host", default="100.68.210.77")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    args = parser.parse_args()
    client = JsonLineClient(args.host, args.port, args.timeout_s)
    sync = client.estimate_clock_offset()
    output = dict(sync.response)
    output["estimated_herbert_clock_offset_s"] = sync.offset_s
    output["round_trip_s"] = sync.round_trip_s
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
