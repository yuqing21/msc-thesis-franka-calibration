from __future__ import annotations

import json

from msc_cali.realsense import RealSenseUnavailableError, list_realsense_devices


def main() -> int:
    try:
        devices = list_realsense_devices()
    except RealSenseUnavailableError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps({"ok": bool(devices), "devices": [device.to_dict() for device in devices]}, indent=2))
    return 0 if devices else 3


if __name__ == "__main__":
    raise SystemExit(main())

