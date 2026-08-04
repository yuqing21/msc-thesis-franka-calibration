from types import SimpleNamespace

import pytest

from msc_cali.realsense import (
    RealSenseConfig,
    RealSenseUnavailableError,
    _load_pyrealsense2,
    list_realsense_devices,
)


def test_realsense_config_rejects_invalid_stream() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        RealSenseConfig(color_width=0)


def test_device_listing_uses_sdk_camera_information() -> None:
    camera_info = SimpleNamespace(
        name="name",
        serial_number="serial",
        firmware_version="firmware",
        usb_type_descriptor="usb",
    )

    class FakeDevice:
        values = {
            "name": "Intel RealSense D435f",
            "serial": "123456",
            "firmware": "5.16.0.1",
            "usb": "3.2",
        }

        def supports(self, key: str) -> bool:
            return key in self.values

        def get_info(self, key: str) -> str:
            return self.values[key]

    fake_rs = SimpleNamespace(
        camera_info=camera_info,
        context=lambda: SimpleNamespace(query_devices=lambda: [FakeDevice()]),
    )
    devices = list_realsense_devices(fake_rs)
    assert len(devices) == 1
    assert devices[0].name == "Intel RealSense D435f"
    assert devices[0].serial == "123456"
    assert devices[0].usb_type == "3.2"


def test_missing_sdk_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr("msc_cali.realsense.importlib.import_module", fail_import)
    with pytest.raises(RealSenseUnavailableError, match="requirements-realsense"):
        _load_pyrealsense2()

