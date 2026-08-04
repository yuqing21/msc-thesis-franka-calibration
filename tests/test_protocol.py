import math

import pytest

from msc_cali.protocol import ProtocolError, decode_message, encode_message, make_ping, make_preview_target, validate_request


def test_ping_round_trip() -> None:
    message = make_ping()
    assert decode_message(encode_message(message))["type"] == "ping"


def test_preview_target_is_normalized() -> None:
    message = make_preview_target([0.4, -0.1, 0.3], 0.9, sent_at_s=1234.5)
    assert message["target_base_m"] == [0.4, -0.1, 0.3]
    assert message["confidence"] == 0.9
    assert message["sent_at_s"] == 1234.5


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_preview_target_rejects_nonfinite_values(bad_value: float) -> None:
    message = make_ping()
    message.update({"type": "preview_target", "target_base_m": [0.2, bad_value, 0.3], "confidence": 1.0})
    with pytest.raises(ProtocolError, match="finite"):
        validate_request(message)


def test_protocol_rejects_motion_commands() -> None:
    message = make_ping()
    message["type"] = "move_arm"
    with pytest.raises(ProtocolError, match="unsupported"):
        validate_request(message)
