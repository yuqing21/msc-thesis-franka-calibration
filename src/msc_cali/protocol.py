from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any


SCHEMA = "cali-control/v1"
MAX_MESSAGE_BYTES = 16_384


class ProtocolError(ValueError):
    pass


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite")
    return result


def make_ping() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "type": "ping",
        "request_id": uuid.uuid4().hex,
        "sent_at_s": time.time(),
    }


def make_preview_target(
    target_base_m: Any,
    confidence: float,
    *,
    sent_at_s: float | None = None,
) -> dict[str, Any]:
    values = list(target_base_m)
    message = {
        "schema": SCHEMA,
        "type": "preview_target",
        "request_id": uuid.uuid4().hex,
        "sent_at_s": time.time() if sent_at_s is None else sent_at_s,
        "target_base_m": values,
        "confidence": confidence,
    }
    return validate_request(message)


def validate_request(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ProtocolError("message must be a JSON object")
    if message.get("schema") != SCHEMA:
        raise ProtocolError("unsupported schema")
    request_id = message.get("request_id")
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 64:
        raise ProtocolError("request_id must be a non-empty string of at most 64 characters")
    sent_at_s = _finite_number(message.get("sent_at_s"), name="sent_at_s")
    message_type = message.get("type")
    normalized: dict[str, Any] = {
        "schema": SCHEMA,
        "type": message_type,
        "request_id": request_id,
        "sent_at_s": sent_at_s,
    }
    if message_type == "ping":
        return normalized
    if message_type != "preview_target":
        raise ProtocolError(f"unsupported message type: {message_type!r}")

    target = message.get("target_base_m")
    if not isinstance(target, list) or len(target) != 3:
        raise ProtocolError("target_base_m must be a list of three metre values")
    normalized_target = [_finite_number(value, name=f"target_base_m[{index}]") for index, value in enumerate(target)]
    if any(abs(value) > 2.0 for value in normalized_target):
        raise ProtocolError("preview target exceeds the protocol's absolute 2 m sanity bound")
    confidence = _finite_number(message.get("confidence"), name="confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ProtocolError("confidence must lie in [0, 1]")
    normalized["target_base_m"] = normalized_target
    normalized["confidence"] = confidence
    return normalized


def encode_message(message: dict[str, Any]) -> bytes:
    encoded = (json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message is too large")
    return encoded


def decode_message(payload: bytes) -> dict[str, Any]:
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message is too large")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid UTF-8 JSON message") from exc
    if not isinstance(decoded, dict):
        raise ProtocolError("response must be a JSON object")
    return decoded
