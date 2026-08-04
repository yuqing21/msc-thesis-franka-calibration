from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any

from .protocol import MAX_MESSAGE_BYTES, ProtocolError, decode_message, encode_message, make_ping


@dataclass(frozen=True)
class ClockSync:
    offset_s: float
    round_trip_s: float
    response: dict[str, Any]


@dataclass(frozen=True)
class JsonLineClient:
    host: str
    port: int = 8765
    timeout_s: float = 3.0

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        payload = encode_message(message)
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as connection:
            connection.settimeout(self.timeout_s)
            connection.sendall(payload)
            chunks: list[bytes] = []
            received = 0
            while True:
                chunk = connection.recv(4096)
                if not chunk:
                    raise ProtocolError("control bridge closed before sending a newline response")
                chunks.append(chunk)
                received += len(chunk)
                if received > MAX_MESSAGE_BYTES:
                    raise ProtocolError("response is too large")
                if b"\n" in chunk:
                    break
        line = b"".join(chunks).split(b"\n", maxsplit=1)[0]
        return decode_message(line)

    def estimate_clock_offset(self) -> ClockSync:
        """Estimate Herbert wall-clock offset using an NTP-style midpoint."""

        started_at_s = time.time()
        response = self.request(make_ping())
        finished_at_s = time.time()
        if response.get("type") != "pong":
            raise ProtocolError("control bridge did not return pong during clock synchronization")
        remote_received_at_s = response.get("received_at_s")
        if isinstance(remote_received_at_s, bool) or not isinstance(remote_received_at_s, (int, float)):
            raise ProtocolError("pong is missing numeric received_at_s")
        midpoint_s = (started_at_s + finished_at_s) / 2.0
        return ClockSync(
            offset_s=float(remote_received_at_s) - midpoint_s,
            round_trip_s=finished_at_s - started_at_s,
            response=response,
        )
