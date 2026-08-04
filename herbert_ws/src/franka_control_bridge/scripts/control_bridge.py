#!/usr/bin/env python3
"""Receive Windows target previews and publish them for inspection only."""

import json
import math
import socket
import time
from typing import Any, Dict, Tuple

import rospy
from geometry_msgs.msg import PointStamped


SCHEMA = "cali-control/v1"


class RequestError(ValueError):
    pass


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError("{} must be numeric".format(name))
    result = float(value)
    if not math.isfinite(result):
        raise RequestError("{} must be finite".format(name))
    return result


class PreviewBridge:
    def __init__(self) -> None:
        self.bind_host = str(rospy.get_param("~bind_host", "100.68.210.77"))
        self.port = int(rospy.get_param("~port", 8765))
        self.allowed_client_ip = str(rospy.get_param("~allowed_client_ip", "100.126.29.85"))
        self.frame_id = str(rospy.get_param("~frame_id", "panda_link0"))
        self.max_message_bytes = int(rospy.get_param("~max_message_bytes", 16384))
        self.max_target_age_s = float(rospy.get_param("~max_target_age_s", 2.0))
        self.workspace_min = [float(value) for value in rospy.get_param("~workspace_min_m")]
        self.workspace_max = [float(value) for value in rospy.get_param("~workspace_max_m")]
        if len(self.workspace_min) != 3 or len(self.workspace_max) != 3:
            raise RuntimeError("workspace bounds must each contain three values")
        self.publisher = rospy.Publisher("/cali/preview_target", PointStamped, queue_size=1)

    def validate(self, message: Any) -> Dict[str, Any]:
        if not isinstance(message, dict):
            raise RequestError("message must be a JSON object")
        if message.get("schema") != SCHEMA:
            raise RequestError("unsupported schema")
        request_id = message.get("request_id")
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 64:
            raise RequestError("invalid request_id")
        sent_at_s = finite_number(message.get("sent_at_s"), "sent_at_s")
        message_type = message.get("type")
        if message_type == "ping":
            return {
                "schema": SCHEMA,
                "type": "ping",
                "request_id": request_id,
                "sent_at_s": sent_at_s,
            }
        if message_type != "preview_target":
            raise RequestError("unsupported command; robot motion is disabled")
        target = message.get("target_base_m")
        if not isinstance(target, list) or len(target) != 3:
            raise RequestError("target_base_m must contain three metre values")
        target = [finite_number(value, "target_base_m") for value in target]
        confidence = finite_number(message.get("confidence"), "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise RequestError("confidence must lie in [0, 1]")
        age_s = abs(time.time() - sent_at_s)
        if age_s > self.max_target_age_s:
            raise RequestError("preview target is stale")
        for axis, (value, lower, upper) in enumerate(zip(target, self.workspace_min, self.workspace_max)):
            if not lower <= value <= upper:
                raise RequestError("target axis {} lies outside preview workspace".format(axis))
        return {
            "schema": SCHEMA,
            "type": "preview_target",
            "request_id": request_id,
            "sent_at_s": sent_at_s,
            "target_base_m": target,
            "confidence": confidence,
        }

    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if message["type"] == "ping":
            return {
                "schema": SCHEMA,
                "type": "pong",
                "request_id": message["request_id"],
                "received_at_s": time.time(),
                "robot_control_enabled": False,
            }
        point = PointStamped()
        point.header.stamp = rospy.Time.now()
        point.header.frame_id = self.frame_id
        point.point.x, point.point.y, point.point.z = message["target_base_m"]
        self.publisher.publish(point)
        rospy.loginfo(
            "Preview target only: frame=%s xyz=[%.4f, %.4f, %.4f] confidence=%.3f",
            self.frame_id,
            point.point.x,
            point.point.y,
            point.point.z,
            message["confidence"],
        )
        return {
            "schema": SCHEMA,
            "type": "ack",
            "request_id": message["request_id"],
            "received_at_s": time.time(),
            "published_topic": "/cali/preview_target",
            "robot_control_enabled": False,
        }

    def receive_line(self, connection: socket.socket) -> bytes:
        chunks = []
        received = 0
        while True:
            chunk = connection.recv(4096)
            if not chunk:
                raise RequestError("connection closed before newline")
            chunks.append(chunk)
            received += len(chunk)
            if received > self.max_message_bytes:
                raise RequestError("message too large")
            if b"\n" in chunk:
                return b"".join(chunks).split(b"\n", 1)[0]

    @staticmethod
    def send(connection: socket.socket, response: Dict[str, Any]) -> None:
        connection.sendall((json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))

    def serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.bind_host, self.port))
            server.listen(4)
            server.settimeout(1.0)
            rospy.loginfo(
                "Preview-only bridge listening on %s:%d; allowed client=%s",
                self.bind_host,
                self.port,
                self.allowed_client_ip,
            )
            while not rospy.is_shutdown():
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(2.0)
                    request_id = "unknown"
                    try:
                        if self.allowed_client_ip and address[0] != self.allowed_client_ip:
                            raise RequestError("client IP is not allowed")
                        payload = self.receive_line(connection)
                        decoded = json.loads(payload.decode("utf-8"))
                        if isinstance(decoded, dict) and isinstance(decoded.get("request_id"), str):
                            request_id = decoded["request_id"][:64]
                        validated = self.validate(decoded)
                        self.send(connection, self.process(validated))
                    except (RequestError, UnicodeDecodeError, json.JSONDecodeError, socket.timeout) as exc:
                        rospy.logwarn("Rejected bridge request from %s: %s", address[0], exc)
                        self.send(
                            connection,
                            {
                                "schema": SCHEMA,
                                "type": "error",
                                "request_id": request_id,
                                "received_at_s": time.time(),
                                "error": str(exc),
                                "robot_control_enabled": False,
                            },
                        )


def main() -> None:
    rospy.init_node("franka_control_bridge")
    PreviewBridge().serve()


if __name__ == "__main__":
    main()

