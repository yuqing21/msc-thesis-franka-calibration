#!/usr/bin/env python3
"""Safety-gated Franka Hand grasp action client."""

import argparse
import json
import sys


CONFIRMATION_TEXT = "ENABLE_GRIPPER"


def bounded(value: float, lower: float, upper: float, name: str) -> float:
    value = float(value)
    if not lower <= value <= upper:
        raise ValueError("{} must lie in [{}, {}]".format(name, lower, upper))
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run or execute a Franka Hand grasp.")
    parser.add_argument("--width-m", type=float, required=True, help="Measured object width, not zero")
    parser.add_argument("--speed-m-s", type=float, default=0.02)
    parser.add_argument("--force-n", type=float, required=True)
    parser.add_argument("--epsilon-inner-m", type=float, default=0.005)
    parser.add_argument("--epsilon-outer-m", type=float, default=0.005)
    parser.add_argument("--server", default="/franka_gripper/grasp")
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--execute", action="store_true", help="Actually send the ROS action")
    parser.add_argument("--confirm", default="", help="Must equal ENABLE_GRIPPER when --execute is used")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        width = bounded(args.width_m, 0.001, 0.080, "width_m")
        speed = bounded(args.speed_m_s, 0.001, 0.100, "speed_m_s")
        force = bounded(args.force_n, 1.0, 70.0, "force_n")
        epsilon_inner = bounded(args.epsilon_inner_m, 0.0, 0.020, "epsilon_inner_m")
        epsilon_outer = bounded(args.epsilon_outer_m, 0.0, 0.020, "epsilon_outer_m")
    except ValueError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2

    plan = {
        "server": args.server,
        "width_m": width,
        "speed_m_s": speed,
        "force_n": force,
        "epsilon_inner_m": epsilon_inner,
        "epsilon_outer_m": epsilon_outer,
        "execute": args.execute,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("DRY RUN: no ROS action was sent")
        return 0
    if args.confirm != CONFIRMATION_TEXT:
        print("ERROR: --execute requires --confirm {}".format(CONFIRMATION_TEXT), file=sys.stderr)
        return 2

    import actionlib
    import rospy
    from franka_gripper.msg import GraspAction, GraspEpsilon, GraspGoal

    rospy.init_node("cali_gripper_grasp", anonymous=True)
    client = actionlib.SimpleActionClient(args.server, GraspAction)
    if not client.wait_for_server(rospy.Duration(5.0)):
        print("ERROR: gripper action server is unavailable", file=sys.stderr)
        return 3
    goal = GraspGoal()
    goal.width = width
    goal.speed = speed
    goal.force = force
    goal.epsilon = GraspEpsilon(inner=epsilon_inner, outer=epsilon_outer)
    client.send_goal(goal)
    if not client.wait_for_result(rospy.Duration(args.timeout_s)):
        client.cancel_goal()
        print("ERROR: grasp timed out and was cancelled", file=sys.stderr)
        return 4
    result = client.get_result()
    print("result={}".format(result))
    return 0 if bool(getattr(result, "success", False)) else 5


if __name__ == "__main__":
    raise SystemExit(main())

