#!/usr/bin/env python3
"""Safety-gated Franka Hand homing action client."""

import argparse
import sys


CONFIRMATION_TEXT = "ENABLE_HOMING"


def main() -> int:
    parser = argparse.ArgumentParser(description="Home the Franka Hand.")
    parser.add_argument("--server", default="/franka_gripper/homing")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    print("server={}".format(args.server))
    print("execute={}".format(args.execute))
    if not args.execute:
        print("DRY RUN: no ROS action was sent")
        return 0
    if args.confirm != CONFIRMATION_TEXT:
        print(
            "ERROR: --execute requires --confirm {}".format(CONFIRMATION_TEXT),
            file=sys.stderr,
        )
        return 2

    import actionlib
    import rospy
    from franka_gripper.msg import HomingAction, HomingGoal

    rospy.init_node("cali_gripper_homing", anonymous=True)
    client = actionlib.SimpleActionClient(args.server, HomingAction)
    if not client.wait_for_server(rospy.Duration(5.0)):
        print("ERROR: homing action server is unavailable", file=sys.stderr)
        return 3
    client.send_goal(HomingGoal())
    if not client.wait_for_result(rospy.Duration(args.timeout_s)):
        client.cancel_goal()
        print("ERROR: homing timed out and was cancelled", file=sys.stderr)
        return 4
    result = client.get_result()
    print("result={}".format(result))
    return 0 if bool(getattr(result, "success", False)) else 5


if __name__ == "__main__":
    raise SystemExit(main())
