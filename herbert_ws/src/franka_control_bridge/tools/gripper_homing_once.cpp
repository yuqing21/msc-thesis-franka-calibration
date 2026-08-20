// Safety-gated one-shot Franka Hand homing using the compatible isolated libfranka.
#include <iostream>
#include <string>

#include <franka/exception.h>
#include <franka/gripper.h>

int main(int argc, char** argv) {
  if (argc != 3 || std::string(argv[2]) != "ENABLE_GRIPPER_HOMING") {
    std::cerr << "Usage: " << argv[0] << " <robot-ip> ENABLE_GRIPPER_HOMING\n";
    return 2;
  }
  try {
    franka::Gripper gripper(argv[1]);
    const franka::GripperState before = gripper.readOnce();
    std::cout << "before_width_m=" << before.width << '\n';
    std::cout << "before_max_width_m=" << before.max_width << '\n';
    if (!gripper.homing()) {
      std::cerr << "homing returned false\n";
      return 3;
    }
    const franka::GripperState after = gripper.readOnce();
    std::cout << "after_width_m=" << after.width << '\n';
    std::cout << "after_max_width_m=" << after.max_width << '\n';
    std::cout << "is_grasped=" << (after.is_grasped ? "true" : "false") << '\n';
  } catch (const franka::Exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
  return 0;
}
