// Safety-gated one-shot Franka Hand grasp using the isolated compatible libfranka.
#include <cmath>
#include <iostream>
#include <string>

#include <franka/exception.h>
#include <franka/gripper.h>

int main(int argc, char** argv) {
  if (argc != 6 || std::string(argv[5]) != "ENABLE_GRIPPER_GRASP") {
    std::cerr << "Usage: " << argv[0]
              << " <robot-ip> <width-m> <speed-m-s> <force-n> ENABLE_GRIPPER_GRASP\n";
    return 2;
  }
  try {
    const double width = std::stod(argv[2]);
    const double speed = std::stod(argv[3]);
    const double force = std::stod(argv[4]);
    if (!std::isfinite(width) || width < 0.001 || width > 0.080 ||
        !std::isfinite(speed) || speed < 0.001 || speed > 0.100 ||
        !std::isfinite(force) || force < 1.0 || force > 10.0) {
      std::cerr << "unsafe or invalid grasp parameter\n";
      return 2;
    }

    franka::Gripper gripper(argv[1]);
    const franka::GripperState before = gripper.readOnce();
    std::cout << "before_width_m=" << before.width << '\n';
    const bool success = gripper.grasp(width, speed, force, 0.008, 0.008);
    const franka::GripperState after = gripper.readOnce();
    std::cout << "success=" << (success ? "true" : "false") << '\n';
    std::cout << "after_width_m=" << after.width << '\n';
    std::cout << "is_grasped=" << (after.is_grasped ? "true" : "false") << '\n';
    return success ? 0 : 3;
  } catch (const franka::Exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 2;
  }
}
