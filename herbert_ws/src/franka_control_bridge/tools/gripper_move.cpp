// Safety-gated one-shot Franka Hand move using the isolated compatible libfranka.
#include <cmath>
#include <iostream>
#include <string>

#include <franka/exception.h>
#include <franka/gripper.h>

int main(int argc, char** argv) {
  if (argc != 5 || std::string(argv[4]) != "ENABLE_GRIPPER_MOVE") {
    std::cerr << "Usage: " << argv[0]
              << " <robot-ip> <width-m> <speed-m-s> ENABLE_GRIPPER_MOVE\n";
    return 2;
  }

  try {
    const double width = std::stod(argv[2]);
    const double speed = std::stod(argv[3]);
    if (!std::isfinite(width) || width < 0.0 || width > 0.080) {
      std::cerr << "width-m must lie in [0.0, 0.080]\n";
      return 2;
    }
    if (!std::isfinite(speed) || speed < 0.001 || speed > 0.100) {
      std::cerr << "speed-m-s must lie in [0.001, 0.100]\n";
      return 2;
    }

    franka::Gripper gripper(argv[1]);
    const franka::GripperState before = gripper.readOnce();
    std::cout << "before_width_m=" << before.width << '\n';
    if (!gripper.move(width, speed)) {
      std::cerr << "gripper move returned false\n";
      return 3;
    }
    const franka::GripperState after = gripper.readOnce();
    std::cout << "after_width_m=" << after.width << '\n';
    std::cout << "max_width_m=" << after.max_width << '\n';
  } catch (const franka::Exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 2;
  }
  return 0;
}
