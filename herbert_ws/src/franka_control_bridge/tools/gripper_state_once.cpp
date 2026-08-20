// Read one Franka Hand state without issuing a command.
#include <iostream>
#include <string>

#include <franka/exception.h>
#include <franka/gripper.h>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-ip>\n";
    return 2;
  }
  try {
    franka::Gripper gripper(argv[1]);
    const franka::GripperState state = gripper.readOnce();
    std::cout << "width_m=" << state.width << '\n';
    std::cout << "max_width_m=" << state.max_width << '\n';
    std::cout << "is_grasped=" << (state.is_grasped ? "true" : "false") << '\n';
    std::cout << "temperature_c=" << state.temperature << '\n';
  } catch (const franka::Exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
  return 0;
}
