// Read one Franka arm state without loading a controller or issuing motion.
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include <franka/exception.h>
#include <franka/robot.h>

template <std::size_t N>
void print_array(const char* name, const std::array<double, N>& values, bool trailing_comma = true) {
  std::cout << "\"" << name << "\":[";
  for (std::size_t index = 0; index < N; ++index) {
    if (index > 0) {
      std::cout << ',';
    }
    std::cout << values[index];
  }
  std::cout << ']';
  if (trailing_comma) {
    std::cout << ',';
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-ip>\n";
    return 2;
  }
  try {
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    const franka::RobotState state = robot.readOnce();
    const double max_abs_dq = *std::max_element(
        state.dq.begin(), state.dq.end(), [](double left, double right) {
          return std::abs(left) < std::abs(right);
        });

    std::cout << std::setprecision(12);
    std::cout << "{\"schema\":\"franka-state-once/v1\",\"robot_ip\":\"" << argv[1] << "\",";
    print_array("O_T_EE", state.O_T_EE);
    print_array("F_T_EE", state.F_T_EE);
    print_array("q", state.q);
    print_array("dq", state.dq);
    std::cout << "\"max_abs_dq_rad_s\":" << std::abs(max_abs_dq) << ',';
    std::cout << "\"current_errors_empty\":" << (state.current_errors ? "false" : "true") << ',';
    std::cout << "\"last_motion_errors_empty\":" << (state.last_motion_errors ? "false" : "true");
    std::cout << "}\n";
  } catch (const franka::Exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
  return 0;
}
