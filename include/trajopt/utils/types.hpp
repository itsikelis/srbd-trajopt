#pragma once

#include <string>

namespace trajopt {
    static const std::string BODY_POS_TRAJECTORY = "body_pos_traj";
    static const std::string BODY_ROT_TRAJECTORY = "body_rot_traj";

    static const std::string FOOT_POS = "foot_pos_traj";
    static const std::string FOOT_FORCE = "foot_force_traj";

    static const std::string COST_MIN_EFFORT = "cost_min_eff";
    static const std::string COST_MIN_TIME = "cost_min_t";

    static const std::string BODY_DYNAMICS = "body_acc_dyn";

    static const std::string BODY_TRAJ_EQUAL_ACC = "body_eq_acc";
} // namespace trajopt
