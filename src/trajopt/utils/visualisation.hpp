#pragma once

#include <string>

#include <robot_dart/gui/magnum/graphics.hpp>
#include <robot_dart/robot_dart_simu.hpp>

#include <ifopt/problem.h>

#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/srbd/srbd.hpp>

#include <trajopt/utils/types.hpp>

namespace trajopt {
    template <class FootTrajectoryVars>
    inline void visualise(const ifopt::Problem& nlp, const SingleRigidBodyDynamicsModel& model, double totalTime, double dt, const std::string vidName = "")
    {
        robot_dart::RobotDARTSimu simu(dt);
        auto graphics = std::make_shared<robot_dart::gui::magnum::Graphics>();
        simu.set_graphics(graphics);
        // simu.set_graphics_freq(500);
        if (vidName != "") {
            graphics->record_video(std::string(SRCPATH) + vidName);
        }

        simu.add_floor();
        // auto terrain = std::make_shared<robot_dart::Robot>(std::string(SRCPATH) + "/step-terrain.urdf");
        // simu.add_visual_robot(terrain);

        auto robot = robot_dart::Robot::create_box(Eigen::Vector3d(0.6, 0.2, 0.15), Eigen::Isometry3d::Identity(), "free");
        simu.add_visual_robot(robot);

        for (size_t k = 0; k < model.numFeet; ++k) {
            auto foot = robot_dart::Robot::create_ellipsoid(Eigen::Vector3d(0.05, 0.05, 0.05), Eigen::Isometry3d::Identity(), "free", 0.1, dart::Color::Red(1.0), "foot" + std::to_string(k));
            simu.add_visual_robot(foot);
        }
        // Visualise trajectory.
        size_t iters = totalTime / dt + 1;

        double t = 0.;
        for (size_t i = 0; i < iters; ++i) {
            Eigen::Vector3d bodyPos = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::BODY_POS_TRAJECTORY))->trajectoryEval(t, 0);
            Eigen::Vector3d bodyRot = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::BODY_ROT_TRAJECTORY))->trajectoryEval(t, 0);

            Eigen::AngleAxisd z_rot(bodyRot[0], Eigen::Vector3d::UnitZ());
            Eigen::AngleAxisd y_rot(bodyRot[1], Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd x_rot(bodyRot[2], Eigen::Vector3d::UnitX());
            auto tf = Eigen::Isometry3d::Identity();
            tf.rotate(z_rot);
            tf.rotate(y_rot);
            tf.rotate(x_rot);

            auto rot = dart::math::logMap(tf);
            simu.robot(1)->set_positions(robot_dart::make_vector({rot[0], rot[1], rot[2], bodyPos[0], bodyPos[1], bodyPos[2]}));

            for (size_t k = 0; k < model.numFeet; ++k) {
                Eigen::Vector3d footPos = std::static_pointer_cast<FootTrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(k)))->trajectoryEval(t, 0);
                simu.robot(k + 2)->set_positions(robot_dart::make_vector({0., 0., 0., footPos[0], footPos[1], footPos[2] + 0.025}));
            }

            simu.step_world();

            t += dt;
        }
        simu.run(1.);
    }

} // namespace trajopt
