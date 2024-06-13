#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <trajopt/utils/utils.hpp>

#include <trajopt/utils/test_problems/create_implicit_nlp.hpp>
#include <trajopt/utils/test_problems/create_pendulum_nlp.hpp>
#include <trajopt/utils/test_problems/create_phased_nlp.hpp>

#if (VIZ)
#include <robot_dart/gui/magnum/graphics.hpp>
#include <robot_dart/robot_dart_simu.hpp>
#endif

static constexpr size_t numKnots = 10;
static constexpr size_t numSamples = 16;

static constexpr double totalTime = 0.5;

// Body position
static constexpr double initBodyPosX = 0.;
static constexpr double initBodyPosY = 0.;
static constexpr double initBodyPosZ = 0.5;

static constexpr double targetBodyPosX = 0.2;
static constexpr double targetBodyPosY = 0.;
static constexpr double targetBodyPosZ = 0.5;

int main()
{
    std::srand(std::time(0));

    trajopt::SingleRigidBodyDynamicsModel model;
    trajopt::init_model_anymal(model);
    // trajopt::init_model_biped(model);

    trajopt::TerrainGrid terrain(200, 200, 1., -100, -100, 100, 100);
    terrain.set_zero();

    Eigen::Vector3d initBodyPos = Eigen::Vector3d(initBodyPosX, initBodyPosY, initBodyPosZ + terrain.height(initBodyPosX, initBodyPosY));
    Eigen::Vector3d targetBodyPos = Eigen::Vector3d(targetBodyPosX, targetBodyPosY, targetBodyPosZ + terrain.height(targetBodyPosX, targetBodyPosY));

    // ifopt::Problem nlp = create_phased_nlp(numKnots, numSamples, totalTime, initBodyPos, targetBodyPos, model, terrain);
    ifopt::Problem nlp = create_implicit_nlp(numKnots, numSamples, totalTime, initBodyPos, targetBodyPos, model, terrain);
    // ifopt::Problem nlp = trajopt::create_pendulum_nlp();

    std::cout << "Solving.." << std::endl;
    ifopt::IpoptSolver ipopt;
    ipopt.SetOption("jacobian_approximation", "exact");
    ipopt.SetOption("max_cpu_time", 1e50);
    ipopt.SetOption("max_iter", static_cast<int>(1000));

    // Solve.
    auto tStart = std::chrono::high_resolution_clock::now();

    ipopt.Solve(nlp);
    nlp.PrintCurrent();
    const auto tEnd = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(tEnd - tStart);
    std::cout << "Wall clock time: " << duration.count() / 1000 << " seconds." << std::endl;

    // Print variables.
    //
    // double dt = 0.;
    // for (size_t i = 0; i < numSamples; ++i) {
    //     std::cout << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::BODY_POS_TRAJECTORY))->trajectoryEval(dt, 0).transpose() << std::endl;
    //
    //     dt += sampleTime;
    // }
    //
    // std::cout << std::endl;
    //
    // for (size_t k = 0; k < 4; k++) {
    //
    //     dt = 0.;
    //     for (size_t i = 0; i < numSamples; ++i) {
    //         std::cout << "p_" << k << ": " << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(k)))->trajectoryEval(dt, 0).transpose() << std::endl;
    //
    //         std::cout << "v_" << k << ": " << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(k)))->trajectoryEval(dt, 1).transpose() << std::endl;
    //
    //         dt += sampleTime;
    //     }
    //
    //     std::cout << std::endl;
    //
    //     dt = 0.;
    //     for (size_t i = 0; i < numSamples; ++i) {
    //         std::cout << "f_" << k << ": " << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_FORCE + "_" + std::to_string(k)))->trajectoryEval(dt, 0).transpose() << std::endl;
    //
    //         dt += sampleTime;
    //     }
    //
    //     std::cout << std::endl;
    // }

#if (VIZ)
    robot_dart::RobotDARTSimu simu(0.001);
    auto graphics = std::make_shared<robot_dart::gui::magnum::Graphics>();
    simu.set_graphics(graphics);
    // simu.set_graphics_freq(500);
    graphics->record_video(std::string(SRCPATH) + "srbd_implicit.mp4");

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
    double dt = 0.001;
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
            Eigen::Vector3d footPos = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(k)))->trajectoryEval(t, 0);
            simu.robot(k + 2)->set_positions(robot_dart::make_vector({0., 0., 0., footPos[0], footPos[1], footPos[2] + 0.025}));
        }

        simu.step_world();

        t += dt;
    }
    simu.run(1.);

#endif

    return 0;
}
