#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <ifopt_sets/variables/trajectory_vars.hpp>
#include <srbd/srbd.hpp>
#include <terrain/terrain_grid.hpp>

#include <ifopt_sets/cost/min_effort.hpp>

#include <ifopt_sets/constraints/common/acceleration.hpp>
#include <ifopt_sets/constraints/common/dynamics.hpp>
#include <ifopt_sets/constraints/common/friction_cone.hpp>

#include <ifopt_sets/constraints/contact_implicit/foot_body_distance_implicit.hpp>
#include <ifopt_sets/constraints/contact_implicit/foot_terrain_distance_implicit.hpp>
#include <ifopt_sets/constraints/contact_implicit/implicit_contact.hpp>
#include <ifopt_sets/constraints/contact_implicit/implicit_velocity.hpp>

#include <utils/types.hpp>
#include <utils/utils.hpp>

// #include <robot_dart/gui/magnum/graphics.hpp>
// #include <robot_dart/robot_dart_simu.hpp>

#define VISUALISE 0

using Eigen::Vector3d;
using Eigen::VectorXd;

static constexpr size_t numKnots = 40;
static constexpr size_t numSamples = 40;

static constexpr double totalTime = 0.5;

// Body position
static constexpr double initBodyPosX = 0.;
static constexpr double initBodyPosY = 0.;
static constexpr double initBodyPosZ = 0.;

static constexpr double targetBodyPosX = 0.2;
static constexpr double targetBodyPosY = 0.;
static constexpr double targetBodyPosZ = 0.;

int main()
{
    trajopt::SingleRigidBodyDynamicsModel model;
    trajopt::init_model_anymal(model);
    // trajopt::init_model_biped(model);

    // TODO: Handle zero grids.
    trajopt::TerrainGrid terrain(200, 200, 1., -100, -100, 100, 100);
    terrain.set_zero();

    // Add variable sets.
    ifopt::Problem nlp;

    double sampleTime = totalTime / static_cast<double>(numSamples - 1.);
    auto polyTimes = VectorXd(numKnots - 1);
    for (size_t i = 0; i < static_cast<size_t>(polyTimes.size()); ++i) {
        polyTimes[i] = totalTime / static_cast<double>(numKnots - 1);
    }

    // Add body pos and rot var sets.
    Vector3d initBodyPos = Vector3d(initBodyPosX, initBodyPosY, initBodyPosZ + terrain.height(initBodyPosX, initBodyPosY));
    Vector3d targetBodyPos = Vector3d(targetBodyPosX, targetBodyPosY, targetBodyPosZ + terrain.height(targetBodyPosX, targetBodyPosX));
    auto bodyPosBounds = trajopt::fillBoundVector(initBodyPos, targetBodyPos, ifopt::NoBound, 6 * numKnots);
    auto initBodyPosVals = VectorXd(3 * 2 * numKnots);

    auto posVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_POS_TRAJECTORY, initBodyPosVals, polyTimes, bodyPosBounds);
    nlp.AddVariableSet(posVars);

    Vector3d initRotPos = Vector3d::Zero();
    Vector3d targetRotPos = Vector3d::Zero();
    auto bodyRotBounds = trajopt::fillBoundVector(initRotPos, targetRotPos, ifopt::NoBound, 6 * numKnots);
    auto initBodyRotVals = VectorXd(3 * 2 * numKnots);

    auto rotVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_ROT_TRAJECTORY, initBodyRotVals, polyTimes, bodyRotBounds);
    nlp.AddVariableSet(rotVars);

    // Add feet variable sets.
    auto initFootPosVals = VectorXd(3 * 2 * numKnots);
    auto initFootForceVals = VectorXd(3 * 2 * numKnots);

    ifopt::Component::VecBound footPosBounds(6 * numKnots, ifopt::NoBound);
    // ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::NoBound);
    double max_force = 2. * model.mass * std::abs(model.gravity[2]);
    ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::Bounds(-max_force, max_force));
    // for (size_t i = 0; i < numKnots; ++i) {
    //     footForceBounds.at(i * 6 + 2) = ifopt::Bounds(-1e-6, max_force);
    // }

    for (size_t i = 0; i < model.numFeet; ++i) {
        // Add initial and final positions for each foot.
        footPosBounds[0] = (i == 0 || i == 1) ? ifopt::Bounds(0.34, 0.34) : ifopt::Bounds(-0.34, -0.34);
        footPosBounds[1] = (i == 1 || i == 3) ? ifopt::Bounds(0.19, 0.19) : ifopt::Bounds(-0.19, -0.19);

        footPosBounds[6 * numKnots - 3] = (i == 0 || i == 1) ? ifopt::Bounds(targetBodyPos(0) + 0.34, targetBodyPos(0) + 0.34) : ifopt::Bounds(targetBodyPos(0) - 0.34, targetBodyPos(0) - 0.34);
        footPosBounds[6 * numKnots - 2] = (i == 1 || i == 3) ? ifopt::Bounds(targetBodyPos(1) + 0.19, targetBodyPos(1) + 0.19) : ifopt::Bounds(targetBodyPos(1) - 0.19, targetBodyPos(1) - 0.19);

        auto footPosVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_POS + "_" + std::to_string(i), initFootPosVals, polyTimes, footPosBounds);
        nlp.AddVariableSet(footPosVars);

        auto footForceVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_FORCE + "_" + std::to_string(i), initFootForceVals, polyTimes, footForceBounds);
        nlp.AddVariableSet(footForceVars);
    }

    // Add regular constraint sets.
    nlp.AddConstraintSet(std::make_shared<trajopt::Dynamics<trajopt::TrajectoryVars>>(model, numSamples, sampleTime));

    nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(posVars));
    nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(rotVars));

    // ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::NoBound);

    for (size_t i = 0; i < model.numFeet; ++i) {
        auto footPosVars = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(i)));
        auto footForceVars = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_FORCE + "_" + std::to_string(i)));

        nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(footPosVars));
        nlp.AddConstraintSet(std::make_shared<trajopt::FrictionCone<trajopt::TrajectoryVars>>(footForceVars, footPosVars, terrain, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootBodyDistanceImplicit>(model, posVars, rotVars, footPosVars, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootTerrainDistanceImplicit>(footPosVars, terrain, numKnots));
        nlp.AddConstraintSet(std::make_shared<trajopt::ImplicitContactConstraints>(footPosVars, footForceVars, terrain, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::ImplicitVelocityConstraints>(footPosVars, footForceVars, terrain, numSamples, sampleTime));

        // Add cost set
        nlp.AddCostSet(std::make_shared<trajopt::MinEffort<trajopt::TrajectoryVars>>(footPosVars, numKnots));
    }

    std::cout << "Solving.." << std::endl;
    ifopt::IpoptSolver ipopt;
    ipopt.SetOption("jacobian_approximation", "exact");
    ipopt.SetOption("max_cpu_time", 1e50);
    ipopt.SetOption("max_iter", static_cast<int>(300));

    // Solve.
    auto tStart = std::chrono::high_resolution_clock::now();

    ipopt.Solve(nlp);
    nlp.PrintCurrent();
    const auto tEnd = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(tEnd - tStart);
    std::cout << "Wall clock time: " << duration.count() / 1000 << " seconds." << std::endl;

    // Print variables.

    double dt = 0.;
    for (size_t i = 0; i < numSamples; ++i) {
        std::cout << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::BODY_POS_TRAJECTORY))->trajectoryEval(dt, 0).transpose() << std::endl;

        dt += sampleTime;
    }

    std::cout << std::endl;

    dt = 0.;
    for (size_t i = 0; i < numSamples; ++i) {
        std::cout << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_0"))->trajectoryEval(dt, 0).transpose() << std::endl;

        dt += sampleTime;
    }

    std::cout << std::endl;

    dt = 0.;
    for (size_t i = 0; i < numSamples; ++i) {
        std::cout << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_FORCE + "_0"))->trajectoryEval(dt, 0).transpose() << std::endl;

        dt += sampleTime;
    }

    std::cout << std::endl;
    // auto body_pos = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::BODY_POS_TRAJECTORY))->GetValues();
    //
    // for (size_t i = 0; i < numKnots; ++i) {
    //     std::cout << body_pos[i * 6 + 0] << ", ";
    //     std::cout << body_pos[i * 6 + 1] << ", ";
    //     std::cout << body_pos[i * 6 + 2] << std::endl;
    // }
    // std::cout << std::endl;
    //
    // auto foot_pos = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_0"))->GetValues();
    //
    // for (size_t i = 0; i < numKnots; ++i) {
    //     std::cout << foot_pos[i * 6 + 0] << ", ";
    //     std::cout << foot_pos[i * 6 + 1] << ", ";
    //     std::cout << foot_pos[i * 6 + 2] << std::endl;
    // }
    // std::cout << std::endl;
    //
    // auto foot_force = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_FORCE + "_0"))->GetValues();
    //
    // for (size_t i = 0; i < numKnots; ++i) {
    //     std::cout << foot_force[i * 6 + 0] << ", ";
    //     std::cout << foot_force[i * 6 + 1] << ", ";
    //     std::cout << foot_force[i * 6 + 2] << std::endl;
    // }
    // std::cout << std::endl;

#if (VISUALISE)
    // Part B: Visualise in simulation.
    // Load robot.
    auto quad = std::make_shared<robot_dart::Robot>(std::string(SRCPATH) + "/../urdf/drone.urdf");
    quad->free_from_world();

    // Create simulation world.
    robot_dart::RobotDARTSimu simu(0.005);
    auto graphics = std::make_shared<robot_dart::gui::magnum::Graphics>();
    simu.set_graphics(graphics);
    // simu.set_graphics_freq(0.01);
    graphics->record_video(std::string(SRCPATH) + "/planar-quad-gasfadfaef.mp4");
    // simu.set_graphics_freq(static_cast<int>(1. / DT));

    rspl::Trajectory<1> trajectory_x = std::static_pointer_cast<planar_quad::XTrajectory>(nlp.GetOptVariables()->GetComponent(planar_quad::X_VARIABLES))->GetTrajectory();
    rspl::Trajectory<1> trajectory_y = std::static_pointer_cast<planar_quad::YTrajectory>(nlp.GetOptVariables()->GetComponent(planar_quad::Y_VARIABLES))->GetTrajectory();
    rspl::Trajectory<1> trajectory_theta = std::static_pointer_cast<planar_quad::ThetaTrajectory>(nlp.GetOptVariables()->GetComponent(planar_quad::THETA_VARIABLES))->GetTrajectory();

    auto jac = quad->get_jacobian();

    simu.add_visual_robot(quad);
    // simu.add_checkerboard_floor();
    t = 0.;
    // Visualise trajectory.
    while (t <= planar_quad::DURATION) {
        quad->set_positions(robot_dart::make_vector({0., -trajectory_theta.position(t)(0), 0., trajectory_x.position(t)(0), 0., trajectory_y.position(t)(0)}));
        if (simu.step_world())
            break;
        // std::cout << trajectory_theta.position(t)(0) << std::endl;
        t += simu.timestep();
    }
#endif

    return 0;
}
