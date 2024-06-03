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

#include <ifopt_sets/constraints/contact_implicit/foot_body_distance_implicit.hpp>
#include <ifopt_sets/constraints/contact_implicit/foot_terrain_distance_implicit.hpp>
#include <ifopt_sets/constraints/contact_implicit/friction_cone_implicit.hpp>
#include <ifopt_sets/constraints/contact_implicit/implicit_contact.hpp>
#include <ifopt_sets/constraints/contact_implicit/implicit_velocity.hpp>

#include <utils/types.hpp>
#include <utils/utils.hpp>

int main()
{
    trajopt::SingleRigidBodyDynamicsModel model;
    trajopt::init_model_anymal(model);

    // TODO: Handle zero grids.
    trajopt::TerrainGrid terrain(200, 200, 1., -100, -100, 100, 100);
    std::vector<double> grid;
    grid.resize(200 * 200);
    for (auto& item : grid) {
        item = 0.;
    }
    terrain.set_grid(grid);

    // Add variable sets.
    ifopt::Problem nlp;

    static constexpr size_t numKnots = 20;
    size_t numSamples = 20;

    double totalTime = 0.5;
    double sampleTime = totalTime / static_cast<double>(numSamples - 1.);
    auto polyTimes = Eigen::VectorXd(numKnots - 1);
    for (size_t i = 0; i < static_cast<size_t>(polyTimes.size()); ++i) {
        polyTimes[i] = totalTime / static_cast<double>(numKnots - 1);
    }

    // Add body pos and rot var sets.
    Eigen::Vector3d initBodyPos = Eigen::Vector3d(0., 0., 0.5 + terrain.height(0., 0.));
    Eigen::Vector3d targetBodyPos = Eigen::Vector3d(0., 0., 0.5 + terrain.height(0., 0.));
    ifopt::Component::VecBound bodyPosBounds = trajopt::fillBoundVector(initBodyPos, targetBodyPos, ifopt::NoBound, 6 * numKnots);
    auto initBodyPosVals = Eigen::VectorXd(3 * 2 * numKnots);

    auto posVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_POS_TRAJECTORY, initBodyPosVals, polyTimes, bodyPosBounds);
    nlp.AddVariableSet(posVars);

    Eigen::Vector3d initRotPos = Eigen::Vector3d::Zero();
    Eigen::Vector3d targetRotPos = Eigen::Vector3d::Zero();
    ifopt::Component::VecBound bodyRotBounds = trajopt::fillBoundVector(initRotPos, targetRotPos, ifopt::NoBound, 6 * numKnots);
    auto initBodyRotVals = Eigen::VectorXd(3 * 2 * numKnots);
    auto rotVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_ROT_TRAJECTORY, initBodyRotVals, polyTimes, bodyRotBounds);
    nlp.AddVariableSet(rotVars);

    // // Add regular constraint sets.
    auto dynamConstr = std::make_shared<trajopt::Dynamics<trajopt::TrajectoryVars>>(model, numSamples, sampleTime);
    nlp.AddConstraintSet(dynamConstr);

    nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(posVars));
    nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(rotVars));

    // size_t numPhasedKnots = numPosSteps + std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
    // size_t numPhasedVars = 3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
    double max_force = 2. * model.mass * std::abs(model.gravity[2]);
    auto initFootPosVals = Eigen::VectorXd(3 * 2 * numKnots);
    auto initFootForceVals = Eigen::VectorXd(3 * 2 * numKnots);

    // std::cout << initFootPosVals.rows() << " , " << initFootForceVals.rows() << std::endl;
    // std::cout << numPhasedKnots << " , " << initFootForceVals.rows() << std::endl;

    ifopt::Component::VecBound footPosBounds(6 * numKnots, ifopt::NoBound);
    // ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::NoBound);
    ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::Bounds(-max_force, max_force));

    for (size_t i = 0; i < model.numFeet; ++i) {
        // Add initial and final positions for each foot.
        footPosBounds[0] = (i == 0 || i == 1) ? ifopt::Bounds(0.34, 0.34) : ifopt::Bounds(-0.34, -0.34);
        footPosBounds[1] = (i == 1 || i == 3) ? ifopt::Bounds(0.19, 0.19) : ifopt::Bounds(-0.19, -0.19);

        footPosBounds[6 * numKnots - 3] = (i == 0 || i == 1) ? ifopt::Bounds(targetBodyPos[0] + 0.34, targetBodyPos[0] + 0.34) : ifopt::Bounds(targetBodyPos[0] - 0.34, targetBodyPos[0] - 0.34);
        footPosBounds[6 * numKnots - 2] = (i == 1 || i == 3) ? ifopt::Bounds(targetBodyPos[1] + 0.19, targetBodyPos[1] + 0.19) : ifopt::Bounds(targetBodyPos[1] - 0.19, targetBodyPos[1] - 0.19);

        auto footPosVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_POS + "_" + std::to_string(i), initFootPosVals, polyTimes, footPosBounds);
        nlp.AddVariableSet(footPosVars);

        nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(footPosVars));

        auto footForceVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_FORCE + "_" + std::to_string(i), initFootForceVals, polyTimes, footForceBounds);
        nlp.AddVariableSet(footForceVars);

        nlp.AddConstraintSet(std::make_shared<trajopt::FrictionConeImplicit>(footForceVars, footPosVars, terrain, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootBodyDistanceImplicit>(model, posVars, rotVars, footPosVars, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootTerrainDistanceImplicit>(footPosVars, terrain, numKnots));
        nlp.AddConstraintSet(std::make_shared<trajopt::ImplicitContactConstraints>(footPosVars, footForceVars, terrain, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::ImplicitVelocityConstraints>(footPosVars, footForceVars, terrain, numSamples, sampleTime));

        // Add cost set
        nlp.AddCostSet(std::make_shared<trajopt::MinEffort<trajopt::TrajectoryVars>>(footForceVars, numKnots));
    }

    std::cout << "Solving.." << std::endl;
    ifopt::IpoptSolver ipopt;
    ipopt.SetOption("jacobian_approximation", "exact");
    // ipopt.SetOption("jacobian_approximation", "finite-difference-values");
    ipopt.SetOption("max_cpu_time", 1e50);
    ipopt.SetOption("max_iter", static_cast<int>(300));

    // Solve.
    auto t_start = std::chrono::high_resolution_clock::now();

    ipopt.Solve(nlp);
    nlp.PrintCurrent();
    const auto t_end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(t_end - t_start);
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

    return 0;
}
