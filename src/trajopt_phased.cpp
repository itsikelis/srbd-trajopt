#include <chrono>
#include <ctime>

#include <Eigen/Dense>

#include <iostream>
#include <memory>
#include <numeric>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <ifopt_sets/variables/phased_trajectory_vars.hpp>
#include <ifopt_sets/variables/trajectory_vars.hpp>
#include <srbd/srbd.hpp>
#include <terrain/terrain_grid.hpp>

#include <ifopt_sets/constraints/common/acceleration.hpp>

#include <ifopt_sets/constraints/phased/dynamics_phased.hpp>
#include <ifopt_sets/constraints/phased/foot_body_distance_phased.hpp>
#include <ifopt_sets/constraints/phased/foot_terrain_distance_phased.hpp>
#include <ifopt_sets/constraints/phased/friction_cone_phased.hpp>
#include <ifopt_sets/constraints/phased/phased_acceleration.hpp>

#include <utils/types.hpp>
#include <utils/utils.hpp>

int main()
{
    trajopt::SingleRigidBodyDynamicsModel model;
    trajopt::init_model_anymal(model);

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
    nlp.AddConstraintSet(std::make_shared<trajopt::DynamicsPhased>(model, numSamples, sampleTime));

    nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(posVars));
    nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(rotVars));

    // Add feet pos and force var sets.
    size_t numPosSteps = 2;
    size_t numForceSteps = 1;
    Eigen::Vector3d phaseTimes = {0.2, 0.1, 0.2};
    std::vector<size_t> posKnotsPerSwing = {3};
    std::vector<size_t> forceKnotsPerSwing = {3, 3};

    // size_t numPhasedKnots = numPosSteps + std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
    // size_t numPhasedVars = 3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
    double max_force = 2. * model.mass * std::abs(model.gravity[2]);
    auto initFootPosVals = Eigen::VectorXd(3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0));
    auto initFootForceVals = Eigen::VectorXd(3 * numForceSteps + 6 * std::accumulate(forceKnotsPerSwing.begin(), forceKnotsPerSwing.end(), 0));

    // std::cout << initFootPosVals.rows() << " , " << initFootForceVals.rows() << std::endl;
    // std::cout << numPhasedKnots << " , " << initFootForceVals.rows() << std::endl;

    ifopt::Component::VecBound footPosBounds(3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0), ifopt::NoBound);
    ifopt::Component::VecBound footForceBounds(3 * numForceSteps + 6 * std::accumulate(forceKnotsPerSwing.begin(), forceKnotsPerSwing.end(), 0), ifopt::Bounds(-max_force, max_force));

    for (size_t i = 0; i < model.numFeet; ++i) {
        auto footPosVars = std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_POS + "_" + std::to_string(i), initFootPosVals, footPosBounds, phaseTimes, posKnotsPerSwing, trajopt::rspl::Phase::Stance);
        nlp.AddVariableSet(footPosVars);

        nlp.AddConstraintSet(std::make_shared<trajopt::PhasedAccelerationConstraints>(footPosVars));
        // nlp.AddConstraintSet(std::make_shared<trajopt::FootPosTerrainConstraints>(footPosVars, terrain, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootTerrainDistancePhased>(footPosVars, terrain, numPosSteps, 1, posKnotsPerSwing));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootBodyDistancePhased>(model, posVars, rotVars, footPosVars, numSamples, sampleTime));

        auto footForceVars = std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_FORCE + "_" + std::to_string(i), initFootForceVals, footForceBounds, phaseTimes, forceKnotsPerSwing, trajopt::rspl::Phase::Swing);
        nlp.AddVariableSet(footForceVars);

        nlp.AddConstraintSet(std::make_shared<trajopt::FrictionConePhased>(footForceVars, footPosVars, terrain, numSamples, sampleTime));
    }

    std::cout << "Solving.." << std::endl;
    ifopt::IpoptSolver ipopt;
    ipopt.SetOption("jacobian_approximation", "exact");
    // ipopt.SetOption("jacobian_approximation", "finite-difference-values");
    ipopt.SetOption("max_cpu_time", 1e50);
    ipopt.SetOption("max_iter", static_cast<int>(1000));

    // Solve.
    auto t_start = std::chrono::high_resolution_clock::now();

    ipopt.Solve(nlp);
    // nlp.PrintCurrent();
    const auto t_end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(t_end - t_start);
    std::cout << "Wall clock time: " << duration.count() / 1000 << " seconds." << std::endl;

    return 0;
}
