#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <trajopt/ifopt_sets/cost/min_effort.hpp>

#include <trajopt/ifopt_sets/constraints/common/acceleration.hpp>
#include <trajopt/ifopt_sets/constraints/common/dynamics.hpp>
#include <trajopt/ifopt_sets/constraints/common/friction_cone.hpp>

#include <trajopt/ifopt_sets/constraints/contact_implicit/foot_body_distance_implicit.hpp>
#include <trajopt/ifopt_sets/constraints/contact_implicit/foot_terrain_distance_implicit.hpp>
#include <trajopt/ifopt_sets/constraints/contact_implicit/implicit_contact.hpp>
#include <trajopt/ifopt_sets/constraints/contact_implicit/implicit_velocity.hpp>

#include <trajopt/utils/types.hpp>
#include <trajopt/utils/utils.hpp>

using Eigen::Vector3d;
using Eigen::VectorXd;

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
    trajopt::SingleRigidBodyDynamicsModel model;
    trajopt::init_model_anymal(model);
    // trajopt::init_model_biped(model);

    // TODO: Handle zero grids.
    trajopt::TerrainGrid terrain(200, 200, 1., -100, -100, 100, 100);
    terrain.set_zero();

    // Add variable sets.
    ifopt::Problem nlp;

    double sampleTime = totalTime / static_cast<double>(numSamples - 1.);
    VectorXd polyTimes = VectorXd::Zero(numKnots - 1);
    for (size_t i = 0; i < static_cast<size_t>(polyTimes.size()); ++i) {
        polyTimes[i] = totalTime / static_cast<double>(numKnots - 1);
    }

    // Add body pos and rot var sets.
    Vector3d initBodyPos = Vector3d(initBodyPosX, initBodyPosY, initBodyPosZ + terrain.height(initBodyPosX, initBodyPosY));
    Vector3d targetBodyPos = Vector3d(targetBodyPosX, targetBodyPosY, targetBodyPosZ + terrain.height(targetBodyPosX, targetBodyPosX));
    auto bodyPosBounds = trajopt::fillBoundVector(initBodyPos, targetBodyPos, ifopt::NoBound, 6 * numKnots);
    VectorXd initBodyPosVals = VectorXd::Zero(3 * 2 * numKnots);

    auto posVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_POS_TRAJECTORY, initBodyPosVals, polyTimes, bodyPosBounds);
    nlp.AddVariableSet(posVars);

    Vector3d initRotPos = Vector3d::Zero();
    Vector3d targetRotPos = Vector3d::Zero();
    auto bodyRotBounds = trajopt::fillBoundVector(initRotPos, targetRotPos, ifopt::NoBound, 6 * numKnots);
    VectorXd initBodyRotVals = VectorXd::Zero(3 * 2 * numKnots);

    auto rotVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_ROT_TRAJECTORY, initBodyRotVals, polyTimes, bodyRotBounds);
    nlp.AddVariableSet(rotVars);

    // Add feet variable sets.
    VectorXd initFootPosVals = VectorXd::Zero(3 * 2 * numKnots);
    VectorXd initFootForceVals = VectorXd::Zero(3 * 2 * numKnots);

    ifopt::Component::VecBound footPosBounds(6 * numKnots, ifopt::NoBound);
    // ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::NoBound);
    double max_force = 2. * model.mass * std::abs(model.gravity[2]);
    ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::Bounds(-max_force, max_force));
    // for (size_t i = 0; i < numKnots; ++i) {
    //     footForceBounds.at(i * 6 + 2) = ifopt::Bounds(-1e-6, max_force);
    // }

    for (size_t i = 0; i < model.numFeet; ++i) {
        // Add initial and final positions for each foot.
        // right feet
        if (i == 0 || i == 2) {
            double bound_x = initBodyPos(0);
            if (i == 0)
                bound_x += 0.34;
            else
                bound_x -= 0.34;
            double bound_y = initBodyPos(1) - 0.19;

            footPosBounds[0] = ifopt::Bounds(bound_x, bound_x);
            footPosBounds[1] = ifopt::Bounds(bound_y, bound_y);

            // bound_x = targetBodyPos(0);
            // bound_y = targetBodyPos(1) - 0.19;
            // if (i == 0)
            //     bound_x += 0.34;
            // else
            //     bound_x -= 0.34;

            // footPosBounds[6 * numKnots - 6] = ifopt::Bounds(bound_x, bound_x);
            // footPosBounds[6 * numKnots - 5] = ifopt::Bounds(bound_y, bound_y);
        }
        else if (i == 1 || i == 3) {
            double bound_x = initBodyPos(0);
            if (i == 1)
                bound_x += 0.34;
            else
                bound_x -= 0.34;
            double bound_y = initBodyPos(1) + 0.19;

            footPosBounds[0] = ifopt::Bounds(bound_x, bound_x);
            footPosBounds[1] = ifopt::Bounds(bound_y, bound_y);

            // bound_x = targetBodyPos(0);
            // bound_y = targetBodyPos(1) + 0.19;
            // if (i == 1)
            //     bound_x += 0.34;
            // else
            //     bound_x -= 0.34;

            // footPosBounds[6 * numKnots - 6] = ifopt::Bounds(bound_x, bound_x);
            // footPosBounds[6 * numKnots - 5] = ifopt::Bounds(bound_y, bound_y);
        }

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
        // nlp.AddCostSet(std::make_shared<trajopt::MinEffort<trajopt::TrajectoryVars>>(footForceVars, numKnots));
    }

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

    double dt = 0.;
    for (size_t i = 0; i < numSamples; ++i) {
        std::cout << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::BODY_POS_TRAJECTORY))->trajectoryEval(dt, 0).transpose() << std::endl;

        dt += sampleTime;
    }

    std::cout << std::endl;

    for (size_t k = 0; k < 4; k++) {

        dt = 0.;
        for (size_t i = 0; i < numSamples; ++i) {
            std::cout << "p_" << k << ": " << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(k)))->trajectoryEval(dt, 0).transpose() << std::endl;

            std::cout << "v_" << k << ": " << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(k)))->trajectoryEval(dt, 1).transpose() << std::endl;

            dt += sampleTime;
        }

        std::cout << std::endl;

        dt = 0.;
        for (size_t i = 0; i < numSamples; ++i) {
            std::cout << "f_" << k << ": " << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_FORCE + "_" + std::to_string(k)))->trajectoryEval(dt, 0).transpose() << std::endl;

            dt += sampleTime;
        }

        std::cout << std::endl;
    }

    return 0;
}
