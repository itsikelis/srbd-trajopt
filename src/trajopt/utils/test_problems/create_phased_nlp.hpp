#pragma once

#include <ifopt/problem.h>

#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <trajopt/ifopt_sets/cost/min_effort.hpp>

#include <trajopt/ifopt_sets/constraints/common/acceleration.hpp>
#include <trajopt/ifopt_sets/constraints/common/dynamics.hpp>
#include <trajopt/ifopt_sets/constraints/common/friction_cone.hpp>

#include <trajopt/ifopt_sets/variables/phased_trajectory_vars.hpp>

#include <trajopt/ifopt_sets/constraints/phased/foot_body_distance_phased.hpp>
#include <trajopt/ifopt_sets/constraints/phased/foot_terrain_distance_phased.hpp>
#include <trajopt/ifopt_sets/constraints/phased/phased_acceleration.hpp>

#include <trajopt/utils/types.hpp>
#include <trajopt/utils/utils.hpp>

namespace trajopt {
    inline ifopt::Problem create_phased_nlp(size_t numKnots, size_t numSamples, double totalTime, const Eigen::Vector3d& initBodyPos, const Eigen::Vector3d& targetBodyPos, const SingleRigidBodyDynamicsModel& model, const TerrainGrid& terrain)
    {
        ifopt::Problem nlp;

        double sampleTime = totalTime / static_cast<double>(numSamples - 1.);
        std::vector<double> polyTimes = std::vector<double>(numKnots - 1);
        for (size_t i = 0; i < static_cast<size_t>(polyTimes.size()); ++i) {
            polyTimes[i] = totalTime / static_cast<double>(numKnots - 1);
        }

        // Add body pos and rot var sets.
        ifopt::Component::VecBound bodyPosBounds = trajopt::fillBoundVector(initBodyPos, targetBodyPos, ifopt::NoBound, 6 * numKnots);
        Eigen::VectorXd initBodyPosVals = Eigen::VectorXd::Zero(3 * 2 * numKnots);

        auto posVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_POS_TRAJECTORY, initBodyPosVals, polyTimes, bodyPosBounds);
        nlp.AddVariableSet(posVars);

        Eigen::Vector3d initRotPos = Eigen::Vector3d::Zero();
        Eigen::Vector3d targetRotPos = Eigen::Vector3d::Zero();
        ifopt::Component::VecBound bodyRotBounds = trajopt::fillBoundVector(initRotPos, targetRotPos, ifopt::NoBound, 6 * numKnots);
        Eigen::VectorXd initBodyRotVals = Eigen::VectorXd::Zero(3 * 2 * numKnots);
        auto rotVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_ROT_TRAJECTORY, initBodyRotVals, polyTimes, bodyRotBounds);
        nlp.AddVariableSet(rotVars);

        // // Add regular constraint sets.
        nlp.AddConstraintSet(std::make_shared<trajopt::Dynamics<trajopt::PhasedTrajectoryVars>>(model, numSamples, sampleTime));

        nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(posVars));
        nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(rotVars));

        // Add feet pos and force var sets.
        size_t numPosSteps = 2;
        size_t numForceSteps = 1;
        std::vector<double> phaseTimes = {0.2, 0.1, 0.2};
        std::vector<size_t> posKnotsPerSwing = {1};
        std::vector<size_t> forceKnotsPerSwing = {5, 5};

        // size_t numPhasedKnots = numPosSteps + std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
        // size_t numPhasedVars = 3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
        double max_force = 2. * model.mass * std::abs(model.gravity[2]);
        Eigen::VectorXd initFootPosVals = Eigen::VectorXd::Zero(3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0));
        Eigen::VectorXd initFootForceVals = Eigen::VectorXd::Zero(3 * numForceSteps + 6 * std::accumulate(forceKnotsPerSwing.begin(), forceKnotsPerSwing.end(), 0));

        // std::cout << initFootPosVals.rows() << " , " << initFootForceVals.rows() << std::endl;
        // std::cout << numPhasedKnots << " , " << initFootForceVals.rows() << std::endl;

        ifopt::Component::VecBound footPosBounds(3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0), ifopt::NoBound);
        ifopt::Component::VecBound footForceBounds(3 * numForceSteps + 6 * std::accumulate(forceKnotsPerSwing.begin(), forceKnotsPerSwing.end(), 0), ifopt::Bounds(-max_force, max_force));

        // std::vector<std::make_shared<trajopt::PhasedTrajectoryVars>> feetForces, feetPos;

        for (size_t i = 0; i < model.numFeet; ++i) {
            auto footPosVars = std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_POS + "_" + std::to_string(i), initFootPosVals, footPosBounds, phaseTimes, posKnotsPerSwing, trajopt::rspl::Phase::Stance);
            nlp.AddVariableSet(footPosVars);

            nlp.AddConstraintSet(std::make_shared<trajopt::PhasedAccelerationConstraints>(footPosVars));
            // nlp.AddConstraintSet(std::make_shared<trajopt::FootPosTerrainConstraints>(footPosVars, terrain, numSamples, sampleTime));
            nlp.AddConstraintSet(std::make_shared<trajopt::FootTerrainDistancePhased>(footPosVars, terrain, numPosSteps, 1, posKnotsPerSwing));
            nlp.AddConstraintSet(std::make_shared<trajopt::FootBodyDistancePhased>(model, posVars, rotVars, footPosVars, numSamples, sampleTime));

            auto footForceVars = std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_FORCE + "_" + std::to_string(i), initFootForceVals, footForceBounds, phaseTimes, forceKnotsPerSwing, trajopt::rspl::Phase::Swing);
            nlp.AddVariableSet(footForceVars);

            nlp.AddConstraintSet(std::make_shared<trajopt::FrictionCone<trajopt::PhasedTrajectoryVars>>(footForceVars, footPosVars, terrain, numSamples, sampleTime));

            // nlp.AddCostSet(std::make_shared<trajopt::MinEffort<trajopt::PhasedTrajectoryVars>>(footPosVars, numKnots));
        }

        return nlp;
    }
} // namespace trajopt
