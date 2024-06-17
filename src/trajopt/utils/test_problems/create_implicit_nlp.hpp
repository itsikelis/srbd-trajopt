#pragma once

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

namespace trajopt {

    inline ifopt::Problem create_implicit_nlp(size_t numKnots, size_t numSamples, double totalTime, const Eigen::Vector3d& initBodyPos, const Eigen::Vector3d& targetBodyPos, const SingleRigidBodyDynamicsModel& model, const TerrainGrid& terrain)
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

        // Add feet variable sets.
        Eigen::VectorXd initFootPosVals = Eigen::VectorXd::Zero(3 * 2 * numKnots);
        Eigen::VectorXd initFootForceVals = Eigen::VectorXd::Zero(3 * 2 * numKnots);

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
            nlp.AddConstraintSet(std::make_shared<trajopt::FootTerrainDistanceImplicit>(footPosVars, terrain, numSamples, sampleTime));
            nlp.AddConstraintSet(std::make_shared<trajopt::ImplicitContactConstraints>(footPosVars, footForceVars, terrain, numSamples, sampleTime));
            nlp.AddConstraintSet(std::make_shared<trajopt::ImplicitVelocityConstraints>(footPosVars, footForceVars, terrain, numSamples, sampleTime));

            // Add cost set
            // nlp.AddCostSet(std::make_shared<trajopt::MinEffort<trajopt::TrajectoryVars>>(footForceVars, numKnots));
        }

        return nlp;
    }
} // namespace trajopt
