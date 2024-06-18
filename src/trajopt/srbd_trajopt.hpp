#pragma once

#include <ctime>
#include <iostream>

#include "trajopt/robo_spline/types.hpp"
#include <Eigen/Core>

#include <numeric>
#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <ifopt/ipopt_solver.h>
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
    class SrbdTrajopt {
    public:
        using Vector = Eigen::VectorXd;
        using Vec3 = Eigen::Vector3d;

        using Model = SingleRigidBodyDynamicsModel;
        using Terrain = TerrainGrid;

        struct Params {
            Vec3 initBodyPos{Vec3::Zero()};
            Vec3 targetBodyPos{Vec3::Zero()};

            Vec3 initBodyRot{Vec3::Zero()};
            Vec3 targetBodyRot{Vec3::Zero()};

            Vec3 initBodyVel{Vec3::Zero()};
            Vec3 targetBodyVel{Vec3::Zero()};

            Vec3 initBodyRotVel{Vec3::Zero()};
            Vec3 targetBodyRotVel{Vec3::Zero()};

            size_t numKnots;
            size_t numSamples;

            size_t numStepPhases;
            size_t numSwingPhases;

            bool addCost = false;
            double maxForce = 1e5; // Usually = 2. * model.mass * std::abs(model.gravity[2]);

            std::vector<size_t> numSteps = {2, 2, 2, 2};

            std::vector<std::vector<double>> phaseTimes = {
                {0.2, 0.1, 0.2},
                {0.2, 0.1, 0.2},
                {0.2, 0.1, 0.2},
                {0.2, 0.1, 0.2}};

            std::vector<std::vector<size_t>> stepKnotsPerSwing = {
                {1},
                {1},
                {1},
                {1}};

            std::vector<std::vector<size_t>> forceKnotsPerSwing = {
                {5, 5},
                {5, 5},
                {5, 5},
                {5, 5}};

            std::vector<rspl::Phase> initialFootPhases{rspl::Phase::Stance, rspl::Phase::Stance, rspl::Phase::Stance, rspl::Phase::Stance};

            // IPOPT Parameters
            std::string jacobianApproximation = "exact";
            double maxCpuTime = 1e50;
            size_t maxIters = 1000;
        };

    public:
        SrbdTrajopt(const Params& params, const Model& model, const Terrain& terrain) : _params(params), _model(model), _terrain(terrain)
        {
            // TODO: assert all phase times are the same.

            _totalTime = std::accumulate(_params.phaseTimes[0].begin(), _params.phaseTimes[0].end(), 0.);
        }

        ~SrbdTrajopt() {}

        void initProblem(ifopt::Problem& nlp)
        {
            // Add body pos and rot var sets.
            std::vector<double> polyTimes = std::vector<double>(_params.numKnots - 1);
            for (size_t i = 0; i < static_cast<size_t>(polyTimes.size()); ++i) {
                polyTimes[i] = _totalTime / static_cast<double>(_params.numKnots - 1);
            }

            double sampleTime = _totalTime / static_cast<double>(_params.numSamples - 1.);

            ifopt::Component::VecBound bodyPosBounds = trajopt::fillBoundVector(_params.initBodyPos, _params.initBodyVel, _params.targetBodyPos, _params.targetBodyVel, ifopt::NoBound, 6 * _params.numKnots);
            Eigen::VectorXd initBodyPosVals = Eigen::VectorXd::Zero(3 * 2 * _params.numKnots);

            auto posVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_POS_TRAJECTORY, initBodyPosVals, polyTimes, bodyPosBounds);
            nlp.AddVariableSet(posVars);

            ifopt::Component::VecBound bodyRotBounds = trajopt::fillBoundVector(_params.initBodyRot, _params.initBodyRotVel, _params.targetBodyRot, _params.targetBodyRotVel, ifopt::NoBound, 6 * _params.numKnots);
            Eigen::VectorXd initBodyRotVals = Eigen::VectorXd::Zero(3 * 2 * _params.numKnots);
            auto rotVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_ROT_TRAJECTORY, initBodyRotVals, polyTimes, bodyRotBounds);
            nlp.AddVariableSet(rotVars);

            // Initialise feet var sets.
            for (size_t i = 0; i < _model.numFeet; ++i) {
                size_t numForceSteps = _params.forceKnotsPerSwing[i].size();
                rspl::Phase initForcePhase = (_params.initialFootPhases[i] == rspl::Phase::Stance) ? rspl::Phase::Swing : rspl::Phase::Stance;

                Eigen::VectorXd initFootPosVals = Eigen::VectorXd::Zero(3 * _params.numSteps[i] + 6 * std::accumulate(_params.stepKnotsPerSwing[i].begin(), _params.stepKnotsPerSwing[i].end(), 0));
                Eigen::VectorXd initFootForceVals = Eigen::VectorXd::Zero(3 * numForceSteps + 6 * std::accumulate(_params.forceKnotsPerSwing[i].begin(), _params.forceKnotsPerSwing[i].end(), 0));

                ifopt::Component::VecBound footPosBounds(3 * _params.numSteps[i] + 6 * std::accumulate(_params.stepKnotsPerSwing[i].begin(), _params.stepKnotsPerSwing[i].end(), 0), ifopt::NoBound);
                ifopt::Component::VecBound footForceBounds(3 * numForceSteps + 6 * std::accumulate(_params.forceKnotsPerSwing[i].begin(), _params.forceKnotsPerSwing[i].end(), 0), ifopt::Bounds(-_params.maxForce, _params.maxForce));

                nlp.AddVariableSet(std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_POS + "_" + std::to_string(i), initFootPosVals, footPosBounds, _params.phaseTimes[i], _params.stepKnotsPerSwing[i], _params.initialFootPhases[i]));
                nlp.AddVariableSet(std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_FORCE + "_" + std::to_string(i), initFootForceVals, footForceBounds, _params.phaseTimes[i], _params.forceKnotsPerSwing[i], initForcePhase));
            }

            // Add constraints
            nlp.AddConstraintSet(std::make_shared<trajopt::Dynamics<trajopt::PhasedTrajectoryVars>>(_model, _params.numSamples, sampleTime));
            nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(posVars));
            nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(rotVars));

            for (size_t i = 0; i < _model.numFeet; ++i) {
                auto footPosVars = std::static_pointer_cast<trajopt::PhasedTrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(i)));
                auto footForceVars = std::static_pointer_cast<trajopt::PhasedTrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_FORCE + "_" + std::to_string(i)));

                nlp.AddConstraintSet(std::make_shared<trajopt::PhasedAccelerationConstraints>(footPosVars));
                nlp.AddConstraintSet(std::make_shared<trajopt::FootTerrainDistancePhased>(footPosVars, _terrain, _params.numSteps[i], 1, _params.stepKnotsPerSwing[i]));
                nlp.AddConstraintSet(std::make_shared<trajopt::FootBodyDistancePhased>(_model, posVars, rotVars, footPosVars, _params.numSamples, sampleTime));
                nlp.AddConstraintSet(std::make_shared<trajopt::FrictionCone<trajopt::PhasedTrajectoryVars>>(footForceVars, footPosVars, _terrain, _params.numSamples, sampleTime));
                if (_params.addCost)
                    nlp.AddCostSet(std::make_shared<trajopt::MinEffort<trajopt::PhasedTrajectoryVars>>(footPosVars, _params.numKnots));
            }
        }

        void solveProblem(ifopt::Problem& nlp)
        {
            ifopt::IpoptSolver ipopt;
            ipopt.SetOption("jacobian_approximation", _params.jacobianApproximation);
            ipopt.SetOption("max_cpu_time", _params.maxCpuTime);
            ipopt.SetOption("max_iter", static_cast<int>(_params.maxIters));

            ipopt.Solve(nlp);

            std::cout << "Ipopt Wall Time: " << ipopt.GetTotalWallclockTime() << " seconds." << std::endl;
        }

    protected:
        Params _params;
        Model _model;
        Terrain _terrain;

        double _totalTime;
    };

} // namespace trajopt