#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include "ifopt_sets/constraints/acceleration.hpp"
#include "ifopt_sets/constraints/dynamics.hpp"
#include "ifopt_sets/constraints/foot_constraints.hpp"
#include "ifopt_sets/variables.hpp"
#include "utils/srbd.hpp"
#include "utils/types.hpp"

namespace trajopt {
class TrajOpt {
public:
  TrajOpt(const SingleRigidBodyDynamicsModel &model, const Terrain &terrain,
          const TrajOptArguments &args)
      : _model(model), _terrain(terrain), _args(args) {}

  void Init() {
    // Generate Phase Times std::vector
    std::vector<double>
        phaseTimes; // Phase times for "Standing At Start" Phased Trajectories
    for (size_t i = 0; i < _args.numSteps - 1; ++i) {
      phaseTimes.push_back(_args.stepPhaseTime);
      phaseTimes.push_back(_args.swingPhaseTime);
    }
    phaseTimes.push_back(
        _args.stepPhaseTime); // Add last step (no swing after).

    // Calculate total time.
    double totalTime =
        std::accumulate(phaseTimes.begin(), phaseTimes.end(), 0.);
    // Calculate spline duration for normal trajectories.
    Eigen::VectorXd polyTimes = Eigen::VectorXd(_args.numKnots);
    for (size_t i = 0; i < _args.numKnots; ++i) {
      polyTimes[i] = totalTime / static_cast<double>(_args.numKnots - 1.);
    }
    _sampleTime = totalTime / static_cast<double>(_args.numSamples - 1.);

    std::cout << "Adding Regular Variable and Constraint Sets.." << std::endl;
    _nlp = ifopt::Problem();

    ifopt::Component::VecBound posBounds =
        GetBoundsRegular(6 * _args.numKnots, "pos");
    auto posVars = std::make_shared<TrajectoryVars>(
        BODY_POS_TRAJECTORY, _args.numKnots, polyTimes, posBounds);
    _nlp.AddVariableSet(posVars);

    ifopt::Component::VecBound rotBounds =
        GetBoundsRegular(6 * _args.numKnots, "rot");
    auto rotVars = std::make_shared<TrajectoryVars>(
        BODY_ROT_TRAJECTORY, _args.numKnots, polyTimes, rotBounds);
    _nlp.AddVariableSet(rotVars);

    // Add Regular Constraint sets.
    auto dynamConstr = std::make_shared<DynamicsConstraint>(
        _model, _args.numSamples, _sampleTime);
    _nlp.AddConstraintSet(dynamConstr);

    _nlp.AddConstraintSet(std::make_shared<AccelerationConstraints>(posVars));
    _nlp.AddConstraintSet(std::make_shared<AccelerationConstraints>(rotVars));

    std::cout << "Adding Phased Variable and Constraint Sets.." << std::endl;

    if (_args.gait == "jumping") {
      size_t numSwings = _args.numSteps - 1;
      // size_t posKnotPoints = 2 * _args.numSteps + _args.numKnotsPerSwing *
      // numSwings; size_t forceKnotPoints = 2 * numSwings +
      // _args.numKnotsPerSwing * _args.numSteps; std::cout << "Pos Knot Points:
      // " << posKnotPoints << std::endl; std::cout << "Force Knot Points: " <<
      // forceKnotPoints << std::endl;

      // Generate gaits (same for all feet).
      std::vector<double> pawPosPhasedTimes = GaitSequencer(
          phaseTimes, _args.numSteps, numSwings, _args.numKnotsPerSwing, true);
      std::vector<double> pawForcePhasedTimes = GaitSequencer(
          phaseTimes, _args.numSteps, numSwings, _args.numKnotsPerSwing, false);

      // Generate Foot Force Bounds (Same for all feet in jumping).
      double max_force = 2. * _model.mass * std::abs(_model.gravity[2]);
      ifopt::Bounds forceBounds = ifopt::Bounds(-max_force, max_force);
      ifopt::Component::VecBound pawForceBounds(
          3 * numSwings + 6 * _args.numSteps * _args.numKnotsPerSwing,
          forceBounds);
      // When not in contact, force should be 0.
      // numSwings in force var set is the number of step phases in foot pos var
      // set.
      for (size_t i = 0; i < numSwings; ++i) {
        pawForceBounds.at((i + 1) * 6 * _args.numKnotsPerSwing + i * 3 + 0) =
            ifopt::BoundZero;
        pawForceBounds.at((i + 1) * 6 * _args.numKnotsPerSwing + i * 3 + 1) =
            ifopt::BoundZero;
        pawForceBounds.at((i + 1) * 6 * _args.numKnotsPerSwing + i * 3 + 2) =
            ifopt::BoundZero;
      }

      for (size_t i = 0; i < _model.numFeet; ++i) {
        // Init foot #i
        ifopt::Component::VecBound pawPosBounds(
            3 * _args.numSteps + 6 * numSwings * _args.numKnotsPerSwing,
            ifopt::NoBound);

        pawPosBounds.at(0) =
            ifopt::Bounds(_model.feetPoses[i][0], _model.feetPoses[i][0]);
        pawPosBounds.at(1) =
            ifopt::Bounds(_model.feetPoses[i][1], _model.feetPoses[i][1]);
        pawPosBounds.at(2) = ifopt::Bounds(
            _terrain.z(_model.feetPoses[i][0], _model.feetPoses[i][0]),
            _terrain.z(_model.feetPoses[i][1], _model.feetPoses[i][1]));

        auto pawPosVars = std::make_shared<PhasedTrajectoryVars>(
            PAW_POS + "_" + std::to_string(i), pawPosPhasedTimes, pawPosBounds,
            _args.numSteps, numSwings, _args.numKnotsPerSwing, true);
        _nlp.AddVariableSet(pawPosVars);

        _nlp.AddConstraintSet(
            std::make_shared<PhasedAccelerationConstraints>(pawPosVars));
        _nlp.AddConstraintSet(std::make_shared<FootPosTerrainConstraints>(
            pawPosVars, _terrain, _args.numSteps, numSwings,
            _args.numKnotsPerSwing));
        _nlp.AddConstraintSet(std::make_shared<FootBodyPosConstraints>(
            _model, posVars, rotVars, pawPosVars, _args.numSamples,
            _sampleTime));

        // Add paw force var set.
        auto pawForceVars = std::make_shared<PhasedTrajectoryVars>(
            PAW_FORCES + "_" + std::to_string(i), pawForcePhasedTimes,
            pawForceBounds, numSwings, _args.numSteps, _args.numKnotsPerSwing,
            false);
        _nlp.AddVariableSet(pawForceVars);

        _nlp.AddConstraintSet(std::make_shared<FrictionConeConstraints>(
            pawForceVars, _terrain, _args.numSamples, _sampleTime));

        // TO-DO: We could also ignore that! This is the control! It could do
        // whatever accelerations!
        // nlp.AddConstraintSet(std::make_shared<AccelerationConstraints>(forceVars));
      }
    } else if (_args.gait == "pace") {
      if (_args.stepPhaseTime != _args.swingPhaseTime) {
        std::cerr << "For pace, only equal swing and step times are supported "
                     "at the moment!"
                  << std::endl;
        _args.swingPhaseTime = _args.stepPhaseTime;
      }

      size_t numSwings = _args.numSteps - 1;

      // Generate gaits for even feet indices.
      std::vector<double> standGait = GaitSequencer(
          phaseTimes, _args.numSteps, numSwings, _args.numKnotsPerSwing, true);
      std::vector<double> swingGait = GaitSequencer(
          phaseTimes, _args.numSteps, numSwings, _args.numKnotsPerSwing, false);

      double max_force = 2. * _model.mass * std::abs(_model.gravity[2]);
      ifopt::Bounds forceBounds = ifopt::Bounds(-max_force, max_force);

      // Even foot indices.
      ifopt::Component::VecBound pawForceBoundsEven(
          3 * numSwings + 6 * _args.numSteps * _args.numKnotsPerSwing,
          forceBounds);
      // When not in contact, force should be 0.
      // numSwings in force var set is the number of step phases in foot pos var
      // set.
      for (size_t i = 0; i < numSwings; ++i) {
        pawForceBoundsEven.at((i + 1) * 6 * _args.numKnotsPerSwing + i * 3 +
                              0) = ifopt::BoundZero;
        pawForceBoundsEven.at((i + 1) * 6 * _args.numKnotsPerSwing + i * 3 +
                              1) = ifopt::BoundZero;
        pawForceBoundsEven.at((i + 1) * 6 * _args.numKnotsPerSwing + i * 3 +
                              2) = ifopt::BoundZero;
      }

      // Even foot indices.
      ifopt::Component::VecBound pawForceBoundsOdd(
          3 * _args.numSteps + 6 * numSwings * _args.numKnotsPerSwing,
          forceBounds);
      // When not in contact, force should be 0.
      for (size_t i = 0; i < _args.numSteps; ++i) {
        pawForceBoundsOdd.at(i * 3 + i * 6 * _args.numKnotsPerSwing + 0) =
            ifopt::BoundZero;
        pawForceBoundsOdd.at(i * 3 + i * 6 * _args.numKnotsPerSwing + 1) =
            ifopt::BoundZero;
        pawForceBoundsOdd.at(i * 3 + i * 6 * _args.numKnotsPerSwing + 2) =
            ifopt::BoundZero;
      }

      for (size_t i = 0; i < _model.numFeet; ++i) {
        if (i % 2 == 0) {
          ifopt::Component::VecBound pawPosBounds(
              3 * _args.numSteps + 6 * numSwings * _args.numKnotsPerSwing,
              ifopt::NoBound);
          pawPosBounds.at(0) =
              ifopt::Bounds(_model.feetPoses[i][0], _model.feetPoses[i][0]);
          pawPosBounds.at(1) =
              ifopt::Bounds(_model.feetPoses[i][1], _model.feetPoses[i][1]);
          pawPosBounds.at(2) = ifopt::Bounds(
              _terrain.z(_model.feetPoses[i][0], _model.feetPoses[i][0]),
              _terrain.z(_model.feetPoses[i][1], _model.feetPoses[i][1]));

          auto pawPosVars = std::make_shared<PhasedTrajectoryVars>(
              PAW_POS + "_" + std::to_string(i), standGait, pawPosBounds,
              _args.numSteps, numSwings, _args.numKnotsPerSwing, true);
          _nlp.AddVariableSet(pawPosVars);

          _nlp.AddConstraintSet(
              std::make_shared<PhasedAccelerationConstraints>(pawPosVars));
          _nlp.AddConstraintSet(std::make_shared<FootPosTerrainConstraints>(
              pawPosVars, _terrain, _args.numSteps, numSwings,
              _args.numKnotsPerSwing));
          _nlp.AddConstraintSet(std::make_shared<FootBodyPosConstraints>(
              _model, posVars, rotVars, pawPosVars, _args.numSamples,
              _sampleTime));

          // Add paw force var set.
          auto pawForceVars = std::make_shared<PhasedTrajectoryVars>(
              PAW_FORCES + "_" + std::to_string(i), swingGait,
              pawForceBoundsEven, numSwings, _args.numSteps,
              _args.numKnotsPerSwing, false);
          _nlp.AddVariableSet(pawForceVars);

          _nlp.AddConstraintSet(std::make_shared<FrictionConeConstraints>(
              pawForceVars, _terrain, _args.numSamples, _sampleTime));
        } else {
          ifopt::Component::VecBound pawPosBounds(
              3 * numSwings + 6 * _args.numSteps * _args.numKnotsPerSwing,
              ifopt::NoBound);
          auto initFootPos = _args.initPos + _model.feetPoses[i];
          pawPosBounds.at(0) = ifopt::Bounds(initFootPos[0], initFootPos[0]);
          pawPosBounds.at(1) = ifopt::Bounds(initFootPos[1], initFootPos[1]);
          pawPosBounds.at(2) =
              ifopt::Bounds(_terrain.z(initFootPos[0], initFootPos[0]),
                            _terrain.z(initFootPos[1], initFootPos[1]));

          auto targetFootPos = _args.targetPos + _model.feetPoses[i];
          pawPosBounds.at(pawPosBounds.size() - 3) =
              ifopt::Bounds(targetFootPos[0], targetFootPos[0]);
          pawPosBounds.at(pawPosBounds.size() - 2) =
              ifopt::Bounds(targetFootPos[1], targetFootPos[1]);
          pawPosBounds.at(pawPosBounds.size() - 1) =
              ifopt::Bounds(_terrain.z(targetFootPos[0], targetFootPos[0]),
                            _terrain.z(targetFootPos[1], targetFootPos[1]));

          auto pawPosVars = std::make_shared<PhasedTrajectoryVars>(
              PAW_POS + "_" + std::to_string(i), swingGait, pawPosBounds,
              numSwings, _args.numSteps, _args.numKnotsPerSwing, false);
          _nlp.AddVariableSet(pawPosVars);

          _nlp.AddConstraintSet(
              std::make_shared<PhasedAccelerationConstraints>(pawPosVars));
          _nlp.AddConstraintSet(std::make_shared<FootPosTerrainConstraints>(
              pawPosVars, _terrain, numSwings, _args.numSteps,
              _args.numKnotsPerSwing));
          _nlp.AddConstraintSet(std::make_shared<FootBodyPosConstraints>(
              _model, posVars, rotVars, pawPosVars, _args.numSamples,
              _sampleTime));

          auto pawForceVars = std::make_shared<PhasedTrajectoryVars>(
              PAW_FORCES + "_" + std::to_string(i), standGait,
              pawForceBoundsOdd, _args.numSteps, numSwings,
              _args.numKnotsPerSwing, true);
          _nlp.AddVariableSet(pawForceVars);

          _nlp.AddConstraintSet(std::make_shared<FrictionConeConstraints>(
              pawForceVars, _terrain, _args.numSamples, _sampleTime));
        }
      }
    } else if (_args.gait == "trot") {
      if (_args.stepPhaseTime != _args.swingPhaseTime) {
        std::cerr << "For trot, only equal swing and step times are supported "
                     "at the moment!"
                  << std::endl;
        _args.swingPhaseTime = _args.stepPhaseTime;
      }

      size_t numSwings = _args.numSteps - 1;

      // Generate gaits for even feet indices.
      std::vector<double> standGait = GaitSequencer(
          phaseTimes, _args.numSteps, numSwings, _args.numKnotsPerSwing, true);
      std::vector<double> swingGait = GaitSequencer(
          phaseTimes, _args.numSteps, numSwings, _args.numKnotsPerSwing, false);

      double max_force = 2. * _model.mass * std::abs(_model.gravity[2]);
      ifopt::Bounds forceBounds = ifopt::Bounds(-max_force, max_force);

      // Even foot indices.
      ifopt::Component::VecBound pawForceBoundsEven(
          3 * numSwings + 6 * _args.numSteps * _args.numKnotsPerSwing,
          forceBounds);
      // When not in contact, force should be 0.
      // numSwings in force var set is the number of step phases in foot pos var
      // set.
      for (size_t i = 0; i < numSwings; ++i) {
        pawForceBoundsEven.at((i + 1) * 6 * _args.numKnotsPerSwing + i * 3 +
                              0) = ifopt::BoundZero;
        pawForceBoundsEven.at((i + 1) * 6 * _args.numKnotsPerSwing + i * 3 +
                              1) = ifopt::BoundZero;
        pawForceBoundsEven.at((i + 1) * 6 * _args.numKnotsPerSwing + i * 3 +
                              2) = ifopt::BoundZero;
      }

      // Even foot indices.
      ifopt::Component::VecBound pawForceBoundsOdd(
          3 * _args.numSteps + 6 * numSwings * _args.numKnotsPerSwing,
          forceBounds);
      // When not in contact, force should be 0.
      for (size_t i = 0; i < _args.numSteps; ++i) {
        pawForceBoundsOdd.at(i * 3 + i * 6 * _args.numKnotsPerSwing + 0) =
            ifopt::BoundZero;
        pawForceBoundsOdd.at(i * 3 + i * 6 * _args.numKnotsPerSwing + 1) =
            ifopt::BoundZero;
        pawForceBoundsOdd.at(i * 3 + i * 6 * _args.numKnotsPerSwing + 2) =
            ifopt::BoundZero;
      }

      for (size_t i = 0; i < _model.numFeet; ++i) {
        if (i == 0 || i == 3) {
          ifopt::Component::VecBound pawPosBounds(
              3 * _args.numSteps + 6 * numSwings * _args.numKnotsPerSwing,
              ifopt::NoBound);
          pawPosBounds.at(0) =
              ifopt::Bounds(_model.feetPoses[i][0], _model.feetPoses[i][0]);
          pawPosBounds.at(1) =
              ifopt::Bounds(_model.feetPoses[i][1], _model.feetPoses[i][1]);
          pawPosBounds.at(2) = ifopt::Bounds(
              _terrain.z(_model.feetPoses[i][0], _model.feetPoses[i][0]),
              _terrain.z(_model.feetPoses[i][1], _model.feetPoses[i][1]));

          auto pawPosVars = std::make_shared<PhasedTrajectoryVars>(
              PAW_POS + "_" + std::to_string(i), standGait, pawPosBounds,
              _args.numSteps, numSwings, _args.numKnotsPerSwing, true);
          _nlp.AddVariableSet(pawPosVars);

          _nlp.AddConstraintSet(
              std::make_shared<PhasedAccelerationConstraints>(pawPosVars));
          _nlp.AddConstraintSet(std::make_shared<FootPosTerrainConstraints>(
              pawPosVars, _terrain, _args.numSteps, numSwings,
              _args.numKnotsPerSwing));
          _nlp.AddConstraintSet(std::make_shared<FootBodyPosConstraints>(
              _model, posVars, rotVars, pawPosVars, _args.numSamples,
              _sampleTime));

          // Add paw force var set.
          auto pawForceVars = std::make_shared<PhasedTrajectoryVars>(
              PAW_FORCES + "_" + std::to_string(i), swingGait,
              pawForceBoundsEven, numSwings, _args.numSteps,
              _args.numKnotsPerSwing, false);
          _nlp.AddVariableSet(pawForceVars);

          _nlp.AddConstraintSet(std::make_shared<FrictionConeConstraints>(
              pawForceVars, _terrain, _args.numSamples, _sampleTime));
        } else {
          ifopt::Component::VecBound pawPosBounds(
              3 * numSwings + 6 * _args.numSteps * _args.numKnotsPerSwing,
              ifopt::NoBound);
          pawPosBounds.at(0) =
              ifopt::Bounds(_model.feetPoses[i][0], _model.feetPoses[i][0]);
          pawPosBounds.at(1) =
              ifopt::Bounds(_model.feetPoses[i][1], _model.feetPoses[i][1]);
          pawPosBounds.at(2) = ifopt::Bounds(
              _terrain.z(_model.feetPoses[i][0], _model.feetPoses[i][0]),
              _terrain.z(_model.feetPoses[i][1], _model.feetPoses[i][1]));

          auto pawPosVars = std::make_shared<PhasedTrajectoryVars>(
              PAW_POS + "_" + std::to_string(i), swingGait, pawPosBounds,
              numSwings, _args.numSteps, _args.numKnotsPerSwing, false);
          _nlp.AddVariableSet(pawPosVars);

          _nlp.AddConstraintSet(
              std::make_shared<PhasedAccelerationConstraints>(pawPosVars));
          _nlp.AddConstraintSet(std::make_shared<FootPosTerrainConstraints>(
              pawPosVars, _terrain, numSwings, _args.numSteps,
              _args.numKnotsPerSwing));
          _nlp.AddConstraintSet(std::make_shared<FootBodyPosConstraints>(
              _model, posVars, rotVars, pawPosVars, _args.numSamples,
              _sampleTime));

          auto pawForceVars = std::make_shared<PhasedTrajectoryVars>(
              PAW_FORCES + "_" + std::to_string(i), standGait,
              pawForceBoundsOdd, _args.numSteps, numSwings,
              _args.numKnotsPerSwing, true);
          _nlp.AddVariableSet(pawForceVars);

          _nlp.AddConstraintSet(std::make_shared<FrictionConeConstraints>(
              pawForceVars, _terrain, _args.numSamples, _sampleTime));
        }
      }
    } else {
      if (_args.stepPhaseTime != _args.swingPhaseTime) {
        std::cerr << "For bound, only equal swing and step times are supported "
                     "at the moment!"
                  << std::endl;
        _args.swingPhaseTime = _args.stepPhaseTime;
      }

      size_t numSwings = _args.numSteps - 1;

      // Generate gaits for even feet indices.
      std::vector<double> standGait = GaitSequencer(
          phaseTimes, _args.numSteps, numSwings, _args.numKnotsPerSwing, true);
      std::vector<double> swingGait = GaitSequencer(
          phaseTimes, _args.numSteps, numSwings, _args.numKnotsPerSwing, false);

      double max_force = 2. * _model.mass * std::abs(_model.gravity[2]);
      ifopt::Bounds forceBounds = ifopt::Bounds(-max_force, max_force);

      // Even foot indices.
      ifopt::Component::VecBound pawForceBoundsEven(
          3 * numSwings + 6 * _args.numSteps * _args.numKnotsPerSwing,
          forceBounds);
      // When not in contact, force should be 0.
      // numSwings in force var set is the number of step phases in foot pos var
      // set.
      for (size_t i = 0; i < numSwings; ++i) {
        pawForceBoundsEven.at((i + 1) * 6 * _args.numKnotsPerSwing + i * 3 +
                              0) = ifopt::BoundZero;
        pawForceBoundsEven.at((i + 1) * 6 * _args.numKnotsPerSwing + i * 3 +
                              1) = ifopt::BoundZero;
        pawForceBoundsEven.at((i + 1) * 6 * _args.numKnotsPerSwing + i * 3 +
                              2) = ifopt::BoundZero;
      }

      // Even foot indices.
      ifopt::Component::VecBound pawForceBoundsOdd(
          3 * _args.numSteps + 6 * numSwings * _args.numKnotsPerSwing,
          forceBounds);
      // When not in contact, force should be 0.
      for (size_t i = 0; i < _args.numSteps; ++i) {
        pawForceBoundsOdd.at(i * 3 + i * 6 * _args.numKnotsPerSwing + 0) =
            ifopt::BoundZero;
        pawForceBoundsOdd.at(i * 3 + i * 6 * _args.numKnotsPerSwing + 1) =
            ifopt::BoundZero;
        pawForceBoundsOdd.at(i * 3 + i * 6 * _args.numKnotsPerSwing + 2) =
            ifopt::BoundZero;
      }

      for (size_t i = 0; i < _model.numFeet; ++i) {
        if (i == 0 || i == 1) {
          ifopt::Component::VecBound pawPosBounds(
              3 * _args.numSteps + 6 * numSwings * _args.numKnotsPerSwing,
              ifopt::NoBound);
          pawPosBounds.at(0) =
              ifopt::Bounds(_model.feetPoses[i][0], _model.feetPoses[i][0]);
          pawPosBounds.at(1) =
              ifopt::Bounds(_model.feetPoses[i][1], _model.feetPoses[i][1]);
          pawPosBounds.at(2) = ifopt::Bounds(
              _terrain.z(_model.feetPoses[i][0], _model.feetPoses[i][0]),
              _terrain.z(_model.feetPoses[i][1], _model.feetPoses[i][1]));

          auto pawPosVars = std::make_shared<PhasedTrajectoryVars>(
              PAW_POS + "_" + std::to_string(i), standGait, pawPosBounds,
              _args.numSteps, numSwings, _args.numKnotsPerSwing, true);
          _nlp.AddVariableSet(pawPosVars);

          _nlp.AddConstraintSet(
              std::make_shared<PhasedAccelerationConstraints>(pawPosVars));
          _nlp.AddConstraintSet(std::make_shared<FootPosTerrainConstraints>(
              pawPosVars, _terrain, _args.numSteps, numSwings,
              _args.numKnotsPerSwing));
          _nlp.AddConstraintSet(std::make_shared<FootBodyPosConstraints>(
              _model, posVars, rotVars, pawPosVars, _args.numSamples,
              _sampleTime));

          // Add paw force var set.
          auto pawForceVars = std::make_shared<PhasedTrajectoryVars>(
              PAW_FORCES + "_" + std::to_string(i), swingGait,
              pawForceBoundsEven, numSwings, _args.numSteps,
              _args.numKnotsPerSwing, false);
          _nlp.AddVariableSet(pawForceVars);

          _nlp.AddConstraintSet(std::make_shared<FrictionConeConstraints>(
              pawForceVars, _terrain, _args.numSamples, _sampleTime));
        } else {
          ifopt::Component::VecBound pawPosBounds(
              3 * numSwings + 6 * _args.numSteps * _args.numKnotsPerSwing,
              ifopt::NoBound);
          pawPosBounds.at(0) =
              ifopt::Bounds(_model.feetPoses[i][0], _model.feetPoses[i][0]);
          pawPosBounds.at(1) =
              ifopt::Bounds(_model.feetPoses[i][1], _model.feetPoses[i][1]);
          pawPosBounds.at(2) = ifopt::Bounds(
              _terrain.z(_model.feetPoses[i][0], _model.feetPoses[i][0]),
              _terrain.z(_model.feetPoses[i][1], _model.feetPoses[i][1]));

          auto pawPosVars = std::make_shared<PhasedTrajectoryVars>(
              PAW_POS + "_" + std::to_string(i), swingGait, pawPosBounds,
              numSwings, _args.numSteps, _args.numKnotsPerSwing, false);
          _nlp.AddVariableSet(pawPosVars);

          _nlp.AddConstraintSet(
              std::make_shared<PhasedAccelerationConstraints>(pawPosVars));
          _nlp.AddConstraintSet(std::make_shared<FootPosTerrainConstraints>(
              pawPosVars, _terrain, numSwings, _args.numSteps,
              _args.numKnotsPerSwing));
          _nlp.AddConstraintSet(std::make_shared<FootBodyPosConstraints>(
              _model, posVars, rotVars, pawPosVars, _args.numSamples,
              _sampleTime));

          auto pawForceVars = std::make_shared<PhasedTrajectoryVars>(
              PAW_FORCES + "_" + std::to_string(i), standGait,
              pawForceBoundsOdd, _args.numSteps, numSwings,
              _args.numKnotsPerSwing, true);
          _nlp.AddVariableSet(pawForceVars);

          _nlp.AddConstraintSet(std::make_shared<FrictionConeConstraints>(
              pawForceVars, _terrain, _args.numSamples, _sampleTime));
        }
      }
    }
    // _nlp.PrintCurrent();
  }

  void Solve() {
    std::cout << "Solving.." << std::endl;
    ifopt::IpoptSolver ipopt;
    ipopt.SetOption("jacobian_approximation", "exact");
    // ipopt.SetOption("jacobian_approximation", "finite-difference-values");
    ipopt.SetOption("max_cpu_time", 1e50);
    ipopt.SetOption("max_iter", static_cast<int>(1000));

    // Solve.
    ipopt.Solve(_nlp);
    // _nlp.PrintCurrent();
  }

  rspl::Trajectory3D GetBodyTrajectory(std::string name) {
    rspl::Trajectory3D traj;
    if (name == "pos") {
      traj = std::static_pointer_cast<TrajectoryVars>(
                 _nlp.GetOptVariables()->GetComponent(BODY_POS_TRAJECTORY))
                 ->traj();
    } else {
      traj = std::static_pointer_cast<TrajectoryVars>(
                 _nlp.GetOptVariables()->GetComponent(BODY_ROT_TRAJECTORY))
                 ->traj();
    }
    return traj;
  }

  rspl::Trajectory3D GetFootPosTrajectory(size_t idx) {
    rspl::Trajectory3D traj = std::static_pointer_cast<PhasedTrajectoryVars>(
                                  _nlp.GetOptVariables()->GetComponent(
                                      PAW_POS + "_" + std::to_string(idx)))
                                  ->traj();
    return traj;
  }

  rspl::Trajectory3D GetFootForceTrajectory(size_t idx) {
    rspl::Trajectory3D traj = std::static_pointer_cast<PhasedTrajectoryVars>(
                                  _nlp.GetOptVariables()->GetComponent(
                                      PAW_FORCES + "_" + std::to_string(idx)))
                                  ->traj();
    return traj;
  }

  double SampleTime() { return _sampleTime; }

  void StoreSamplesToCsv(const std::string &filename,
                         bool absolute_path = false) {
    // Store results in a CSV for plotting.
    // std::cout << "Writing output to CSV File" << std::endl;
    std::string path;
    if (absolute_path) {
      path = filename;
    } else {
      path = std::string(SRCPATH) + "/" + filename;
    }

    std::ofstream csv(path);

    if (!csv.is_open()) {
      std::cerr << "Error opening the CSV file!" << std::endl;
    }

    csv << "index,time,x_b,y_b,z_b,th_z,th_y,th_x";

    for (size_t k = 0; k < _model.numFeet; ++k) {
      csv << ","
          << "x" << std::to_string(k) << ","
          << "y" << std::to_string(k) << ","
          << "z" << std::to_string(k) << ","
          << "fx" << std::to_string(k) << ","
          << "fy" << std::to_string(k) << ","
          << "fz" << std::to_string(k);
    }

    csv << std::endl;

    double t = 0.;
    for (size_t i = 1; i < _args.numSamples; ++i) {

      csv << i << "," << t;

      // Body Positions
      double x_b = GetBodyTrajectory("pos").position(t)(0);
      double y_b = GetBodyTrajectory("pos").position(t)(1);
      double z_b = GetBodyTrajectory("pos").position(t)(2);

      double th_z = GetBodyTrajectory("rot").position(t)(0);
      double th_y = GetBodyTrajectory("rot").position(t)(1);
      double th_x = GetBodyTrajectory("rot").position(t)(2);

      csv << "," << x_b << "," << y_b << "," << z_b << "," << th_z << ","
          << th_y << "," << th_x;

      for (size_t k = 0; k < _model.numFeet; ++k) {
        double x = GetFootPosTrajectory(k).position(t)(0);
        double y = GetFootPosTrajectory(k).position(t)(1);
        double z = GetFootPosTrajectory(k).position(t)(2);

        double fx = GetFootForceTrajectory(k).position(t)(0);
        double fy = GetFootForceTrajectory(k).position(t)(1);
        double fz = GetFootForceTrajectory(k).position(t)(2);

        csv << "," << x << "," << y << "," << z << "," << fx << "," << fy << ","
            << fz;
      }
      csv << std::endl;
      t += _sampleTime;
    }

    // Close the CSV file
    csv.close();
    std::cout << "CSV file created successfully!" << std::endl;
  }

protected:
  std::vector<double> GaitSequencer(const std::vector<double> &phaseTimes,
                                    size_t numSteps, size_t numSwings,
                                    size_t numKnotsPerSwing,
                                    bool standingAtStart) {
    size_t numPhases = numSteps + numSwings;
    // double phaseTime = totalTime / static_cast<double>(numPhases);

    std::vector<double> polyTimes;

    bool isStance = standingAtStart;

    for (size_t k = 0; k < numPhases; k++) {
      double phaseTime = phaseTimes[k];
      if (isStance) {
        polyTimes.push_back(phaseTime);
      } else {
        size_t totalN = numKnotsPerSwing;
        if (k > 0 && k < (numPhases - 1))
          totalN += 1;
        double swingSplineTime =
            phaseTime / static_cast<double>(
                            totalN); // duration of each spline in a swing phase
        for (size_t j = 0; j < totalN; j++) {
          polyTimes.push_back(swingSplineTime);
        }
      }

      isStance = !isStance;
    }

    // std::cout << polyTimes.size() << ": ";
    // for (size_t i = 0; i < polyTimes.size(); ++i) {
    //     std::cout << "#" << polyTimes.at(i) << "#";
    // }

    // std::cout << std::endl;

    // std::cout << "Sum Time: " << std::accumulate(polyTimes.begin(),
    // polyTimes.end(), 0.) << std::endl;

    return polyTimes;
  }

  ifopt::Component::VecBound GetBoundsRegular(size_t size, std::string varSet) {
    ifopt::Component::VecBound bounds(size, ifopt::NoBound);
    if (varSet == "pos") {
      bounds.at(0) = ifopt::Bounds(_args.initPos[0], _args.initPos[0]);
      bounds.at(1) = ifopt::Bounds(_args.initPos[1], _args.initPos[1]);
      bounds.at(2) = ifopt::Bounds(_args.initPos[2], _args.initPos[2]);
      bounds.at(3) = ifopt::Bounds(_args.initVel[0], _args.initVel[0]);
      bounds.at(4) = ifopt::Bounds(_args.initVel[1], _args.initVel[1]);
      bounds.at(5) = ifopt::Bounds(_args.initVel[2], _args.initVel[2]);
      bounds.at(6 * _args.numKnots - 6) =
          ifopt::Bounds(_args.targetPos[0], _args.targetPos[0]);
      bounds.at(6 * _args.numKnots - 5) =
          ifopt::Bounds(_args.targetPos[1], _args.targetPos[1]);
      bounds.at(6 * _args.numKnots - 4) =
          ifopt::Bounds(_args.targetPos[2], _args.targetPos[2]);
      bounds.at(6 * _args.numKnots - 3) =
          ifopt::Bounds(_args.targetVel[0], _args.targetVel[0]);
      bounds.at(6 * _args.numKnots - 2) =
          ifopt::Bounds(_args.targetVel[1], _args.targetVel[1]);
      bounds.at(6 * _args.numKnots - 1) =
          ifopt::Bounds(_args.targetVel[2], _args.targetVel[2]);
    } else if (varSet == "rot") {
      bounds.at(0) = ifopt::Bounds(_args.initRot[0], _args.initRot[0]);
      bounds.at(1) = ifopt::Bounds(_args.initRot[1], _args.initRot[1]);
      bounds.at(2) = ifopt::Bounds(_args.initRot[2], _args.initRot[2]);
      bounds.at(3) = ifopt::Bounds(_args.initAngVel[0], _args.initAngVel[0]);
      bounds.at(4) = ifopt::Bounds(_args.initAngVel[1], _args.initAngVel[1]);
      bounds.at(5) = ifopt::Bounds(_args.initAngVel[2], _args.initAngVel[2]);
      bounds.at(6 * _args.numKnots - 6) =
          ifopt::Bounds(_args.targetRot[0], _args.targetRot[0]);
      bounds.at(6 * _args.numKnots - 5) =
          ifopt::Bounds(_args.targetRot[1], _args.targetRot[1]);
      bounds.at(6 * _args.numKnots - 4) =
          ifopt::Bounds(_args.targetRot[2], _args.targetRot[2]);
      bounds.at(6 * _args.numKnots - 3) =
          ifopt::Bounds(_args.targetAngVel[0], _args.targetAngVel[0]);
      bounds.at(6 * _args.numKnots - 2) =
          ifopt::Bounds(_args.targetAngVel[1], _args.targetAngVel[1]);
      bounds.at(6 * _args.numKnots - 1) =
          ifopt::Bounds(_args.targetAngVel[2], _args.targetAngVel[2]);
    }

    return bounds;
  }

protected:
  const SingleRigidBodyDynamicsModel _model;
  const Terrain _terrain;
  TrajOptArguments _args;

  ifopt::Problem _nlp;

  double _sampleTime;
};

} // namespace trajopt