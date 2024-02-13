#pragma once

#include <cstdlib>
#include <numeric>
#include <robo_spline/trajectory.hpp>
#include <string>

#include <ifopt/variable_set.h>
#include <robo_spline/trajectory.hpp>

#include "../utils/types.hpp"

namespace trajopt {
class TrajectoryVars : public ifopt::VariableSet {
public:
  using Jacobian = ifopt::Component::Jacobian;

  TrajectoryVars(const std::string &name, uint32_t numKnotPoints,
                 const Eigen::VectorXd &polyTimes, const VecBound &bounds)
      : VariableSet(numKnotPoints * 6, name), _numKnotPoints(numKnotPoints),
        _polyTimes(polyTimes), _bounds(bounds) {
    // SetVariables(Eigen::VectorXd::Random(GetRows())); // initialize to zero
    SetVariables(Eigen::VectorXd::Zero(GetRows())); // initialize to zero
  }

  void SetVariables(const Eigen::VectorXd &x) override {
    _values = x;

    UpdateTrajectory();
  }

  Eigen::VectorXd GetValues() const override { return _values; }

  VecBound GetBounds() const override { return _bounds; }

  uint32_t numKnotPoints() const { return _numKnotPoints; }
  Eigen::VectorXd polyTimes() const { return _polyTimes; }

  Eigen::Vector3d position(uint32_t k) const {
    return _values.segment(k * 6, 3);
  }
  Eigen::Vector3d velocity(uint32_t k) const {
    return _values.segment(k * 6 + 3, 3);
  }

  Eigen::Vector3d position(double t) const { return _traj.position(t); }
  Eigen::Vector3d velocity(double t) const { return _traj.velocity(t); }
  Eigen::Vector3d acceleration(double t) const { return _traj.acceleration(t); }

  // TO-DO: How can I avoid the transpose here?
  Jacobian jacobianPosition(double t) const {
    rspl::Trajectory3D::SparseJacobian jac;
    rspl::Trajectory3D::SplineIndex index;
    std::tie(index, jac) = _traj.jacobian_pos(t);

    // std::cout << jac << std::endl;

    Jacobian fullJac(GetRows(), 3);
    fullJac.middleRows(index * 6, jac.cols()) = jac.transpose();

    return fullJac.transpose();
  }

  Jacobian jacobianVelocity(double t) const {
    rspl::Trajectory3D::SparseJacobian jac;
    rspl::Trajectory3D::SplineIndex index;
    std::tie(index, jac) = _traj.jacobian_vel(t);

    Jacobian fullJac(GetRows(), 3);
    fullJac.middleRows(index * 6, jac.cols()) = jac.transpose();

    return fullJac.transpose();
  }

  Jacobian jacobianAcceleration(double t) const {
    rspl::Trajectory3D::SparseJacobian jac;
    rspl::Trajectory3D::SplineIndex index;
    std::tie(index, jac) = _traj.jacobian_acc(t);

    Jacobian fullJac(GetRows(), 3);
    fullJac.middleRows(index * 6, jac.cols()) = jac.transpose();

    return fullJac.transpose();
  }

  void UpdateTrajectory() {
    // Re-create Trajectory with new points.
    _traj.clear();

    for (uint32_t i = 0; i < _numKnotPoints; i++) {
      double dt = (i == 0) ? 0. : _polyTimes[i];
      _traj.add_point(_values.segment(i * 6, 3), _values.segment(i * 6 + 3, 3),
                      dt);
    }
  }

  rspl::Trajectory3D &traj() { return _traj; }

protected:
  uint32_t _numKnotPoints;
  Eigen::VectorXd _polyTimes;
  VecBound _bounds;
  Eigen::VectorXd _values;
  rspl::Trajectory3D _traj;
};

class PhasedTrajectoryVars : public ifopt::VariableSet {
public:
  using Jacobian = ifopt::Component::Jacobian;

  PhasedTrajectoryVars(const std::string &name,
                       const std::vector<double> &polyTimes,
                       const VecBound &bounds, size_t numStepPhases,
                       size_t numSwingPhases, size_t numKnotsPerSwing,
                       bool standingAtStart)
      : VariableSet(3 * numStepPhases + 6 * numKnotsPerSwing * numSwingPhases,
                    name),
        _polyTimes(polyTimes),
        _totalTime(std::accumulate(polyTimes.begin(), polyTimes.end(), 0.)),
        _bounds(bounds), _numTotalPhases(numStepPhases + numSwingPhases),
        _numStepPhases(numStepPhases), _numSwingPhases(numSwingPhases),
        _numKnotsPerSwing(numKnotsPerSwing), _standingAtStart(standingAtStart) {
    size_t totalKnotPoints =
        2 * _numStepPhases + _numKnotsPerSwing * _numSwingPhases;

    _isStance.resize(totalKnotPoints - 1);
    _varStart.resize(totalKnotPoints - 1);

    bool isStance = _standingAtStart;
    size_t vIdx = 0;
    size_t kIdx = 0;
    for (size_t k = 0; k < _numTotalPhases; k++) {
      if (isStance) {
        _isStance[kIdx] = isStance;
        _varStart[kIdx++] = vIdx;

        vIdx += 3;
      } else {
        if (vIdx >= 3)
          vIdx -= 3;
        for (size_t j = 0; j < _numKnotsPerSwing; j++) {
          _isStance[kIdx] = isStance;
          _varStart[kIdx++] = vIdx;

          if (j == 0 && kIdx > 1)
            vIdx += 3;
          else
            vIdx += 6;
        }
        if (k > 0 && kIdx < (_isStance.size() - 1)) {
          _isStance[kIdx] = isStance;
          _varStart[kIdx++] = vIdx;
          vIdx += 6;
        }
      }

      isStance = !isStance;
    }

    // for (size_t k = 0; k < _isStance.size(); k++) {
    //     std::cout << "#" << k << ": " << (_isStance[k] ? "Stance" : "Swing")
    //     << " -> " << _varStart[k] << std::endl;
    // }
    // std::cout << _values.size() << std::endl;

    // SetVariables(Eigen::VectorXd::Random(GetRows())); // initialize to random
    SetVariables(Eigen::VectorXd::Zero(GetRows())); // initialize to zero
  }

  void SetVariables(const Eigen::VectorXd &x) override {
    _values = x;

    _updateTrajectory();
  }

  Eigen::VectorXd GetValues() const override { return _values; }

  VecBound GetBounds() const override { return _bounds; }

  size_t numKnotPoints() const { return _traj.splines().size() + 1; }
  const std::vector<double> &polyTimes() const { return _polyTimes; }

  Eigen::Vector3d position(double t) const { return _traj.position(t); }
  Eigen::Vector3d velocity(double t) const { return _traj.velocity(t); }
  Eigen::Vector3d acceleration(double t) const { return _traj.acceleration(t); }

  // TO-DO: How can I avoid the transpose here?
  Jacobian jacobianPosition(double t) const {
    // Let's get the Jacobian and Index from the trajectory
    rspl::Trajectory3D::SparseJacobian jac;
    rspl::Trajectory3D::SplineIndex index;
    std::tie(index, jac) = _traj.jacobian_pos(t);

    // Now let's fill the Jacobian
    Jacobian fullJac(GetRows(), 3);

    _fillJac(fullJac, jac, index);

    return fullJac.transpose();
  }

  // TO-DO: How can I avoid the transpose here?
  Jacobian jacobianVelocity(double t) const {
    // Let's get the Jacobian and Index from the trajectory
    rspl::Trajectory3D::SparseJacobian jac;
    rspl::Trajectory3D::SplineIndex index;
    std::tie(index, jac) = _traj.jacobian_vel(t);

    // Now let's fill the Jacobian
    Jacobian fullJac(GetRows(), 3);

    _fillJac(fullJac, jac, index);

    return fullJac.transpose();
  }

  // TO-DO: How can I avoid the transpose here?
  Jacobian jacobianAcceleration(double t) const {
    // Let's get the Jacobian and Index from the trajectory
    rspl::Trajectory3D::SparseJacobian jac;
    rspl::Trajectory3D::SplineIndex index;
    std::tie(index, jac) = _traj.jacobian_acc(t);

    // Now let's fill the Jacobian
    Jacobian fullJac(GetRows(), 3);

    _fillJac(fullJac, jac, index);

    return fullJac.transpose();
  }

  rspl::Trajectory3D &traj() { return _traj; }

  size_t numKnotsPerSwing() const { return _numKnotsPerSwing; }

  bool standingAtStart() const { return _standingAtStart; }

  bool isSplineStance(size_t k) const { return _isStance[k]; }
  size_t varStart(size_t k) const { return _varStart[k]; }

  double totalTime() const { return _totalTime; }

protected:
  void _updateTrajectory() {
    _traj.clear();

    const Eigen::Vector3d zeroVel = Eigen::Vector3d::Zero();

    bool isStance = _standingAtStart;
    int kIdx = -1;
    size_t vIdx = 0;

    Eigen::Vector3d pos, vel;
    double dt = 0.;

    for (size_t k = 0; k < _numTotalPhases; k++) {
      if (isStance) {
        pos = _values.segment(vIdx, 3);
        _traj.add_point(pos, zeroVel, dt);
        // std::cout << "Adding point: (" << pos.transpose() << "), (" <<
        // zeroVel.transpose() << ") -> " << dt << std::endl;

        dt = _polyTimes[++kIdx];
        _traj.add_point(pos, zeroVel, dt);
        // std::cout << "Adding point: (" << pos.transpose() << "), (" <<
        // zeroVel.transpose() << ") -> " << dt << std::endl;

        vIdx += 3;
        kIdx++;
        dt = _polyTimes[kIdx];
      } else {
        for (size_t j = 0; j < _numKnotsPerSwing; j++) {
          pos = _values.segment(vIdx, 3);
          vel = _values.segment(vIdx + 3, 3);

          _traj.add_point(pos, vel, dt);

          // std::cout << "Adding point: (" << pos.transpose() << "), (" <<
          // vel.transpose() << ") -> " << dt << std::endl;

          kIdx++;
          vIdx += 6;
          dt = _polyTimes[kIdx];
        }
      }

      isStance = !isStance;
    }
  }

  void _fillJac(Jacobian &fullJac, const Jacobian &jac, size_t index) const {
    bool isStance = _isStance[index];
    size_t sIdx = _varStart[index];

    if (isStance) {
      fullJac.middleRows(sIdx, 3) = jac.block(0, 0, jac.rows(), 3).transpose() +
                                    jac.block(0, 6, jac.rows(), 3).transpose();
    } else {
      bool hasPrev = (index > 0);
      bool hasNext = (index < (_isStance.size() - 1));
      bool prevStance = (index == 0) ? false : _isStance[index - 1];
      bool nextStance =
          (index >= (_isStance.size() - 1)) ? false : _isStance[index + 1];

      bool firstNode = !hasPrev && hasNext;
      bool lastNode = hasPrev && !hasNext;
      bool middleNode = hasPrev && hasNext;

      size_t pIdx = hasPrev ? _varStart[index - 1] : sIdx;
      size_t nIdx = hasNext ? _varStart[index + 1] : sIdx;

      // First node cases
      if (firstNode && !nextStance)
        fullJac.middleRows(sIdx, jac.cols()) = jac.transpose();
      else if (firstNode && nextStance) {
        fullJac.middleRows(sIdx, 6) =
            jac.block(0, 0, jac.rows(), 6).transpose();
        fullJac.middleRows(nIdx, 3) =
            jac.block(0, 6, jac.rows(), 3).transpose();
      }

      // Last node cases
      if (lastNode && prevStance) {
        fullJac.middleRows(pIdx, 3) =
            jac.block(0, 0, jac.rows(), 3).transpose();
        fullJac.middleRows(sIdx, 6) =
            jac.block(0, 6, jac.rows(), 6).transpose();
      } else if (lastNode && !prevStance) {
        fullJac.middleRows(sIdx, jac.cols()) = jac.transpose();
      }

      // Middle node cases
      if (middleNode) {
        if (prevStance) {
          fullJac.middleRows(pIdx, 3) =
              jac.block(0, 0, jac.rows(), 3).transpose();
          fullJac.middleRows(nIdx, 6) =
              jac.block(0, 6, jac.rows(), 6).transpose();
        } else if (!prevStance && !nextStance) {
          fullJac.middleRows(sIdx, jac.cols()) = jac.transpose();
        } else if (!prevStance && nextStance) {
          fullJac.middleRows(sIdx, 9) =
              jac.block(0, 0, jac.rows(), 9).transpose();
        }
      }
    }
  }

  const std::vector<double> _polyTimes;
  const double _totalTime;
  const VecBound _bounds;
  const size_t _numTotalPhases;
  const size_t _numStepPhases;
  const size_t _numSwingPhases;
  const size_t _numKnotsPerSwing;
  const bool _standingAtStart;

  std::vector<bool> _isStance;
  std::vector<size_t> _varStart;

  Eigen::VectorXd _values;
  rspl::Trajectory3D _traj;
};

} // namespace trajopt
