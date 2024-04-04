#pragma once

#include <cstdlib>
#include <robo_spline/cubic_hermite_spline.hpp>
#include <robo_spline/trajectory.hpp>
#include <robo_spline/phased_trajectory.hpp>
#include <string>

#include <ifopt/variable_set.h>
#include <robo_spline/trajectory.hpp>

namespace trajopt {
class TrajectoryVars : public ifopt::VariableSet {
public:
  using Jacobian = ifopt::Component::Jacobian;

  TrajectoryVars(const std::string& name, const Eigen::VectorXd& initVals, const Eigen::VectorXd& splineTimes, const VecBound &bounds)
                : VariableSet(kSpecifyLater, name), _values(initVals), _bounds(bounds), _splineTimes(splineTimes) {
    // SetVariables(Eigen::VectorXd::Random(GetRows())); // initialize to zero
    // SetVariables(Eigen::VectorXd::Zero(GetRows())); // initialize to zero
    SetVariables(initVals);
    SetRows(_traj.num_vars());
  }

  void SetVariables(const Eigen::VectorXd &x) override {
    _values = x;
    
    _traj.clear(); // Sanity check.
    _traj = rspl::Trajectory<rspl::CubicHermiteSpline<3>>(_values, _splineTimes); // Recreate trajectory with new points.
  }

  Eigen::VectorXd GetValues() const override { return _values; }
  VecBound GetBounds() const override { return _bounds; }

  size_t numKnotPoints() const { return _traj.num_knot_points(); }
  size_t numSplines() const { return _traj.num_knot_points() - 1; }
  double splineDuration(size_t idx) const { return _traj.spline(idx)->duration(); }
 
  inline Eigen::Vector3d trajectoryEval(double t, size_t order) const { return _traj.eval(t, order); }
  inline Jacobian trajectoryJacobian(double t, size_t order) const { return _traj.jac_block(t, order); }

  inline Eigen::Vector3d splineEval(size_t idx, double t, size_t order) const { return _traj.spline(idx)->eval(t, order); }
  inline Jacobian splineJacobian(size_t spline_idx, double t, size_t order ) const { return _traj.jac_block(spline_idx, t, order); }
 
protected:
  Eigen::VectorXd _values;
  VecBound _bounds;
  Eigen::VectorXd _splineTimes;

  rspl::Trajectory<rspl::CubicHermiteSpline<3>> _traj;
};

class PhasedTrajectoryVars : public ifopt::VariableSet {
public:
  using Jacobian = ifopt::Component::Jacobian;

  PhasedTrajectoryVars(const std::string &name, const Eigen::VectorXd& initVals, const VecBound& bounds, const Eigen::VectorXd& phaseTimes, const std::vector<size_t> knotsPerSwing, rspl::Phase initPhase)
                      : VariableSet(kSpecifyLater, name), _values(initVals), _bounds(bounds), _phaseTimes(phaseTimes), _knotsPerSwing(knotsPerSwing), _initPhase(initPhase) 
  {
    // SetVariables(Eigen::VectorXd::Random(GetRows())); // initialize to random
    // SetVariables(Eigen::VectorXd::Zero(GetRows())); // initialize to zero
    SetVariables(initVals);
    SetRows(_traj.num_vars());
  }

  void SetVariables(const Eigen::VectorXd &x) override {
    _values = x;

    _traj.clear(); // Sanity check.
    _traj = rspl::PhasedTrajectory<rspl::CubicHermiteSpline<3>>(_values, _phaseTimes, _knotsPerSwing, _initPhase ); // Recreate trajectory with new points.
  }

  Eigen::VectorXd GetValues() const override { return _values; }
  VecBound GetBounds() const override { return _bounds; }

  size_t numKnotPoints() const { return _traj.num_knot_points(); }
  size_t numSplines() const { return _traj.num_knot_points() - 1; }
  double splineDuration(size_t idx) const { return _traj.spline(idx)->duration(); }

  inline Eigen::Vector3d trajectoryEval(double t, size_t order) const { return _traj.eval(t, order); }
  inline Jacobian trajectoryJacobian(double t, size_t order) const { return _traj.jac_block(t, order); }

  inline Eigen::Vector3d splineEval(size_t idx, double t, size_t order) const { return _traj.spline(idx)->eval(t, order); }
  inline Jacobian splineJacobian(size_t idx, double t, size_t order ) const { return _traj.jac_block(idx, t, order); }

  inline bool standingAt(double t) const { return (_traj.phase_at(t) == rspl::Phase::Stance ) ? true : false; }
  inline size_t varStartAt(double t) const { return _traj.var_start_at(t); } 

protected:
  Eigen::VectorXd _values;
  VecBound _bounds;
  Eigen::VectorXd _phaseTimes;
  std::vector<size_t> _knotsPerSwing;
  rspl::Phase _initPhase;

  rspl::PhasedTrajectory<rspl::CubicHermiteSpline<3>> _traj;
};

} // namespace trajopt
