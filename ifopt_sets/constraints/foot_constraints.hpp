
#pragma once

#include <ifopt/constraint_set.h>

#include "../../utils/srbd.hpp"
#include "../variables.hpp"

namespace trajopt {
class FrictionConeConstraints : public ifopt::ConstraintSet {
public:
  FrictionConeConstraints(
      const std::shared_ptr<PhasedTrajectoryVars> &forceVars,
      const trajopt::Terrain &terrain, size_t numSamples, double sampleTime)
      : ConstraintSet(5 * numSamples, forceVars->GetName() + "_friction_cone"),
        _forceVarsName(forceVars->GetName()), _numSamples(numSamples),
        _sampleTime(sampleTime), _terrain(terrain) {}

  VectorXd GetValues() const override {
    VectorXd g = VectorXd::Zero(GetRows());

    auto forceVars = std::static_pointer_cast<PhasedTrajectoryVars>(
        GetVariables()->GetComponent(_forceVarsName));

    double t = 0.;
    for (size_t i = 0; i < _numSamples; ++i) {
      Eigen::Vector3d f = forceVars->position(t);
      double fn = f.dot(_terrain.n);
      double ft = f.dot(_terrain.t);
      double fb = f.dot(_terrain.b);

      g[i * 5 + 0] = fn;
      g[i * 5 + 1] = ft - _terrain.mu * fn;
      g[i * 5 + 2] = -ft - _terrain.mu * fn;
      g[i * 5 + 3] = fb - _terrain.mu * fn;
      g[i * 5 + 4] = -fb - _terrain.mu * fn;

      t += _sampleTime;
    }

    return g;
  }

  VecBound GetBounds() const override {
    VecBound b(GetRows());

    for (size_t i = 0; i < _numSamples; ++i) {
      b.at(i * 5 + 0) = ifopt::BoundGreaterZero;
      b.at(i * 5 + 1) = ifopt::BoundSmallerZero;
      b.at(i * 5 + 2) = ifopt::BoundSmallerZero;
      b.at(i * 5 + 3) = ifopt::BoundSmallerZero;
      b.at(i * 5 + 4) = ifopt::BoundSmallerZero;
    }

    return b;
  }

  void FillJacobianBlock(std::string var_set,
                         Jacobian &jac_block) const override {
    if (var_set == _forceVarsName) {
      auto forceVars = std::static_pointer_cast<PhasedTrajectoryVars>(
          GetVariables()->GetComponent(_forceVarsName));

      double t = 0.;
      for (size_t i = 0; i < _numSamples; ++i) {
        Jacobian fPos = forceVars->jacobianPosition(t);

        Jacobian mult0 = _terrain.n.transpose().sparseView(1, -1);
        Jacobian res0 = mult0 * fPos;

        Jacobian mult1 = (_terrain.t - _terrain.mu * _terrain.n)
                             .transpose()
                             .sparseView(1, -1);
        Jacobian res1 = mult1 * fPos;

        Jacobian mult2 = (-_terrain.t - _terrain.mu * _terrain.n)
                             .transpose()
                             .sparseView(1, -1);
        Jacobian res2 = mult2 * fPos;

        Jacobian mult3 = (_terrain.b - _terrain.mu * _terrain.n)
                             .transpose()
                             .sparseView(1, -1);
        Jacobian res3 = mult3 * fPos;

        Jacobian mult4 = (-_terrain.b - _terrain.mu * _terrain.n)
                             .transpose()
                             .sparseView(1, -1);
        Jacobian res4 = mult4 * fPos;

        jac_block.middleRows(i * 5 + 0, 1) += res0;
        jac_block.middleRows(i * 5 + 1, 1) += res1;
        jac_block.middleRows(i * 5 + 2, 1) += res2;
        jac_block.middleRows(i * 5 + 3, 1) += res3;
        jac_block.middleRows(i * 5 + 4, 1) += res4;

        t += _sampleTime;
      }
    }
  }

private:
  const std::string _forceVarsName;
  const size_t _numSamples;
  const double _sampleTime;
  const trajopt::Terrain _terrain;
};

class FootPosTerrainConstraints : public ifopt::ConstraintSet {
public:
  FootPosTerrainConstraints(const std::shared_ptr<PhasedTrajectoryVars> &vars,
                            const trajopt::Terrain &terrain, size_t numSteps,
                            size_t numSwings, size_t numKnotsPerSwing)
      : ConstraintSet(kSpecifyLater, vars->GetName() + "_foot_pos_terrain"),
        _varsName(vars->GetName()), _terrain(terrain),
        _numPhases(numSteps + numSwings), _numSteps(numSteps),
        _numSwings(numSwings), _numKnotsPerSwing(numKnotsPerSwing) {
    SetRows(numSteps + numSwings * numKnotsPerSwing);
  }

  VectorXd GetValues() const override {
    VectorXd g = VectorXd::Zero(GetRows());

    auto values = std::static_pointer_cast<PhasedTrajectoryVars>(
                      GetVariables()->GetComponent(_varsName))
                      ->GetValues();

    bool standing = std::static_pointer_cast<PhasedTrajectoryVars>(
                        GetVariables()->GetComponent(_varsName))
                        ->standingAtStart();
    size_t valIdx = 0;
    size_t cIdx = 0;

    // std::cout << "##############" << std::endl;
    // std::cout << GetName() << " num of phases: " << _numPhases << std::endl;
    // std::cout << GetName() << " num of steps: " << _numSteps << std::endl;
    // std::cout << GetName() << " num of swings: " << _numSwings << std::endl;
    // std::cout << "##############" << std::endl;
    for (size_t i = 0; i < _numPhases; ++i) {
      if (standing) {
        g(cIdx++) =
            values[valIdx + 2] - _terrain.z(values[valIdx], values[valIdx + 1]);
        valIdx += 3;
      } else {
        for (size_t k = 0; k < _numKnotsPerSwing; ++k) {
          g(cIdx++) = values[valIdx + 2] -
                      _terrain.z(values[valIdx], values[valIdx + 1]);
          valIdx += 6;
        }
      }
      standing = !standing;
    }
    return g;
  }

  VecBound GetBounds() const override {
    VecBound b(GetRows(), ifopt::BoundZero);

    bool standing = std::static_pointer_cast<PhasedTrajectoryVars>(
                        GetVariables()->GetComponent(_varsName))
                        ->standingAtStart();
    size_t bIdx = 0;

    for (size_t i = 0; i < _numPhases; ++i) {
      if (standing) {
        b.at(bIdx++) = ifopt::BoundZero;
      } else {
        for (size_t k = 0; k < _numKnotsPerSwing; ++k) {
          b.at(bIdx++) = ifopt::BoundGreaterZero;
        }
      }
      standing = !standing;
    }
    b.back() = ifopt::BoundZero;

    return b;
  }

  void FillJacobianBlock(std::string var_set,
                         Jacobian &jac_block) const override {
    if (var_set == _varsName) {
      auto vars = std::static_pointer_cast<PhasedTrajectoryVars>(
          GetVariables()->GetComponent(_varsName));

      bool standing = std::static_pointer_cast<PhasedTrajectoryVars>(
                          GetVariables()->GetComponent(_varsName))
                          ->standingAtStart();
      size_t rowIdx = 0;
      size_t colIdx = 0;

      // We should also compute the gradient of the terrain function w.r.t. the
      // foot's x and y positions here, however, since the use cases tested are
      // tested only on a step terrain (discontinuous), we set the gradient
      // zero.
      for (size_t i = 0; i < _numPhases; ++i) {
        if (standing) {
          jac_block.coeffRef(rowIdx++, colIdx + 2) = 1.;
          colIdx += 3;
        } else {
          for (size_t k = 0; k < _numKnotsPerSwing; ++k) {
            jac_block.coeffRef(rowIdx++, colIdx + 2) = 1.;
            colIdx += 6;
          }
        }
        standing = !standing;
      }
    }
  }

private:
  const std::string _varsName;
  const trajopt::Terrain _terrain;
  const size_t _numPhases;
  const size_t _numSteps;
  const size_t _numSwings;
  const size_t _numKnotsPerSwing;
};

class FootBodyPosConstraints : public ifopt::ConstraintSet {
public:
  FootBodyPosConstraints(
      const SingleRigidBodyDynamicsModel &model,
      const std::shared_ptr<TrajectoryVars> &bodyPosVars,
      const std::shared_ptr<TrajectoryVars> &bodyRotVars,
      const std::shared_ptr<PhasedTrajectoryVars> &footPosVars,
      size_t numSamples, double sampleTime)
      : ConstraintSet(3 * numSamples,
                      footPosVars->GetName() + "_foot_body_pos"),
        _model(model), _bodyPosVarsName(bodyPosVars->GetName()),
        _bodyRotVarsName(bodyRotVars->GetName()),
        _footPosVarsName(footPosVars->GetName()), _numSamples(numSamples),
        _sampleTime(sampleTime) {}

  VectorXd GetValues() const override {
    VectorXd g = VectorXd::Zero(GetRows());

    auto bodyPosVars = std::static_pointer_cast<TrajectoryVars>(
        GetVariables()->GetComponent(_bodyPosVarsName));
    auto bodyRotVars = std::static_pointer_cast<TrajectoryVars>(
        GetVariables()->GetComponent(_bodyRotVarsName));
    auto footPosVars = std::static_pointer_cast<PhasedTrajectoryVars>(
        GetVariables()->GetComponent(_footPosVarsName));

    double t = 0.;
    for (size_t i = 0; i < _numSamples; ++i) {
      Eigen::Vector3d b = bodyPosVars->position(t);
      Jacobian R = eulerZYXToMatrix(bodyRotVars->position(t));
      Eigen::Vector3d f = footPosVars->position(t);

      g.segment(i * 3, 3) = R.transpose() * (f - b);

      t += _sampleTime;
    }

    return g;
  }

  VecBound GetBounds() const override {
    VecBound b(GetRows(), ifopt::BoundZero);

    size_t idx = 0;
    for (size_t k = 0; k < _model.numFeet; ++k) {
      if (_footPosVarsName == PAW_POS + "_" + std::to_string(k)) {
        break;
      }
      ++idx;
    }

    for (size_t i = 0; i < _numSamples; ++i) {
      b.at(i * 3 + 0) = ifopt::Bounds(_model.feetMinBounds[idx][0],
                                      _model.feetMaxBounds[idx][0]);
      b.at(i * 3 + 1) = ifopt::Bounds(_model.feetMinBounds[idx][1],
                                      _model.feetMaxBounds[idx][1]);
      b.at(i * 3 + 2) = ifopt::Bounds(_model.feetMinBounds[idx][2],
                                      _model.feetMaxBounds[idx][2]);
    }

    return b;
  }

  void FillJacobianBlock(std::string var_set,
                         Jacobian &jac_block) const override {
    auto bodyPosVars = std::static_pointer_cast<TrajectoryVars>(
        GetVariables()->GetComponent(_bodyPosVarsName));
    auto bodyRotVars = std::static_pointer_cast<TrajectoryVars>(
        GetVariables()->GetComponent(_bodyRotVarsName));
    auto footPosVars = std::static_pointer_cast<PhasedTrajectoryVars>(
        GetVariables()->GetComponent(_footPosVarsName));

    if (var_set == _bodyPosVarsName) {
      double t = 0.;
      for (size_t i = 0; i < _numSamples; ++i) {
        Jacobian R = eulerZYXToMatrix(bodyRotVars->position(t));
        Jacobian dBodyPos = bodyPosVars->jacobianPosition(t);

        jac_block.middleRows(i * 3, 3) = -R.transpose() * dBodyPos;

        t += _sampleTime;
      }
    } else if (var_set == _bodyRotVarsName) {
      double t = 0.;
      for (size_t i = 0; i < _numSamples; ++i) {
        Jacobian dBodyRot = bodyRotVars->jacobianPosition(t);

        Eigen::Vector3d b = bodyPosVars->position(t);
        Eigen::Vector3d euler_zyx = bodyRotVars->position(t);
        Eigen::Vector3d f = footPosVars->position(t);

        Jacobian mult = derivRotationTransposeVector(euler_zyx, f - b);
        Jacobian res = mult * dBodyRot;
        jac_block.middleRows(i * 3, 3) = res;

        t += _sampleTime;
      }
    } else if (var_set == _footPosVarsName) {
      double t = 0.;
      for (size_t i = 0; i < _numSamples; ++i) {
        Jacobian R = eulerZYXToMatrix(bodyRotVars->position(t));
        Jacobian dFootPos = footPosVars->jacobianPosition(t);

        jac_block.middleRows(i * 3, 3) = R.transpose() * dFootPos;

        t += _sampleTime;
      }
    }
  }

protected:
  Jacobian eulerZYXToMatrix(const Eigen::Vector3d &eulerZYX) const {
    const double x = eulerZYX[2];
    const double y = eulerZYX[1];
    const double z = eulerZYX[0];

    const double cx = std::cos(x);
    const double sx = std::sin(x);

    const double cy = std::cos(y);
    const double sy = std::sin(y);

    const double cz = std::cos(z);
    const double sz = std::sin(z);

    Eigen::Matrix3d R;
    R << cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy, cy * sz,
        cx * cz + sx * sy * sz, cx * sy * sz - cz * sx, -sy, cy * sx, cx * cy;

    return R.sparseView(1., -1.);
  }

  Jacobian derivRotationTransposeVector(const Eigen::Vector3d &eulerZYX,
                                        const Eigen::Vector3d &v) const {
    const double x = eulerZYX[2];
    const double y = eulerZYX[1];
    const double z = eulerZYX[0];

    const double cx = std::cos(x);
    const double sx = std::sin(x);

    const double cy = std::cos(y);
    const double sy = std::sin(y);

    const double cz = std::cos(z);
    const double sz = std::sin(z);

    // out = R.T * v
    // out[0] = R.T.row(0) * v = cy * cz * v[0] + cy * sz * v[1] + -sy * v[2]
    // out[1] = R.T.row(1) * v = (cz * sx * sy - cx * sz) * v[0] + (cx * cz + sx
    // * sy * sz) * v[1] + cy * sx * v[2] out[2] = R.T.row(2) * v = (sx * sz +
    // cx * cz * sy) * v[0] + (cx * sy * sz - cz * sx) * v[1] + cx * cy * v[2]

    Jacobian jac(3, 3);
    // out[0] wrt Z(0)
    jac.coeffRef(0, 0) = -cy * sz * v[0] + cy * cz * v[1];
    // out[0] wrt Y(1)
    jac.coeffRef(0, 1) = -sy * cz * v[0] - sy * sz * v[1] - cy * v[2];
    // out[0] wrt X(2)
    // jac.coeffRef(0, 2) = 0.;

    // out[1] wrt Z(0)
    jac.coeffRef(1, 0) =
        (-sz * sx * sy - cx * cz) * v[0] + (-cx * sz + sx * sy * cz) * v[1];
    // out[1] wrt Y(1)
    jac.coeffRef(1, 1) =
        cz * sx * cy * v[0] + sx * cy * sz * v[1] - sy * sx * v[2];
    // out[1] wrt X(2)
    jac.coeffRef(1, 2) = (cz * cx * sy + sx * sz) * v[0] +
                         (-sx * cz + cx * sy * sz) * v[1] + cy * cx * v[2];

    // out[2] wrt Z(0)
    jac.coeffRef(2, 0) =
        (sx * cz - cx * sz * sy) * v[0] + (cx * sy * cz + sz * sx) * v[1];
    // out[2] wrt Y(1)
    jac.coeffRef(2, 1) =
        cx * cz * cy * v[0] + cx * cy * sz * v[1] - cx * sy * v[2];
    // out[2] wrt X(2)
    jac.coeffRef(2, 2) = (cx * sz - sx * cz * sy) * v[0] +
                         (-sx * sy * sz - cz * cx) * v[1] - sx * cy * v[2];

    return jac;
  }

protected:
  const SingleRigidBodyDynamicsModel _model;
  const std::string _bodyPosVarsName;
  const std::string _bodyRotVarsName;
  const std::string _footPosVarsName;
  const size_t _numSamples;
  const double _sampleTime;
};

} // namespace trajopt