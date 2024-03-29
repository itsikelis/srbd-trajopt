#pragma once

#include <ifopt/constraint_set.h>

#include "../../utils/srbd.hpp"
#include "../variables.hpp"

namespace trajopt {
class DynamicsConstraint : public ifopt::ConstraintSet {
public:
  DynamicsConstraint(const SingleRigidBodyDynamicsModel &model,
                     unsigned int numSamplePoints, double sampleTime)
      : ConstraintSet(6 * numSamplePoints, BODY_DYNAMICS), _model(model),
        _numSamplePoints(numSamplePoints), _sampleTime(sampleTime) {}

  VectorXd GetValues() const override {
    VectorXd g = VectorXd::Zero(GetRows());

    auto positionVars = std::static_pointer_cast<TrajectoryVars>(
        GetVariables()->GetComponent(BODY_POS_TRAJECTORY));
    auto rotationVars = std::static_pointer_cast<TrajectoryVars>(
        GetVariables()->GetComponent(BODY_ROT_TRAJECTORY));

    double t = 0.;
    for (unsigned int i = 0; i < _numSamplePoints; i++) {
      Eigen::Vector3d acc = positionVars->acceleration(t);
      Eigen::Vector3d accEuler = rotationVars->acceleration(t);
      Eigen::Vector3d velEuler = rotationVars->velocity(t);

      Eigen::Vector3d f = Eigen::Vector3d::Zero();
      Eigen::Vector3d tau = Eigen::Vector3d::Zero();
      for (unsigned int k = 0; k < _model.numFeet; k++) {
        auto forceVars = std::static_pointer_cast<PhasedTrajectoryVars>(
            GetVariables()->GetComponent(PAW_FORCES + "_" + std::to_string(k)));
        Eigen::Vector3d force = forceVars->position(t);

        auto pawPosVars = std::static_pointer_cast<PhasedTrajectoryVars>(
            GetVariables()->GetComponent(PAW_POS + "_" + std::to_string(k)));
        Eigen::Vector3d bodyPos = positionVars->position(t);
        Eigen::Vector3d pawPos = pawPosVars->position(t);
        tau += (pawPos - bodyPos).cross(force);

        // tau += _model.feetPoses[k].cross(force);

        f += force;
      }

      Eigen::Vector3d eulerZYX = rotationVars->position(t);
      Eigen::Matrix3d R = eulerZYXToMatrix(eulerZYX);
      Eigen::Matrix3d E = eulerZYXToOmega(eulerZYX);
      Eigen::Matrix3d Edot = eulerZYXToOmegaDot(eulerZYX, velEuler);
      Eigen::Vector3d omega = E * velEuler;
      Eigen::Vector3d omegaDot = E * accEuler + Edot * velEuler;

      g.segment(i * 6, 3) = acc - f / _model.mass - _model.gravity;
      g.segment(i * 6 + 3, 3) =
          omegaDot -
          (_model.inertia.inverse() *
           (R.transpose() * tau - omega.cross(_model.inertia * omega)));

      t += _sampleTime;
    }

    return g;
  }

  VecBound GetBounds() const override {
    // All constraints equal to zero.
    VecBound b(GetRows());
    for (int i = 0; i < GetRows(); i++) {
      b.at(i) = ifopt::BoundZero;
    }
    return b;
  }

  void FillJacobianBlock(std::string var_set,
                         Jacobian &jac_block) const override {
    Jacobian inertia =
        _model.inertia.sparseView(1., -1.); // TO-DO: Make this better
    Jacobian inertia_inv = _model.inertia.inverse().sparseView(1., -1.);

    if (var_set == BODY_POS_TRAJECTORY) {
      auto positionVars = std::static_pointer_cast<TrajectoryVars>(
          GetVariables()->GetComponent(BODY_POS_TRAJECTORY));
      auto rotationVars = std::static_pointer_cast<TrajectoryVars>(
          GetVariables()->GetComponent(BODY_ROT_TRAJECTORY));

      double t = 0.;
      for (size_t i = 0; i < _numSamplePoints; ++i) {
        jac_block.middleRows(i * 6, 3) = positionVars->jacobianAcceleration(t);

        Jacobian dPos = positionVars->jacobianPosition(t);

        Jacobian derivTauSum(3, 3);
        for (size_t k = 0; k < _model.numFeet; ++k) {
          std::string fVar = PAW_FORCES + "_" + std::to_string(k);
          std::string pVar = PAW_FORCES + "_" + std::to_string(k);

          Eigen::Vector3d pawPos =
              std::static_pointer_cast<PhasedTrajectoryVars>(
                  GetVariables()->GetComponent(pVar))
                  ->position(t);
          Eigen::Vector3d f = std::static_pointer_cast<PhasedTrajectoryVars>(
                                  GetVariables()->GetComponent(fVar))
                                  ->position(t);
          Eigen::Vector3d bodyPos = positionVars->position(t);

          derivTauSum -= derivSkewMultiplyVector((pawPos - bodyPos), f);
        }

        Eigen::Vector3d eulerZYX = rotationVars->position(t);
        Jacobian R = eulerZYXToMatrix(eulerZYX);

        Jacobian multiplier = inertia_inv * R.transpose() * derivTauSum;
        Jacobian res = -multiplier * dPos;
        jac_block.middleRows(i * 6 + 3, 3) += res;

        t += _sampleTime;
      }
    } else if (var_set == BODY_ROT_TRAJECTORY) {
      auto positionVars = std::static_pointer_cast<TrajectoryVars>(
          GetVariables()->GetComponent(BODY_POS_TRAJECTORY));
      auto rotationVars = std::static_pointer_cast<TrajectoryVars>(
          GetVariables()->GetComponent(BODY_ROT_TRAJECTORY));
      double t = 0.;
      for (unsigned int i = 0; i < _numSamplePoints; i++) {
        Jacobian dAcc = rotationVars->jacobianAcceleration(t);
        Jacobian dVel = rotationVars->jacobianVelocity(t);
        Jacobian dPos = rotationVars->jacobianPosition(t);
        Eigen::Vector3d eulerZYX = rotationVars->position(t);
        Eigen::Vector3d velEuler = rotationVars->velocity(t);
        Eigen::Vector3d accEuler = rotationVars->acceleration(t);

        Jacobian E = eulerZYXToOmega(eulerZYX);
        Jacobian Edot = eulerZYXToOmegaDot(eulerZYX, velEuler);

        // TO-DO: Optimize ALL this
        jac_block.middleRows(i * 6 + 3, 3) =
            E * dAcc +
            derivEulerZYXToOmegaMultiplyVec(eulerZYX, accEuler) * dPos +
            Edot * dVel +
            derivEulerZYXToOmegaDotMultiplyVec(eulerZYX, velEuler, velEuler) *
                dPos +
            derivDotEulerZYXToOmegaDotMultiplyVec(eulerZYX, velEuler,
                                                  velEuler) *
                dVel;

        Eigen::Vector3d tau = Eigen::Vector3d::Zero();
        for (unsigned int k = 0; k < _model.numFeet; k++) {
          auto forceVars = std::static_pointer_cast<PhasedTrajectoryVars>(
              GetVariables()->GetComponent(PAW_FORCES + "_" +
                                           std::to_string(k)));
          Eigen::Vector3d force = forceVars->position(t);

          auto pawPosVars = std::static_pointer_cast<PhasedTrajectoryVars>(
              GetVariables()->GetComponent(PAW_POS + "_" + std::to_string(k)));
          Eigen::Vector3d bodyPos = positionVars->position(t);
          Eigen::Vector3d pawPos = pawPosVars->position(t);
          tau += (pawPos - bodyPos).cross(force);

          // tau += _model.feetPoses[k].cross(force);
        }

        Jacobian multiplier =
            (inertia_inv * derivRotationTransposeVector(eulerZYX, tau));
        Jacobian res = -multiplier * dPos;
        jac_block.middleRows(i * 6 + 3, 3) += res;

        Eigen::Vector3d omega = E * velEuler;

        Jacobian dEvdx = derivEulerZYXToOmegaMultiplyVec(eulerZYX, velEuler);
        Jacobian dSkewIw = derivSkewMultiplyVector(omega, inertia * omega);
        Jacobian dWdx = toSkewSymmetric(omega) * inertia;
        Jacobian multiplier2 = inertia_inv * ((dSkewIw + dWdx) * dEvdx);

        Jacobian dSkewdv = dSkewIw * E;
        Jacobian dWdv = toSkewSymmetric(omega) * inertia * E;
        Jacobian multiplier3 = inertia_inv * (dSkewdv + dWdv);
        Jacobian res2 = multiplier2 * dPos + multiplier3 * dVel;
        jac_block.middleRows(i * 6 + 3, 3) += res2;

        t += _sampleTime;
      }
    } else {
      for (unsigned int k = 0; k < _model.numFeet; k++) {
        std::string fVar = PAW_FORCES + "_" + std::to_string(k);
        std::string pVar = PAW_POS + "_" + std::to_string(k);
        if (var_set == fVar) {
          auto positionVars = std::static_pointer_cast<TrajectoryVars>(
              GetVariables()->GetComponent(BODY_POS_TRAJECTORY));
          auto rotationVars = std::static_pointer_cast<TrajectoryVars>(
              GetVariables()->GetComponent(BODY_ROT_TRAJECTORY));
          auto forceVars = std::static_pointer_cast<PhasedTrajectoryVars>(
              GetVariables()->GetComponent(fVar));
          auto pawPosVars = std::static_pointer_cast<PhasedTrajectoryVars>(
              GetVariables()->GetComponent(pVar));

          double t = 0.;
          for (unsigned int i = 0; i < _numSamplePoints; i++) {
            Jacobian dPos = forceVars->jacobianPosition(t);
            jac_block.middleRows(i * 6, 3) = -dPos / _model.mass;

            Eigen::Vector3d eulerZYX = rotationVars->position(t);
            Jacobian R = eulerZYXToMatrix(eulerZYX);

            Eigen::Vector3d pawPos = pawPosVars->position(t);
            Eigen::Vector3d bodyPos = positionVars->position(t);

            Jacobian multiplier = (inertia_inv * R.transpose() *
                                   toSkewSymmetric(pawPos - bodyPos));
            Jacobian res = -multiplier * dPos;
            jac_block.middleRows(i * 6 + 3, 3) = res;

            t += _sampleTime;
          }
        } else if (var_set == pVar) {
          auto positionVars = std::static_pointer_cast<TrajectoryVars>(
              GetVariables()->GetComponent(BODY_POS_TRAJECTORY));
          auto rotationVars = std::static_pointer_cast<TrajectoryVars>(
              GetVariables()->GetComponent(BODY_ROT_TRAJECTORY));
          auto pawPosVars = std::static_pointer_cast<PhasedTrajectoryVars>(
              GetVariables()->GetComponent(pVar));
          auto forceVars = std::static_pointer_cast<PhasedTrajectoryVars>(
              GetVariables()->GetComponent(fVar));

          double t = 0.;
          for (size_t i = 0; i < _numSamplePoints; ++i) {
            Jacobian dPos = pawPosVars->jacobianPosition(t);

            Eigen::Vector3d eulerZYX = rotationVars->position(t);
            Jacobian R = eulerZYXToMatrix(eulerZYX);

            Eigen::Vector3d f = forceVars->position(t);

            Eigen::Vector3d pawPos = pawPosVars->position(t);
            Eigen::Vector3d bodyPos = positionVars->position(t);

            Jacobian multiplier =
                inertia_inv * R.transpose() *
                derivSkewMultiplyVector((pawPos - bodyPos), f);
            Jacobian res = -multiplier * dPos;
            jac_block.middleRows(i * 6 + 3, 3) = res;

            t += _sampleTime;
          }
        }
      }
    }
  }

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

  Jacobian derivRotationVector(const Eigen::Vector3d &eulerZYX,
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

    // out = R * v
    // out[0] = R.row(0) * v = cy * cz * v[0] + (cz * sx * sy - cx * sz) * v[1]
    // + (sx * sz + cx * cz * sy) * v[2] out[1] = R.row(1) * v = cy * sz * v[0]
    // + (cx * cz + sx * sy * sz) * v[1] + (cx * sy * sz - cz * sx) * v[2]
    // out[2] = R.row(2) * v = -sy * v[0] + cy * sx * v[1] + cx * cy * v[2]

    Jacobian jac(3, 3);
    // out[0] wrt Z(0)
    jac.coeffRef(0, 0) = -cy * sz * v[0] + (-sz * sx * sy - cx * cz) * v[1] +
                         (sx * cz - cx * sz * sy) * v[2];
    // out[0] wrt Y(1)
    jac.coeffRef(0, 1) =
        -sy * cz * v[0] + cz * sx * cy * v[1] + cx * cz * cy * v[2];
    // out[0] wrt X(2)
    jac.coeffRef(0, 2) =
        (cz * cx * sy + sx * sz) * v[1] + (-cx * sz + sx * cz * sy) * v[2];

    // out[1] wrt Z(0)
    jac.coeffRef(1, 0) = cy * cz * v[0] + (-cx * sz + sx * sy * cz) * v[1] +
                         (cx * sy * cz + sz * sx) * v[2];
    // out[1] wrt Y(1)
    jac.coeffRef(1, 1) =
        -sy * sz * v[0] + sx * cy * sz * v[1] + cx * cy * sz * v[2];
    // out[1] wrt X(2)
    jac.coeffRef(1, 2) =
        (-sx * cz + cx * sy * sz) * v[1] + (-sx * sy * sz - cz * cx) * v[2];

    // out[2] wrt Z(0)
    // jac.coeffRef(2, 0) = 0.;
    // out[2] wrt Y(1)
    jac.coeffRef(2, 1) = -cy * v[0] - sy * sx * v[1] - cx * sy * v[2];
    // out[2] wrt X(2)
    jac.coeffRef(2, 2) = cy * cx * v[1] - sx * cy * v[2];

    return jac;
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

  Jacobian eulerZYXToOmega(const Eigen::Vector3d &eulerZYX) const {
    const double x = eulerZYX[2];
    const double y = eulerZYX[1];

    const double cx = std::cos(x);
    const double sx = std::sin(x);

    const double cy = std::cos(y);
    const double sy = std::sin(y);

    // Eigen::Matrix3d E;
    // E << -sy, 0., 1.,
    //     cy * sx, cx, 0.,
    //     cx * cy, -sx, 0.;
    Jacobian E(3, 3);
    E.coeffRef(0, 0) = -sy;
    E.coeffRef(0, 2) = 1.;
    E.coeffRef(1, 0) = cy * sx;
    E.coeffRef(1, 1) = cx;
    E.coeffRef(2, 0) = cx * cy;
    E.coeffRef(2, 1) = -sx;

    return E;
  }

  Jacobian derivEulerZYXToOmegaMultiplyVec(const Eigen::Vector3d &eulerZYX,
                                           const Eigen::Vector3d &v) const {
    const double x = eulerZYX[2];
    const double y = eulerZYX[1];

    const double cx = std::cos(x);
    const double sx = std::sin(x);

    const double cy = std::cos(y);
    const double sy = std::sin(y);

    // out = E * v
    // out[0] = -sy * v[0] + v[2]
    // out[1] = cy * sx * v[0] + cx * v[1]
    // out[2] = cx * cy * v[0] - sx * v[1]

    Jacobian jac(3, 3);
    // out[0] wrt Z(0)
    // jac.coeffRef(0, 0) = 0.;
    // out[0] wrt Y(1)
    jac.coeffRef(0, 1) = -cy * v[0];
    // out[0] wrt X(2)
    // jac.coeffRef(0, 2) = 0.;

    // out[1] wrt Z(0)
    // jac.coeffRef(1, 0) = 0.;
    // out[1] wrt Y(1)
    jac.coeffRef(1, 1) = -sy * sx * v[0];
    // out[1] wrt X(2)
    jac.coeffRef(1, 2) = cy * cx * v[0] - sx * v[1];

    // out[2] wrt Z(0)
    // jac.coeffRef(2, 0) = 0.;
    // out[2] wrt Y(1)
    jac.coeffRef(2, 1) = -cx * sy * v[0];
    // out[2] wrt X(2)
    jac.coeffRef(2, 2) = -sx * cy * v[0] - cx * v[1];

    return jac;
  }

  Jacobian eulerZYXToOmegaDot(const Eigen::Vector3d &eulerZYX,
                              const Eigen::Vector3d &eulerZYXDot) const {
    const double x = eulerZYX[2];
    const double y = eulerZYX[1];
    const double dx = eulerZYXDot[2];
    const double dy = eulerZYXDot[1];

    const double cx = std::cos(x);
    const double sx = std::sin(x);

    const double cy = std::cos(y);
    const double sy = std::sin(y);

    // Eigen::Matrix3d Edot;
    // Edot << -cy * dy, 0., 0.,
    //     -sy * sx * dy + cy * cx * dx, -sx * dx, 0.,
    //     -sx * cy * dx - cx * sy * dy, -cx * dx, 0.;
    Jacobian Edot(3, 3);
    Edot.coeffRef(0, 0) = -cy * dy;
    Edot.coeffRef(1, 0) = -sy * sx * dy + cy * cx * dx;
    Edot.coeffRef(1, 1) = -sx * dx;
    Edot.coeffRef(2, 0) = -sx * cy * dx - cx * sy * dy;
    Edot.coeffRef(2, 1) = -cx * dx;

    return Edot;
  }

  Jacobian
  derivEulerZYXToOmegaDotMultiplyVec(const Eigen::Vector3d &eulerZYX,
                                     const Eigen::Vector3d &eulerZYXDot,
                                     const Eigen::Vector3d &v) const {
    const double x = eulerZYX[2];
    const double y = eulerZYX[1];
    const double dx = eulerZYXDot[2];
    const double dy = eulerZYXDot[1];

    const double cx = std::cos(x);
    const double sx = std::sin(x);

    const double cy = std::cos(y);
    const double sy = std::sin(y);

    // out = Edot * v
    // out[0] = -cy * dy * v[0]
    // out[1] = (-sy * sx * dy + cy * cx * dx) * v[0] + -sx * dx * v[1]
    // out[2] = (-sx * cy * dx - cx * sy * dy) * v[0] -cx * dx * v[1]

    Jacobian jac(3, 3);
    // out[0] wrt Z(0)
    // jac.coeffRef(0, 0) = 0.;
    // out[0] wrt Y(1)
    jac.coeffRef(0, 1) = sy * dy * v[0];
    // out[0] wrt X(2)
    // jac.coeffRef(0, 2) = 0.;

    // out[1] wrt Z(0)
    // jac.coeffRef(1, 0) = 0.;
    // out[1] wrt Y(1)
    jac.coeffRef(1, 1) = (-cy * sx * dy - sy * cx * dx) * v[0];
    // out[1] wrt X(2)
    jac.coeffRef(1, 2) = (-sy * cx * dy - cy * sx * dx) * v[0] - cx * dx * v[1];

    // out[2] wrt Z(0)
    // jac.coeffRef(2, 0) = 0.;
    // out[2] wrt Y(1)
    jac.coeffRef(2, 1) = (sx * sy * dx - cx * cy * dy) * v[0];
    // out[2] wrt X(2)
    jac.coeffRef(2, 2) = (-cx * cy * dx + sx * sy * dy) * v[0] + sx * dx * v[1];

    return jac;
  }

  Jacobian
  derivDotEulerZYXToOmegaDotMultiplyVec(const Eigen::Vector3d &eulerZYX,
                                        const Eigen::Vector3d &,
                                        const Eigen::Vector3d &v) const {
    const double x = eulerZYX[2];
    const double y = eulerZYX[1];

    const double cx = std::cos(x);
    const double sx = std::sin(x);

    const double cy = std::cos(y);
    const double sy = std::sin(y);

    // out = Edot * v
    // out[0] = -cy * dy * v[0]
    // out[1] = (-sy * sx * dy + cy * cx * dx) * v[0] + -sx * dx * v[1]
    // out[2] = (-sx * cy * dx - cx * sy * dy) * v[0] -cx * dx * v[1]

    Jacobian jac(3, 3);
    // out[0] wrt dZ(0)
    // jac.coeffRef(0, 0) = 0.;
    // out[0] wrt dY(1)
    jac.coeffRef(0, 1) = -cy * v[0];
    // out[0] wrt dX(2)
    // jac.coeffRef(0, 2) = 0.;

    // out[1] wrt dZ(0)
    // jac.coeffRef(1, 0) = 0.;
    // out[1] wrt dY(1)
    jac.coeffRef(1, 1) = -sy * sx * v[0];
    // out[1] wrt dX(2)
    jac.coeffRef(1, 2) = cy * cx * v[0] - sx * v[1];

    // out[2] wrt dZ(0)
    // jac.coeffRef(2, 0) = 0.;
    // out[2] wrt dY(1)
    jac.coeffRef(2, 1) = -cx * sy * v[0];
    // out[2] wrt dX(2)
    jac.coeffRef(2, 2) = -sx * cy * v[0] - cx * v[1];

    return jac;
  }

  Jacobian toSkewSymmetric(const Eigen::Vector3d &vec) const {
    // Eigen::Matrix3d skew;
    // skew << 0., -vec[2], vec[1],
    //     vec[2], 0., -vec[0],
    //     -vec[1], vec[0], 0.;
    Jacobian skew(3, 3);
    skew.coeffRef(0, 1) = -vec[2];
    skew.coeffRef(0, 2) = vec[1];
    skew.coeffRef(1, 0) = vec[2];
    skew.coeffRef(1, 2) = -vec[0];
    skew.coeffRef(2, 0) = -vec[1];
    skew.coeffRef(2, 1) = vec[0];
    return skew;
  }

  Jacobian derivSkewMultiplyVector(const Eigen::Vector3d &,
                                   const Eigen::Vector3d &v) const {
    // out = skew * v
    // out[0] = -vec[2]*v[1] + vec[1]*v[2]
    // out[1] = vec[2]*v[0] - vec[0]*v[2]
    // out[2] = -vec[1]*v[0] + vec[0]*v[1]

    Jacobian jac(3, 3);
    // out[0] wrt vec[0]
    // jac.coeffRef(0, 0) = 0.;
    // out[0] wrt vec[1]
    jac.coeffRef(0, 1) = v[2];
    // out[0] wrt vec[2]
    jac.coeffRef(0, 2) = -v[1];

    // out[1] wrt vec[0]
    jac.coeffRef(1, 0) = -v[2];
    // out[1] wrt vec[1]
    // jac.coeffRef(1, 1) = 0.;
    // out[1] wrt vec[2]
    jac.coeffRef(1, 2) = v[0];

    // out[2] wrt vec[0]
    jac.coeffRef(2, 0) = v[1];
    // out[2] wrt vec[1]
    jac.coeffRef(2, 1) = -v[0];
    // out[2] wrt vec[2]
    // jac.coeffRef(2, 2) = 0.;

    return jac;
  }

protected:
  SingleRigidBodyDynamicsModel _model;
  unsigned int _numSamplePoints;
  double _sampleTime;
};

} // namespace trajopt