#pragma once

#include <ifopt/constraint_set.h>

#include "include/ifopt_sets/variables/phased_trajectory_vars.hpp"
#include "include/ifopt_sets/variables/trajectory_vars.hpp"
#include "include/srbd/srbd.hpp"
#include "include/utils/types.hpp"

namespace trajopt {
    class FootBodyDistance : public ifopt::ConstraintSet {
    public:
        FootBodyDistance(
            const SingleRigidBodyDynamicsModel& model,
            const std::shared_ptr<TrajectoryVars>& bodyPosVars,
            const std::shared_ptr<TrajectoryVars>& bodyRotVars,
            const std::shared_ptr<PhasedTrajectoryVars>& footPosVars,
            size_t numSamples, double sampleTime)
            : ConstraintSet(3 * numSamples, footPosVars->GetName() + "_foot_body_pos"),
              _model(model),
              _bodyPosVarsName(bodyPosVars->GetName()),
              _bodyRotVarsName(bodyRotVars->GetName()),
              _footPosVarsName(footPosVars->GetName()),
              _numSamples(numSamples),
              _sampleTime(sampleTime)
        {
        }

        VectorXd GetValues() const override
        {
            VectorXd g = VectorXd::Zero(GetRows());

            auto bodyPosVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_bodyPosVarsName));
            auto bodyRotVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_bodyRotVarsName));
            auto footPosVars = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_footPosVarsName));

            double t = 0.;
            for (size_t i = 0; i < _numSamples; ++i) {
                Eigen::Vector3d b = bodyPosVars->trajectoryEval(t, 0);
                Jacobian R = eulerZYXToMatrix(bodyRotVars->trajectoryEval(t, 0));
                Eigen::Vector3d f = footPosVars->trajectoryEval(t, 0);

                g.segment(i * 3, 3) = R.transpose() * (f - b);

                t += _sampleTime;
            }

            return g;
        }

        VecBound GetBounds() const override
        {

            VecBound b(GetRows(), ifopt::BoundZero);

            size_t idx = 0;
            for (size_t k = 0; k < _model.numFeet; ++k) {
                if (_footPosVarsName == FOOT_POS + "_" + std::to_string(k)) {
                    break;
                }
                ++idx;
            }

            for (size_t i = 0; i < _numSamples; ++i) {
                b.at(i * 3 + 0) = ifopt::Bounds(_model.feetMinBounds[idx][0], _model.feetMaxBounds[idx][0]);
                b.at(i * 3 + 1) = ifopt::Bounds(_model.feetMinBounds[idx][1], _model.feetMaxBounds[idx][1]);
                b.at(i * 3 + 2) = ifopt::Bounds(_model.feetMinBounds[idx][2], _model.feetMaxBounds[idx][2]);
            }

            return b;
        }

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
        {
            auto bodyPosVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_bodyPosVarsName));
            auto bodyRotVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_bodyRotVarsName));
            auto footPosVars = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_footPosVarsName));

            if (var_set == _bodyPosVarsName) {
                double t = 0.;
                for (size_t i = 0; i < _numSamples; ++i) {
                    Jacobian R = eulerZYXToMatrix(bodyRotVars->trajectoryEval(t, 0));
                    Jacobian dBodyPos = bodyPosVars->trajectoryJacobian(t, 0);

                    jac_block.middleRows(i * 3, 3) = -R.transpose() * dBodyPos;

                    t += _sampleTime;
                }
            }
            else if (var_set == _bodyRotVarsName) {
                double t = 0.;
                for (size_t i = 0; i < _numSamples; ++i) {
                    Jacobian dBodyRot = bodyRotVars->trajectoryJacobian(t, 0);

                    Eigen::Vector3d b = bodyPosVars->trajectoryEval(t, 0);
                    Eigen::Vector3d euler_zyx = bodyRotVars->trajectoryEval(t, 0);
                    Eigen::Vector3d f = footPosVars->trajectoryEval(t, 0);

                    Jacobian mult = derivRotationTransposeVector(euler_zyx, f - b);
                    Jacobian res = mult * dBodyRot;
                    jac_block.middleRows(i * 3, 3) = res;

                    t += _sampleTime;
                }
            }
            else if (var_set == _footPosVarsName) {
                double t = 0.;
                for (size_t i = 0; i < _numSamples; ++i) {
                    Jacobian R = eulerZYXToMatrix(bodyRotVars->trajectoryEval(t, 0));
                    Jacobian dFootPos = footPosVars->trajectoryJacobian(t, 0);

                    Jacobian res = R.transpose() * dFootPos;
                    jac_block.middleRows(i * 3, 3) = res;

                    t += _sampleTime;
                }
            }
        }

    protected:
        Jacobian eulerZYXToMatrix(const Eigen::Vector3d& eulerZYX) const
        {
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

        Jacobian derivRotationTransposeVector(const Eigen::Vector3d& eulerZYX,
            const Eigen::Vector3d& v) const
        {
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
            jac.coeffRef(1, 0) = (-sz * sx * sy - cx * cz) * v[0] + (-cx * sz + sx * sy * cz) * v[1];
            // out[1] wrt Y(1)
            jac.coeffRef(1, 1) = cz * sx * cy * v[0] + sx * cy * sz * v[1] - sy * sx * v[2];
            // out[1] wrt X(2)
            jac.coeffRef(1, 2) = (cz * cx * sy + sx * sz) * v[0] + (-sx * cz + cx * sy * sz) * v[1] + cy * cx * v[2];

            // out[2] wrt Z(0)
            jac.coeffRef(2, 0) = (sx * cz - cx * sz * sy) * v[0] + (cx * sy * cz + sz * sx) * v[1];
            // out[2] wrt Y(1)
            jac.coeffRef(2, 1) = cx * cz * cy * v[0] + cx * cy * sz * v[1] - cx * sy * v[2];
            // out[2] wrt X(2)
            jac.coeffRef(2, 2) = (cx * sz - sx * cz * sy) * v[0] + (-sx * sy * sz - cz * cx) * v[1] - sx * cy * v[2];

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
