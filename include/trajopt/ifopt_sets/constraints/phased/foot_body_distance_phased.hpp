#pragma once

#include <ifopt/constraint_set.h>

#include <trajopt/ifopt_sets/variables/phased_trajectory_vars.hpp>
#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/srbd/srbd.hpp>
#include <trajopt/utils/types.hpp>

namespace trajopt {
    class FootBodyDistancePhased : public ifopt::ConstraintSet {
    public:
        FootBodyDistancePhased(
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
    protected:
        const SingleRigidBodyDynamicsModel _model;
        const std::string _bodyPosVarsName;
        const std::string _bodyRotVarsName;
        const std::string _footPosVarsName;
        const size_t _numSamples;
        const double _sampleTime;
    };
} // namespace trajopt
