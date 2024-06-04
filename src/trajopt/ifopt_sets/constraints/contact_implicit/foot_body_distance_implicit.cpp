#include "foot_body_distance_implicit.hpp"

using namespace trajopt;

FootBodyDistanceImplicit::FootBodyDistanceImplicit(const SingleRigidBodyDynamicsModel& model, const std::shared_ptr<TrajectoryVars>& bodyPosVars, const std::shared_ptr<TrajectoryVars>& bodyRotVars, const std::shared_ptr<TrajectoryVars>& footPosVars, size_t numSamples, double sampleTime)
    : ConstraintSet(3 * numSamples, footPosVars->GetName() + "_foot_body_distance"), _model(model), _bodyPosVarsName(bodyPosVars->GetName()), _bodyRotVarsName(bodyRotVars->GetName()), _footPosVarsName(footPosVars->GetName()), _numSamples(numSamples), _sampleTime(sampleTime) {}

FootBodyDistanceImplicit::VectorXd FootBodyDistanceImplicit::GetValues() const
{
    VectorXd g = VectorXd::Zero(GetRows());

    auto bodyPosVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_bodyPosVarsName));
    auto bodyRotVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_bodyRotVarsName));
    auto footPosVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_footPosVarsName));

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

FootBodyDistanceImplicit::VecBound FootBodyDistanceImplicit::GetBounds() const
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

void FootBodyDistanceImplicit::FillJacobianBlock(std::string var_set, FootBodyDistanceImplicit::Jacobian& jac_block) const
{
    auto bodyPosVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_bodyPosVarsName));
    auto bodyRotVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_bodyRotVarsName));
    auto footPosVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_footPosVarsName));

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
