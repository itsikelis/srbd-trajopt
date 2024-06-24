#include "implicit_velocity.hpp"

using namespace trajopt;

ImplicitVelocityConstraints::ImplicitVelocityConstraints(const std::shared_ptr<TrajectoryVars>& pos_vars, const std::shared_ptr<TrajectoryVars>& force_vars, const trajopt::TerrainGrid& terrain, size_t num_samples, double sample_time) : ConstraintSet(3 * (num_samples - 1), pos_vars->GetName() + "_implicit_velocity"), _posVarsName(pos_vars->GetName()), _forceVarsName(force_vars->GetName()), _terrain(terrain), _numSamples(num_samples), _sampleTime(sample_time) {}

ImplicitVelocityConstraints::VectorXd ImplicitVelocityConstraints::GetValues() const
{
    VectorXd g = VectorXd::Zero(GetRows());

    auto posVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName));
    auto forceVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_forceVarsName));

    double dt = 0.;
    for (size_t i = 1; i < _numSamples; ++i) {
        Eigen::VectorXd pos_prev = posVars->trajectoryEval(dt, 0);
        Eigen::VectorXd force = forceVars->trajectoryEval(dt + _sampleTime, 0);
        Eigen::VectorXd pos_next = posVars->trajectoryEval(dt + _sampleTime, 0);

        double f_n = force.dot(_terrain.GetNormalizedBasis(TerrainGrid::Normal, pos_next[0], pos_next[1]));
        g.segment((i - 1) * 3, 3) = f_n * (pos_next - pos_prev);

        dt += _sampleTime;
    }

    return g;
}

ImplicitVelocityConstraints::VecBound ImplicitVelocityConstraints::GetBounds() const
{
    VecBound b(GetRows(), ifopt::BoundZero);
    return b;
}

void ImplicitVelocityConstraints::FillJacobianBlock(std::string var_set, ImplicitVelocityConstraints::Jacobian& jac_block) const
{
    auto posVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName));
    auto forceVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_forceVarsName));

    if (var_set == _forceVarsName) {

        double dt = 0.;
        for (size_t i = 1; i < _numSamples; ++i) {
            Jacobian dForce = forceVars->trajectoryJacobian(dt + _sampleTime, 0);

            Eigen::VectorXd pos_prev = posVars->trajectoryEval(dt, 0);
            Eigen::VectorXd pos_next = posVars->trajectoryEval(dt + _sampleTime, 0);
            Eigen::VectorXd vel = (pos_next - pos_prev);

            Eigen::Vector3d n = _terrain.GetNormalizedBasis(TerrainGrid::Normal, pos_next[0], pos_next[1]);

            Jacobian mult(3, 3);
            for (size_t k = 0; k < 3; ++k) {
                auto res = vel[k] * n.transpose();
                for (size_t j = 0; j < 3; ++j) {
                    mult.coeffRef(k, j) = res.coeff(j);
                }
            }

            jac_block.middleRows((i - 1) * 3, 3) = mult * dForce;

            dt += _sampleTime;
        }
    }
    else if (var_set == _posVarsName) {
        double dt = 0.;
        for (size_t i = 1; i < _numSamples; ++i) {
            Jacobian dPos_prev = posVars->trajectoryJacobian(dt, 0);
            Jacobian dPos_next = posVars->trajectoryJacobian(dt + _sampleTime, 0);

            Eigen::VectorXd pos_next = posVars->trajectoryEval(dt + _sampleTime, 0);
            Eigen::VectorXd force = forceVars->trajectoryEval(dt + _sampleTime, 0);

            double f_n = force.dot(_terrain.GetNormalizedBasis(TerrainGrid::Normal, pos_next[0], pos_next[1]));

            jac_block.middleRows((i - 1) * 3, 3) = f_n * (dPos_next - dPos_prev);
            dt += _sampleTime;
        }
    }
}
