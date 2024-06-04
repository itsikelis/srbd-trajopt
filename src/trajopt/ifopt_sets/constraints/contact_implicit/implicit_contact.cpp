#include "implicit_contact.hpp"

using namespace trajopt;

ImplicitContactConstraints::ImplicitContactConstraints(const std::shared_ptr<TrajectoryVars>& pos_vars, const std::shared_ptr<TrajectoryVars>& force_vars, const trajopt::TerrainGrid& terrain, size_t num_samples, double sample_time) : ConstraintSet(num_samples, pos_vars->GetName() + "_implicit_contact"), _posVarsName(pos_vars->GetName()), _forceVarsName(force_vars->GetName()), _terrain(terrain), _numSamples(num_samples), _sampleTime(sample_time) {}

ImplicitContactConstraints::VectorXd ImplicitContactConstraints::GetValues() const
{
    VectorXd g = VectorXd::Zero(GetRows());

    auto pos_traj = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName));
    auto force_traj = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_forceVarsName));

    double dt = 0.;
    for (size_t i = 0; i < static_cast<size_t>(GetRows()); ++i) {
        Eigen::VectorXd pos = pos_traj->trajectoryEval(dt, 0);
        Eigen::VectorXd force = force_traj->trajectoryEval(dt, 0);

        double f_n = force.dot(_terrain.n(pos[0], pos[1]));
        double phi = pos[2] - _terrain.height(pos[0], pos[1]);
        g[i] = f_n * phi;

        dt += _sampleTime;
    }

    return g;
}

ImplicitContactConstraints::VecBound ImplicitContactConstraints::GetBounds() const
{
    VecBound b(GetRows(), ifopt::BoundZero);
    return b;
}

void ImplicitContactConstraints::FillJacobianBlock(std::string var_set, ImplicitContactConstraints::Jacobian& jac_block) const
{
    auto pos_traj = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName));
    auto force_traj = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_forceVarsName));

    if (var_set == _forceVarsName) {
        double dt = 0.;
        for (size_t i = 0; i < static_cast<size_t>(GetRows()); ++i) {
            Jacobian dForce = force_traj->trajectoryJacobian(dt, 0);

            Eigen::VectorXd pos = pos_traj->trajectoryEval(dt, 0);
            auto phi = pos[2] - _terrain.height(pos[0], pos[1]);

            Eigen::Vector3d n = _terrain.n(pos[0], pos[1]);

            // TODO: Find a better way to do this.
            jac_block.middleRows(i, 1) += n[0] * phi * dForce.middleRows(0, 1);
            jac_block.middleRows(i, 1) += n[1] * phi * dForce.middleRows(1, 1);
            jac_block.middleRows(i, 1) += n[2] * phi * dForce.middleRows(2, 1);
            // jac_block.coeffRef(i, i * 6 + 1) = n[1] * phi;
            // jac_block.coeffRef(i, i * 6 + 2) = n[2] * phi;

            dt += _sampleTime;
        }
    }
    else if (var_set == _posVarsName) {
        double dt = 0.;
        for (size_t i = 0; i < static_cast<size_t>(GetRows()); ++i) {
            Jacobian dPos = force_traj->trajectoryJacobian(dt, 0);

            Eigen::VectorXd pos = pos_traj->trajectoryEval(dt, 0);
            Eigen::VectorXd force = force_traj->trajectoryEval(dt, 0);

            auto terrain_d = _terrain.jacobian(pos[0], pos[1]);

            double f_n = force.dot(_terrain.n(pos[0], pos[1]));

            jac_block.middleRows(i, 1) = f_n * (-terrain_d[0]) * dPos.middleRows(0, 1);
            jac_block.middleRows(i, 1) = f_n * (-terrain_d[1]) * dPos.middleRows(1, 1);
            jac_block.middleRows(i, 1) = f_n * 1. * dPos.middleRows(2, 1);

            dt += _sampleTime;
        }
    }
}
