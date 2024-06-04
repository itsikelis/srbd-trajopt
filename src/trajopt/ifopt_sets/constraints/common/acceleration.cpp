#include "acceleration.hpp"

using namespace trajopt;

AccelerationConstraints::AccelerationConstraints(const std::shared_ptr<TrajectoryVars>& vars) : ConstraintSet((vars->numKnotPoints() - 2) * 3, vars->GetName() + "_equal_acc"), _varSetName(vars->GetName()) {}

AccelerationConstraints::VectorXd AccelerationConstraints::GetValues() const
{
    VectorXd g = VectorXd::Zero(GetRows());

    auto vars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_varSetName));

    auto iters = vars->numSplines();
    for (unsigned int i = 0; i < iters - 1; i++) {
        g.segment(i * 3, 3) = vars->splineEval(i, vars->splineDuration(i), 2) - vars->splineEval(i + 1, 0., 2);
    }

    return g;
}

AccelerationConstraints::VecBound AccelerationConstraints::GetBounds() const
{
    // All constraints equal to zero.
    VecBound b(GetRows());
    for (int i = 0; i < GetRows(); i++) {
        b.at(i) = ifopt::BoundZero;
    }
    return b;
}

void AccelerationConstraints::FillJacobianBlock(std::string var_set, Jacobian& jac_block) const
{
    if (var_set == _varSetName) {
        auto vars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_varSetName));

        auto iters = vars->numSplines() - 1;
        for (unsigned int i = 0; i < iters; i++) {
            Jacobian jacAccStart = vars->splineJacobian(i, vars->splineDuration(i), 2);
            Jacobian jacAccEnd = vars->splineJacobian(i + 1, 0., 2);

            jac_block.middleRows(i * 3, 3) = jacAccStart - jacAccEnd;
        }
    }
}
