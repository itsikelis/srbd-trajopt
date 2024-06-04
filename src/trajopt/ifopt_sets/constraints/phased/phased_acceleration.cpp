#include "phased_acceleration.hpp"

using namespace trajopt;

PhasedAccelerationConstraints::PhasedAccelerationConstraints(const std::shared_ptr<PhasedTrajectoryVars>& vars) : ConstraintSet((vars->numKnotPoints() - 2) * 3, vars->GetName() + "_equal_acc"), _variableSetName(vars->GetName()) {}

PhasedAccelerationConstraints::VectorXd PhasedAccelerationConstraints::GetValues() const
{
    VectorXd g = VectorXd::Zero(GetRows());

    auto vars = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_variableSetName));

    auto iters = vars->numSplines() - 1;
    for (size_t i = 0; i < iters; i++) {
        g.segment(i * 3, 3) = vars->splineEval(i, vars->splineDuration(i), 2) - vars->splineEval(i + 1, 0., 2);
    }

    return g;
}

PhasedAccelerationConstraints::VecBound PhasedAccelerationConstraints::GetBounds() const
{
    // All constraints equal to zero.
    VecBound b(GetRows());
    for (int i = 0; i < GetRows(); i++) {
        b.at(i) = ifopt::BoundZero;
    }
    return b;
}

void PhasedAccelerationConstraints::FillJacobianBlock(std::string var_set, PhasedAccelerationConstraints::Jacobian& jac_block) const
{
    if (var_set == _variableSetName) {
        auto vars = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_variableSetName));

        auto iters = vars->numSplines() - 1;
        for (unsigned int i = 0; i < iters; i++) {
            Jacobian jacAccStart = vars->splineJacobian(i, vars->splineDuration(i), 2);
            Jacobian jacAccEnd = vars->splineJacobian(i + 1, 0., 2);

            jac_block.middleRows(i * 3, 3) = jacAccStart - jacAccEnd;
        }
    }
}
