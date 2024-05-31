#pragma once

#include <ifopt/constraint_set.h>

#include <ifopt_sets/variables/phased_trajectory_vars.hpp>

namespace trajopt {
    class PhasedAccelerationConstraints : public ifopt::ConstraintSet {
    public:
        PhasedAccelerationConstraints(const std::shared_ptr<PhasedTrajectoryVars>& vars) : ConstraintSet(kSpecifyLater, vars->GetName() + "_equal_acc"), _variableSetName(vars->GetName())
        {
            SetRows((vars->numKnotPoints() - 2) * 3);
        }

        VectorXd GetValues() const override
        {
            VectorXd g = VectorXd::Zero(GetRows());

            auto vars = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_variableSetName));

            auto iters = vars->numSplines() - 1;
            for (size_t i = 0; i < iters; i++) {
                g.segment(i * 3, 3) = vars->splineEval(i, vars->splineDuration(i), 2) - vars->splineEval(i + 1, 0., 2);
            }

            return g;
        }

        VecBound GetBounds() const override
        {
            // All constraints equal to zero.
            VecBound b(GetRows());
            for (int i = 0; i < GetRows(); i++) {
                b.at(i) = ifopt::BoundZero;
            }
            return b;
        }

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
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

    protected:
        std::string _variableSetName;
    };
} // namespace trajopt
