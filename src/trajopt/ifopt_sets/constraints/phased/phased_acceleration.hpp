#pragma once

#include <ifopt/constraint_set.h>

#include <trajopt/ifopt_sets/variables/phased_trajectory_vars.hpp>

namespace trajopt {
    class PhasedAccelerationConstraints : public ifopt::ConstraintSet {
    public:
        PhasedAccelerationConstraints(const std::shared_ptr<PhasedTrajectoryVars>& vars);

        VectorXd GetValues() const override;

        VecBound GetBounds() const override;

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

    protected:
        std::string _variableSetName;
    };
} // namespace trajopt
