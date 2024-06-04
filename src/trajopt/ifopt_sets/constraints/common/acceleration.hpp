#pragma once

#include <ifopt/constraint_set.h>

#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>

namespace trajopt {
    class AccelerationConstraints : public ifopt::ConstraintSet {
    public:
        AccelerationConstraints(const std::shared_ptr<TrajectoryVars>& vars);

        VectorXd GetValues() const override;

        VecBound GetBounds() const override;

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

    protected:
        std::string _varSetName;
    };
} // namespace trajopt
