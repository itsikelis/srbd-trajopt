#pragma once

#include <ifopt/cost_term.h>

#include <ifopt_sets/variables/phased_trajectory_vars.hpp>
#include <ifopt_sets/variables/trajectory_vars.hpp>

#include <srbd/srbd.hpp>
#include <utils/types.hpp>
#include <utils/utils.hpp>

namespace trajopt {
    template <typename TrajectoryVarsType>
    class MinEffort : public ifopt::CostTerm {
    public:
        MinEffort(const std::shared_ptr<TrajectoryVars>& footForceVars, size_t numKnots)
            : CostTerm(footForceVars->GetName() + "cost_min_effort"),
              _varSetName(footForceVars->GetName()),
              _numKnots(numKnots)
        {
        }

        double GetCost() const override
        {
            double c = 0.;

            Eigen::VectorXd force_knots = GetVariables()->GetComponent(_varSetName)->GetValues();

            for (size_t i = 0; i < _numKnots; ++i) {
                Eigen::VectorXd f = force_knots.segment(i * 3, 3);
                c += f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
            }

            return c;
        }

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
        {
            if (var_set == _varSetName) {
                Eigen::VectorXd force_knots = GetVariables()->GetComponent(_varSetName)->GetValues();
                for (size_t i = 0; i < _numKnots; ++i) {
                    Eigen::VectorXd f = force_knots.segment(i * 3, 3);
                    jac_block.coeffRef(0, i * 3 + 0) += 2. * f[0];
                    jac_block.coeffRef(0, i * 3 + 1) += 2. * f[1];
                    jac_block.coeffRef(0, i * 3 + 2) += 2. * f[2];
                }
            }
        }

    protected:
        const std::string _varSetName;
        const size_t _numKnots;
    };
} // namespace trajopt
