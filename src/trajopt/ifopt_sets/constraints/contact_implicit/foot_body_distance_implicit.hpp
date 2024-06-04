#pragma once

#include <ifopt/constraint_set.h>

#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/srbd/srbd.hpp>

#include <trajopt/utils/types.hpp>
#include <trajopt/utils/utils.hpp>

namespace trajopt {

    class FootBodyDistanceImplicit : public ifopt::ConstraintSet {
    public:
        FootBodyDistanceImplicit(const SingleRigidBodyDynamicsModel& model, const std::shared_ptr<TrajectoryVars>& bodyPosVars, const std::shared_ptr<TrajectoryVars>& bodyRotVars, const std::shared_ptr<TrajectoryVars>& footPosVars, size_t numSamples, double sampleTime);

        VectorXd GetValues() const override;

        VecBound GetBounds() const override;

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

    protected:
        const SingleRigidBodyDynamicsModel _model;
        const std::string _bodyPosVarsName;
        const std::string _bodyRotVarsName;
        const std::string _footPosVarsName;
        const size_t _numSamples;
        const double _sampleTime;
    };
} // namespace trajopt
