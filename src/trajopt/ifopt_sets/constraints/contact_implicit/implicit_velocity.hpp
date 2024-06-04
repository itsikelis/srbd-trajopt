#pragma once

#include <ifopt/constraint_set.h>

#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

namespace trajopt {
    class ImplicitVelocityConstraints : public ifopt::ConstraintSet {
    public:
        ImplicitVelocityConstraints(const std::shared_ptr<TrajectoryVars>& pos_vars, const std::shared_ptr<TrajectoryVars>& force_vars, const trajopt::TerrainGrid& terrain, size_t num_samples, double sample_time);

        VectorXd GetValues() const override;

        VecBound GetBounds() const override;

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

    protected:
        const std::string _posVarsName;
        const std::string _forceVarsName;
        const trajopt::TerrainGrid _terrain;
        const size_t _numSamples;
        const double _sampleTime;
    };
} // namespace trajopt
