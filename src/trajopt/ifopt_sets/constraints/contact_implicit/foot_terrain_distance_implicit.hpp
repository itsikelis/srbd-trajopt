#pragma once

#include <ifopt/constraint_set.h>

#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

namespace trajopt {
    class FootTerrainDistanceImplicit : public ifopt::ConstraintSet {
    public:
        FootTerrainDistanceImplicit(const std::shared_ptr<TrajectoryVars>& posVars, const trajopt::TerrainGrid& terrain, size_t numSamples, double sampleTime);

        VectorXd GetValues() const override;

        VecBound GetBounds() const override;

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

    protected:
        const std::string _posVarsName;
        const trajopt::TerrainGrid _terrain;

        size_t _numSamples;
        double _sampleTime;
    };
} // namespace trajopt
