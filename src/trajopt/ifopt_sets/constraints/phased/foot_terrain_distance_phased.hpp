#pragma once

#include <ifopt/constraint_set.h>

#include <trajopt/ifopt_sets/variables/phased_trajectory_vars.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

namespace trajopt {
    class FootTerrainDistancePhased : public ifopt::ConstraintSet {
    public:
        FootTerrainDistancePhased(const std::shared_ptr<PhasedTrajectoryVars>& vars, const trajopt::TerrainGrid& terrain, size_t numSteps, size_t numSwings, std::vector<size_t> numKnotsPerSwing);

        VectorXd GetValues() const override;

        VecBound GetBounds() const override;

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

    private:
        const std::string _varsName;
        const trajopt::TerrainGrid _terrain;
        const size_t _numPhases;
        const std::vector<size_t> _numKnotsPerSwing;
    };
} // namespace trajopt
