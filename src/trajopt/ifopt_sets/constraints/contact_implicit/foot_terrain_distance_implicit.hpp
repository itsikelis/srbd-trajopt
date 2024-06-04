#pragma once

#include <ifopt/constraint_set.h>

#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

namespace trajopt {
    class FootTerrainDistanceImplicit : public ifopt::ConstraintSet {
    public:
        FootTerrainDistanceImplicit(const std::shared_ptr<TrajectoryVars>& pos_vars, const trajopt::TerrainGrid& terrain, size_t num_knots);

        VectorXd GetValues() const override;

        VecBound GetBounds() const override;

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

    protected:
        const std::string _posVarsName;
        const trajopt::TerrainGrid _terrain;
    };
} // namespace trajopt
