#pragma once

#include <ifopt/constraint_set.h>

#include "include/ifopt_sets/variables/trajectory_vars.hpp"
#include "include/terrain/terrain_grid.hpp"

namespace trajopt {
    class FootTerrainDistanceImplicit : public ifopt::ConstraintSet {
    public:
        FootTerrainDistanceImplicit(const std::shared_ptr<TrajectoryVars>& pos_vars, const trajopt::TerrainGrid& terrain, size_t num_knots)
            : ConstraintSet(num_knots, pos_vars->GetName() + "_terrain_pos"),
              _posVarsName(pos_vars->GetName()),
              _terrain(terrain)
        {
        }

        VectorXd GetValues() const override
        {
            VectorXd g = VectorXd::Zero(GetRows());

            auto pos_knots = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName))->GetValues();

            for (size_t i = 0; i < static_cast<size_t>(GetRows()); ++i) {
                Eigen::VectorXd pos = pos_knots.segment(i * 6, 3);

                double phi = pos[2] - _terrain.height(pos[0], pos[1]);
                g[i] = phi;
            }

            return g;
        }

        VecBound GetBounds() const override
        {
            VecBound b(GetRows(), ifopt::BoundGreaterZero);
            return b;
        }

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
        {
            auto pos_knots = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName))->GetValues();

            if (var_set == _posVarsName) {
                for (size_t i = 0; i < static_cast<size_t>(GetRows()); ++i) {
                    Eigen::VectorXd pos = pos_knots.segment(i * 6, 3);

                    auto terrain_d = _terrain.jacobian(pos[0], pos[1]);

                    jac_block.coeffRef(i, i * 6) = -terrain_d[0];
                    jac_block.coeffRef(i, i * 6 + 1) = -terrain_d[1];
                    jac_block.coeffRef(i, i * 6 + 2) = 1.;
                }
            }
        }

    protected:
        const std::string _posVarsName;
        const trajopt::TerrainGrid _terrain;
    };
} // namespace trajopt
