#include "foot_terrain_distance_implicit.hpp"

using namespace trajopt;

FootTerrainDistanceImplicit::FootTerrainDistanceImplicit(const std::shared_ptr<TrajectoryVars>& pos_vars, const trajopt::TerrainGrid& terrain, size_t num_knots)
    : ConstraintSet(num_knots, pos_vars->GetName() + "_foot_terrain_distance"), _posVarsName(pos_vars->GetName()), _terrain(terrain) {}

FootTerrainDistanceImplicit::VectorXd FootTerrainDistanceImplicit::GetValues() const
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

FootTerrainDistanceImplicit::VecBound FootTerrainDistanceImplicit::GetBounds() const
{
    VecBound b(GetRows(), ifopt::BoundGreaterZero);
    return b;
}

void FootTerrainDistanceImplicit::FillJacobianBlock(std::string var_set, FootTerrainDistanceImplicit::Jacobian& jac_block) const
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
