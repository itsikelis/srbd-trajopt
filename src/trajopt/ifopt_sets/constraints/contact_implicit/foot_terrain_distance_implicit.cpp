#include "foot_terrain_distance_implicit.hpp"

using namespace trajopt;

FootTerrainDistanceImplicit::FootTerrainDistanceImplicit(const std::shared_ptr<TrajectoryVars>& posVars,
    const trajopt::TerrainGrid& terrain,
    size_t numSamples,
    double sampleTime)
    : ConstraintSet(numSamples, posVars->GetName() + "_foot_terrain_distance"),
      _posVarsName(posVars->GetName()),
      _terrain(terrain),
      _numSamples(numSamples),
      _sampleTime(sampleTime) {}

FootTerrainDistanceImplicit::VectorXd FootTerrainDistanceImplicit::GetValues() const
{
    VectorXd g = VectorXd::Zero(GetRows());

    auto vars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName));

    double t = 0.;
    for (size_t i = 0; i < _numSamples; ++i) {
        Eigen::VectorXd pos = vars->trajectoryEval(t, 0);

        double phi = pos[2] - _terrain.GetHeight(pos[0], pos[1]);
        g[i] = phi;

        t += _sampleTime;
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
    auto vars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName));

    if (var_set == _posVarsName) {
        double t = 0.;
        for (size_t i = 0; i < _numSamples; ++i) {
            Eigen::VectorXd pos = vars->trajectoryEval(t, 0);
            Jacobian dPos = vars->trajectoryJacobian(t, 0);

            auto terrain_dx = _terrain.GetDerivativeOfHeightWrt(TerrainGrid::X_, pos[0], pos[1]);
            auto terrain_dy = _terrain.GetDerivativeOfHeightWrt(TerrainGrid::Y_, pos[0], pos[1]);

            jac_block.middleRows(i, 1) = -terrain_dx * dPos.middleRows(0, 1);
            jac_block.middleRows(i, 1) = -terrain_dy * dPos.middleRows(1, 1);
            jac_block.middleRows(i, 1) = 1. * dPos.middleRows(2, 1);

            t += _sampleTime;
        }
    }
}
