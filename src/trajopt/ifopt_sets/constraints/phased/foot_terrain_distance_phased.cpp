#include "foot_terrain_distance_phased.hpp"

using namespace trajopt;

FootTerrainDistancePhased::FootTerrainDistancePhased(const std::shared_ptr<PhasedTrajectoryVars>& vars, const trajopt::TerrainGrid& terrain, size_t numSteps, size_t numSwings, std::vector<size_t> numKnotsPerSwing) : ConstraintSet(numSteps + std::accumulate(numKnotsPerSwing.begin(), numKnotsPerSwing.end(), 0), vars->GetName() + "_foot_pos_terrain"), _varsName(vars->GetName()), _terrain(terrain), _numPhases(numSteps + numSwings), _numKnotsPerSwing(numKnotsPerSwing) {}

FootTerrainDistancePhased::VectorXd FootTerrainDistancePhased::GetValues() const
{
    VectorXd g = VectorXd::Zero(GetRows());

    auto values = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_varsName))->GetValues();

    bool standing = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_varsName))->standingAt(0.);
    size_t valIdx = 0;
    size_t cIdx = 0;
    size_t sIdx = 0;
    for (size_t i = 0; i < _numPhases; ++i) {
        if (standing) {
            g(cIdx++) = values[valIdx + 2] - _terrain.height(values[valIdx], values[valIdx + 1]);
            valIdx += 3;
        }
        else {
            size_t swingKnots = _numKnotsPerSwing[sIdx];
            sIdx++;
            for (size_t k = 0; k < swingKnots; ++k) {
                g(cIdx++) = values[valIdx + 2] - _terrain.height(values[valIdx], values[valIdx + 1]);
                valIdx += 6;
            }
        }
        standing = !standing;
    }
    return g;
}

FootTerrainDistancePhased::VecBound FootTerrainDistancePhased::GetBounds() const
{
    VecBound b(GetRows(), ifopt::BoundZero);

    bool standing = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_varsName))->standingAt(0.);
    size_t bIdx = 0;

    size_t sIdx = 0;
    for (size_t i = 0; i < _numPhases; ++i) {
        if (standing) {
            b.at(bIdx++) = ifopt::BoundZero;
        }
        else {
            size_t swingKnots = _numKnotsPerSwing[sIdx];
            sIdx++;
            for (size_t k = 0; k < swingKnots; ++k) {
                b.at(bIdx++) = ifopt::BoundGreaterZero;
            }
        }
        standing = !standing;
    }
    b.back() = ifopt::BoundZero;

    return b;
}

void FootTerrainDistancePhased::FillJacobianBlock(std::string var_set, FootTerrainDistancePhased::Jacobian& jac_block) const
{
    if (var_set == _varsName) {
        Eigen::VectorXd vars = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_varsName))->GetValues();

        bool standing = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_varsName))->standingAt(0.);
        size_t rowIdx = 0;
        size_t colIdx = 0;

        // We should also compute the gradient of the terrain function w.r.t. the
        // foot's x and y positions here, however, since the use cases tested are
        // tested only on a step terrain (discontinuous), we set the gradient
        // zero.
        size_t sIdx = 0;
        for (size_t i = 0; i < _numPhases; ++i) {

            if (standing) {
                Eigen::VectorXd pos = vars.segment(colIdx, 3);
                auto terrain_d = _terrain.jacobian(pos[0], pos[1]);

                jac_block.coeffRef(rowIdx, colIdx + 0) = -terrain_d[0];
                jac_block.coeffRef(rowIdx, colIdx + 1) = -terrain_d[1];
                jac_block.coeffRef(rowIdx, colIdx + 2) = 1.;
                rowIdx++;
                colIdx += 3;
            }
            else {
                Eigen::VectorXd pos = vars.segment(colIdx, 3);
                auto terrain_d = _terrain.jacobian(pos[0], pos[1]);

                size_t swingKnots = _numKnotsPerSwing[sIdx];
                sIdx++;
                for (size_t k = 0; k < swingKnots; ++k) {
                    jac_block.coeffRef(rowIdx, colIdx + 0) = -terrain_d[0];
                    jac_block.coeffRef(rowIdx, colIdx + 1) = -terrain_d[1];
                    jac_block.coeffRef(rowIdx, colIdx + 2) = 1.;
                    rowIdx++;
                    colIdx += 6;
                }
            }
            standing = !standing;
        }
    }
}
