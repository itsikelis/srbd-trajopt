#pragma once

#include <ifopt/constraint_set.h>

#include "include/ifopt_sets/variables/phased_trajectory_vars.hpp"
#include "include/terrain/terrain_grid.hpp"

namespace trajopt {
    class FootTerrainDistance : public ifopt::ConstraintSet {
    public:
        FootTerrainDistance(const std::shared_ptr<PhasedTrajectoryVars>& vars,
            const trajopt::TerrainGrid& terrain, size_t numSteps,
            size_t numSwings, std::vector<size_t> numKnotsPerSwing)
            : ConstraintSet(kSpecifyLater, vars->GetName() + "_foot_pos_terrain"),
              _varsName(vars->GetName()),
              _terrain(terrain),
              _numPhases(numSteps + numSwings),
              _numKnotsPerSwing(numKnotsPerSwing)
        {
            SetRows(numSteps + std::accumulate(numKnotsPerSwing.begin(), numKnotsPerSwing.end(), 0));
        }

        VectorXd GetValues() const override
        {
            VectorXd g = VectorXd::Zero(GetRows());

            auto values = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_varsName))->GetValues();

            bool standing = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_varsName))->standingAt(0.);
            size_t valIdx = 0;
            size_t cIdx = 0;

            // std::cout << "##############" << std::endl;
            // std::cout << GetName() << " num of phases: " << _numPhases << std::endl;
            // std::cout << GetName() << " num of steps: " << _numSteps << std::endl;
            // std::cout << GetName() << " num of swings: " << _numSwings << std::endl;
            // std::cout << "##############" << std::endl;
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

        VecBound GetBounds() const override
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

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
        {
            if (var_set == _varsName) {
                auto vars = std::static_pointer_cast<PhasedTrajectoryVars>(
                    GetVariables()->GetComponent(_varsName));

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
                        jac_block.coeffRef(rowIdx++, colIdx + 2) = 1.;
                        colIdx += 3;
                    }
                    else {
                        size_t swingKnots = _numKnotsPerSwing[sIdx];
                        sIdx++;
                        for (size_t k = 0; k < swingKnots; ++k) {
                            jac_block.coeffRef(rowIdx++, colIdx + 2) = 1.;
                            colIdx += 6;
                        }
                    }
                    standing = !standing;
                }
            }
        }

    private:
        const std::string _varsName;
        const trajopt::TerrainGrid _terrain;
        const size_t _numPhases;
        const std::vector<size_t> _numKnotsPerSwing;
    };

    // class FootPosTerrainConstraints : public ifopt::ConstraintSet {
    // public:
    //     FootPosTerrainConstraints(
    //         const std::shared_ptr<PhasedTrajectoryVars>& vars,
    //         const trajopt::Terrain& terrain,
    //         size_t numSamples,
    //         double sampleTime)
    //         : ConstraintSet(numSamples, vars->GetName() + "_foot_pos_terrain"),
    //           _varSetName(vars->GetName()),
    //           _terrain(terrain),
    //           _numSamples(numSamples),
    //           _sampleTime(sampleTime) {}
    //
    //     VectorXd GetValues() const override
    //     {
    //         VectorXd g = VectorXd::Zero(GetRows());
    //         auto values = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_varSetName))->GetValues();
    //
    //         // std::cout << "##############" << std::endl;
    //         // std::cout << GetName() << " num of phases: " << _numPhases << std::endl;
    //         // std::cout << GetName() << " num of steps: " << _numSteps << std::endl;
    //         // std::cout << GetName() << " num of swings: " << _numSwings << std::endl;
    //         // std::cout << "##############" << std::endl;
    //
    //         auto vars = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_varSetName));
    //
    //         double t = 0.;
    //         for (size_t i = 0; i < _numSamples; ++i) {
    //             auto footPos = vars->trajectoryEval(t, 0);
    //             auto footZ = footPos[2];
    //             auto terrainZ = _terrain.z(footPos[0], footPos[1]);
    //             g[i] = footZ - terrainZ;
    //
    //             t += _sampleTime;
    //         }
    //         return g;
    //     }
    //
    //     VecBound GetBounds() const override
    //     {
    //         VecBound b(GetRows(), ifopt::BoundZero);
    //         auto vars = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_varSetName));
    //
    //         double t = 0.;
    //         for (size_t i = 0; i < _numSamples; ++i) {
    //             if (vars->standingAt(t)) {
    //                 b.at(i) = ifopt::BoundZero;
    //             }
    //             else {
    //                 b.at(i) = ifopt::BoundGreaterZero;
    //             }
    //
    //             t += _sampleTime;
    //         }
    //         return b;
    //     }
    //
    //     void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
    //     {
    //         if (var_set == _varSetName) {
    //             auto vars = std::static_pointer_cast<PhasedTrajectoryVars>(GetVariables()->GetComponent(_varSetName));
    //
    //             double t = 0.;
    //             for (size_t i = 0; i < _numSamples; ++i) {
    //                 Jacobian dPosZ = vars->trajectoryJacobian(t, 0).row(2);
    //                 size_t varIdx = vars->varStartAt(t);
    //
    //                 jac_block.coeffRef(i, varIdx + 2) = 1.;
    //                 Jacobian res = jac_block.row(i);
    //                 for (size_t j = 0; j < static_cast<size_t>(jac_block.row(i).cols()); ++j) {
    //                     res.coeffRef(0, j) *= dPosZ.coeffRef(0, j);
    //                 }
    //                 jac_block.row(i) = res;
    //             }
    //         }
    //     }
    //
    // private:
    //     const std::string _varSetName;
    //     const trajopt::Terrain _terrain;
    //     size_t _numSamples;
    //     double _sampleTime;
    // };
} // namespace trajopt
