#pragma once

#include <ifopt/constraint_set.h>

#include <trajopt/ifopt_sets/variables/phased_trajectory_vars.hpp>
#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

namespace trajopt {
    template <typename FootTrajectoryVars>
    class FrictionCone : public ifopt::ConstraintSet {
    public:
        FrictionCone(
            const std::shared_ptr<FootTrajectoryVars>& forceVars,
            const std::shared_ptr<FootTrajectoryVars>& posVars,
            const trajopt::TerrainGrid& terrain,
            size_t numSamples,
            double sampleTime)
            : ConstraintSet(5 * numSamples, forceVars->GetName() + "_friction_cone"),
              _forceVarsName(forceVars->GetName()),
              _posVarsName(posVars->GetName()),
              _numSamples(numSamples),
              _sampleTime(sampleTime),
              _terrain(terrain) {}

        VectorXd GetValues() const override
        {
            VectorXd g = VectorXd::Zero(GetRows());

            auto forceVars = std::static_pointer_cast<FootTrajectoryVars>(GetVariables()->GetComponent(_forceVarsName));
            auto posVars = std::static_pointer_cast<FootTrajectoryVars>(GetVariables()->GetComponent(_posVarsName));

            double t = 0.;
            for (size_t i = 0; i < _numSamples; ++i) {
                Eigen::Vector3d f = forceVars->trajectoryEval(t, 0);
                Eigen::Vector3d pos = posVars->trajectoryEval(t, 0);
                double x = pos[0];
                double y = pos[1];

                double fn = f.dot(_terrain.GetNormalizedBasis(TerrainGrid::Normal, x, y));
                double ft1 = f.dot(_terrain.GetNormalizedBasis(TerrainGrid::Tangent1, x, y));
                double ft2 = f.dot(_terrain.GetNormalizedBasis(TerrainGrid::Tangent2, x, y));

                g[i * 5 + 0] = fn;
                g[i * 5 + 1] = ft1 - _terrain.GetFrictionCoeff() * fn;
                g[i * 5 + 2] = -ft1 - _terrain.GetFrictionCoeff() * fn;
                g[i * 5 + 3] = ft2 - _terrain.GetFrictionCoeff() * fn;
                g[i * 5 + 4] = -ft2 - _terrain.GetFrictionCoeff() * fn;

                t += _sampleTime;
            }

            return g;
        }

        VecBound GetBounds() const override
        {
            VecBound b(GetRows());

            for (size_t i = 0; i < _numSamples; ++i) {
                b.at(i * 5 + 0) = ifopt::BoundGreaterZero;
                b.at(i * 5 + 1) = ifopt::BoundSmallerZero;
                b.at(i * 5 + 2) = ifopt::BoundSmallerZero;
                b.at(i * 5 + 3) = ifopt::BoundSmallerZero;
                b.at(i * 5 + 4) = ifopt::BoundSmallerZero;
            }

            return b;
        }

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
        {
            if (var_set == _forceVarsName) {
                auto forceVars = std::static_pointer_cast<FootTrajectoryVars>(GetVariables()->GetComponent(_forceVarsName));
                auto posVars = std::static_pointer_cast<FootTrajectoryVars>(GetVariables()->GetComponent(_posVarsName));

                double t = 0.;
                for (size_t i = 0; i < _numSamples; ++i) {
                    Jacobian fPos = forceVars->trajectoryJacobian(t, 0);
                    Eigen::Vector3d footPos = posVars->trajectoryEval(t, 0);
                    auto x = footPos[0];
                    auto y = footPos[1];

                    Eigen::Vector3d n = _terrain.GetNormalizedBasis(TerrainGrid::Normal, x, y);
                    Eigen::Vector3d t1 = _terrain.GetNormalizedBasis(TerrainGrid::Tangent1, x, y);
                    Eigen::Vector3d t2 = _terrain.GetNormalizedBasis(TerrainGrid::Tangent2, x, y);
                    Jacobian mult0 = n.transpose().sparseView(1, -1);
                    Jacobian res0 = mult0 * fPos;

                    Jacobian mult1 = (t1 - _terrain.GetFrictionCoeff() * n).transpose().sparseView(1, -1);
                    Jacobian res1 = mult1 * fPos;

                    Jacobian mult2 = (-t1 - _terrain.GetFrictionCoeff() * n).transpose().sparseView(1, -1);
                    Jacobian res2 = mult2 * fPos;

                    Jacobian mult3 = (t2 - _terrain.GetFrictionCoeff() * n).transpose().sparseView(1, -1);
                    Jacobian res3 = mult3 * fPos;

                    Jacobian mult4 = (-t2 - _terrain.GetFrictionCoeff() * n).transpose().sparseView(1, -1);
                    Jacobian res4 = mult4 * fPos;

                    jac_block.middleRows(i * 5 + 0, 1) += res0;
                    jac_block.middleRows(i * 5 + 1, 1) += res1;
                    jac_block.middleRows(i * 5 + 2, 1) += res2;
                    jac_block.middleRows(i * 5 + 3, 1) += res3;
                    jac_block.middleRows(i * 5 + 4, 1) += res4;

                    t += _sampleTime;
                }
            }
        }

    private:
        const std::string _forceVarsName, _posVarsName;
        const size_t _numSamples;
        const double _sampleTime;
        const trajopt::TerrainGrid _terrain;
    };
} // namespace trajopt
