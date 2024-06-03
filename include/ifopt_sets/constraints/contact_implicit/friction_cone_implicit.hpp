#pragma once

#include <ifopt/constraint_set.h>

#include <ifopt_sets/variables/trajectory_vars.hpp>
#include <terrain/terrain_grid.hpp>

namespace trajopt {
    class FrictionConeImplicit : public ifopt::ConstraintSet {
    public:
        FrictionConeImplicit(
            const std::shared_ptr<TrajectoryVars>& forceVars,
            const std::shared_ptr<TrajectoryVars>& posVars,
            const trajopt::TerrainGrid& terrain,
            size_t numSamples,
            double sampleTime)
            : ConstraintSet(5 * numSamples, forceVars->GetName() + "_friction_cone"),
              _posVarsName(posVars->GetName()),
              _forceVarsName(forceVars->GetName()),
              _terrain(terrain),
              _numSamples(numSamples),
              _sampleTime(sampleTime)
        {
        }

        VectorXd GetValues() const override
        {
            VectorXd g = VectorXd::Zero(GetRows());

            auto forceVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_forceVarsName));
            auto posVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName));

            double t = 0.;
            for (size_t i = 0; i < _numSamples; ++i) {
                Eigen::Vector3d f = forceVars->trajectoryEval(t, 0);
                Eigen::Vector3d pos = posVars->trajectoryEval(t, 0);
                double x = pos[0];
                double y = pos[1];

                double fn = f.dot(_terrain.n(x, y));
                double ft = f.dot(_terrain.t(x, y));
                double fb = f.dot(_terrain.b(x, y));

                g[i * 5 + 0] = fn;
                g[i * 5 + 1] = ft - _terrain.mu() * fn;
                g[i * 5 + 2] = -ft - _terrain.mu() * fn;
                g[i * 5 + 3] = fb - _terrain.mu() * fn;
                g[i * 5 + 4] = -fb - _terrain.mu() * fn;

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
                auto forceVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_forceVarsName));
                auto posVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName));

                double t = 0.;
                for (size_t i = 0; i < _numSamples; ++i) {
                    Jacobian fPos = forceVars->trajectoryJacobian(t, 0);
                    Eigen::Vector3d footPos = posVars->trajectoryEval(t, 0);
                    auto x = footPos[0];
                    auto y = footPos[1];

                    Jacobian mult0 = _terrain.n(x, y).transpose().sparseView(1, -1);
                    Jacobian res0 = mult0 * fPos;

                    Jacobian mult1 = (_terrain.t(x, y) - _terrain.mu() * _terrain.n(x, y)).transpose().sparseView(1, -1);
                    Jacobian res1 = mult1 * fPos;

                    Jacobian mult2 = (-_terrain.t(x, y) - _terrain.mu() * _terrain.n(x, y)).transpose().sparseView(1, -1);
                    Jacobian res2 = mult2 * fPos;

                    Jacobian mult3 = (_terrain.b(x, y) - _terrain.mu() * _terrain.n(x, y)).transpose().sparseView(1, -1);
                    Jacobian res3 = mult3 * fPos;

                    Jacobian mult4 = (-_terrain.b(x, y) - _terrain.mu() * _terrain.n(x, y)).transpose().sparseView(1, -1);
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

    protected:
        const std::string _posVarsName;
        const std::string _forceVarsName;
        const trajopt::TerrainGrid _terrain;
        const size_t _numSamples;
        const double _sampleTime;
    };
} // namespace trajopt
