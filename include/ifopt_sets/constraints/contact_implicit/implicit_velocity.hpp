#pragma once

#include <ifopt/constraint_set.h>

#include "include/ifopt_sets/variables/trajectory_vars.hpp"
#include "include/terrain/terrain_grid.hpp"

namespace trajopt {
    class ImplicitVelocityConstraints : public ifopt::ConstraintSet {
    public:
        ImplicitVelocityConstraints(const std::shared_ptr<TrajectoryVars>& pos_vars, const std::shared_ptr<TrajectoryVars>& force_vars, const trajopt::TerrainGrid& terrain, size_t num_samples, double sample_time)
            : ConstraintSet(3 * num_samples, pos_vars->GetName() + "_implicit_velocity"),
              _posVarsName(pos_vars->GetName()),
              _forceVarsName(force_vars->GetName()),
              _terrain(terrain),
              _numSamples(num_samples),
              _sampleTime(sample_time)
        {
        }

        VectorXd GetValues() const override
        {
            VectorXd g = VectorXd::Zero(GetRows());

            auto posVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName));
            auto forceVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_forceVarsName));

            double dt = 0.;
            for (size_t i = 0; i < _numSamples; ++i) {
                Eigen::VectorXd pos = posVars->trajectoryEval(dt, 0);
                Eigen::VectorXd vel = posVars->trajectoryEval(dt, 1);
                Eigen::VectorXd force = forceVars->trajectoryEval(dt, 0);

                double f_n = force.dot(_terrain.n(pos[0], pos[1]));
                g.segment(i * 3, 3) = f_n * vel;

                dt += _sampleTime;
            }

            return g;
        }

        VecBound GetBounds() const override
        {
            VecBound b(GetRows(), ifopt::BoundZero);
            return b;
        }

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
        {
            auto posVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName));
            auto forceVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_forceVarsName));

            if (var_set == _forceVarsName) {

                double dt = 0.;
                for (size_t i = 0; i < _numSamples; ++i) {
                    Jacobian dForce = forceVars->trajectoryJacobian(dt, 0);

                    Eigen::VectorXd pos = posVars->trajectoryEval(dt, 0);
                    Eigen::VectorXd vel = posVars->trajectoryEval(dt, 1);

                    Eigen::Vector3d n = _terrain.n(pos[0], pos[1]);

                    Jacobian mult(3, 3);
                    for (size_t i = 0; i < 3; ++i) {
                        auto res = vel[i] * n.transpose();
                        for (size_t j = 0; j < 3; ++j) {
                            mult.coeffRef(i, j) = res.coeff(j);
                        }
                    }

                    jac_block.middleRows(i * 3, 3) = mult * dForce;

                    dt += _sampleTime;
                }
            }
            else if (var_set == _posVarsName) {
                double dt = 0.;
                for (size_t i = 0; i < _numSamples; ++i) {
                    Jacobian dVel = posVars->trajectoryJacobian(dt, 1);

                    Eigen::VectorXd pos = posVars->trajectoryEval(dt, 0);
                    Eigen::VectorXd vel = posVars->trajectoryEval(dt, 1);
                    Eigen::VectorXd force = forceVars->trajectoryEval(dt, 0);

                    double f_n = force.dot(_terrain.n(pos[0], pos[1]));

                    jac_block.middleRows(i * 3, 3) = f_n * dVel;
                    dt += _sampleTime;
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