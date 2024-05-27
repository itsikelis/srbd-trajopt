#pragma once

#include <ifopt/constraint_set.h>

#include "../../utils/srbd.hpp"
#include "../../utils/terrain.hpp"
#include "../../utils/types.hpp"
#include "../variables.hpp"

namespace trajopt {
    class FrictionConeConstraints : public ifopt::ConstraintSet {
    public:
        FrictionConeConstraints(
            const std::shared_ptr<TrajectoryVars>& forceVars,
            const std::shared_ptr<TrajectoryVars>& posVars,
            const trajopt::TerrainGrid<200, 200>& terrain,
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

    private:
        const std::string _forceVarsName, _posVarsName;
        const size_t _numSamples;
        const double _sampleTime;
        const trajopt::TerrainGrid<200, 200> _terrain;
    };

    class ImplicitContactConstraints : public ifopt::ConstraintSet {
    public:
        ImplicitContactConstraints(const std::shared_ptr<TrajectoryVars>& pos_vars, const std::shared_ptr<TrajectoryVars>& force_vars, const trajopt::TerrainGrid<200, 200>& terrain, size_t num_knots)
            : ConstraintSet(num_knots, pos_vars->GetName() + "_implicit_contact"),
              _posVarsName(pos_vars->GetName()),
              _forceVarsName(force_vars->GetName()),
              _terrain(terrain)
        {
        }

        VectorXd GetValues() const override
        {
            VectorXd g = VectorXd::Zero(GetRows());

            // std::cout << "##############" << std::endl;
            // std::cout << GetName() << " num of phases: " << _numPhases << std::endl;
            // std::cout << GetName() << " num of steps: " << _numSteps << std::endl;
            // std::cout << GetName() << " num of swings: " << _numSwings << std::endl;
            // std::cout << "##############" << std::endl;

            auto pos_knots = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName))->GetValues();
            auto force_knots = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_forceVarsName))->GetValues();

            for (size_t i = 0; i < static_cast<size_t>(GetRows()); ++i) {
                Eigen::VectorXd pos = pos_knots.segment(i * 6, 3);
                Eigen::VectorXd force = force_knots.segment(i * 6, 3);

                double f_n = force.dot(_terrain.n(pos[0], pos[1]));
                double phi = pos[2] - _terrain.height(pos[0], pos[1]);
                g[i] = f_n * phi;
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
            auto pos_knots = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_posVarsName))->GetValues();
            auto force_knots = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_forceVarsName))->GetValues();

            if (var_set == _forceVarsName) {
                for (size_t i = 0; i < static_cast<size_t>(GetRows()); ++i) {
                    Eigen::VectorXd pos = pos_knots.segment(i * 6, 3);
                    auto phi = pos[2] - _terrain.height(pos[0], pos[1]);

                    Eigen::Vector3d n = _terrain.n(pos[0], pos[1]);
                    jac_block.coeffRef(i, i * 6) = n[0] * phi;
                    jac_block.coeffRef(i, i * 6 + 1) = n[1] * phi;
                    jac_block.coeffRef(i, i * 6 + 2) = n[2] * phi;
                }
            }
            else if (var_set == _posVarsName) {
                for (size_t i = 0; i < static_cast<size_t>(GetRows()); ++i) {
                    jac_block.coeffRef(i, i * 6 + 2) = 1; // Assume ground is level.
                }
            }
        }

    private:
        const std::string _posVarsName;
        const std::string _forceVarsName;
        const trajopt::TerrainGrid<200, 200> _terrain;
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

    class FootBodyPosConstraints : public ifopt::ConstraintSet {
    public:
        FootBodyPosConstraints(
            const SingleRigidBodyDynamicsModel& model,
            const std::shared_ptr<TrajectoryVars>& bodyPosVars,
            const std::shared_ptr<TrajectoryVars>& bodyRotVars,
            const std::shared_ptr<TrajectoryVars>& footPosVars,
            size_t numSamples, double sampleTime)
            : ConstraintSet(3 * numSamples, footPosVars->GetName() + "_foot_body_pos"),
              _model(model),
              _bodyPosVarsName(bodyPosVars->GetName()),
              _bodyRotVarsName(bodyRotVars->GetName()),
              _footPosVarsName(footPosVars->GetName()),
              _numSamples(numSamples),
              _sampleTime(sampleTime)
        {
        }

        VectorXd GetValues() const override
        {
            VectorXd g = VectorXd::Zero(GetRows());

            auto bodyPosVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_bodyPosVarsName));
            auto bodyRotVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_bodyRotVarsName));
            auto footPosVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_footPosVarsName));

            double t = 0.;
            for (size_t i = 0; i < _numSamples; ++i) {
                Eigen::Vector3d b = bodyPosVars->trajectoryEval(t, 0);
                Jacobian R = eulerZYXToMatrix(bodyRotVars->trajectoryEval(t, 0));
                Eigen::Vector3d f = footPosVars->trajectoryEval(t, 0);

                g.segment(i * 3, 3) = R.transpose() * (f - b);

                t += _sampleTime;
            }

            return g;
        }

        VecBound GetBounds() const override
        {

            VecBound b(GetRows(), ifopt::BoundZero);

            size_t idx = 0;
            for (size_t k = 0; k < _model.numFeet; ++k) {
                if (_footPosVarsName == FOOT_POS + "_" + std::to_string(k)) {
                    break;
                }
                ++idx;
            }

            for (size_t i = 0; i < _numSamples; ++i) {
                b.at(i * 3 + 0) = ifopt::Bounds(_model.feetMinBounds[idx][0], _model.feetMaxBounds[idx][0]);
                b.at(i * 3 + 1) = ifopt::Bounds(_model.feetMinBounds[idx][1], _model.feetMaxBounds[idx][1]);
                b.at(i * 3 + 2) = ifopt::Bounds(_model.feetMinBounds[idx][2], _model.feetMaxBounds[idx][2]);
            }

            return b;
        }

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
        {
            auto bodyPosVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_bodyPosVarsName));
            auto bodyRotVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_bodyRotVarsName));
            auto footPosVars = std::static_pointer_cast<TrajectoryVars>(GetVariables()->GetComponent(_footPosVarsName));

            if (var_set == _bodyPosVarsName) {
                double t = 0.;
                for (size_t i = 0; i < _numSamples; ++i) {
                    Jacobian R = eulerZYXToMatrix(bodyRotVars->trajectoryEval(t, 0));
                    Jacobian dBodyPos = bodyPosVars->trajectoryJacobian(t, 0);

                    jac_block.middleRows(i * 3, 3) = -R.transpose() * dBodyPos;

                    t += _sampleTime;
                }
            }
            else if (var_set == _bodyRotVarsName) {
                double t = 0.;
                for (size_t i = 0; i < _numSamples; ++i) {
                    Jacobian dBodyRot = bodyRotVars->trajectoryJacobian(t, 0);

                    Eigen::Vector3d b = bodyPosVars->trajectoryEval(t, 0);
                    Eigen::Vector3d euler_zyx = bodyRotVars->trajectoryEval(t, 0);
                    Eigen::Vector3d f = footPosVars->trajectoryEval(t, 0);

                    Jacobian mult = derivRotationTransposeVector(euler_zyx, f - b);
                    Jacobian res = mult * dBodyRot;
                    jac_block.middleRows(i * 3, 3) = res;

                    t += _sampleTime;
                }
            }
            else if (var_set == _footPosVarsName) {
                double t = 0.;
                for (size_t i = 0; i < _numSamples; ++i) {
                    Jacobian R = eulerZYXToMatrix(bodyRotVars->trajectoryEval(t, 0));
                    Jacobian dFootPos = footPosVars->trajectoryJacobian(t, 0);

                    Jacobian res = R.transpose() * dFootPos;
                    jac_block.middleRows(i * 3, 3) = res;

                    t += _sampleTime;
                }
            }
        }

    protected:
        Jacobian eulerZYXToMatrix(const Eigen::Vector3d& eulerZYX) const
        {
            const double x = eulerZYX[2];
            const double y = eulerZYX[1];
            const double z = eulerZYX[0];

            const double cx = std::cos(x);
            const double sx = std::sin(x);

            const double cy = std::cos(y);
            const double sy = std::sin(y);

            const double cz = std::cos(z);
            const double sz = std::sin(z);

            Eigen::Matrix3d R;
            R << cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy, cy * sz,
                cx * cz + sx * sy * sz, cx * sy * sz - cz * sx, -sy, cy * sx, cx * cy;

            return R.sparseView(1., -1.);
        }

        Jacobian derivRotationTransposeVector(const Eigen::Vector3d& eulerZYX,
            const Eigen::Vector3d& v) const
        {
            const double x = eulerZYX[2];
            const double y = eulerZYX[1];
            const double z = eulerZYX[0];

            const double cx = std::cos(x);
            const double sx = std::sin(x);

            const double cy = std::cos(y);
            const double sy = std::sin(y);

            const double cz = std::cos(z);
            const double sz = std::sin(z);

            // out = R.T * v
            // out[0] = R.T.row(0) * v = cy * cz * v[0] + cy * sz * v[1] + -sy * v[2]
            // out[1] = R.T.row(1) * v = (cz * sx * sy - cx * sz) * v[0] + (cx * cz + sx
            // * sy * sz) * v[1] + cy * sx * v[2] out[2] = R.T.row(2) * v = (sx * sz +
            // cx * cz * sy) * v[0] + (cx * sy * sz - cz * sx) * v[1] + cx * cy * v[2]

            Jacobian jac(3, 3);
            // out[0] wrt Z(0)
            jac.coeffRef(0, 0) = -cy * sz * v[0] + cy * cz * v[1];
            // out[0] wrt Y(1)
            jac.coeffRef(0, 1) = -sy * cz * v[0] - sy * sz * v[1] - cy * v[2];
            // out[0] wrt X(2)
            // jac.coeffRef(0, 2) = 0.;

            // out[1] wrt Z(0)
            jac.coeffRef(1, 0) = (-sz * sx * sy - cx * cz) * v[0] + (-cx * sz + sx * sy * cz) * v[1];
            // out[1] wrt Y(1)
            jac.coeffRef(1, 1) = cz * sx * cy * v[0] + sx * cy * sz * v[1] - sy * sx * v[2];
            // out[1] wrt X(2)
            jac.coeffRef(1, 2) = (cz * cx * sy + sx * sz) * v[0] + (-sx * cz + cx * sy * sz) * v[1] + cy * cx * v[2];

            // out[2] wrt Z(0)
            jac.coeffRef(2, 0) = (sx * cz - cx * sz * sy) * v[0] + (cx * sy * cz + sz * sx) * v[1];
            // out[2] wrt Y(1)
            jac.coeffRef(2, 1) = cx * cz * cy * v[0] + cx * cy * sz * v[1] - cx * sy * v[2];
            // out[2] wrt X(2)
            jac.coeffRef(2, 2) = (cx * sz - sx * cz * sy) * v[0] + (-sx * sy * sz - cz * cx) * v[1] - sx * cy * v[2];

            return jac;
        }

    protected:
        const SingleRigidBodyDynamicsModel _model;
        const std::string _bodyPosVarsName;
        const std::string _bodyRotVarsName;
        const std::string _footPosVarsName;
        const size_t _numSamples;
        const double _sampleTime;
    };
} // namespace trajopt
