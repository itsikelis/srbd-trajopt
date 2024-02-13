#pragma once

#include <ifopt/constraint_set.h>

#include "../variables.hpp"

namespace trajopt {
class AccelerationConstraints : public ifopt::ConstraintSet {
public:
  AccelerationConstraints(const std::shared_ptr<TrajectoryVars> &vars)
      : ConstraintSet(kSpecifyLater, vars->GetName() + "_equal_acc"),
        _variableSet(vars->GetName()) {
    SetRows((vars->numKnotPoints() - 2) * 3);
  }

  VectorXd GetValues() const override {
    VectorXd g = VectorXd::Zero(GetRows());

    auto vars = std::static_pointer_cast<TrajectoryVars>(
        GetVariables()->GetComponent(_variableSet));
    auto &splines = vars->traj().splines();

    for (unsigned int i = 0; i < splines.size() - 1; i++) {
      g.segment(i * 3, 3) = splines[i]->acceleration(splines[i]->duration()) -
                            splines[i + 1]->acceleration(0.);
    }

    return g;
  }

  VecBound GetBounds() const override {
    // All constraints equal to zero.
    VecBound b(GetRows());
    for (int i = 0; i < GetRows(); i++) {
      b.at(i) = ifopt::BoundZero;
    }
    return b;
  }

  void FillJacobianBlock(std::string var_set,
                         Jacobian &jac_block) const override {
    if (var_set == _variableSet) {
      auto vars = std::static_pointer_cast<TrajectoryVars>(
          GetVariables()->GetComponent(_variableSet));

      auto &splines = vars->traj().splines();

      for (unsigned int i = 0; i < splines.size() - 1; i++) {
        Jacobian jacStart = splines[i]->jacobian_acc(splines[i]->duration());
        Jacobian jacEnd = splines[i + 1]->jacobian_acc(0.);

        for (int k = 0; k < jacStart.outerSize(); ++k)
          for (Jacobian::InnerIterator it(jacStart, k); it; ++it) {
            jac_block.coeffRef(i * 3 + it.row(), i * 6 + it.col()) +=
                it.value();
          }
        for (int k = 0; k < jacEnd.outerSize(); ++k)
          for (Jacobian::InnerIterator it(jacEnd, k); it; ++it) {
            jac_block.coeffRef(i * 3 + it.row(), (i + 1) * 6 + it.col()) +=
                -it.value();
          }
      }
    }
  }

protected:
  std::string _variableSet;
};

class PhasedAccelerationConstraints : public ifopt::ConstraintSet {
public:
  PhasedAccelerationConstraints(
      const std::shared_ptr<PhasedTrajectoryVars> &vars)
      : ConstraintSet(kSpecifyLater, vars->GetName() + "_equal_acc"),
        _variableSetName(vars->GetName()) {
    SetRows((vars->numKnotPoints() - 2) * 3);
  }

  VectorXd GetValues() const override {
    VectorXd g = VectorXd::Zero(GetRows());

    auto vars = std::static_pointer_cast<PhasedTrajectoryVars>(
        GetVariables()->GetComponent(_variableSetName));
    auto &splines = vars->traj().splines();

    for (size_t i = 0; i < splines.size() - 1; i++) {
      g.segment(i * 3, 3) = splines[i]->acceleration(splines[i]->duration()) -
                            splines[i + 1]->acceleration(0.);
    }

    return g;
  }

  VecBound GetBounds() const override {
    // All constraints equal to zero.
    VecBound b(GetRows());
    for (int i = 0; i < GetRows(); i++) {
      b.at(i) = ifopt::BoundZero;
    }
    return b;
  }

  void FillJacobianBlock(std::string var_set,
                         Jacobian &jac_block) const override {
    if (var_set == _variableSetName) {
      auto vars = std::static_pointer_cast<PhasedTrajectoryVars>(
          GetVariables()->GetComponent(_variableSetName));

      auto &splines = vars->traj().splines();

      for (size_t i = 0; i < splines.size() - 1; i++) {
        Jacobian jacStart = splines[i]->jacobian_acc(splines[i]->duration());
        Jacobian jacEnd = splines[i + 1]->jacobian_acc(0.);

        bool isStanceStart = vars->isSplineStance(i);
        size_t sIdxStart = vars->varStart(i);

        bool isStanceEnd = vars->isSplineStance(i + 1);
        size_t sIdxEnd = vars->varStart(i + 1);

        // wrt to first spline
        if (isStanceStart) {
          Jacobian newJac = jacStart.block(0, 0, jacStart.rows(), 3) +
                            jacStart.block(0, 6, jacStart.rows(), 3);
          for (int k = 0; k < newJac.outerSize(); ++k)
            for (Jacobian::InnerIterator it(newJac, k); it; ++it) {
              jac_block.coeffRef(i * 3 + it.row(), sIdxStart + it.col()) +=
                  it.value();
            }
        } else {
          bool hasPrev = (i > 0);

          if (i == 0) {
            Jacobian newJac = jacStart.block(0, 0, jacStart.rows(), 6);
            for (int k = 0; k < newJac.outerSize(); ++k)
              for (Jacobian::InnerIterator it(newJac, k); it; ++it) {
                jac_block.coeffRef(i * 3 + it.row(), sIdxStart + it.col()) +=
                    it.value();
              }
          }

          if (hasPrev) {
            // size_t pIdx = vars->varStart(i - 1);
            bool isStancePrev = vars->isSplineStance(i - 1);
            if (isStancePrev) {
              Jacobian newJac = jacStart.block(0, 0, jacStart.rows(), 3);
              for (int k = 0; k < newJac.outerSize(); ++k)
                for (Jacobian::InnerIterator it(newJac, k); it; ++it) {
                  jac_block.coeffRef(i * 3 + it.row(), sIdxStart + it.col()) +=
                      it.value();
                }
            } else {
              Jacobian newJac = jacStart.block(0, 0, jacStart.rows(), 6);
              for (int k = 0; k < newJac.outerSize(); ++k)
                for (Jacobian::InnerIterator it(newJac, k); it; ++it) {
                  jac_block.coeffRef(i * 3 + it.row(), sIdxStart + it.col()) +=
                      it.value();
                }
            }
          }

          if (isStanceEnd) {
            Jacobian newJac = jacStart.block(0, 6, jacStart.rows(), 3);
            for (int k = 0; k < newJac.outerSize(); ++k)
              for (Jacobian::InnerIterator it(newJac, k); it; ++it) {
                jac_block.coeffRef(i * 3 + it.row(), sIdxEnd + it.col()) +=
                    it.value();
              }
          } else {
            Jacobian newJac = jacStart.block(0, 6, jacStart.rows(), 6);
            for (int k = 0; k < newJac.outerSize(); ++k)
              for (Jacobian::InnerIterator it(newJac, k); it; ++it) {
                jac_block.coeffRef(i * 3 + it.row(), sIdxEnd + it.col()) +=
                    it.value();
              }
          }
        }

        // wrt to second spline
        if (isStanceEnd) {
          Jacobian newJac = jacEnd.block(0, 0, jacEnd.rows(), 3) +
                            jacEnd.block(0, 6, jacEnd.rows(), 3);
          for (int k = 0; k < newJac.outerSize(); ++k)
            for (Jacobian::InnerIterator it(newJac, k); it; ++it) {
              jac_block.coeffRef(i * 3 + it.row(), sIdxEnd + it.col()) -=
                  it.value();
            }
        } else {
          if (i == (splines.size() - 2)) {
            size_t addIdx = 6;
            if (isStanceStart)
              addIdx = 3;
            Jacobian newJac = jacEnd.block(0, 6, jacEnd.rows(), 6);
            for (int k = 0; k < newJac.outerSize(); ++k)
              for (Jacobian::InnerIterator it(newJac, k); it; ++it) {
                jac_block.coeffRef(i * 3 + it.row(),
                                   sIdxEnd + addIdx + it.col()) -= it.value();
              }
          }

          bool hasNext = (i < (splines.size() - 2));
          // previous
          {
            if (isStanceStart) {
              Jacobian newJac = jacEnd.block(0, 0, jacEnd.rows(), 3);
              for (int k = 0; k < newJac.outerSize(); ++k)
                for (Jacobian::InnerIterator it(newJac, k); it; ++it) {
                  jac_block.coeffRef(i * 3 + it.row(), sIdxEnd + it.col()) -=
                      it.value();
                }
            } else {
              Jacobian newJac = jacEnd.block(0, 0, jacEnd.rows(), 6);
              for (int k = 0; k < newJac.outerSize(); ++k)
                for (Jacobian::InnerIterator it(newJac, k); it; ++it) {
                  jac_block.coeffRef(i * 3 + it.row(), sIdxEnd + it.col()) -=
                      it.value();
                }
            }
          }
          // next
          if (hasNext) {
            size_t nIdx = vars->varStart(i + 2);
            bool isStanceNext = vars->isSplineStance(i + 2);

            if (isStanceNext) {
              Jacobian newJac = jacEnd.block(0, 6, jacEnd.rows(), 3);
              for (int k = 0; k < newJac.outerSize(); ++k)
                for (Jacobian::InnerIterator it(newJac, k); it; ++it) {
                  jac_block.coeffRef(i * 3 + it.row(), nIdx + it.col()) -=
                      it.value();
                }
            } else {
              Jacobian newJac = jacEnd.block(0, 6, jacEnd.rows(), 6);
              for (int k = 0; k < newJac.outerSize(); ++k)
                for (Jacobian::InnerIterator it(newJac, k); it; ++it) {
                  jac_block.coeffRef(i * 3 + it.row(), nIdx + it.col()) -=
                      it.value();
                }
            }
          }
        }
      }
    }
  }

protected:
  std::string _variableSetName;
};

} // namespace trajopt
