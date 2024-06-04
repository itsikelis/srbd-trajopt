#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#include <Eigen/Dense>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <trajopt/ifopt_sets/cost/min_effort.hpp>

#include <trajopt/ifopt_sets/constraints/common/acceleration.hpp>
#include <trajopt/ifopt_sets/constraints/common/dynamics.hpp>
#include <trajopt/ifopt_sets/constraints/common/friction_cone.hpp>

#include <trajopt/ifopt_sets/constraints/contact_implicit/foot_body_distance_implicit.hpp>
#include <trajopt/ifopt_sets/constraints/contact_implicit/foot_terrain_distance_implicit.hpp>
#include <trajopt/ifopt_sets/constraints/contact_implicit/implicit_contact.hpp>
#include <trajopt/ifopt_sets/constraints/contact_implicit/implicit_velocity.hpp>

#include <trajopt/utils/types.hpp>
#include <trajopt/utils/utils.hpp>

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

static const std::vector<std::string> VarSetNames = {trajopt::BODY_POS_TRAJECTORY, trajopt::BODY_ROT_TRAJECTORY, trajopt::FOOT_POS + "_0", trajopt::FOOT_FORCE + "_0", trajopt::FOOT_POS + "_1", trajopt::FOOT_FORCE + "_1", trajopt::FOOT_POS + "_2", trajopt::FOOT_FORCE + "_2", trajopt::FOOT_POS + "_3", trajopt::FOOT_FORCE + "_3"};
static const double Tolerance = 1e-5;
static const bool Viz = false;

void test_constr_jacobians(const ifopt::Problem& nlp, const std::vector<std::string>& var_set_names, const std::shared_ptr<ifopt::ConstraintSet>& constr_sets, double tol, bool viz);

VectorXd random_uniform_vector(size_t rows, double lower, double upper);

int main()
{
    std::srand(std::time(0));

    trajopt::SingleRigidBodyDynamicsModel model;
    trajopt::init_model_anymal(model);

    // TODO: Handle zero grids.
    trajopt::TerrainGrid terrain(200, 200, 0.7, -100, -100, 100, 100);
    std::vector<double> grid;
    grid.resize(200 * 200);
    for (auto& item : grid) {
        item = 0.;
    }
    terrain.set_grid(grid);

    // Add variable sets.
    ifopt::Problem nlp;

    static constexpr size_t numKnots = 20;
    size_t numSamples = 24;

    double totalTime = 0.5;
    double sampleTime = totalTime / static_cast<double>(numSamples - 1.);
    auto polyTimes = VectorXd(numKnots - 1);
    for (size_t i = 0; i < static_cast<size_t>(polyTimes.size()); ++i) {
        polyTimes[i] = totalTime / static_cast<double>(numKnots - 1);
    }

    // Add body pos and rot var sets.
    Vector3d initBodyPos = Vector3d(0., 0., 0.5 + terrain.height(0., 0.));
    Vector3d targetBodyPos = Vector3d(0., 0., 0.5 + terrain.height(0., 0.));
    ifopt::Component::VecBound bodyPosBounds = trajopt::fillBoundVector(initBodyPos, targetBodyPos, ifopt::NoBound, 6 * numKnots);
    auto initBodyPosVals = random_uniform_vector(3 * 2 * numKnots, -1., 1.);

    auto posVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_POS_TRAJECTORY, initBodyPosVals, polyTimes, bodyPosBounds);
    nlp.AddVariableSet(posVars);

    Vector3d initRotPos = Vector3d::Zero();
    Vector3d targetRotPos = Vector3d::Zero();
    ifopt::Component::VecBound bodyRotBounds = trajopt::fillBoundVector(initRotPos, targetRotPos, ifopt::NoBound, 6 * numKnots);
    auto initBodyRotVals = random_uniform_vector(3 * 2 * numKnots, -1., 1.);

    auto rotVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_ROT_TRAJECTORY, initBodyRotVals, polyTimes, bodyRotBounds);
    nlp.AddVariableSet(rotVars);

    // Add feet variable sets.
    auto initFootPosVals = random_uniform_vector(3 * 2 * numKnots, -1., 1.);
    auto initFootForceVals = random_uniform_vector(3 * 2 * numKnots, -1., 1.);

    ifopt::Component::VecBound footPosBounds(6 * numKnots, ifopt::NoBound);
    ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::NoBound);

    for (size_t i = 0; i < model.numFeet; ++i) {
        // Add initial and final positions for each foot.
        footPosBounds[0] = (i == 0 || i == 1) ? ifopt::Bounds(0.34, 0.34) : ifopt::Bounds(-0.34, -0.34);
        footPosBounds[1] = (i == 1 || i == 3) ? ifopt::Bounds(0.19, 0.19) : ifopt::Bounds(-0.19, -0.19);

        footPosBounds[6 * numKnots - 3] = (i == 0 || i == 1) ? ifopt::Bounds(0.34, 0.34) : ifopt::Bounds(-0.34, -0.34);
        footPosBounds[6 * numKnots - 2] = (i == 1 || i == 3) ? ifopt::Bounds(0.19, 0.19) : ifopt::Bounds(-0.19, -0.19);

        auto footPosVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_POS + "_" + std::to_string(i), initFootPosVals, polyTimes, footPosBounds);
        nlp.AddVariableSet(footPosVars);

        auto footForceVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_FORCE + "_" + std::to_string(i), initFootForceVals, polyTimes, footForceBounds);
        nlp.AddVariableSet(footForceVars);
    }

    // Add regular constraint sets.
    auto dynamConstr = std::make_shared<trajopt::Dynamics<trajopt::TrajectoryVars>>(model, numSamples, sampleTime);
    nlp.AddConstraintSet(dynamConstr);
    test_constr_jacobians(nlp, VarSetNames, dynamConstr, Tolerance, Viz);

    auto posAccConstr = std::make_shared<trajopt::AccelerationConstraints>(posVars);
    nlp.AddConstraintSet(posAccConstr);
    test_constr_jacobians(nlp, VarSetNames, posAccConstr, Tolerance, Viz);

    auto rotAccConstr = std::make_shared<trajopt::AccelerationConstraints>(rotVars);
    nlp.AddConstraintSet(rotAccConstr);
    test_constr_jacobians(nlp, VarSetNames, rotAccConstr, Tolerance, Viz);

    // Add feet constraints.
    for (size_t i = 0; i < model.numFeet; ++i) {

        auto footPosVars = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(i)));
        auto footForceVars = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_FORCE + "_" + std::to_string(i)));

        auto footPosAccConstr = std::make_shared<trajopt::AccelerationConstraints>(footPosVars);
        nlp.AddConstraintSet(footPosAccConstr);
        test_constr_jacobians(nlp, VarSetNames, footPosAccConstr, Tolerance, Viz);

        // nlp.AddConstraintSet(std::make_shared<trajopt::FootPosTerrainConstraints>(footPosVars, terrain, numSamples, sampleTime));
        auto footBodyDistConstr = std::make_shared<trajopt::FootBodyDistanceImplicit>(model, posVars, rotVars, footPosVars, numSamples, sampleTime);
        nlp.AddConstraintSet(footBodyDistConstr);
        test_constr_jacobians(nlp, VarSetNames, footBodyDistConstr, Tolerance, Viz);

        auto footDistTerrainConstr = std::make_shared<trajopt::FootTerrainDistanceImplicit>(footPosVars, terrain, numKnots);
        nlp.AddConstraintSet(footDistTerrainConstr);
        test_constr_jacobians(nlp, VarSetNames, footDistTerrainConstr, Tolerance, Viz);

        auto frictionConeConstr = std::make_shared<trajopt::FrictionCone<trajopt::TrajectoryVars>>(footForceVars, footPosVars, terrain, numSamples, sampleTime);
        nlp.AddConstraintSet(frictionConeConstr);
        test_constr_jacobians(nlp, VarSetNames, frictionConeConstr, Tolerance, Viz);

        auto implicitContactConstr = std::make_shared<trajopt::ImplicitContactConstraints>(footPosVars, footForceVars, terrain, numSamples, sampleTime);
        nlp.AddConstraintSet(implicitContactConstr);
        test_constr_jacobians(nlp, VarSetNames, implicitContactConstr, Tolerance, Viz);

        auto implicitVelocityConstr = std::make_shared<trajopt::ImplicitVelocityConstraints>(footPosVars, footForceVars, terrain, numSamples, sampleTime);
        nlp.AddConstraintSet(implicitVelocityConstr);
        test_constr_jacobians(nlp, VarSetNames, implicitVelocityConstr, Tolerance, Viz);

        // Add cost set
        auto minEffortCost = std::make_shared<trajopt::MinEffort<trajopt::TrajectoryVars>>(footPosVars, numKnots);
        nlp.AddCostSet(minEffortCost);
        test_constr_jacobians(nlp, VarSetNames, minEffortCost, Tolerance, Viz);
    }

    return 0;
}

/////// Jacobian Evaluation ////////
void test_constr_jacobians(const ifopt::Problem& nlp, const std::vector<std::string>& var_set_names, const std::shared_ptr<ifopt::ConstraintSet>& myConstr, double tol, bool viz)
{
    for (auto& var_set_name : var_set_names) {
        auto myVars = nlp.GetOptVariables()->GetComponent(var_set_name);

        // Get Calculated Jacobian.
        auto jac = Eigen::SparseMatrix<double, Eigen::RowMajor>(myConstr->GetRows(), myVars->GetRows());
        myConstr->FillJacobianBlock(var_set_name, jac);
        auto jacDense = MatrixXd(jac);

        // std::cout << jacDense.rows() << ", " << jacDense.cols() << std::endl;

        // Calculate Jacobian using finite differences.
        MatrixXd myJac(myConstr->GetRows(), myVars->GetRows());
        double eps = 1e-6;
        VectorXd vals = myVars->GetValues();

        for (int colIdx = 0; colIdx < myVars->GetRows(); ++colIdx) {
            VectorXd vals_p = vals;
            VectorXd vals_m = vals;
            vals_p[colIdx] += eps;
            vals_m[colIdx] -= eps;

            myVars->SetVariables(vals_p);
            auto dynam_p = myConstr->GetValues();

            myVars->SetVariables(vals_m);
            auto dynam_m = myConstr->GetValues();

            myJac.col(colIdx) = (dynam_p - dynam_m) / (2 * eps);
        }

        // Test if differences are near zero.
        double err = abs((myJac - jacDense).norm());
        if (err > tol) {
            std::cout << "Constraint Set: " << myConstr->GetName() << ", \t";
            std::cout << "Variable Set: " << var_set_name << ", \t";
            std::cout << "Norm of difference: " << err << std::endl;

            if (viz) {
                std::cout << std::setprecision(3);
                for (int i = 0; i < myJac.rows(); i++) {
                    std::cout << "#" << i << std::endl;
                    std::cout << "  Approx: ";
                    for (int colIdx = 0; colIdx < myVars->GetRows(); ++colIdx)
                        std::cout << std::setw(8) << std::fixed << myJac(i, colIdx) << " ";
                    std::cout << std::endl;
                    //   << myJac.col(colIdx).transpose() << std::endl;
                    std::cout << "  Actual: ";
                    for (int colIdx = 0; colIdx < myVars->GetRows(); ++colIdx)
                        std::cout << std::setw(8) << std::fixed << jacDense(i, colIdx) << " ";
                    std::cout << std::endl;
                    //   << jacDense.col(colIdx).transpose() << std::endl;
                }
            }
        }
    }
}

VectorXd random_uniform_vector(size_t rows, double lower, double upper)
{
    std::uniform_real_distribution<double> unif(lower, upper);
    std::default_random_engine re;
    re.seed(time(0));

    VectorXd rand(rows);
    for (size_t i = 0; i < rows; ++i) {
        rand[i] = unif(re);
    }

    return rand;
}
