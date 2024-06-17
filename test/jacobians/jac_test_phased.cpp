#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <Eigen/Dense>

#include <trajopt/ifopt_sets/variables/phased_trajectory_vars.hpp>
#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <trajopt/ifopt_sets/constraints/common/acceleration.hpp>
#include <trajopt/ifopt_sets/constraints/common/dynamics.hpp>
#include <trajopt/ifopt_sets/constraints/common/friction_cone.hpp>

#include <trajopt/ifopt_sets/constraints/phased/foot_body_distance_phased.hpp>
#include <trajopt/ifopt_sets/constraints/phased/foot_terrain_distance_phased.hpp>
#include <trajopt/ifopt_sets/constraints/phased/phased_acceleration.hpp>

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

    // Create a Terrain Model (Select from predefined ones)
    // trajopt::Terrain terrain("");
    trajopt::TerrainGrid terrain(200, 200, 0.7, 0, 0, 200, 200);
    std::vector<double> grid;
    grid.resize(200 * 200);
    for (auto& item : grid) {
        item = 0.;
    }
    terrain.set_grid(grid);

    // Add variable sets.
    ifopt::Problem nlp;

    static constexpr size_t numKnots = 30;
    size_t numSamples = 30;

    double totalTime = 0.5;
    double sampleTime = totalTime / static_cast<double>(numSamples - 1.);
    std::vector<double> polyTimes = std::vector<double>(numKnots - 1);
    for (size_t i = 0; i < static_cast<size_t>(polyTimes.size()); ++i) {
        polyTimes[i] = totalTime / static_cast<double>(numKnots - 1);
    }

    // Add body pos and rot var sets.
    Vector3d initBodyPos = Vector3d(0., 0., 0.5);
    Vector3d targetBodyPos = Vector3d(0., 0., 0.5 + terrain.height(0., 0.));
    ifopt::Component::VecBound bodyPosBounds = trajopt::fillBoundVector(initBodyPos, targetBodyPos, ifopt::NoBound, 6 * numKnots);
    auto initBodyPosVals = VectorXd(3 * 2 * numKnots);

    auto posVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_POS_TRAJECTORY, initBodyPosVals, polyTimes, bodyPosBounds);
    nlp.AddVariableSet(posVars);

    Vector3d initRotPos = Vector3d::Zero();
    Vector3d targetRotPos = Vector3d::Zero();
    ifopt::Component::VecBound bodyRotBounds = trajopt::fillBoundVector(initRotPos, targetRotPos, ifopt::NoBound, 6 * numKnots);
    auto initBodyRotVals = VectorXd(3 * 2 * numKnots);
    auto rotVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_ROT_TRAJECTORY, initBodyRotVals, polyTimes, bodyRotBounds);
    nlp.AddVariableSet(rotVars);

    // Add feet pos and force var sets.
    size_t numPosSteps = 2;
    size_t numForceSteps = 1;
    std::vector<double> phaseTimes = {0.2, 0.1, 0.2};
    std::vector<size_t> posKnotsPerSwing = {3};
    std::vector<size_t> forceKnotsPerSwing = {3, 3};

    // size_t numPhasedKnots = numPosSteps + std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
    // size_t numPhasedVars = 3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
    double max_force = 2. * model.mass * std::abs(model.gravity[2]);
    // auto initFootPosVals = Eigen::VectorXd(3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0));
    // auto initFootForceVals = Eigen::VectorXd(3 * numForceSteps + 6 * std::accumulate(forceKnotsPerSwing.begin(), forceKnotsPerSwing.end(), 0));

    auto initFootPosVals = random_uniform_vector(3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0), 0., 0.5);
    auto initFootForceVals = random_uniform_vector(3 * numForceSteps + 6 * std::accumulate(forceKnotsPerSwing.begin(), forceKnotsPerSwing.end(), 0), 0., 0.5);

    ifopt::Component::VecBound footPosBounds(3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0), ifopt::NoBound);
    ifopt::Component::VecBound footForceBounds(3 * numForceSteps + 6 * std::accumulate(forceKnotsPerSwing.begin(), forceKnotsPerSwing.end(), 0), ifopt::Bounds(-max_force, max_force));

    for (size_t i = 0; i < model.numFeet; ++i) {
        // Add initial and final positions for each foot.
        footPosBounds[0] = (i == 0 || i == 1) ? ifopt::Bounds(0.34, 0.34) : ifopt::Bounds(-0.34, -0.34);
        footPosBounds[1] = (i == 1 || i == 3) ? ifopt::Bounds(0.19, 0.19) : ifopt::Bounds(-0.19, -0.19);

        footPosBounds[footPosBounds.size() - 3] = (i == 0 || i == 1) ? ifopt::Bounds(0.34, 0.34) : ifopt::Bounds(-0.34, -0.34);
        footPosBounds[footPosBounds.size() - 2] = (i == 1 || i == 3) ? ifopt::Bounds(0.19, 0.19) : ifopt::Bounds(-0.19, -0.19);

        auto footPosVars = std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_POS + "_" + std::to_string(i), initFootPosVals, footPosBounds, phaseTimes, posKnotsPerSwing, trajopt::rspl::Phase::Stance);
        nlp.AddVariableSet(footPosVars);

        auto footForceVars = std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_FORCE + "_" + std::to_string(i), initFootForceVals, footForceBounds, phaseTimes, forceKnotsPerSwing, trajopt::rspl::Phase::Swing);
        nlp.AddVariableSet(footForceVars);
    }

    // // Add regular constraint sets.
    auto dynamConstr = std::make_shared<trajopt::Dynamics<trajopt::PhasedTrajectoryVars>>(model, numSamples, sampleTime);
    nlp.AddConstraintSet(dynamConstr);
    test_constr_jacobians(nlp, VarSetNames, dynamConstr, Tolerance, Viz);

    auto posAccConstr = std::make_shared<trajopt::AccelerationConstraints>(posVars);
    nlp.AddConstraintSet(posAccConstr);
    test_constr_jacobians(nlp, VarSetNames, posAccConstr, Tolerance, Viz);

    auto rotAccConstr = std::make_shared<trajopt::AccelerationConstraints>(rotVars);
    nlp.AddConstraintSet(rotAccConstr);
    test_constr_jacobians(nlp, VarSetNames, rotAccConstr, Tolerance, Viz);

    for (size_t i = 0; i < model.numFeet; ++i) {
        auto footPosVars = std::static_pointer_cast<trajopt::PhasedTrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(i)));
        auto footForceVars = std::static_pointer_cast<trajopt::PhasedTrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_FORCE + "_" + std::to_string(i)));

        auto phasedAccConstr = std::make_shared<trajopt::PhasedAccelerationConstraints>(footPosVars);
        nlp.AddConstraintSet(phasedAccConstr);
        test_constr_jacobians(nlp, VarSetNames, phasedAccConstr, Tolerance, Viz);

        // nlp.AddConstraintSet(std::make_shared<trajopt::FootPosTerrainConstraints>(footPosVars, terrain, numSamples, sampleTime));
        auto footTerrainDistancePhased = std::make_shared<trajopt::FootTerrainDistancePhased>(footPosVars, terrain, numPosSteps, 1, posKnotsPerSwing);
        nlp.AddConstraintSet(footTerrainDistancePhased);
        test_constr_jacobians(nlp, VarSetNames, footTerrainDistancePhased, Tolerance, Viz);

        auto footBodyDistancePhased = std::make_shared<trajopt::FootBodyDistancePhased>(model, posVars, rotVars, footPosVars, numSamples, sampleTime);
        nlp.AddConstraintSet(footBodyDistancePhased);
        test_constr_jacobians(nlp, VarSetNames, footBodyDistancePhased, Tolerance, Viz);

        auto frictionCone = std::make_shared<trajopt::FrictionCone<trajopt::PhasedTrajectoryVars>>(footForceVars, footPosVars, terrain, numSamples, sampleTime);
        nlp.AddConstraintSet(frictionCone);
        test_constr_jacobians(nlp, VarSetNames, frictionCone, Tolerance, Viz);
    }

    return 0;
}

//////// Jacobian Evaluation ////////
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
} ////// Jacobian Evaluation ////////
