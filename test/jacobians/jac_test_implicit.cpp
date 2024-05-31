#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#include <Eigen/Dense>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <ifopt_sets/variables/trajectory_vars.hpp>
#include <srbd/srbd.hpp>
#include <terrain/terrain_grid.hpp>

#include <ifopt_sets/constraints/common/acceleration.hpp>

#include <ifopt_sets/constraints/contact_implicit/dynamics_implicit.hpp>
#include <ifopt_sets/constraints/contact_implicit/foot_body_distance_implicit.hpp>
#include <ifopt_sets/constraints/contact_implicit/foot_terrain_distance_implicit.hpp>
#include <ifopt_sets/constraints/contact_implicit/friction_cone_implicit.hpp>
#include <ifopt_sets/constraints/contact_implicit/implicit_contact.hpp>
#include <ifopt_sets/constraints/contact_implicit/implicit_velocity.hpp>

#include <utils/types.hpp>
#include <utils/utils.hpp>

// Return 3D inertia tensor from 6D vector.

void test_jacobians(const ifopt::Problem& nlp, const std::vector<std::string>& var_set_names, const std::vector<std::shared_ptr<ifopt::ConstraintSet>>& constr_sets, bool viz);

Eigen::VectorXd random_uniform_vector(size_t rows, double lower, double upper)
{
    std::uniform_real_distribution<double> unif(lower, upper);
    std::default_random_engine re;
    re.seed(time(0));

    Eigen::VectorXd rand(rows);
    for (size_t i = 0; i < rows; ++i) {
        rand[i] = unif(re);
    }

    return rand;
}

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
    auto polyTimes = Eigen::VectorXd(numKnots - 1);
    for (size_t i = 0; i < static_cast<size_t>(polyTimes.size()); ++i) {
        polyTimes[i] = totalTime / static_cast<double>(numKnots - 1);
    }

    // Add body pos and rot var sets.
    Eigen::Vector3d initBodyPos = Eigen::Vector3d(0., 0., 0.5 + terrain.height(0., 0.));
    Eigen::Vector3d targetBodyPos = Eigen::Vector3d(0., 0., 0.5 + terrain.height(0., 0.));
    ifopt::Component::VecBound bodyPosBounds = trajopt::fillBoundVector(initBodyPos, targetBodyPos, ifopt::NoBound, 6 * numKnots);
    auto initBodyPosVals = random_uniform_vector(3 * 2 * numKnots, -1., 1.);

    auto posVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_POS_TRAJECTORY, initBodyPosVals, polyTimes, bodyPosBounds);
    nlp.AddVariableSet(posVars);

    Eigen::Vector3d initRotPos = Eigen::Vector3d::Zero();
    Eigen::Vector3d targetRotPos = Eigen::Vector3d::Zero();
    ifopt::Component::VecBound bodyRotBounds = trajopt::fillBoundVector(initRotPos, targetRotPos, ifopt::NoBound, 6 * numKnots);
    auto initBodyRotVals = random_uniform_vector(3 * 2 * numKnots, -1., 1.);
    ;
    auto rotVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_ROT_TRAJECTORY, initBodyRotVals, polyTimes, bodyRotBounds);
    nlp.AddVariableSet(rotVars);

    // // Add regular constraint sets.
    auto dynamConstr = std::make_shared<trajopt::DynamicsImplicit>(model, numSamples, sampleTime);
    nlp.AddConstraintSet(dynamConstr);

    auto posAccConstr = std::make_shared<trajopt::AccelerationConstraints>(posVars);
    nlp.AddConstraintSet(posAccConstr);
    auto rotAccConstr = std::make_shared<trajopt::AccelerationConstraints>(rotVars);
    nlp.AddConstraintSet(rotAccConstr);

    // Feet
    auto initFootPosVals = random_uniform_vector(3 * 2 * numKnots, -1., 1.);
    auto initFootForceVals = random_uniform_vector(3 * 2 * numKnots, -1., 1.);

    ifopt::Component::VecBound footPosBounds(6 * numKnots, ifopt::NoBound);
    ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::NoBound);
    // ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::Bounds(-max_force, max_force));

    footPosBounds[0] = ifopt::Bounds(0.34, 0.34);
    footPosBounds[1] = ifopt::Bounds(0.19, 0.19);

    footPosBounds[6 * numKnots - 3] = ifopt::Bounds(0.34, 0.34);
    footPosBounds[6 * numKnots - 2] = ifopt::Bounds(0.19, 0.19);

    // Add first foot.
    auto footPosVars0 = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_POS + "_0", initFootPosVals, polyTimes, footPosBounds);
    nlp.AddVariableSet(footPosVars0);

    auto footPosAccConstr = std::make_shared<trajopt::AccelerationConstraints>(footPosVars0);
    nlp.AddConstraintSet(footPosAccConstr);

    auto footBodyPosConstr = std::make_shared<trajopt::FootBodyDistanceImplicit>(model, posVars, rotVars, footPosVars0, numSamples, sampleTime);
    nlp.AddConstraintSet(footBodyPosConstr);

    auto footTerrainPosConstr = std::make_shared<trajopt::FootTerrainDistanceImplicit>(footPosVars0, terrain, numKnots);
    nlp.AddConstraintSet(footTerrainPosConstr);

    auto footForceVars0 = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_FORCE + "_0", initFootForceVals, polyTimes, footForceBounds);
    nlp.AddVariableSet(footForceVars0);

    auto footForceFrictionConstr = std::make_shared<trajopt::FrictionConeImplicit>(footForceVars0, footPosVars0, terrain, numSamples, sampleTime);
    nlp.AddConstraintSet(footForceFrictionConstr);

    auto implicitContactConstr = std::make_shared<trajopt::ImplicitContactConstraints>(footPosVars0, footForceVars0, terrain, numSamples, sampleTime);
    nlp.AddConstraintSet(implicitContactConstr);

    auto implicitVelocityConstr = std::make_shared<trajopt::ImplicitVelocityConstraints>(footPosVars0, footForceVars0, terrain, numSamples, sampleTime);
    nlp.AddConstraintSet(implicitVelocityConstr);

    // Add rest of feet.
    for (size_t i = 1; i < model.numFeet; ++i) {
        // Add initial and final positions for each foot.
        footPosBounds[0] = (i == 0 || i == 1) ? ifopt::Bounds(0.34, 0.34) : ifopt::Bounds(-0.34, -0.34);
        footPosBounds[1] = (i == 1 || i == 3) ? ifopt::Bounds(0.19, 0.19) : ifopt::Bounds(-0.19, -0.19);

        footPosBounds[6 * numKnots - 3] = (i == 0 || i == 1) ? ifopt::Bounds(0.34, 0.34) : ifopt::Bounds(-0.34, -0.34);
        footPosBounds[6 * numKnots - 2] = (i == 1 || i == 3) ? ifopt::Bounds(0.19, 0.19) : ifopt::Bounds(-0.19, -0.19);

        auto footPosVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_POS + "_" + std::to_string(i), initFootPosVals, polyTimes, footPosBounds);
        nlp.AddVariableSet(footPosVars);

        nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(footPosVars));
        // nlp.AddConstraintSet(std::make_shared<trajopt::FootPosTerrainConstraints>(footPosVars, terrain, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootBodyDistanceImplicit>(model, posVars, rotVars, footPosVars, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootTerrainDistanceImplicit>(footPosVars, terrain, numKnots));

        auto footForceVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_FORCE + "_" + std::to_string(i), initFootForceVals, polyTimes, footForceBounds);
        nlp.AddVariableSet(footForceVars);

        nlp.AddConstraintSet(std::make_shared<trajopt::FrictionConeImplicit>(footForceVars, footPosVars, terrain, numSamples, sampleTime));

        nlp.AddConstraintSet(std::make_shared<trajopt::ImplicitContactConstraints>(footPosVars, footForceVars, terrain, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::ImplicitVelocityConstraints>(footPosVars, footForceVars, terrain, numSamples, sampleTime));
    }

    std::vector<std::string> var_set_names = {trajopt::BODY_POS_TRAJECTORY, trajopt::BODY_ROT_TRAJECTORY, trajopt::FOOT_POS + "_0", trajopt::FOOT_FORCE + "_0"};
    std::vector<std::shared_ptr<ifopt::ConstraintSet>> constr_sets = {dynamConstr, posAccConstr, rotAccConstr, footPosAccConstr, footBodyPosConstr, footForceFrictionConstr, implicitContactConstr, implicitVelocityConstr, footTerrainPosConstr};

    test_jacobians(nlp, var_set_names, constr_sets, true);

    return 0;
}

/////// Jacobian Evaluation ////////
void test_jacobians(const ifopt::Problem& nlp, const std::vector<std::string>& var_set_names, const std::vector<std::shared_ptr<ifopt::ConstraintSet>>& constr_sets, bool viz)
{
    for (auto& var_set_name : var_set_names) {
        auto myVars = nlp.GetOptVariables()->GetComponent(var_set_name);

        for (auto& myConstr : constr_sets) {

            // Get Calculated Jacobian.
            auto jac = Eigen::SparseMatrix<double, Eigen::RowMajor>(myConstr->GetRows(), myVars->GetRows());
            myConstr->FillJacobianBlock(var_set_name, jac);
            auto jacDense = Eigen::MatrixXd(jac);

            // std::cout << jacDense.rows() << ", " << jacDense.cols() << std::endl;

            // Calculate Jacobian using finite differences.
            Eigen::MatrixXd myJac(myConstr->GetRows(), myVars->GetRows());
            double eps = 1e-6;
            Eigen::VectorXd vals = myVars->GetValues();

            for (int colIdx = 0; colIdx < myVars->GetRows(); ++colIdx) {
                Eigen::VectorXd vals_p = vals;
                Eigen::VectorXd vals_m = vals;
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
            if (err > 1e-4) {
                std::cout << "Variable Set: " << var_set_name << ", \t";
                std::cout << "Constraint Set: " << myConstr->GetName() << ", \t";
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
}
