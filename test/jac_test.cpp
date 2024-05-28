#include <ctime>

#include <Eigen/Dense>

#include <iomanip>
#include <memory>
#include <numeric>
#include <random>

#include <ifopt/constraint_set.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include "../include/ifopt_sets/constraints/acceleration.hpp"
#include "../include/ifopt_sets/constraints/dynamics.hpp"
#include "../include/ifopt_sets/constraints/foot_constraints.hpp"

#include "../include/ifopt_sets/variables.hpp"
#include "../include/srbd/srbd.hpp"
#include "../include/utils/terrain.hpp"
#include "../include/utils/types.hpp"

// Return 3D inertia tensor from 6D vector.
inline Eigen::Matrix3d InertiaTensor(double Ixx, double Iyy, double Izz, double Ixy, double Ixz, double Iyz);

void init_model(trajopt::SingleRigidBodyDynamicsModel& model);

ifopt::Component::VecBound fillBoundVector(Eigen::Vector3d init, Eigen::Vector3d target, size_t size);

void test_jacobians(const ifopt::Problem& nlp, const std::string& var_set_name, const std::shared_ptr<ifopt::ConstraintSet>& myConstr);

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
    init_model(model);

    // Create a Terrain Model (Select from predefined ones)
    // trajopt::Terrain terrain("");
    trajopt::TerrainGrid<200, 200> terrain(0.7, 0, 0, 200, 200);

    // Add variable sets.
    ifopt::Problem nlp;

    static constexpr size_t numKnots = 30;
    size_t numSamples = 30;

    double totalTime = 0.5;
    double sampleTime = totalTime / static_cast<double>(numSamples - 1.);
    auto polyTimes = Eigen::VectorXd(numKnots - 1);
    for (size_t i = 0; i < static_cast<size_t>(polyTimes.size()); ++i) {
        polyTimes[i] = totalTime / static_cast<double>(numKnots - 1);
    }

    // Add body pos and rot var sets.
    Eigen::Vector3d initBodyPos = Eigen::Vector3d(0., 0., 0.5);
    Eigen::Vector3d targetBodyPos = Eigen::Vector3d(0., 0., 0.5 + terrain.height(0., 0.));
    ifopt::Component::VecBound bodyPosBounds = fillBoundVector(initBodyPos, targetBodyPos, 6 * numKnots);
    auto initBodyPosVals = Eigen::VectorXd(3 * 2 * numKnots);

    auto posVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_POS_TRAJECTORY, initBodyPosVals, polyTimes, bodyPosBounds);
    nlp.AddVariableSet(posVars);

    Eigen::Vector3d initRotPos = Eigen::Vector3d::Zero();
    Eigen::Vector3d targetRotPos = Eigen::Vector3d::Zero();
    ifopt::Component::VecBound bodyRotBounds = fillBoundVector(initRotPos, targetRotPos, 6 * numKnots);
    auto initBodyRotVals = Eigen::VectorXd(3 * 2 * numKnots);
    auto rotVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_ROT_TRAJECTORY, initBodyRotVals, polyTimes, bodyRotBounds);
    nlp.AddVariableSet(rotVars);

    // // Add regular constraint sets.
    auto dynamConstr = std::make_shared<trajopt::DynamicsConstraint>(model, numSamples, sampleTime);
    nlp.AddConstraintSet(dynamConstr);

    auto posAccConstr = std::make_shared<trajopt::AccelerationConstraints>(posVars);
    nlp.AddConstraintSet(posAccConstr);
    auto rotAccConstr = std::make_shared<trajopt::AccelerationConstraints>(rotVars);
    nlp.AddConstraintSet(rotAccConstr);

    // Add feet pos and force var sets.
    size_t numPosSteps = 2;
    size_t numForceSteps = 1;
    Eigen::Vector3d phaseTimes = {0.2, 0.1, 0.2};
    std::vector<size_t> posKnotsPerSwing = {3};
    std::vector<size_t> forceKnotsPerSwing = {3, 3};

    // size_t numPhasedKnots = numPosSteps + std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
    // size_t numPhasedVars = 3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
    double max_force = 2. * model.mass * std::abs(model.gravity[2]);
    // auto initFootPosVals = Eigen::VectorXd(3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0));
    // auto initFootForceVals = Eigen::VectorXd(3 * numForceSteps + 6 * std::accumulate(forceKnotsPerSwing.begin(), forceKnotsPerSwing.end(), 0));

    auto initFootPosVals = random_uniform_vector(3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0), 0., 0.5);
    auto initFootForceVals = random_uniform_vector(3 * numForceSteps + 6 * std::accumulate(forceKnotsPerSwing.begin(), forceKnotsPerSwing.end(), 0), 0., 0.5);

    // std::cout << initFootPosVals.rows() << " , " << initFootForceVals.rows() << std::endl;
    // std::cout << numPhasedKnots << " , " << initFootForceVals.rows() << std::endl;

    ifopt::Component::VecBound footPosBounds(3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0), ifopt::NoBound);
    ifopt::Component::VecBound footForceBounds(3 * numForceSteps + 6 * std::accumulate(forceKnotsPerSwing.begin(), forceKnotsPerSwing.end(), 0), ifopt::Bounds(-max_force, max_force));

    // Add first foot.
    auto footPosVars0 = std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_POS + "_0", initFootPosVals, footPosBounds, phaseTimes, posKnotsPerSwing, rspl::Phase::Stance);
    nlp.AddVariableSet(footPosVars0);

    auto footPosAccConstr = std::make_shared<trajopt::PhasedAccelerationConstraints>(footPosVars0);
    nlp.AddConstraintSet(footPosAccConstr);

    // auto footPosTerrainConstr = std::make_shared<trajopt::FootPosTerrainConstraints>(footPosVars0, terrain, numSamples, sampleTime);
    auto footPosTerrainConstr = std::make_shared<trajopt::FootPosTerrainConstraints>(footPosVars0, terrain, numPosSteps, 1, posKnotsPerSwing);
    nlp.AddConstraintSet(footPosTerrainConstr);

    auto footBodyPosConstr = std::make_shared<trajopt::FootBodyPosConstraints>(model, posVars, rotVars, footPosVars0, numSamples, sampleTime);
    nlp.AddConstraintSet(footBodyPosConstr);

    auto footForceVars0 = std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_FORCE + "_0", initFootForceVals, footForceBounds, phaseTimes, forceKnotsPerSwing, rspl::Phase::Swing);
    nlp.AddVariableSet(footForceVars0);

    auto footForceFrictionConstr = std::make_shared<trajopt::FrictionConeConstraints>(footForceVars0, footPosVars0, terrain, numSamples, sampleTime);
    nlp.AddConstraintSet(footForceFrictionConstr);

    // Add rest of feet.
    for (size_t i = 1; i < model.numFeet; ++i) {
        auto footPosVars = std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_POS + "_" + std::to_string(i), initFootPosVals, footPosBounds, phaseTimes, posKnotsPerSwing, rspl::Phase::Stance);
        nlp.AddVariableSet(footPosVars);

        nlp.AddConstraintSet(std::make_shared<trajopt::PhasedAccelerationConstraints>(footPosVars));
        // nlp.AddConstraintSet(std::make_shared<trajopt::FootPosTerrainConstraints>(footPosVars, terrain, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootPosTerrainConstraints>(footPosVars, terrain, numPosSteps, 1, posKnotsPerSwing));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootBodyPosConstraints>(model, posVars, rotVars, footPosVars, numSamples, sampleTime));

        auto footForceVars = std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_FORCE + "_" + std::to_string(i), initFootForceVals, footForceBounds, phaseTimes, forceKnotsPerSwing, rspl::Phase::Swing);
        nlp.AddVariableSet(footForceVars);

        nlp.AddConstraintSet(std::make_shared<trajopt::FrictionConeConstraints>(footForceVars, footPosVars, terrain, numSamples, sampleTime));
    }

    // Solve
    // std::cout << "Solving.." << std::endl;
    // ifopt::IpoptSolver ipopt;
    // ipopt.SetOption("jacobian_approximation", "exact");
    // ipopt.SetOption("jacobian_approximation", "finite-difference-values");
    // ipopt.SetOption("max_cpu_time", 1e50);
    // ipopt.SetOption("max_iter", static_cast<int>(1000));

    // Solve.
    // ipopt.Solve(nlp);
    // nlp.PrintCurrent();

    // std::string var_set_name = trajopt::BODY_POS_TRAJECTORY;
    // std::string var_set_name = trajopt::BODY_ROT_TRAJECTORY;
    std::string var_set_name = trajopt::FOOT_POS + "_0";
    // std::string var_set_name = trajopt::FOOT_FORCE + "_0";

    // auto constr_set = dynamConstr;
    // auto constr_set = posAccConstr;
    // auto constr_set = rotAccConstr;

    // auto constr_set = footPosAccConstr;
    // auto constr_set = footPosAccConstr;
    auto constr_set = footPosTerrainConstr;
    // auto constr_set = footBodyPosConstr;
    // auto constr_set = footForceFrictionConstr;

    test_jacobians(nlp, var_set_name, constr_set);

    return 0;
}

inline Eigen::Matrix3d InertiaTensor(double Ixx, double Iyy, double Izz, double Ixy, double Ixz, double Iyz)
{
    Eigen::Matrix3d I;
    I << Ixx, -Ixy, -Ixz, -Ixy, Iyy, -Iyz, -Ixz, -Iyz, Izz;
    return I;
}

void init_model(trajopt::SingleRigidBodyDynamicsModel& model)
{
    //   Anymal characteristics
    Eigen::Matrix3d inertia = InertiaTensor(0.88201174, 1.85452968, 1.97309185, 0.00137526, 0.00062895, 0.00018922);
    const double m_b = 30.4213964625;
    const double x_nominal_b = 0.34;
    const double y_nominal_b = 0.19;
    const double z_nominal_b = -0.42;

    const double dx = 0.15;
    const double dy = 0.1;
    const double dz = 0.1;

    model.mass = m_b;
    model.inertia = inertia;
    model.numFeet = 4;

    // Right fore
    model.feetPoses.push_back(Eigen::Vector3d(x_nominal_b, -y_nominal_b, 0.));
    model.feetMinBounds.push_back(Eigen::Vector3d(x_nominal_b - dx, -y_nominal_b - dy, z_nominal_b - dz));
    model.feetMaxBounds.push_back(Eigen::Vector3d(x_nominal_b + dx, -y_nominal_b + dy, z_nominal_b + dz));

    // Left fore
    model.feetPoses.push_back(Eigen::Vector3d(x_nominal_b, y_nominal_b, 0.));
    model.feetMinBounds.push_back(Eigen::Vector3d(x_nominal_b - dx, y_nominal_b - dy, z_nominal_b - dz));
    model.feetMaxBounds.push_back(Eigen::Vector3d(x_nominal_b + dx, y_nominal_b + dy, z_nominal_b + dz));

    // Right hind
    model.feetPoses.push_back(Eigen::Vector3d(-x_nominal_b, -y_nominal_b, 0.));
    model.feetMinBounds.push_back(Eigen::Vector3d(-x_nominal_b - dx, -y_nominal_b - dy, z_nominal_b - dz));
    model.feetMaxBounds.push_back(Eigen::Vector3d(-x_nominal_b + dx, -y_nominal_b + dy, z_nominal_b + dz));

    // Left hind
    model.feetPoses.push_back(Eigen::Vector3d(-x_nominal_b, y_nominal_b, 0.));
    model.feetMinBounds.push_back(Eigen::Vector3d(-x_nominal_b - dx, y_nominal_b - dy, z_nominal_b - dz));
    model.feetMaxBounds.push_back(Eigen::Vector3d(-x_nominal_b + dx, y_nominal_b + dy, z_nominal_b + dz));
}

ifopt::Component::VecBound fillBoundVector(Eigen::Vector3d init, Eigen::Vector3d target, size_t size)
{
    ifopt::Component::VecBound bounds(size, ifopt::NoBound);
    bounds.at(0) = ifopt::Bounds(init[0], init[0]);
    bounds.at(1) = ifopt::Bounds(init[1], init[1]);
    bounds.at(2) = ifopt::Bounds(init[2], init[2]);
    bounds.at(3) = ifopt::Bounds(init[0], init[0]);
    bounds.at(4) = ifopt::Bounds(init[1], init[1]);
    bounds.at(5) = ifopt::Bounds(init[2], init[2]);
    bounds.at(size - 6) = ifopt::Bounds(target[0], target[0]);
    bounds.at(size - 5) = ifopt::Bounds(target[1], target[1]);
    bounds.at(size - 4) = ifopt::Bounds(target[2], target[2]);
    bounds.at(size - 3) = ifopt::Bounds(target[0], target[0]);
    bounds.at(size - 2) = ifopt::Bounds(target[1], target[1]);
    bounds.at(size - 1) = ifopt::Bounds(target[2], target[2]);

    return bounds;
}

/////// Jacobian Evaluation ////////
void test_jacobians(const ifopt::Problem& nlp, const std::string& var_set_name, const std::shared_ptr<ifopt::ConstraintSet>& myConstr)
{
    // std::string name = trajopt::BODY_POS_TRAJECTORY;
    // std::string name = trajopt::BODY_ROT_TRAJECTORY;
    // std::string name = trajopt::PAW_FORCES + "_0";
    // std::string name = trajopt::PAW_POS + "_0";
    auto myVars = nlp.GetOptVariables()->GetComponent(var_set_name);

    // Possible Constraint Sets to test:
    // dynamConstr
    // frictionConstr
    // phasedAccConstr
    // pawPosTerrainConstr
    // pawBodyPosConstr

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
    std::cout << "Norm of difference: " << abs((myJac - jacDense).norm()) << std::endl;

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
