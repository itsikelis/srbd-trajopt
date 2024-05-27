#include <chrono>
#include <ctime>

#include <Eigen/Dense>

#include <iostream>
#include <memory>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include "ifopt_sets/variables.hpp"
#include "utils/srbd.hpp"
#include "utils/terrain.hpp"
#include "utils/types.hpp"

#include "ifopt_sets/constraints/acceleration.hpp"
#include "ifopt_sets/constraints/dynamics.hpp"
#include "ifopt_sets/constraints/foot_constraints.hpp"

// Return 3D inertia tensor from 6D vector.
inline Eigen::Matrix3d InertiaTensor(double Ixx, double Iyy, double Izz, double Ixy, double Ixz, double Iyz);

void init_model(trajopt::SingleRigidBodyDynamicsModel& model);

ifopt::Component::VecBound fillBoundVector(Eigen::Vector3d init, Eigen::Vector3d target, size_t size);

int main()
{
    trajopt::SingleRigidBodyDynamicsModel model;
    init_model(model);

    // TODO: Handle zero grids.
    trajopt::TerrainGrid<200, 200> terrain(0.7, -100, -100, 100, 100);
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

    nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(posVars));
    nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(rotVars));

    // Add feet pos and force var sets.
    // size_t numPosSteps = 2;
    // size_t numForceSteps = 1;
    // Eigen::Vector3d phaseTimes = {0.2, 0.1, 0.2};
    // std::vector<size_t> posKnotsPerSwing = {3};
    // std::vector<size_t> forceKnotsPerSwing = {3, 3};

    // size_t numPhasedKnots = numPosSteps + std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
    // size_t numPhasedVars = 3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
    // double max_force = 2. * model.mass * std::abs(model.gravity[2]);
    auto initFootPosVals = Eigen::VectorXd(3 * 2 * numKnots);
    auto initFootForceVals = Eigen::VectorXd(3 * 2 * numKnots);

    // std::cout << initFootPosVals.rows() << " , " << initFootForceVals.rows() << std::endl;
    // std::cout << numPhasedKnots << " , " << initFootForceVals.rows() << std::endl;

    ifopt::Component::VecBound footPosBounds(6 * numKnots, ifopt::NoBound);
    ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::NoBound);
    // ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::Bounds(-max_force, max_force));

    for (size_t i = 0; i < model.numFeet; ++i) {
        auto footPosBoundsNew = footPosBounds;

        // Add initial and final positions for each foot.
        footPosBoundsNew[0] = (i == 0 || i == 1) ? ifopt::Bounds(0.34, 0.34) : ifopt::Bounds(-0.34, -0.34);
        footPosBoundsNew[1] = (i == 1 || i == 3) ? ifopt::Bounds(0.19, 0.19) : ifopt::Bounds(-0.19, -0.19);

        footPosBoundsNew[6 * numKnots - 3] = (i == 0 || i == 1) ? ifopt::Bounds(0.34, 0.34) : ifopt::Bounds(-0.34, -0.34);
        footPosBoundsNew[6 * numKnots - 2] = (i == 1 || i == 3) ? ifopt::Bounds(0.19, 0.19) : ifopt::Bounds(-0.19, -0.19);

        auto footPosVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_POS + "_" + std::to_string(i), initFootPosVals, polyTimes, footPosBoundsNew);
        nlp.AddVariableSet(footPosVars);

        nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(footPosVars));
        // nlp.AddConstraintSet(std::make_shared<trajopt::FootPosTerrainConstraints>(footPosVars, terrain, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootBodyPosConstraints>(model, posVars, rotVars, footPosVars, numSamples, sampleTime));

        auto footForceVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_FORCE + "_" + std::to_string(i), initFootForceVals, polyTimes, footForceBounds);
        nlp.AddVariableSet(footForceVars);

        nlp.AddConstraintSet(std::make_shared<trajopt::FrictionConeConstraints>(footForceVars, footPosVars, terrain, numSamples, sampleTime));

        nlp.AddConstraintSet(std::make_shared<trajopt::ImplicitContactConstraints>(footPosVars, footForceVars, terrain, numKnots));
    }

    std::cout << "Solving.." << std::endl;
    ifopt::IpoptSolver ipopt;
    ipopt.SetOption("jacobian_approximation", "exact");
    // ipopt.SetOption("jacobian_approximation", "finite-difference-values");
    ipopt.SetOption("max_cpu_time", 1e50);
    ipopt.SetOption("max_iter", static_cast<int>(200));

    // Solve.
    auto t_start = std::chrono::high_resolution_clock::now();

    ipopt.Solve(nlp);
    nlp.PrintCurrent();
    const auto t_end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(t_end - t_start);
    std::cout << "Wall clock time: " << duration.count() / 1000 << " seconds." << std::endl;

    // Get trajectory.
    // std::string var_set_name = trajopt::BODY_POS_TRAJECTORY;
    // std::string var_set_name = trajopt::BODY_ROT_TRAJECTORY;
    // std::string var_set_name = trajopt::FOOT_POS + "_0";
    // std::string var_set_name = trajopt::FOOT_FORCE + "_0";

    auto body_pos = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::BODY_POS_TRAJECTORY))->GetValues();

    for (auto& item : body_pos) {
        std::cout << item << ", ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    auto foot_pos = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_0"))->GetValues();

    for (auto& item : foot_pos) {
        std::cout << item << ", ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    auto foot_force = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_FORCE + "_0"))->GetValues();

    for (auto& item : foot_force) {
        std::cout << item << ", ";
    }
    std::cout << std::endl;

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
