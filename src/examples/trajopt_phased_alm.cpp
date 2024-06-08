#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>

#include <trajopt/alm/alm.hpp>

#include "alm_problem.hpp"

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <trajopt/ifopt_sets/cost/min_effort.hpp>

#include <trajopt/ifopt_sets/constraints/common/acceleration.hpp>
#include <trajopt/ifopt_sets/constraints/common/dynamics.hpp>
#include <trajopt/ifopt_sets/constraints/common/friction_cone.hpp>

#include <trajopt/ifopt_sets/variables/phased_trajectory_vars.hpp>

#include <trajopt/ifopt_sets/constraints/phased/foot_body_distance_phased.hpp>
#include <trajopt/ifopt_sets/constraints/phased/foot_terrain_distance_phased.hpp>
#include <trajopt/ifopt_sets/constraints/phased/phased_acceleration.hpp>

#include <trajopt/utils/types.hpp>
#include <trajopt/utils/utils.hpp>

static constexpr size_t numKnots = 5;
static constexpr size_t numSamples = 8;

static constexpr double totalTime = 0.2;

// Body position
static constexpr double initBodyPosX = 0.;
static constexpr double initBodyPosY = 0.;
static constexpr double initBodyPosZ = 0.5;

static constexpr double targetBodyPosX = 0.;
static constexpr double targetBodyPosY = 0.;

static constexpr double targetBodyPosZ = 0.5;

using fit_t = AlmProblem<>;
using algo_t = numopt::algo::AugmentedLagrangianMethod<fit_t>;

// fit_t::g_t finite_diff(fit_t& f, const fit_t::x_t& x, double eps = 1e-6)
// {
//     fit_t::g_t fdiff = fit_t::g_t::Zero(x.size());
//     for (int i = 0; i < x.size(); i++) {
//         fit_t::x_t xp = x;
//         xp[i] += eps;
//         fit_t::x_t xm = x;
//         xm[i] -= eps;
//         double fp = f.f(xp);
//         double fm = f.f(xm);
//
//         fdiff[i] = (fp - fm) / (2. * eps);
//     }
//
//     return fdiff;
// }

// fit_t::mat_t finite_diff_hessian(fit_t& f, const fit_t::x_t& x, double eps = 1e-6)
// {
//     fit_t::mat_t fdiff = fit_t::mat_t::Zero(x.size(), x.size());
//     for (int i = 0; i < x.size(); i++) {
//         fit_t::x_t xp = x;
//         xp[i] += eps;
//         fit_t::x_t xm = x;
//         xm[i] -= eps;
//
//         fit_t::g_t fp = f.df(xp);
//         fit_t::g_t fm = f.df(xm);
//         fdiff.row(i) = (fp - fm) / (2. * eps);
//     }
//
//     return fdiff;
// }

// fit_t::mat_t finite_diff_dc(fit_t& f, const fit_t::x_t& x, double eps = 1e-6)
// {
//     fit_t::mat_t fdiff = fit_t::mat_t::Zero(f.dim_eq() + f.dim_ineq(), f.dim());
//     for (int i = 0; i < x.size(); i++) {
//         fit_t::x_t xp = x;
//         xp[i] += eps;
//         fit_t::x_t xm = x;
//         xm[i] -= eps;
//
//         fit_t::x_t fp = f.c(xp);
//         fit_t::x_t fm = f.c(xm);
//         fdiff.col(i) = (fp - fm) / (2. * eps);
//     }
//
//     return fdiff;
// }

int main()
{
    std::srand(std::time(0));

    trajopt::SingleRigidBodyDynamicsModel model;
    trajopt::init_model_anymal(model);

    trajopt::TerrainGrid terrain(200, 200, 1., -100, -100, 100, 100);
    std::vector<double> grid;
    grid.resize(200 * 200);
    for (auto& item : grid) {
        item = 0.;
    }
    terrain.set_grid(grid);

    // Add variable sets.
    ifopt::Problem nlp;

    double sampleTime = totalTime / static_cast<double>(numSamples - 1.);
    Eigen::VectorXd polyTimes = Eigen::VectorXd::Zero(numKnots - 1);
    for (size_t i = 0; i < static_cast<size_t>(polyTimes.size()); ++i) {
        polyTimes[i] = totalTime / static_cast<double>(numKnots - 1);
    }

    // Add body pos and rot var sets.
    Eigen::Vector3d initBodyPos = Eigen::Vector3d(initBodyPosX, initBodyPosY, initBodyPosZ + terrain.height(initBodyPosX, initBodyPosY));
    Eigen::Vector3d targetBodyPos = Eigen::Vector3d(targetBodyPosX, targetBodyPosY, targetBodyPosZ + terrain.height(targetBodyPosX, targetBodyPosY));
    ifopt::Component::VecBound bodyPosBounds = trajopt::fillBoundVector(initBodyPos, targetBodyPos, ifopt::NoBound, 6 * numKnots);
    Eigen::VectorXd initBodyPosVals = Eigen::VectorXd::Zero(3 * 2 * numKnots);

    auto posVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_POS_TRAJECTORY, initBodyPosVals, polyTimes, bodyPosBounds);
    nlp.AddVariableSet(posVars);

    Eigen::Vector3d initRotPos = Eigen::Vector3d::Zero();
    Eigen::Vector3d targetRotPos = Eigen::Vector3d::Zero();
    ifopt::Component::VecBound bodyRotBounds = trajopt::fillBoundVector(initRotPos, targetRotPos, ifopt::NoBound, 6 * numKnots);
    Eigen::VectorXd initBodyRotVals = Eigen::VectorXd::Zero(3 * 2 * numKnots);
    auto rotVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::BODY_ROT_TRAJECTORY, initBodyRotVals, polyTimes, bodyRotBounds);
    nlp.AddVariableSet(rotVars);

    // // Add regular constraint sets.
    nlp.AddConstraintSet(std::make_shared<trajopt::Dynamics<trajopt::PhasedTrajectoryVars>>(model, numSamples, sampleTime));

    nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(posVars));
    nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(rotVars));

    // Add feet pos and force var sets.
    size_t numPosSteps = 2;
    size_t numForceSteps = 1;
    Eigen::Vector3d phaseTimes = {0.2, 0.1, 0.2};
    std::vector<size_t> posKnotsPerSwing = {1};
    std::vector<size_t> forceKnotsPerSwing = {5, 5};

    // size_t numPhasedKnots = numPosSteps + std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
    // size_t numPhasedVars = 3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0);
    double max_force = 2. * model.mass * std::abs(model.gravity[2]);
    Eigen::VectorXd initFootPosVals = Eigen::VectorXd::Zero(3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0));
    Eigen::VectorXd initFootForceVals = Eigen::VectorXd::Zero(3 * numForceSteps + 6 * std::accumulate(forceKnotsPerSwing.begin(), forceKnotsPerSwing.end(), 0));

    // std::cout << initFootPosVals.rows() << " , " << initFootForceVals.rows() << std::endl;
    // std::cout << numPhasedKnots << " , " << initFootForceVals.rows() << std::endl;

    ifopt::Component::VecBound footPosBounds(3 * numPosSteps + 6 * std::accumulate(posKnotsPerSwing.begin(), posKnotsPerSwing.end(), 0), ifopt::NoBound);
    ifopt::Component::VecBound footForceBounds(3 * numForceSteps + 6 * std::accumulate(forceKnotsPerSwing.begin(), forceKnotsPerSwing.end(), 0), ifopt::Bounds(-max_force, max_force));

    // std::vector<std::make_shared<trajopt::PhasedTrajectoryVars>> feetForces, feetPos;

    for (size_t i = 0; i < model.numFeet; ++i) {
        auto footPosVars = std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_POS + "_" + std::to_string(i), initFootPosVals, footPosBounds, phaseTimes, posKnotsPerSwing, trajopt::rspl::Phase::Stance);
        nlp.AddVariableSet(footPosVars);

        nlp.AddConstraintSet(std::make_shared<trajopt::PhasedAccelerationConstraints>(footPosVars));
        // nlp.AddConstraintSet(std::make_shared<trajopt::FootPosTerrainConstraints>(footPosVars, terrain, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootTerrainDistancePhased>(footPosVars, terrain, numPosSteps, 1, posKnotsPerSwing));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootBodyDistancePhased>(model, posVars, rotVars, footPosVars, numSamples, sampleTime));

        auto footForceVars = std::make_shared<trajopt::PhasedTrajectoryVars>(trajopt::FOOT_FORCE + "_" + std::to_string(i), initFootForceVals, footForceBounds, phaseTimes, forceKnotsPerSwing, trajopt::rspl::Phase::Swing);
        nlp.AddVariableSet(footForceVars);

        nlp.AddConstraintSet(std::make_shared<trajopt::FrictionCone<trajopt::PhasedTrajectoryVars>>(footForceVars, footPosVars, terrain, numSamples, sampleTime));

        // nlp.AddCostSet(std::make_shared<trajopt::MinEffort<trajopt::PhasedTrajectoryVars>>(footPosVars, numKnots));
    }

    fit_t fit(nlp);
    algo_t::Params params;
    params.dim = fit.dim();
    params.dim_eq = fit.dim_eq();
    params.dim_ineq = fit.dim_ineq();

    params.initial_x = algo_t::x_t::Zero(fit.dim()); // Random(fit_t::dim);
    // Initialization
    // params.initial_x.head(fit_t::T * fit_t::Ad) = algo_t::x_t::Constant(fit_t::T * fit_t::Ad, fit_t::m * fit_t::g / 2.);
    // for (unsigned int i = 0; i < (fit_t::T - 1); i++) {
    //     params.initial_x.segment(fit_t::T * fit_t::Ad + i * fit_t::D, fit_t::D) = fit.x0 + (fit.xN - fit.x0) * (i + 1) / static_cast<double>(fit_t::T);
    // }
    params.initial_lambda = algo_t::x_t::Zero(fit.dim_eq() + fit.dim_ineq());
    params.initial_rho = 100.;
    params.rho_a = 10.;

    algo_t algo(params, fit);

    std::cout << "Number of variables: " << fit.dim() << std::endl;
    std::cout << "Number of equality constraints: " << fit.dim_eq() << std::endl;
    std::cout << "Number of inequality constraints: " << fit.dim_ineq() << std::endl;

    // std::cout << "f: " << fit.f(nlp.GetVariableValues()) << std::endl;
    // std::cout << "df: " << fit.df(nlp.GetVariableValues()) << std::endl;
    // std::cout << "c: " << fit.c(nlp.GetVariableValues()).transpose() << std::endl;
    // std::cout << "dc: " << fit.dc(nlp.GetVariableValues()) << std::endl;

    // auto df = fit.df(algo.x());
    // auto df_finite = finite_diff(fit, algo.x());
    // std::cout << "df: " << (df - df_finite).norm() << std::endl;
    // auto ddf = fit.ddf(algo.x());
    // auto ddf_finite = finite_diff_hessian(fit, algo.x());
    // std::cout << "ddf: " << (ddf - ddf_finite).norm() << std::endl;
    // auto dc = fit.dc(algo.x());
    // auto dc_finite = finite_diff_dc(fit, algo.x());
    // std::cout << "dc: " << (dc - dc_finite).norm() << std::endl;

    // std::cout << "f: " << fit.f(algo.x()) << std::endl;
    // std::cout << "df: " << fit.df(algo.x()) << std::endl;
    // std::cout << "df: " << finite_diff(fit, algo.x()) << std::endl;
    // std::cout << "ddf:\n" << fit.ddf(algo.x()) << std::endl;
    // std::cout << "ddf:\n" << finite_diff_hessian(fit, algo.x()) << std::endl;
    // std::cout << "c: " << fit.c(algo.x()) << std::endl;
    // std::cout << "dc:\n" << fit.dc(algo.x()) << std::endl;

    // Solve.
    auto tStart = std::chrono::high_resolution_clock::now();

    unsigned int iters = 200;
    for (unsigned int i = 0; i < iters; ++i) {
        auto log = algo.step();
        // std::cout << algo.x().transpose() << " -> " << fit.f(algo.x()) << std::endl;
        std::cout << "f: " << log.f << std::endl;
        std::cout << "c: " << log.c.norm() << std::endl;
        // std::cout << " " << log.func_evals << " " << log.cons_evals << " " << log.grad_evals << " " << log.hessian_evals << " " << log.cjac_evals << std::endl;
    }

    nlp.PrintCurrent();
    nlp.SetVariables(algo.x().data());
    nlp.PrintCurrent();
    const auto tEnd = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(tEnd - tStart);
    std::cout << "Wall clock time: " << duration.count() / 1000 << " seconds." << std::endl;

    // fit_t::x_t x = fit.x0;
    // for (size_t k = 0; k < fit_t::T; k++) {
    //     std::cout << x.transpose() << " with " << algo.x().segment(k * fit_t::Ad, fit_t::Ad).transpose() << " --> ";
    //     if (k < fit_t::T - 1)
    //         x = algo.x().segment(fit_t::T * fit_t::Ad + k * fit_t::D, fit_t::D);
    //     else
    //         x = fit.xN;
    //     std::cout << x.transpose() << std::endl;
    // }

    return 0;
}
