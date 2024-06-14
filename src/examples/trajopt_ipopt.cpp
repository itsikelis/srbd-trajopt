#include <chrono>
#include <ctime>
#include <iostream>

#include <Eigen/Dense>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <trajopt/utils/utils.hpp>

#include <trajopt/utils/test_problems/create_implicit_nlp.hpp>
#include <trajopt/utils/test_problems/create_pendulum_nlp.hpp>
#include <trajopt/utils/test_problems/create_phased_nlp.hpp>
#include <trajopt/utils/visualisation.hpp>

static constexpr size_t numKnots = 10;
static constexpr size_t numSamples = 16;

static constexpr double totalTime = 0.5;

// Body position
static constexpr double initBodyPosX = 0.;
static constexpr double initBodyPosY = 0.;
static constexpr double initBodyPosZ = 0.5;

static constexpr double targetBodyPosX = 0.2;
static constexpr double targetBodyPosY = 0.;
static constexpr double targetBodyPosZ = 0.5;

int main()
{
    std::srand(std::time(0));

    trajopt::SingleRigidBodyDynamicsModel model;
    trajopt::init_model_anymal(model);
    // trajopt::init_model_biped(model);

    trajopt::TerrainGrid terrain(200, 200, 1., -100, -100, 100, 100);
    terrain.set_zero();

    Eigen::Vector3d initBodyPos = Eigen::Vector3d(initBodyPosX, initBodyPosY, initBodyPosZ + terrain.height(initBodyPosX, initBodyPosY));
    Eigen::Vector3d targetBodyPos = Eigen::Vector3d(targetBodyPosX, targetBodyPosY, targetBodyPosZ + terrain.height(targetBodyPosX, targetBodyPosY));

    ifopt::Problem nlp = create_phased_nlp(numKnots, numSamples, totalTime, initBodyPos, targetBodyPos, model, terrain);
    // ifopt::Problem nlp = create_implicit_nlp(numKnots, numSamples, totalTime, initBodyPos, targetBodyPos, model, terrain);
    // ifopt::Problem nlp = trajopt::create_pendulum_nlp();

    std::cout << "Solving.." << std::endl;
    ifopt::IpoptSolver ipopt;
    ipopt.SetOption("jacobian_approximation", "exact");
    ipopt.SetOption("max_cpu_time", 1e50);
    ipopt.SetOption("max_iter", static_cast<int>(1000));

    // Solve.
    auto tStart = std::chrono::high_resolution_clock::now();

    ipopt.Solve(nlp);
    nlp.PrintCurrent();
    const auto tEnd = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(tEnd - tStart);
    std::cout << "Wall clock time: " << duration.count() / 1000 << " seconds." << std::endl;

#if VIZ
    // For phased problem
    trajopt::visualise<trajopt::PhasedTrajectoryVars>(nlp, model, totalTime, 0.001);
    // For implicit problem
    // trajopt::visualise<trajopt::TrajectoryVars>(nlp, model, totalTime, 0.001);
#endif

    // Print variables.
    //
    // double dt = 0.;
    // for (size_t i = 0; i < numSamples; ++i) {
    //     std::cout << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::BODY_POS_TRAJECTORY))->trajectoryEval(dt, 0).transpose() << std::endl;
    //
    //     dt += sampleTime;
    // }
    //
    // std::cout << std::endl;
    //
    // for (size_t k = 0; k < 4; k++) {
    //
    //     dt = 0.;
    //     for (size_t i = 0; i < numSamples; ++i) {
    //         std::cout << "p_" << k << ": " << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(k)))->trajectoryEval(dt, 0).transpose() << std::endl;
    //
    //         std::cout << "v_" << k << ": " << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(k)))->trajectoryEval(dt, 1).transpose() << std::endl;
    //
    //         dt += sampleTime;
    //     }
    //
    //     std::cout << std::endl;
    //
    //     dt = 0.;
    //     for (size_t i = 0; i < numSamples; ++i) {
    //         std::cout << "f_" << k << ": " << std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_FORCE + "_" + std::to_string(k)))->trajectoryEval(dt, 0).transpose() << std::endl;
    //
    //         dt += sampleTime;
    //     }
    //
    //     std::cout << std::endl;
    // }

    return 0;
}
