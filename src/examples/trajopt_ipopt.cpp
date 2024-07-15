#include <ctime>

#include <Eigen/Core>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <trajopt/utils/test_problems/create_implicit_nlp.hpp>
#include <trajopt/utils/test_problems/create_pendulum_nlp.hpp>
#include <trajopt/utils/test_problems/create_phased_nlp.hpp>
#include <trajopt/utils/utils.hpp>

#include <trajopt/srbd_trajopt.hpp>

#include <trajopt/robo_spline/types.hpp>

#include <trajopt/utils/gait_profiler.hpp>

#include <trajopt/utils/visualisation.hpp>

int main()
{
    std::srand(std::time(0));

    trajopt::SingleRigidBodyDynamicsModel model;
    // trajopt::init_model_anymal(model);
    trajopt::init_model_biped(model);

    trajopt::TerrainGrid terrain(20, 20, 1., -10, -10, 10, 10);
    // terrain.SetZero();

    trajopt::TerrainGrid::Grid grid;
    grid.resize(20 * 20);
    for (auto& g : grid) {
        g = 0.;
    }
    for (size_t i = 0; i < 20; i++) {
        for (size_t j = 0; j < 20; j++) {
            grid.at(i * 20 + j) = -static_cast<double>(i) / 10.;
        }
    }
    terrain.SetGrid(grid);

    // trajopt::TerrainGrid::Grid grid;
    //
    // // Create a random grid for the terrain.
    // double lower = -0.1;
    // double upper = 0.1;
    // std::uniform_real_distribution<double> unif(lower, upper);
    // std::default_random_engine re;
    // re.seed(std::time(0));
    //
    // grid.resize(20 * 20);
    // for (auto& item : grid) {
    //     double val = unif(re);
    //     std::cout << "Val: " << val << std::endl;
    //     item = val;
    // }
    // terrain.SetGrid(grid);

    trajopt::SrbdTrajopt::Params params;

    params.numKnots = 20;
    params.numSamples = 24;

    params.initBodyPos = Eigen::Vector3d(0., 0., 0.5 + terrain.GetHeight(0., 0.));
    params.targetBodyPos = Eigen::Vector3d(1.5, 0., 0.5 + terrain.GetHeight(1.5, 0.));

    params.initBodyRot = Eigen::Vector3d::Zero();
    params.targetBodyRot = Eigen::Vector3d(0., 0., 0.);

    params.numSteps = {3, 3, 3, 3};

    // params.maxForce = 2. * model.mass * std::abs(model.gravity[2]);
    params.maxForce = 1e6;
    params.addCost = false;

    params.initialFootPhases = {trajopt::rspl::Phase::Stance, trajopt::rspl::Phase::Stance, trajopt::rspl::Phase::Stance, trajopt::rspl::Phase::Stance};

    params.phaseTimes.resize(model.numFeet);
    params.stepKnotsPerSwing.resize(model.numFeet);
    params.forceKnotsPerSwing.resize(model.numFeet);

    for (size_t i = 0; i < model.numFeet; ++i) {
        std::tie(params.phaseTimes[i], params.stepKnotsPerSwing[i], params.forceKnotsPerSwing[i]) = trajopt::createGait(params.numSteps[i], 0.2, 0.4, 1, 5, params.initialFootPhases[i]);
    }

    trajopt::fixDurations(params.phaseTimes);
    auto to = trajopt::SrbdTrajopt(params, model, terrain);

    ifopt::Problem nlp;
    to.initProblem(nlp);
    to.solveProblem(nlp);

    nlp.PrintCurrent();

#if VIZ
    double totalTime = std::accumulate(params.phaseTimes[0].begin(), params.phaseTimes[0].end(), 0.);
    trajopt::visualise<trajopt::PhasedTrajectoryVars>(nlp, model, 2 * totalTime, 0.001, "biped.mp4");
#endif

    return 0;
}
