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
    trajopt::init_model_anymal(model);

    trajopt::TerrainGrid terrain(200, 200, 1., -100, -100, 100, 100);
    terrain.set_zero();

    trajopt::SrbdTrajopt::Params params;

    params.numKnots = 20;
    params.numSamples = 24;

    params.initBodyPos = Eigen::Vector3d(0., 0., 0.5 + terrain.height(0., 0.));
    params.targetBodyPos = Eigen::Vector3d(1.5, 0., 0.5 + terrain.height(1.5, 0.));

    params.initBodyRot = Eigen::Vector3d::Zero();
    params.targetBodyRot = Eigen::Vector3d::Zero();

    params.maxForce = 2. * model.mass * std::abs(model.gravity[2]);
    params.addCost = false;

    params.numSteps = {4, 4, 5, 4};
    params.initialFootPhases = {trajopt::rspl::Phase::Stance, trajopt::rspl::Phase::Stance, trajopt::rspl::Phase::Stance, trajopt::rspl::Phase::Stance};

    params.phaseTimes.resize(model.numFeet);
    params.stepKnotsPerSwing.resize(model.numFeet);
    params.forceKnotsPerSwing.resize(model.numFeet);

    for (size_t i = 0; i < model.numFeet; ++i) {
        std::tie(params.phaseTimes[i], params.stepKnotsPerSwing[i], params.forceKnotsPerSwing[i]) = trajopt::createGait(params.numSteps[i], 0.2, 0.1, 1, 5, params.initialFootPhases[i]);
    }

    trajopt::fixDurations(params.phaseTimes);
    auto to = trajopt::SrbdTrajopt(params, model, terrain);

    ifopt::Problem nlp;
    to.initProblem(nlp);
    to.solveProblem(nlp);

#if VIZ
    double totalTime = std::accumulate(params.phaseTimes[3].begin(), params.phaseTimes[0].end(), 0.);
    trajopt::visualise<trajopt::PhasedTrajectoryVars>(nlp, model, 2 * totalTime, 0.001);
#endif

    return 0;
}
