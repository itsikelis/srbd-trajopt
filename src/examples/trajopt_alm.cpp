#include <chrono>
#include <ctime>
#include <iostream>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <trajopt/alm/alm.hpp>
#include <trajopt/alm/alm_problem.hpp>

#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <trajopt/utils/test_problems/create_implicit_nlp.hpp>
#include <trajopt/utils/test_problems/create_pendulum_nlp.hpp>
#include <trajopt/utils/test_problems/create_phased_nlp.hpp>
#include <trajopt/utils/utils.hpp>
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

using fit_t = alm ::AlmProblem<>;
using algo_t = numopt::algo::AugmentedLagrangianMethod<fit_t>;

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

    // Solve nlp with ipopt and pass it to augmented lagrangian
    std::cout << "Solving.." << std::endl;
    ifopt::IpoptSolver ipopt;
    ipopt.SetOption("jacobian_approximation", "exact");
    ipopt.SetOption("max_wall_time", 1e50);
    ipopt.SetOption("max_iter", static_cast<int>(1000));
    ipopt.Solve(nlp);

    // Solve with ALM
    fit_t fit(nlp);
    algo_t::Params params;
    params.dim = fit.dim();
    params.dim_eq = fit.dim_eq();
    params.dim_ineq = fit.dim_ineq();

    // params.initial_x = algo_t::x_t::Zero(fit.dim()); // Random(fit_t::dim);
    // Initialization
    params.initial_x = nlp.GetVariableValues();
    size_t vals = 10;
    for (size_t i = 0; i < vals; ++i) {
        params.initial_x[(rand() % fit.dim())] += 0.1;
    }

    params.initial_lambda = algo_t::x_t::Zero(fit.dim_eq() + fit.dim_ineq());
    params.max_rho = 1e20;
    params.initial_rho = algo_t::x_t::Ones(fit.dim_eq() + fit.dim_ineq()) * 100.;
    params.rho_a = 10.;

    algo_t algo(params, fit);

    std::cout << "Number of variables: " << fit.dim() << std::endl;
    std::cout << "Number of equality constraints: " << fit.dim_eq() << std::endl;
    std::cout << "Number of inequality constraints: " << fit.dim_ineq() << std::endl;

    // std::cout << "f: " << fit.f(nlp.GetVariableValues()) << std::endl;
    // std::cout << "df: " << fit.df(nlp.GetVariableValues()) << std::endl;
    // std::cout << "c: " << fit.c(nlp.GetVariableValues()).transpose() << std::endl;
    // std::cout << "dc: " << fit.dc(nlp.GetVariableValues()) << std::endl;

    // Solve.
    auto tStart = std::chrono::high_resolution_clock::now();

    unsigned int iters = 2000;
    for (unsigned int i = 0; i < iters; ++i) {
        auto log = algo.step();
        // std::cout << algo.x().transpose() << " -> " << fit.f(algo.x()) << std::endl;
        std::cout << "#" << (i + 1) << std::endl;
        // std::cout << "  ρ: " << algo.rho() << std::endl;
        std::cout << "  f: " << log.f << std::endl;
        std::cout << "  c: " << log.c.norm() << std::endl;
        // std::cout << " " << log.func_evals << " " << log.cons_evals << " " << log.grad_evals << " " << log.hessian_evals << " " << log.cjac_evals << std::endl;
    }

    nlp.PrintCurrent();
    nlp.SetVariables(algo.x().data());
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
