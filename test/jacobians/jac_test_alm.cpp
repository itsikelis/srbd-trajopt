#include <ctime>
#include <iostream>
#include <random>

#include <ifopt/problem.h>

#include <trajopt/alm/alm.hpp>
#include <trajopt/alm/alm_problem.hpp>

#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <trajopt/utils/utils.hpp>

#include <trajopt/utils/test_problems/create_implicit_nlp.hpp>
#include <trajopt/utils/test_problems/create_pendulum_nlp.hpp>
#include <trajopt/utils/test_problems/create_phased_nlp.hpp>

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

fit_t::g_t finite_diff(fit_t& f, const fit_t::x_t& x, double eps = 1e-6)
{
    fit_t::g_t fdiff = fit_t::g_t::Zero(x.size());
    for (int i = 0; i < x.size(); i++) {
        fit_t::x_t xp = x;
        xp[i] += eps;
        fit_t::x_t xm = x;
        xm[i] -= eps;
        double fp = f.f(xp);
        double fm = f.f(xm);

        fdiff[i] = (fp - fm) / (2. * eps);
    }

    return fdiff;
}

fit_t::mat_t finite_diff_hessian(fit_t& f, const fit_t::x_t& x, double eps = 1e-6)
{
    fit_t::mat_t fdiff = fit_t::mat_t::Zero(x.size(), x.size());
    for (int i = 0; i < x.size(); i++) {
        fit_t::x_t xp = x;
        xp[i] += eps;
        fit_t::x_t xm = x;
        xm[i] -= eps;

        fit_t::g_t fp = f.df(xp);
        fit_t::g_t fm = f.df(xm);
        fdiff.row(i) = (fp - fm) / (2. * eps);
    }

    return fdiff;
}

fit_t::mat_t finite_diff_dc(fit_t& f, const fit_t::x_t& x, double eps = 1e-6)
{
    fit_t::mat_t fdiff = fit_t::mat_t::Zero(f.dim_eq() + f.dim_ineq(), f.dim());
    for (int i = 0; i < x.size(); i++) {
        fit_t::x_t xp = x;
        xp[i] += eps;
        fit_t::x_t xm = x;
        xm[i] -= eps;

        fit_t::x_t fp = f.c(xp);
        fit_t::x_t fm = f.c(xm);
        fdiff.col(i) = (fp - fm) / (2. * eps);
    }

    return fdiff;
}

int main()
{
    std::srand(std::time(0));

    trajopt::SingleRigidBodyDynamicsModel model;
    trajopt::init_model_anymal(model);
    // trajopt::init_model_biped(model);

    trajopt::TerrainGrid terrain(200, 200, 0.7, 0, 0, 200, 200);
    std::vector<double> grid;

    // Create a random grid for the terrain.
    double lower = -0.1;
    double upper = 0.1;
    std::uniform_real_distribution<double> unif(lower, upper);
    std::default_random_engine re;
    re.seed(std::time(0));

    grid.resize(200 * 200);
    for (auto& item : grid) {
        item = unif(re);
    }
    terrain.SetGrid(grid);

    Eigen::Vector3d initBodyPos = Eigen::Vector3d(initBodyPosX, initBodyPosY, initBodyPosZ + terrain.GetHeight(initBodyPosX, initBodyPosY));
    Eigen::Vector3d targetBodyPos = Eigen::Vector3d(targetBodyPosX, targetBodyPosY, targetBodyPosZ + terrain.GetHeight(targetBodyPosX, targetBodyPosY));

    // ifopt::Problem nlp = create_phased_nlp(numKnots, numSamples, totalTime, initBodyPos, targetBodyPos, model, terrain);
    ifopt::Problem nlp = create_implicit_nlp(numKnots, numSamples, totalTime, initBodyPos, targetBodyPos, model, terrain);
    // ifopt::Problem nlp = trajopt::create_pendulum_nlp();

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

    auto df = fit.df(algo.x());
    auto df_finite = finite_diff(fit, algo.x());
    std::cout << "df: " << (df - df_finite).norm() << std::endl;
    auto ddf = fit.ddf(algo.x());
    auto ddf_finite = finite_diff_hessian(fit, algo.x());
    std::cout << "ddf: " << (ddf - ddf_finite).norm() << std::endl;
    auto dc = fit.dc(algo.x());
    auto dc_finite = finite_diff_dc(fit, algo.x());
    std::cout << "dc: " << (dc - dc_finite).norm() << std::endl;

    // std::cout << "f: " << fit.f(algo.x()) << std::endl;
    // std::cout << "df: " << fit.df(algo.x()) << std::endl;
    // std::cout << "df: " << finite_diff(fit, algo.x()) << std::endl;
    // std::cout << "ddf:\n" << fit.ddf(algo.x()) << std::endl;
    // std::cout << "ddf:\n" << finite_diff_hessian(fit, algo.x()) << std::endl;
    // std::cout << "c: " << fit.c(algo.x()) << std::endl;
    // std::cout << "dc:\n" << fit.dc(algo.x()) << std::endl;

    return 0;
}
