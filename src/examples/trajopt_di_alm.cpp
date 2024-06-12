#include <chrono>
#include <ctime>
#include <iostream>

#include <trajopt/alm/alm.hpp>

#include <trajopt/alm/alm_problem.hpp>
#include <trajopt/utils/utils.hpp>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

using fit_t = alm::AlmProblem<>;
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

    // nlp.AddCostSet(std::make_shared<ifopt::InvertedPendulumCost>());
    ifopt::Problem nlp = trajopt::create_pendulum_nlp();
    nlp.PrintCurrent();

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
    params.max_rho = 1e6;
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

    unsigned int iters = 1000;
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
