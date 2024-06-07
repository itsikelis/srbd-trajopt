#include <Eigen/Dense>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>

#include <trajopt/alm/alm.hpp>

#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <trajopt/ifopt_sets/variables/trajectory_vars.hpp>
#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <trajopt/ifopt_sets/cost/min_effort.hpp>

#include <trajopt/ifopt_sets/constraints/common/acceleration.hpp>
#include <trajopt/ifopt_sets/constraints/common/dynamics.hpp>
#include <trajopt/ifopt_sets/constraints/common/friction_cone.hpp>

#include <trajopt/ifopt_sets/constraints/contact_implicit/foot_body_distance_implicit.hpp>
#include <trajopt/ifopt_sets/constraints/contact_implicit/foot_terrain_distance_implicit.hpp>
#include <trajopt/ifopt_sets/constraints/contact_implicit/implicit_contact.hpp>
#include <trajopt/ifopt_sets/constraints/contact_implicit/implicit_velocity.hpp>

#include <trajopt/utils/types.hpp>
#include <trajopt/utils/utils.hpp>

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

template <typename Scalar = double>
class AlmProblem {
public:
    static constexpr unsigned int D = 6;
    static constexpr unsigned int Ad = 2;
    static constexpr unsigned int T = 51;
    static constexpr double dt = 0.05;

    static constexpr unsigned int dim = T * Ad + (T - 1) * D;
    static constexpr unsigned int dim_eq = T * D;
    static constexpr unsigned int dim_ineq = 2 * T * Ad;

    using mat_t = Eigen::Matrix<Scalar, -1, -1>;
    using x_t = Eigen::Matrix<Scalar, -1, 1>;
    using g_t = Eigen::Matrix<Scalar, 1, -1>;

    AlmProblem(const ifopt::Problem& nlp) : _dim(static_cast<size_t>(nlp.GetNumberOfOptimizationVariables()))
    {
        _nlp = nlp;

        _calc_constr_dims();
    }

    Scalar f(const x_t& x)
    {
        Scalar cost = _nlp.EvaluateCostFunction(x.data());
        return cost;
    }

    g_t df(const x_t& x)
    {
        _nlp.SetVariables(x.data()); // TODO: Maybe this is not needed.
        g_t grad = _nlp.EvaluateCostFunctionGradient(x.data());

        return grad;
    }

    // mat_t ddf(const x_t& x)
    // {
    //     mat_t hessian = mat_t::Zero(dim, dim);
    //     // Action dimensions
    //     for (unsigned int i = 0; i < T; i++) {
    //         hessian.block(i * Ad, i * Ad, Ad, Ad) = R;
    //     }
    //     // State dimensions
    //     for (unsigned int i = 0; i < (T - 1); i++) {
    //         hessian.block(T * Ad + i * D, T * Ad + i * D, D, D) = Q;
    //     }
    //
    //     return hessian;
    // }

    x_t c(const x_t& x)
    {
        if (_neq < 0 || _nin < 0) {
            // Determine equality and inequality constraint dims.
            _calc_constr_dims();
        }

        return x;
        // return out;
    }

    mat_t dc(const x_t& x)
    {
        mat_t C = mat_t::Zero(dim_eq + dim_ineq, dim);
        if (_neq < 0 || _nin < 0) {
            // Determine equlaity and inequality constraint dims.
        }
        return C;
    }

protected:
    void _calc_constr_dims()
    {
        _neq = 0;
        _nin = 0;

        // From constraint sets
        for (const auto& ct : _nlp.GetConstraints().GetComponents()) {
            int n = ct->GetRows();
            if (n < 0)
                continue;
            for (int i = 0; i < n; i++) {
                auto bounds = ct->GetBounds()[i];
                if (std::abs(bounds.lower_ - bounds.upper_) < 1e-8) {
                    ++_neq;
                    std::cout << "Equality" << " == " << bounds.upper_ << std::endl;
                }
                else {
                    if (bounds.lower_ > -1e20) {
                        ++_nin;
                        std::cout << "Inequality" << " >= " << bounds.lower_ << std::endl;
                    }

                    if (bounds.upper_ < 1e20) {
                        ++_nin;
                        std::cout << "Inequality" << " <= " << bounds.upper_ << std::endl;
                    }
                }
            }
        }

        // From variable bounds
        const auto& bounds = _nlp.GetBoundsOnOptimizationVariables();
        for (size_t i = 0; i < _dim; ++i) {
            if (std::abs(bounds[i].lower_ - bounds[i].upper_) < 1e-8) {
                std::cout << "Equality" << " == " << bounds[i].upper_ << std::endl;
                ++_neq;
            }
            else {
                if (bounds[i].lower_ > -1e20) {
                    std::cout << "Inequality" << " >= " << bounds[i].lower_ << std::endl;
                    ++_nin;
                }

                if (bounds[i].upper_ < 1e20) {
                    ++_nin;
                    std::cout << "Inequality" << " <= " << bounds[i].upper_ << std::endl;
                }
            }
        }
    }

protected:
    ifopt::Problem _nlp;

    size_t _dim{0};

    size_t _neq{0}; // Number of equality constraints.
    size_t _nin{0}; // Number of inequality constraints.
};

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
//
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
//
// fit_t::mat_t finite_diff_dc(fit_t& f, const fit_t::x_t& x, double eps = 1e-6)
// {
//     fit_t::mat_t fdiff = fit_t::mat_t::Zero(fit_t::dim_eq + fit_t::dim_ineq, fit_t::dim);
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
    // trajopt::init_model_biped(model);

    trajopt::TerrainGrid terrain(200, 200, 1., -100, -100, 100, 100);
    terrain.set_zero();

    // Add variable sets.
    ifopt::Problem nlp;

    double sampleTime = totalTime / static_cast<double>(numSamples - 1.);
    Eigen::VectorXd polyTimes = Eigen::VectorXd::Zero(numKnots - 1);
    for (size_t i = 0; i < static_cast<size_t>(polyTimes.size()); ++i) {
        polyTimes[i] = totalTime / static_cast<double>(numKnots - 1);
    }

    // Add body pos and rot var sets.
    Eigen::Vector3d initBodyPos = Eigen::Vector3d(initBodyPosX, initBodyPosY, initBodyPosZ + terrain.height(initBodyPosX, initBodyPosY));
    Eigen::Vector3d targetBodyPos = Eigen::Vector3d(targetBodyPosX, targetBodyPosY, targetBodyPosZ + terrain.height(targetBodyPosX, targetBodyPosX));
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

    // Add feet variable sets.
    Eigen::VectorXd initFootPosVals = Eigen::VectorXd::Zero(3 * 2 * numKnots);
    Eigen::VectorXd initFootForceVals = Eigen::VectorXd::Zero(3 * 2 * numKnots);

    ifopt::Component::VecBound footPosBounds(6 * numKnots, ifopt::NoBound);
    // ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::NoBound);
    double max_force = 2. * model.mass * std::abs(model.gravity[2]);
    ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::Bounds(-max_force, max_force));
    // for (size_t i = 0; i < numKnots; ++i) {
    //     footForceBounds.at(i * 6 + 2) = ifopt::Bounds(-1e-6, max_force);
    // }

    for (size_t i = 0; i < model.numFeet; ++i) {
        // Add initial and final positions for each foot.
        // right feet
        if (i == 0 || i == 2) {
            double bound_x = initBodyPos(0);
            if (i == 0)
                bound_x += 0.34;
            else
                bound_x -= 0.34;
            double bound_y = initBodyPos(1) - 0.19;

            footPosBounds[0] = ifopt::Bounds(bound_x, bound_x);
            footPosBounds[1] = ifopt::Bounds(bound_y, bound_y);

            // bound_x = targetBodyPos(0);
            // bound_y = targetBodyPos(1) - 0.19;
            // if (i == 0)
            //     bound_x += 0.34;
            // else
            //     bound_x -= 0.34;

            // footPosBounds[6 * numKnots - 6] = ifopt::Bounds(bound_x, bound_x);
            // footPosBounds[6 * numKnots - 5] = ifopt::Bounds(bound_y, bound_y);
        }
        else if (i == 1 || i == 3) {
            double bound_x = initBodyPos(0);
            if (i == 1)
                bound_x += 0.34;
            else
                bound_x -= 0.34;
            double bound_y = initBodyPos(1) + 0.19;

            footPosBounds[0] = ifopt::Bounds(bound_x, bound_x);
            footPosBounds[1] = ifopt::Bounds(bound_y, bound_y);

            // bound_x = targetBodyPos(0);
            // bound_y = targetBodyPos(1) + 0.19;
            // if (i == 1)
            //     bound_x += 0.34;
            // else
            //     bound_x -= 0.34;

            // footPosBounds[6 * numKnots - 6] = ifopt::Bounds(bound_x, bound_x);
            // footPosBounds[6 * numKnots - 5] = ifopt::Bounds(bound_y, bound_y);
        }

        auto footPosVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_POS + "_" + std::to_string(i), initFootPosVals, polyTimes, footPosBounds);
        nlp.AddVariableSet(footPosVars);

        auto footForceVars = std::make_shared<trajopt::TrajectoryVars>(trajopt::FOOT_FORCE + "_" + std::to_string(i), initFootForceVals, polyTimes, footForceBounds);
        nlp.AddVariableSet(footForceVars);
    }

    // Add regular constraint sets.
    nlp.AddConstraintSet(std::make_shared<trajopt::Dynamics<trajopt::TrajectoryVars>>(model, numSamples, sampleTime));

    nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(posVars));
    nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(rotVars));

    // ifopt::Component::VecBound footForceBounds(6 * numKnots, ifopt::NoBound);

    for (size_t i = 0; i < model.numFeet; ++i) {
        auto footPosVars = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_POS + "_" + std::to_string(i)));
        auto footForceVars = std::static_pointer_cast<trajopt::TrajectoryVars>(nlp.GetOptVariables()->GetComponent(trajopt::FOOT_FORCE + "_" + std::to_string(i)));

        nlp.AddConstraintSet(std::make_shared<trajopt::AccelerationConstraints>(footPosVars));
        nlp.AddConstraintSet(std::make_shared<trajopt::FrictionCone<trajopt::TrajectoryVars>>(footForceVars, footPosVars, terrain, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootBodyDistanceImplicit>(model, posVars, rotVars, footPosVars, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::FootTerrainDistanceImplicit>(footPosVars, terrain, numKnots));
        nlp.AddConstraintSet(std::make_shared<trajopt::ImplicitContactConstraints>(footPosVars, footForceVars, terrain, numSamples, sampleTime));
        nlp.AddConstraintSet(std::make_shared<trajopt::ImplicitVelocityConstraints>(footPosVars, footForceVars, terrain, numSamples, sampleTime));

        // Add cost set
        // nlp.AddCostSet(std::make_shared<trajopt::MinEffort<trajopt::TrajectoryVars>>(footForceVars, numKnots));
    }

    fit_t fit(nlp);
    // algo_t::Params params;
    // params.dim = fit_t::dim;
    // params.dim_eq = fit_t::dim_eq;
    // params.dim_ineq = fit_t::dim_ineq;
    //
    // params.initial_x = algo_t::x_t::Zero(fit_t::dim); // Random(fit_t::dim);
    // // Initialization
    // params.initial_x.head(fit_t::T * fit_t::Ad) = algo_t::x_t::Constant(fit_t::T * fit_t::Ad, fit_t::m * fit_t::g / 2.);
    // for (unsigned int i = 0; i < (fit_t::T - 1); i++) {
    //     params.initial_x.segment(fit_t::T * fit_t::Ad + i * fit_t::D, fit_t::D) = fit.x0 + (fit.xN - fit.x0) * (i + 1) / static_cast<double>(fit_t::T);
    // }
    // params.initial_lambda = algo_t::x_t::Zero(fit_t::dim_eq + fit_t::dim_ineq);
    // params.initial_rho = 100.;
    // params.rho_a = 10.;

    // algo_t algo(params);

    std::cout << "Number of variables: " << fit_t::dim << std::endl;
    std::cout << "Number of equality constraints: " << fit_t::dim_eq << std::endl;
    std::cout << "Number of inequality constraints: " << fit_t::dim_ineq << std::endl;

    std::cout << "f: " << fit.f(nlp.GetVariableValues()) << std::endl;
    std::cout << "df: " << fit.df(nlp.GetVariableValues()) << std::endl;
    std::cout << "c: " << fit.c(nlp.GetVariableValues()).transpose() << std::endl;

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

    // unsigned int iters = 1;
    // for (unsigned int i = 0; i < iters; ++i) {
    //     auto log = algo.step();
    //     std::cout << algo.x().transpose() << " -> " << fit.f(algo.x()) << std::endl;
    //     std::cout << "f: " << log.f << std::endl;
    //     std::cout << "c: " << log.c.norm() << std::endl;
    //     std::cout << " " << log.func_evals << " " << log.cons_evals << " " << log.grad_evals << " " << log.hessian_evals << " " << log.cjac_evals << std::endl;
    // }

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