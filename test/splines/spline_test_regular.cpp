#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <trajopt/robo_spline/cubic_hermite_spline.hpp>
#include <trajopt/robo_spline/trajectory.hpp>
#include <vector>

namespace rspl = trajopt::rspl;

static constexpr size_t NumKnotPoints = 5;
static constexpr size_t Dim = 3;
static constexpr double eps = 1e-6;

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

// Test calculated.
template <typename _Traj>
bool test_duration(const _Traj& traj, const std::vector<double>& times)
{
    // std::cout << "Trajectory Duration Test" << std::endl;

    // std::cout << "Duration 1, Duration 2 " << std::endl;
    // std::cout << times.sum() << ", " << traj.duration() << " (these should be approx. equal)" << std::endl;
    return (std::abs(std::accumulate(times.begin(), times.end(), 0.) - traj.duration()) < eps);
}

// Test pos and vel at knot points are approximately equal.
template <typename _Traj>
bool test_splines(const _Traj& traj)
{
    bool flag = true;

    for (size_t deriv_order = 0; deriv_order < 2; ++deriv_order) {

        for (size_t i = 0; i < traj.num_knot_points() - 2; i++) {
            double T = traj.spline(i)->duration();
            auto pos1 = traj.spline(i)->eval(T, deriv_order);
            auto pos2 = traj.spline(i + 1)->eval(0., deriv_order);
            auto res = std::abs((pos1 - pos2).norm());
            if (res > eps)
                flag = false;
        }
    }
    return flag;
}

// Compare calculated with estimated Jacobians
template <typename _Traj>
bool test_jacobians(const _Traj& traj, const Eigen::VectorXd& knots, const std::vector<double>& times)
{
    bool flag = true;
    Eigen::VectorXd test_times(times.size() - 1);
    for (size_t i = 0; i < static_cast<size_t>(times.size()) - 1; ++i) {
        if (i == 0) {
            test_times[i] = (times[i + 1] - times[i]) / 2.;
            continue;
        }
        test_times[i] = times[i] + ((times[i + 1] - times[i]) / 2.);
    }
    // Evaluate all Jacobian orders.
    for (size_t order = 0; order < 3; ++order) {
        for (size_t i = 0; i < static_cast<size_t>(test_times.rows()); ++i) {
            double t = test_times[i];

            // t = test_times[0] + 1e-3;

            rspl::Jacobian jac = traj.jac_block(t, order);

            Eigen::MatrixXd jac_est = Eigen::MatrixXd(Dim, knots.rows());

            for (size_t j = 0.; j < static_cast<size_t>(knots.rows()); ++j) {
                auto knots_p = knots;
                auto knots_m = knots;
                knots_p[j] += eps;
                knots_m[j] -= eps;

                auto traj_p = rspl::Trajectory<Dim>(knots_p, times);
                auto traj_m = rspl::Trajectory<Dim>(knots_m, times);

                auto pos_p = traj_p.eval(t, order);
                auto pos_m = traj_m.eval(t, order);

                jac_est.col(j) = (pos_p - pos_m) / (2 * eps);
            }

            // std::cout << "Jacobian Block:" << std::endl;
            // std::cout << Eigen::MatrixXd(jac) << std::endl;

            // std::cout << "Jacobian Block Approximation:" << std::endl;
            // std::cout << jac_est << std::endl;

            // std::cout << "Absolute norm of difference: (eps: " << eps << ")" << std::endl;
            // std::cout << abs((jac - jac_est).norm()) << std::endl;
            double err = abs((jac - jac_est).norm());

            if (err > 2 * eps) {
                std::cerr << "Absolute norm of difference: (eps: " << eps << ")" << std::endl;
                std::cerr << abs((jac - jac_est).norm()) << std::endl;
                flag = false;
                break;
            }
        }
    }

    return flag;
}

int main()
{
    srand(static_cast<size_t>(time(0)));
    // for (size_t iters = 0; iters < 1000; ++iters) {
    Eigen::VectorXd knots = random_uniform_vector(NumKnotPoints * 2 * Dim, -10., 10.);

    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
    std::uniform_real_distribution<double> dist{0., 3.};
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };
    std::vector<double> times(NumKnotPoints - 1);
    std::generate(begin(times), end(times), gen);

    rspl::Trajectory<Dim> traj(knots, times);

    if (!test_duration(traj, times)) {
        std::cerr << "Test Failed: test_duration" << std::endl;
        return -1;
    }

    if (!test_splines(traj)) {
        std::cerr << "Test Failed: test_splines" << std::endl;
        return -1;
    }

    if (!test_jacobians(traj, knots, times)) {
        std::cerr << "Test Failed: test_jacobians" << std::endl;
        return -1;
    }
    // }

    // std::cout << "Positions:" << std::endl;
    // for (double t = 0.; t <= (total_duration + 1e-6); t += 0.1) {
    //     std::cout << t << ": " << traj.position(t).transpose() << std::endl;
    // }

    // std::cout << "Velocities:" << std::endl;
    // for (double t = 0.; t <= (total_duration + 1e-6); t += 0.1) {
    //     std::cout << t << ": " << traj.velocity(t).transpose() << std::endl;
    // }

    // std::cout << "Accelerations:" << std::endl;
    // for (double t = 0.; t <= (total_duration + 1e-6); t += 0.1) {
    //     std::cout << t << ": " << traj.acceleration(t).transpose() << std::endl;
    // }

    // std::cout << "Derivatives:" << std::endl;
    // size_t idx = 0;
    // std::cout << pols[idx].spline->deriv_pos(0.125).transpose() << std::endl
    //           << std::endl;
    // std::cout << pols[idx].spline->deriv_vel(0.125).transpose() << std::endl
    //           << std::endl;
    // std::cout << pols[idx].spline->deriv_acc(0.125).transpose() << std::endl
    //           << std::endl;

    // // Test derivatives
    // double t = 1.25;
    // size_t idx;
    // rspl::Jacobian jac_pos;
    // std::tie(idx, jac_pos) = traj.jacobian_position(t);

    // std::cout << "Spline Index:" << idx << std::endl;
    // std::cout << Eigen::MatrixXd(jac_pos) << std::endl;

    // std::cout << pols[idx].spline->deriv_pos(0.125).transpose();

    // const size_t num_knot_points = 3;
    // const size_t dim = 2;

    // Eigen::Vector2d p0(0., 0.);
    // Eigen::Vector2d v0(0., 0.);

    // Eigen::Vector2d p1(1., 2.);
    // Eigen::Vector2d v1(1., 1.);

    // Eigen::Vector2d p2(2., 3.);
    // Eigen::Vector2d v2(0., 0.);

    // double t0 = 3.;
    // double t1 = 2.;

    // Eigen::VectorXd points(num_knot_points * 2 * dim);
    // Eigen::VectorXd times(num_knot_points - 1);

    // points << p0, v0, p1, v1, p2, v2;
    // times << t0, t1;

    // rspl::Trajectory<rspl::CubicHermiteSpline<dim>, num_knot_points> traj(points, times);

    std::cout << "All tests successful!" << std::endl;

    return 0;
}
