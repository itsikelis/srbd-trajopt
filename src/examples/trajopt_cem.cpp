#include <iostream>

#include <algevo/algo/cem_discrete.hpp>

#include <trajopt/srbd/srbd.hpp>
#include <trajopt/terrain/terrain_grid.hpp>

#include <trajopt/srbd_trajopt.hpp>
#include <trajopt/utils/gait_profiler.hpp>

size_t GetNumViolations(const ifopt::Composite& comp, double tol = 1e-4);
size_t GetNumViolations(const ifopt::Problem& nlp, double tol = 1e-4);

template <typename Scalar = double>
struct FitDiscrete {
public:
    static constexpr unsigned int dim = 4;
    static constexpr unsigned int num_values = 5;

    using x_t = Eigen::Matrix<unsigned int, 1, dim>;

    FitDiscrete()
    {
        trajopt::init_model_anymal(_model);
        _terrain.SetZero();
    }

    Scalar eval(const x_t& x)
    {
        // std::cout << "Num Steps: " << x[0] + 2 << ", " << x[1] + 2 << ", " << x[2] + 2 << ", " << x[3] + 2 << std::endl;
        trajopt::SrbdTrajopt::Params params;

        params.numKnots = 20;
        params.numSamples = 24;

        size_t posKnotsInSwingPhase = 1;
        size_t forceKnotsInStancePhase = 3;

        params.initBodyPos = Eigen::Vector3d(0., 0., 0.5 + _terrain.GetHeight(0., 0.));
        params.targetBodyPos = Eigen::Vector3d(1.5, 0., 0.5 + _terrain.GetHeight(1.5, 0.));

        params.initBodyRot = Eigen::Vector3d::Zero();
        params.targetBodyRot = Eigen::Vector3d(M_PI, 0., 0.);

        params.maxForce = 2. * _model.mass * std::abs(_model.gravity[2]);
        params.addCost = false;

        params.numSteps = {x[0] + 2, x[1] + 2, x[2] + 2, x[3] + 2};
        params.initialFootPhases = {trajopt::rspl::Phase::Stance, trajopt::rspl::Phase::Stance, trajopt::rspl::Phase::Stance, trajopt::rspl::Phase::Stance};

        params.phaseTimes.resize(_model.numFeet);
        params.stepKnotsPerSwing.resize(_model.numFeet);
        params.forceKnotsPerSwing.resize(_model.numFeet);

        params.maxWallTime = 20.;
        params.printLevel = 5;

        for (size_t i = 0; i < _model.numFeet; ++i) {
            std::tie(params.phaseTimes[i], params.stepKnotsPerSwing[i], params.forceKnotsPerSwing[i]) = trajopt::createGait(params.numSteps[i], 0.2, 0.1, posKnotsInSwingPhase, forceKnotsInStancePhase, params.initialFootPhases[i]);
        }

        trajopt::fixDurations(params.phaseTimes);

        auto to = trajopt::SrbdTrajopt(params, _model, _terrain);

        ifopt::Problem nlp;
        to.initProblem(nlp);
        to.solveProblem(nlp);

        double cost = x.sum() + GetNumViolations(nlp);

        return -cost;
    }

protected:
    trajopt::SingleRigidBodyDynamicsModel _model;
    trajopt::TerrainGrid _terrain{trajopt::TerrainGrid(200, 200, 1., -100, -100, 100, 100)};
};

// Typedefs
using FitD = FitDiscrete<>;
using Algo = algevo::algo::CrossEntropyMethodDiscrete<FitD>;

int main()
{
    Algo::Params params;
    params.dim = FitD::dim;
    params.pop_size = 24;
    params.num_elites = params.pop_size * 0.8;
    params.num_values = {FitD::num_values, FitD::num_values, FitD::num_values, FitD::num_values};
    params.init_probs = Algo::p_t::Ones(FitD::dim, FitD::num_values) / static_cast<double>(FitD::num_values);

    Algo cem(params);

    for (unsigned int i = 0; i < 50; i++) {
        auto log = cem.step();

        std::cout << log.iterations << "(" << log.func_evals << "): " << log.best_value << std::endl;
        std::cout << log.best.transpose() << std::endl;
        std::cout << cem.probabilities() << std::endl
                  << std::endl;
    }
    return 0;
}

size_t GetNumViolations(const ifopt::Composite& comp, double tol)
{
    size_t n = 0;
    for (auto& c : comp.GetComponents()) {
        Eigen::VectorXd x = c->GetValues();
        ifopt::Component::VecBound bounds = c->GetBounds();
        for (std::size_t i = 0; i < bounds.size(); ++i) {
            double lower = bounds.at(i).lower_;
            double upper = bounds.at(i).upper_;
            double val = x(i);
            if (val < lower - tol || upper + tol < val)
                n++;
        }
    }

    return n;
}

size_t GetNumViolations(const ifopt::Problem& nlp, double tol)
{
    auto constraints = nlp.GetConstraints();
    size_t n = GetNumViolations(constraints, tol);
    auto vars = nlp.GetOptVariables();
    n += GetNumViolations(*vars, tol);
    return n;
}
