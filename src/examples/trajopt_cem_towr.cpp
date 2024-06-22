#include <iostream>

#include <algevo/algo/cem_discrete.hpp>

#include <ifopt/composite.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <towr/nlp_formulation.h>
#include <towr/terrain/examples/height_map_examples.h>

#include <trajopt/utils/gait_profiler.hpp>

size_t GetNumViolations(const ifopt::Composite& comp, double tol = 1e-4);
size_t GetNumViolations(const ifopt::Problem& nlp, double tol = 1e-4);

void towrHopperNlp(ifopt::Problem& nlp, size_t numSteps, double swingDuration, double stanceDuration, bool standingAtStart, bool optimiseTime = false);
void towrAnymalNlp(ifopt::Problem& nlp, Eigen::Matrix<unsigned int, 1, 4> numSteps, double stanceDuration, double swingDuration, std::vector<bool> standingAtStart, size_t numKnots, size_t numSamples, bool optimiseTime = true);

template <typename Scalar = double>
struct FitDiscrete {
public:
    static constexpr unsigned int dim = 4;
    static constexpr unsigned int num_values = 5;

    using x_t = Eigen::Matrix<unsigned int, 1, dim>;

    FitDiscrete()
    {
    }

    Scalar eval(const x_t& x)
    {
        ifopt::Problem nlp;

        size_t numKnots = 10;
        size_t numSamples = 16;

        // // Hopper Problem
        // size_t numSteps = x[0] + 2;
        // double swingDuration = 0.1;
        // double stanceDuration = 0.2;
        // bool standingAtStart = true;
        // bool optimiseTime = true;
        // towrHopperNlp(nlp, numSteps, stanceDuration, swingDuration, standingAtStart, optimiseTime);

        // Anymal Problem
        x_t numSteps = x;
        for (auto& item : numSteps)
            item += 2;
        std::vector<bool> standingAtStart(4, true);
        double swingDuration = 0.1;
        double stanceDuration = 0.2;
        bool optimiseTime = true;
        towrAnymalNlp(nlp, numSteps, stanceDuration, swingDuration, standingAtStart, numKnots, numSamples, optimiseTime);

        auto solver = std::make_shared<ifopt::IpoptSolver>();
        solver->SetOption("jacobian_approximation", "exact");
        solver->SetOption("max_wall_time", 12.);
        solver->SetOption("max_iter", 1000);
        solver->SetOption("print_level", 0);

        solver->Solve(nlp);

        std::cout << "Wall Time: " << solver->GetTotalWallclockTime() << std::endl;

        double cost = x.sum() + GetNumViolations(nlp);

        return -cost;
    }
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

void towrHopperNlp(ifopt::Problem& nlp, size_t numSteps, double stanceDuration, double swingDuration, bool standingAtStart, bool optimiseTime)
{
    towr::NlpFormulation formulation;

    formulation.terrain_ = std::make_shared<towr::FlatGround>(0.0);
    formulation.model_ = towr::RobotModel(towr::RobotModel::Monoped);

    formulation.initial_base_.lin.at(towr::kPos).z() = 0.5;
    formulation.initial_ee_W_.push_back(Eigen::Vector3d::Zero());

    formulation.final_base_.lin.at(towr::kPos) << 1.0, 0.0, 0.5;

    if (optimiseTime) {
        formulation.params_.bound_phase_duration_ = std::make_pair(0.1, numSteps * 0.2 + (numSteps - 1) * 0.1);
        formulation.params_.OptimizePhaseDurations();
    }

    std::vector<double> durations;

    bool isStance = standingAtStart;
    size_t stepCounter = 0;
    while (stepCounter < numSteps) {
        if (isStance) {
            stepCounter++;
            durations.push_back(stanceDuration);
        }
        else {
            durations.push_back(swingDuration);
        }
        isStance = !isStance;
    }

    for (auto d : durations) {
        std::cout << d << ", ";
    }
    std::cout << std::endl;

    formulation.params_.ee_phase_durations_.push_back(durations);
    formulation.params_.ee_in_contact_at_start_.push_back(standingAtStart);

    towr::SplineHolder solution;
    for (auto c : formulation.GetVariableSets(solution))
        nlp.AddVariableSet(c);
    for (auto c : formulation.GetConstraints(solution))
        nlp.AddConstraintSet(c);
    for (auto c : formulation.GetCosts())
        nlp.AddCostSet(c);
}

void towrAnymalNlp(
    ifopt::Problem& nlp,
    Eigen::Matrix<unsigned int, 1, 4> numSteps,
    double stanceDuration, double swingDuration,
    std::vector<bool> standingAtStart,
    size_t numKnots,
    size_t numSamples,
    bool optimiseTime)
{
    towr::NlpFormulation formulation;

    formulation.terrain_ = std::make_shared<towr::FlatGround>(0.0);
    formulation.model_ = towr::RobotModel(towr::RobotModel::Anymal);

    formulation.initial_base_.lin.at(towr::kPos).z() = 0.5;
    formulation.final_base_.lin.at(towr::kPos) << 1.5, 0.0, 0.5;

    formulation.final_base_.ang.at(towr::kPos).z() = M_PI;

    formulation.params_.force_polynomials_per_stance_phase_ = 4;
    formulation.params_.ee_polynomials_per_swing_phase_ = 2;

    if (optimiseTime) {
        // Find max num of steps
        unsigned int max = 0.;
        for (auto& s : numSteps) {
            if (s > max) {
                max = s;
            }
        }
        formulation.params_.bound_phase_duration_ = std::make_pair(0.1, max * 0.2 + (max - 1) * 0.1);
        formulation.params_.OptimizePhaseDurations();
    }

    for (size_t i = 0; i < 4; ++i) {
        double init_x = formulation.initial_base_.lin.at(towr::kPos).x();
        double init_y = formulation.initial_base_.lin.at(towr::kPos).y();
        double init_z = 0.;
        if (i == 0 || i == 2) {
            if (i == 0)
                init_x += 0.34;
            else
                init_x -= 0.34;
            init_y += 0.19;

            init_z = formulation.terrain_->GetHeight(init_x, init_y);
        }
        else if (i == 1 || i == 3) {
            if (i == 1)
                init_x += 0.34;
            else
                init_x -= 0.34;
            init_y -= 0.19;

            init_z = formulation.terrain_->GetHeight(init_x, init_y);
        }

        Eigen::Vector3d initPos(init_x, init_y, init_z);
        // std::cout << "Init pos " << i << ": " << initPos.transpose() << std::endl;
        formulation.initial_ee_W_.push_back(initPos);
    }

    for (size_t i = 0; i < 4; ++i) {
        std::vector<double> durations;
        size_t steps = numSteps[i];

        // std::cout << "Num Steps: " << numSteps << std::endl;

        bool isStance = standingAtStart[i];
        size_t stepCounter = 0;
        while (stepCounter < steps) {
            if (isStance) {
                stepCounter++;
                durations.push_back(stanceDuration);
            }
            else {
                durations.push_back(swingDuration);
            }
            isStance = !isStance;
        }

        // for (auto& d : durations) {
        //     std::cout << d << ", " << std::endl;
        // }

        formulation.params_.ee_phase_durations_.push_back(durations);

        formulation.params_.ee_in_contact_at_start_.push_back(true);
    }

    trajopt::fixDurations(formulation.params_.ee_phase_durations_);

    double totalTime = std::accumulate(formulation.params_.ee_phase_durations_[0].begin(), formulation.params_.ee_phase_durations_[0].end(), 0.);
    formulation.params_.duration_base_polynomial_ = totalTime / static_cast<double>(numKnots - 1);

    formulation.params_.dt_constraint_dynamic_ = totalTime / static_cast<double>(numSamples - 1.);
    formulation.params_.dt_constraint_base_motion_ = totalTime / static_cast<double>(numSamples - 1.);
    formulation.params_.dt_constraint_range_of_motion_ = totalTime / static_cast<double>(numSamples - 1.);

    towr::SplineHolder solution;
    for (auto c : formulation.GetVariableSets(solution))
        nlp.AddVariableSet(c);
    for (auto c : formulation.GetConstraints(solution))
        nlp.AddConstraintSet(c);
    for (auto c : formulation.GetCosts())
        nlp.AddCostSet(c);
}
