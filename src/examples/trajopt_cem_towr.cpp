#include <iostream>

#include <algevo/algo/cem_discrete.hpp>

#include <ifopt/composite.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>

#include <towr/nlp_formulation.h>
#include <towr/terrain/examples/height_map_examples.h>

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
    }

    Scalar eval(const x_t& x)
    {
        towr::NlpFormulation formulation;

        formulation.terrain_ = std::make_shared<towr::FlatGround>(0.0);

        // Kinematic limits and dynamic parameters of the hopper
        formulation.model_ = towr::RobotModel(towr::RobotModel::Anymal);

        formulation.initial_base_.lin.at(towr::kPos).z() = 0.5;
        formulation.final_base_.lin.at(towr::kPos) << 1.0, 0.0, 0.5;

        for (size_t i = 0; i < 4; ++i) {
            formulation.initial_ee_W_.push_back(Eigen::Vector3d::Zero());

            // Parameters that define the motion. See c'tor for default values or
            // other values that can be modified.
            // First we define the initial phase durations, that can however be changed
            // by the optimizer. The number of swing and stance phases however is fixed.
            // alternating stance and swing:     ____-----_____-----_____-----_____
            formulation.params_.ee_in_contact_at_start_.push_back(true);
        }

        double swingDuration = 0.1;
        double stanceDuration = 0.2;

        for (size_t i = 0; i < 4; ++i) {
            std::vector<double> durations;
            size_t numSteps = x[i] + 2;
            bool isStance = formulation.params_.ee_in_contact_at_start_[i];
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

            // Print durations
            for (auto& d : durations) {
                std::cout << d << ", ";
            }
            std::cout << std::endl;

            formulation.params_.ee_phase_durations_.push_back(durations);
        }

        ifopt::Problem nlp;
        towr::SplineHolder solution;
        for (auto c : formulation.GetVariableSets(solution))
            nlp.AddVariableSet(c);
        for (auto c : formulation.GetConstraints(solution))
            nlp.AddConstraintSet(c);
        for (auto c : formulation.GetCosts())
            nlp.AddCostSet(c);

        auto solver = std::make_shared<ifopt::IpoptSolver>();
        solver->SetOption("jacobian_approximation", "exact");
        solver->SetOption("max_cpu_time", 2.);
        solver->SetOption("max_iter", 1000);
        solver->SetOption("print_level", 0);

        solver->Solve(nlp);

        double cost = x.sum() + GetNumViolations(nlp);

        return -cost;
    }

protected:
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
