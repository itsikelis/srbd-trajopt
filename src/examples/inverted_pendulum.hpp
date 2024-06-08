#include <ifopt/cost_term.h>
#include <ifopt/variable_set.h>

#define b 0.01
#define m 1.
#define l 1.
#define grav 9.82

#define N 20
#define N_state (N + 1)

#define DURATION 4.

#define DT static_cast<double>(DURATION) / static_cast<double>(N)

namespace ifopt {
    using Eigen::VectorXd;

    class InvertedPendulumVariables : public VariableSet {
    public:
        InvertedPendulumVariables() : InvertedPendulumVariables("var_set1"){};
        InvertedPendulumVariables(const std::string& name) : VariableSet(2 * N_state + N, name)
        {
            _values = VectorXd::Zero(GetRows());
            // Initial state is [0, 0]
            _values[0] = 0.;
            _values[N_state] = 0.;
            // Final state is [pi, 0]
            _values[N_state - 1] = M_PI;
            _values[2 * N_state - 1] = 0.;

            // Interpolation
            double dist = _values[N_state - 1] - _values[0];
            double avg_vel = dist / DURATION;
            for (int i = 1; i < N_state - 1; i++) {
                double p = i / static_cast<double>(N_state - 1);
                _values[i] = _values[0] + p * _values[N_state - 1];
                _values[N_state + i] = avg_vel;
            }
        }

        void SetVariables(const VectorXd& x) override
        {
            _values = x;
        }

        Eigen::VectorXd GetValues() const override
        {
            return _values;
        }

        VecBound GetBounds() const override
        {
            VecBound bounds(GetRows());
            for (int i = 0; i < GetRows(); i++) {
                bounds.at(i) = NoBound;
            }

            // Initial state is [0, 0]
            bounds.at(0) = BoundZero;
            bounds.at(N_state) = BoundZero;
            // Final state is [pi, 0]
            bounds.at(N_state - 1) = Bounds(M_PI, M_PI);
            bounds.at(2 * N_state - 1) = BoundZero;

            // Actuation bounds
            for (int i = 2 * N_state; i < GetRows(); i++) {
                bounds.at(i) = Bounds(-2.5, 2.5);
            }

            return bounds;
        }

    private:
        VectorXd _values;
    };

    class InvertedPendulumCost : public CostTerm {
    public:
        InvertedPendulumCost() : InvertedPendulumCost("cost_term1") {}
        InvertedPendulumCost(const std::string& name) : CostTerm(name) {}

        double GetCost() const override
        {
            // Get values and split to x and y.
            Eigen::VectorXd valueVector = GetVariables()->GetComponent("var_set1")->GetValues();

            // Take the last N values (controls of the returned values vector.)
            Eigen::VectorXd u = valueVector.tail(N);

            // Calculate cost function (sum of controls squared).
            double cost = 0;
            for (int i = 0; i < u.size(); i++) {
                cost += u(i) * u(i);
            }
            return cost;
        }

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
        {
            if (var_set == "var_set1") {
                VectorXd valueVector = GetVariables()->GetComponent("var_set1")->GetValues();
                Eigen::VectorXd u = valueVector.tail(N);

                for (int i = 0; i < u.size(); i++) {
                    jac_block.coeffRef(0, 2 * N_state + i) = 2 * u(i);
                }
            }
        }
    };

    class InvertedPendulumConstraints : public ConstraintSet {
    public:
        InvertedPendulumConstraints() : InvertedPendulumConstraints("constraint1") {}

        InvertedPendulumConstraints(const std::string& name) : ConstraintSet(2 * N, name) {}

        VectorXd GetValues() const override
        {
            VectorXd res(GetRows());
            VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();
            VectorXd q = x.head(N_state);
            VectorXd qDot = x.segment(N_state, N_state);
            VectorXd u = x.tail(N);

            double l_squared = l * l;

            for (int i = 0; i < N; i++) {
                // q euler integration
                res(i) = q[i + 1] - (q[i] + qDot[i] * DT);
                // qDot euler integration
                double qDDot = (u[i] - b * qDot[i] - m * grav * l * std::sin(q[i]) / 2.) / (m * l_squared / 3.);
                res(N + i) = qDot[i + 1] - (qDot[i] + qDDot * DT);
            }

            return res;
        }

        // All constraints are equal to zero.
        VecBound GetBounds() const override
        {
            // All constraints equal to zero.
            VecBound bounds(GetRows());
            for (int i = 0; i < GetRows(); i++) {
                bounds.at(i) = BoundZero;
            }
            return bounds;
        }

        void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override
        {
            if (var_set == "var_set1") {
                double l_squared = l * l;

                VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();
                VectorXd q = x.head(N_state);

                for (int i = 0; i < N; i++) {
                    // q euler integration derivatives
                    jac_block.coeffRef(i, i) = -1.; // w.r.t. q[i]
                    jac_block.coeffRef(i, i + 1) = 1.; // w.r.t. q[i+1]
                    jac_block.coeffRef(i, N_state + i) = -DT; // w.r.t. qDot[i]

                    // qDot euler integration derivatives
                    jac_block.coeffRef(N + i, i) = 3. * grav * std::cos(q[i]) * DT / (2. * l); // w.r.t. q[i]
                    jac_block.coeffRef(N + i, N_state + i) = -(1. - b * DT / (m * l_squared / 3.)); // w.r.t. qDot[i]
                    jac_block.coeffRef(N + i, N_state + i + 1) = 1.; // w.r.t. qDot[i+1]
                    jac_block.coeffRef(N + i, 2 * N_state + i) = -DT / (m * l_squared / 3.); // w.r.t. u[i]
                }
            }
        }
    };
} // namespace ifopt
