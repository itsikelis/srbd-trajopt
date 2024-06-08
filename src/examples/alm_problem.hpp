#include <Eigen/Dense>

#include <ifopt/problem.h>

template <typename Scalar = double>
class AlmProblem {
public:
    using mat_t = Eigen::Matrix<Scalar, -1, -1>;
    using x_t = Eigen::Matrix<Scalar, -1, 1>;
    using g_t = Eigen::Matrix<Scalar, 1, -1>;

    AlmProblem(const ifopt::Problem& nlp) : _nlp(nlp)
    {
        calc_dims_from_nlp(_nlp);
    }

    size_t dim() const { return _dim; }
    size_t dim_eq() const { return _dim_eq; }
    size_t dim_ineq() const { return _dim_ineq; }

    Scalar f(const x_t& x)
    {
        _nlp.SetVariables(x.data()); // TODO: Maybe this is not needed.
        Scalar cost = _nlp.EvaluateCostFunction(x.data());
        return cost;
    }

    g_t df(const x_t& x)
    {
        _nlp.SetVariables(x.data()); // TODO: Maybe this is not needed.
        g_t grad = _nlp.EvaluateCostFunctionGradient(x.data());
        // g_t grad = g_t::Zero(_dim);

        return grad;
    }

    mat_t ddf(const x_t& x)
    {
        mat_t hessian = mat_t::Zero(_dim, _dim);
        return hessian;
    }

    x_t c(const x_t& x)
    {
        x_t out = x_t::Zero(_dim_eq + _dim_ineq);
        x_t c = x_t::Zero(_dim_eq);
        x_t g = x_t::Zero(_dim_ineq);

        _nlp.SetVariables(x.data());
        fill_constraint_vectors(x, c, g);
        out << c, g;

        return out;
    }

    mat_t dc(const x_t& x)
    {
        mat_t Out = mat_t::Zero(_dim_eq + _dim_ineq, _dim);
        mat_t C = mat_t::Zero(_dim_eq, _dim);
        mat_t G = mat_t::Zero(_dim_ineq, _dim);

        _nlp.SetVariables(x.data());
        fill_constraint_jacobians(C, G);

        Out.block(0, 0, _dim_eq, _dim) = C;
        Out.block(_dim_eq, 0, _dim_ineq, _dim) = G;
        return Out;
    }

protected:
    void calc_dims_from_nlp(const ifopt::Problem& nlp)
    {
        _dim = static_cast<size_t>(nlp.GetNumberOfOptimizationVariables());

        _dim_eq = 0;
        _dim_ineq = 0;

        // From constraint sets
        for (const auto& ct : nlp.GetConstraints().GetComponents()) {
            int n = ct->GetRows();
            if (n < 0)
                continue;
            for (int i = 0; i < n; i++) {
                auto bounds = ct->GetBounds()[i];
                if (std::abs(bounds.lower_ - bounds.upper_) < 1e-8) {
                    // std::cout << "Equality" << " == " << bounds.upper_ << std::endl;
                    ++_dim_eq;
                }
                else {
                    if (bounds.lower_ > -1e20) {
                        // std::cout << "Inequality" << " >= " << bounds.lower_ << std::endl;
                        ++_dim_ineq;
                    }

                    if (bounds.upper_ < 1e20) {
                        // std::cout << "Inequality" << " <= " << bounds.upper_ << std::endl;
                        ++_dim_ineq;
                    }
                }
            }
        }

        // From variable bounds
        const auto& bounds = nlp.GetBoundsOnOptimizationVariables();
        for (size_t i = 0; i < _dim; ++i) {
            if (std::abs(bounds[i].lower_ - bounds[i].upper_) < 1e-8) {
                ++_dim_eq;
            }
            else {
                if (bounds[i].lower_ > -1e20) {
                    ++_dim_ineq;
                }

                if (bounds[i].upper_ < 1e20) {
                    ++_dim_ineq;
                }
            }
        }
    }

    void fill_constraint_vectors(const x_t& x, x_t& c, x_t& g)
    {
        // Fill equality and inequality constraint vectors.
        size_t next_eq = 0;
        size_t next_in = 0;

        // From constraint sets
        for (const auto& ct : _nlp.GetConstraints().GetComponents()) {
            int n = ct->GetRows();
            if (n < 0)
                continue;
            x_t vals = ct->GetValues();
            for (int i = 0; i < n; i++) {
                auto bounds = ct->GetBounds()[i];
                if (std::abs(bounds.lower_ - bounds.upper_) < 1e-8) {
                    c[next_eq] = vals[i] - bounds.lower_;
                    next_eq++;
                }
                else {
                    if (bounds.lower_ > -1e20) {
                        g[next_in] = -vals[i] + bounds.lower_;
                        next_in++;
                    }

                    if (bounds.upper_ < 1e20) {
                        g[next_in] = -bounds.upper_ + vals[i];
                        next_in++;
                    }
                }
            }
        }

        // From variable bounds
        const auto& bounds = _nlp.GetBoundsOnOptimizationVariables();
        for (size_t i = 0; i < _dim; i++) {
            if (std::abs(bounds[i].lower_ - bounds[i].upper_) < 1e-8) {
                c[next_eq] = x[i] - bounds[i].lower_;
                next_eq++;
            }
            else {
                if (bounds[i].lower_ > -1e20) {
                    g[next_in] = -x[i] + bounds[i].lower_;
                    next_in++;
                }

                if (bounds[i].upper_ < 1e20) {
                    g[next_in] = -bounds[i].upper_ + x[i];
                    next_in++;
                }
            }
        }
    }

    void fill_constraint_jacobians(mat_t& C, mat_t& G) const
    {
        // Fill equality and inequality jacobians
        size_t next_eq = 0;
        size_t next_in = 0;

        // From constraint sets
        for (const auto& ct : _nlp.GetConstraints().GetComponents()) {
            int n = ct->GetRows();
            if (n < 0)
                continue;
            mat_t cons = ct->GetJacobian();
            for (int i = 0; i < n; i++) {
                auto bounds = ct->GetBounds()[i];
                if (std::abs(bounds.lower_ - bounds.upper_) < 1e-8) {
                    C.row(next_eq) = cons.row(i);

                    next_eq++;
                }
                else {
                    if (bounds.lower_ > -1e20) {
                        G.row(next_in) = -cons.row(i);

                        next_in++;
                    }

                    if (bounds.upper_ < 1e20) {
                        G.row(next_in) = cons.row(i);

                        next_in++;
                    }
                }
            }
        }

        // From variable bounds
        const auto& bounds = _nlp.GetBoundsOnOptimizationVariables();
        for (size_t i = 0; i < _dim; ++i) {
            if (std::abs(bounds[i].lower_ - bounds[i].upper_) < 1e-8) {
                C(next_eq, i) = 1.;
                next_eq++;
            }
            else {
                if (bounds[i].lower_ > -1e20) {
                    G(next_in, i) = -1.;
                    next_in++;
                }

                if (bounds[i].upper_ < 1e20) {
                    G(next_in, i) = 1.;
                    next_in++;
                }
            }
        }
    }

protected:
    ifopt::Problem _nlp;

    size_t _dim{0};

    size_t _dim_eq{0}; // Number of equality constraints.
    size_t _dim_ineq{0}; // Number of inequality constraints.
};
