//|
//|    Copyright (c) 2022-2023 Computational Intelligence Lab, University of Patras, Greece
//|    Copyright (c) 2023 Laboratory of Automation and Robotics, University of Patras, Greece
//|    Copyright (c) 2022-2023 Konstantinos Chatzilygeroudis
//|    Authors:  Konstantinos Chatzilygeroudis
//|    email:    costashatz@gmail.com
//|    website:  https://nosalro.github.io/
//|              http://cilab.math.upatras.gr/
//|              https://lar.ece.upatras.gr/
//|
//|    This file is part of numopt.
//|
//|    All rights reserved.
//|
//|    Redistribution and use in source and binary forms, with or without
//|    modification, are permitted provided that the following conditions are met:
//|
//|    1. Redistributions of source code must retain the above copyright notice, this
//|       list of conditions and the following disclaimer.
//|
//|    2. Redistributions in binary form must reproduce the above copyright notice,
//|       this list of conditions and the following disclaimer in the documentation
//|       and/or other materials provided with the distribution.
//|
//|    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//|    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//|    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//|    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//|    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//|    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//|    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//|    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//|    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//|    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//|
#ifndef NUMOPT_ALGO_ALM_HPP
#define NUMOPT_ALGO_ALM_HPP

#include <Eigen/Core>
#include <Eigen/Dense>

namespace numopt {
    namespace algo {

        template <typename Fit, typename Scalar = double>
        class AugmentedLagrangianMethod {
        public:
            using mat_t = Eigen::Matrix<Scalar, -1, -1>;
            using x_t = Eigen::Matrix<Scalar, -1, 1>;
            using g_t = Eigen::Matrix<Scalar, 1, -1>;

            struct Params {
                x_t initial_x;
                x_t initial_lambda;
                x_t initial_rho;

                unsigned int dim = 0;
                unsigned int dim_eq = 0;
                unsigned int dim_ineq = 0;

                Scalar rho_a = 10.;
                Scalar max_rho = 10000.;
                Scalar rho_eps = 1e-6;

                unsigned int newton_max_iters = 50;
                Scalar newton_eps = 1e-3;

                unsigned int eq_newton_max_iters = 1;
                Scalar eq_newton_eps = 1e-3;

                unsigned int cg_solve_max_iters = 400;
                Scalar cg_solve_eps = 1e-6;

                Scalar armijo_beta = 1e-3;
                Scalar armijo_min_alpha = 1e-6;
                Scalar armijo_decay = 0.5;

                Scalar eps = 1e-6;
            };

            struct IterationLog {
                unsigned int iterations = 0;
                unsigned int func_evals = 0;
                unsigned int cons_evals = 0;
                unsigned int grad_evals = 0;
                unsigned int hessian_evals = 0;
                unsigned int cjac_evals = 0;

                Scalar f;
                x_t c;
            };

            AugmentedLagrangianMethod(const Params& params, const Fit& fit) : _params(params), _fit(fit)
            {
                assert(params.initial_x.size() == params.dim && "Initial point size and dimensions should match!");
                assert(params.initial_lambda.size() == (params.dim_eq + params.dim_ineq) && "Initial lambda size and constraint dimensions should match!");
                assert((params.initial_rho.array() > 0.).all() && "Initial rho should be bigger than zero!");
                assert(params.rho_a > 1. && "rho update needs to be great than one!");
                _x_k = params.initial_x;
                _lambda_k = params.initial_lambda;
                _rho = params.initial_rho;
            }

            const x_t x() const { return _x_k; }
            const x_t lambda() const { return _lambda_k; }
            const x_t rho() const { return _rho; }

            IterationLog step()
            {
                const auto dim_eq = _params.dim_eq;
                const auto dim_ineq = _params.dim_ineq;
                const auto rho_a = _params.rho_a;

                // First let's solve only the equality constraints
                if (_log.iterations == 0 && dim_eq > 0) {
                    std::tie(_x_k, _lambda_k) = _solve_equality(_x_k, _lambda_k);
                }

                // Solve unconstrained problem
                x_t c;
                std::tie(_x_k, c) = _newton_solve(_x_k, _lambda_k, _rho);

                // Update equality duals
                for (unsigned int i = 0; i < dim_eq; i++) {
                    _lambda_k[i] = _lambda_k[i] + _rho[i] * c[i];
                }

                // Update inequality duals
                for (unsigned int i = dim_eq; i < dim_eq + dim_ineq; i++) {
                    _lambda_k[i] = std::max(Scalar(0.), _lambda_k[i] + _rho[i] * c[i]);
                }

                // Update rhos
                for (unsigned int i = 0; i < dim_eq + dim_ineq; i++) {
                    Scalar cv = std::abs(c[i]);
                    if (cv > _params.rho_eps && (_log.iterations == 0 || std::abs(_log.c[i]) < cv))
                        _rho[i] = std::min(rho_a * _rho[i], _params.max_rho);
                    else if (cv <= _params.rho_eps) {
                        _rho[i] = std::max(_rho[i] / rho_a, 0.5 * std::sqrt(std::abs(_lambda_k[i]) / _params.rho_eps));
                    }
                }

                _log.iterations++;
                _log.c = c;
                _log.f = _eval(_x_k); // TO-DO: How can we get this without doing an extra evaluation!?

                return _log;
            }

        protected:
            // Parameters
            Params _params;

            // Current Estimate
            x_t _x_k;

            // Current Dual Variables
            x_t _lambda_k;

            // Current rho
            x_t _rho;

            // Fitness structure
            Fit _fit;

            // Iteration Log
            IterationLog _log;

            Scalar _eval(const x_t& x)
            {
                _log.func_evals++;
                return _fit.f(x);
            }

            x_t _c_eval(const x_t& x)
            {
                _log.cons_evals++;
                return _fit.c(x);
            }

            g_t _grad_eval(const x_t& x)
            {
                _log.grad_evals++;
                return _fit.df(x);
            }

            mat_t _hessian_eval(const x_t& x)
            {
                _log.hessian_evals++;
                return _fit.ddf(x);
            }

            mat_t _cjac_eval(const x_t& x)
            {
                _log.cjac_evals++;
                return _fit.dc(x);
            }

            std::pair<x_t, x_t> _solve_equality(const x_t& x, const x_t& l)
            {
                const auto dim = _params.dim;
                const auto dim_eq = _params.dim_eq;
                const auto max_iters = (_params.eq_newton_max_iters > 0) ? _params.eq_newton_max_iters : _params.newton_max_iters;
                const auto newton_eps = (_params.eq_newton_eps > 0.) ? _params.eq_newton_eps : _params.newton_eps;
                const auto armijo_beta = _params.armijo_beta;
                const auto armijo_min_alpha = _params.armijo_min_alpha;
                const auto armijo_decay = _params.armijo_decay;

                x_t delta = x_t::Ones(dim + dim_eq);

                x_t x_k = x;
                x_t l_k = l;
                x_t r_k = _rho;

                unsigned int iter = 0;
                while (delta.norm() > newton_eps) {
                    x_t c = _c_eval(x_k).head(dim_eq);
                    mat_t C = _cjac_eval(x_k).block(0, 0, dim_eq, dim);

                    mat_t H = mat_t::Zero(dim + dim_eq, dim + dim_eq);
                    H.block(0, 0, dim, dim) = _hessian_eval(x_k);
                    H.block(dim, 0, dim_eq, dim) = C;
                    H.block(0, dim, dim, dim_eq) = C.transpose();

                    x_t v(dim + dim_eq);
                    v.head(dim) = -(_grad_eval(x_k).transpose() + C.transpose() * l_k.head(dim_eq));
                    v.tail(dim_eq) = -c;

                    // TO-DO: Add scaling/pre-conditioner/regularization?
                    // TO-DO: Check how I can avoid squaring (H.T@..)
                    delta = _cg_solve(H.transpose() * H, H.transpose() * v, x_t::Zero(v.size()));

                    // Armijo rule line search with AL as the merit function
                    Scalar a = 1.;
                    // TO-DO: Keep grad, c, c_jac in memory to avoid re-calling the same thing
                    Scalar prev_val = _al(x_k, l_k, r_k);
                    g_t prev_grad = _dal(x_k, l_k, r_k, true).head(dim + dim_eq);
                    x_t l_tmp = l_k;
                    l_tmp.head(dim_eq) = l_tmp.head(dim_eq) + a * delta.tail(dim_eq);
                    while (_al(x_k + a * delta.head(dim), l_tmp, r_k) > prev_val + armijo_beta * a * prev_grad * delta) {
                        a = armijo_decay * a;

                        l_tmp = l_k;
                        l_tmp.head(dim_eq) += a * delta.tail(dim_eq);

                        if (a < armijo_min_alpha)
                            break;
                    }

                    x_k = x_k + a * delta.head(dim);
                    l_k = l_tmp;

                    iter++;
                    if (iter >= max_iters)
                        break;
                }

                return {x_k, l_k};
            }

            x_t _cg_solve(const mat_t& A, const x_t& b, const x_t& x0) const
            {
                x_t x_k = x0;
                x_t g_k = b - A * x0;
                x_t g_k_1 = g_k;
                x_t d_k, d_k_1;

                for (unsigned int iter = 0; iter < _params.cg_solve_max_iters; iter++) {
                    if (g_k.norm() < _params.cg_solve_eps)
                        break;
                    if (iter >= 1) {
                        d_k_1 = d_k;
                        Scalar beta_k = -(g_k.transpose() * g_k)[0] / (g_k_1.transpose() * g_k_1)[0];
                        d_k = g_k - beta_k * d_k_1;
                    }
                    else
                        d_k = g_k;
                    g_k_1 = g_k;
                    Scalar a_k = (g_k.transpose() * g_k)[0] / (d_k.transpose() * A * d_k)[0];
                    x_k = x_k + a_k * d_k;
                    g_k = g_k - a_k * A * d_k;
                }

                return x_k;
            }

            std::pair<x_t, x_t> _newton_solve(const x_t& x, const x_t& l, const x_t& rho)
            {
                const auto dim_eq = _params.dim_eq;
                const auto dim_ineq = _params.dim_ineq;
                const auto armijo_beta = _params.armijo_beta;
                const auto armijo_min_alpha = _params.armijo_min_alpha;
                const auto armijo_decay = _params.armijo_decay;

                x_t x_k = x;
                x_t l_k = l;
                x_t r_k = rho;

                x_t c;
                mat_t C;
                mat_t residual;

                auto compute_residual = [&c, &C, &residual, &x_k, &l_k, &r_k, this, dim_eq, dim_ineq]() {
                    c = _c_eval(x_k);
                    C = _cjac_eval(x_k);
                    for (unsigned int i = dim_eq; i < dim_eq + dim_ineq; i++) {
                        if (c[i] < 0.) {
                            C.row(i).setZero();
                            c[i] = 0.;
                        }
                    }
                    residual = _grad_eval(x_k) + x_t(l_k.array() + r_k.array() * c.array()).transpose() * C;
                };

                compute_residual();
                g_t prev_grad = _dal(x_k, l_k, r_k);

                unsigned int iter = 0;
                while (residual.norm() > _params.newton_eps && prev_grad.norm() > _params.newton_eps) {
                    // Compute Hessian (Gauss-Newton, we skip the second derivatives of the constraints)
                    // TO-DO: We are doing too many copies here!
                    mat_t CC = C;
                    for (unsigned int i = 0; i < C.rows(); i++) {
                        CC.row(i) *= r_k[i];
                    }
                    mat_t H = _hessian_eval(x_k) + CC.transpose() * C;

                    // Compute Δx
                    // TO-DO: Add scaling/pre-conditioner/regularization?
                    x_t delta_x = _cg_solve(H, -residual.transpose(), x_t::Zero(x_k.size()));

                    // Armijo rule line search with AL as the merit function
                    Scalar a = 1.;
                    // TO-DO: Keep grad, c, c_jac in memory to avoid re-calling the same thing
                    Scalar prev_val = _al(x_k, l_k, r_k);
                    prev_grad = _dal(x_k, l_k, r_k);
                    while (_al(x_k + a * delta_x, l_k, r_k) > prev_val + armijo_beta * a * prev_grad * delta_x) {
                        a = armijo_decay * a;
                        if (a < armijo_min_alpha)
                            break;
                    }
                    // Step iterate x_k
                    x_k = x_k + a * delta_x;

                    // Compute residual
                    compute_residual();

                    iter++;
                    if (iter >= _params.newton_max_iters || delta_x.norm() <= _params.newton_eps)
                        break;
                }

                return {x_k, c};
            }

            Scalar _al(const x_t& x, const x_t& l, const x_t& rho)
            {
                const auto dim_eq = _params.dim_eq;
                const auto dim_ineq = _params.dim_ineq;

                x_t c = _c_eval(x);
                for (unsigned int i = dim_eq; i < dim_eq + dim_ineq; i++) {
                    if (c[i] < 0.) {
                        c[i] = 0.;
                    }
                }

                Scalar pen = static_cast<Scalar>(0.);
                for (unsigned int i = 0; i < dim_eq + dim_ineq; i++) {
                    pen += rho[i] / 2. * c[i] * c[i];
                }

                // return _eval(x) + l.transpose() * c + rho / 2. * c.transpose() * c;
                return _eval(x) + l.transpose() * c + pen;
            }

            g_t _dal(const x_t& x, const x_t& l, const x_t& rho, bool full = false)
            {
                const auto dim = _params.dim;
                const auto dim_eq = _params.dim_eq;
                const auto dim_ineq = _params.dim_ineq;

                x_t c = _c_eval(x);
                for (unsigned int i = dim_eq; i < dim_eq + dim_ineq; i++) {
                    if (c[i] < 0.) {
                        c[i] = 0.;
                    }
                }

                if (full) {
                    g_t dal(dim + dim_eq + dim_ineq);
                    dal << _grad_eval(x) + x_t(l.array() + rho.array() * c.array()).transpose() * _cjac_eval(x), c.transpose();
                    return dal;
                }

                // else
                g_t dal = _grad_eval(x) + x_t(l.array() + rho.array() * c.array()).transpose() * _cjac_eval(x);

                return dal;
            }
        };
    } // namespace algo
} // namespace numopt

#endif
