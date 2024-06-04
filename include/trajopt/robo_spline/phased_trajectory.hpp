#pragma once

#include <cassert>
#include <numeric>

#include <trajopt/robo_spline/component.hpp>
#include <trajopt/robo_spline/trajectory.hpp>
#include <trajopt/robo_spline/types.hpp>

namespace trajopt::rspl {
    template <size_t _Dim>
    class PhasedTrajectory : public Trajectory<_Dim> {
    public:
        using VecD = typename Component<_Dim>::VecD;

    public:
        PhasedTrajectory() : Trajectory<_Dim>() {}

        PhasedTrajectory(const Vector& knot_points, const Vector& phase_times,
            const std::vector<size_t>& knots_per_swing, Phase init_phase)
            : Trajectory<_Dim>(knot_points.rows())
        {
            size_t num_phases = phase_times.rows();
            size_t num_swing_phases = knots_per_swing.size();
            size_t num_stance_phases = num_phases - num_swing_phases;

            // std::cout << "Total, Swing, Step: " << num_phases << ", " << num_swing_phases << ", " << num_stance_phases << std::endl;

            size_t num_knot_points = 2 * num_stance_phases + std::accumulate(knots_per_swing.begin(), knots_per_swing.end(), 0);
            // std::cout << "Total Knot Points: " << num_knot_points << std::endl;

            _is_stance.resize(num_knot_points - 1);
            _var_start.resize(num_knot_points - 1);

            bool is_stance = (init_phase == Phase::Stance) ? true : false;

            size_t v_idx = 0;
            int k_idx = 0; // TODO: change to size_t
            size_t swing_idx = 0;

            for (size_t k = 0; k < num_phases; ++k) {
                if (is_stance) {
                    _is_stance[k_idx] = is_stance;
                    _var_start[k_idx++] = v_idx;

                    v_idx += _Dim;
                }
                else {
                    if (v_idx >= _Dim) {
                        v_idx -= _Dim;
                    }

                    size_t swing_knots = knots_per_swing[swing_idx];
                    swing_idx++;
                    for (size_t j = 0; j < swing_knots; ++j) {
                        _is_stance[k_idx] = is_stance;
                        _var_start[k_idx++] = v_idx;

                        if (j == 0 && k_idx > 1)
                            v_idx += _Dim;
                        else
                            v_idx += 2 * _Dim;
                    }

                    if (k > 0 && k_idx < static_cast<int>(_is_stance.size() - 1)) {
                        _is_stance[k_idx] = is_stance;
                        _var_start[k_idx++] = v_idx;
                        v_idx += 2 * _Dim;
                    }
                }

                is_stance = !is_stance;
            }

            // generate poly_times vector.
            is_stance = _is_stance[0];
            bool starts_with_swing = (!_is_stance[0]);
            bool ends_with_swing = (!_is_stance.back());
            std::vector<Time> poly_times;
            Time t = 0;
            size_t t_idx = 0;
            swing_idx = 0;

            for (size_t i = 0; i < num_phases; ++i) {
                if (is_stance) {
                    t = phase_times[t_idx];
                    poly_times.push_back(t);
                    t_idx++;
                }
                else {
                    size_t iters = 0;

                    size_t swing_knots = knots_per_swing[swing_idx];
                    bool first_swing = (swing_idx == 0);
                    bool last_swing = (swing_idx == knots_per_swing.size() - 1);

                    if (first_swing && starts_with_swing) {
                        iters = swing_knots;
                    }
                    else if (last_swing && ends_with_swing) {
                        iters = swing_knots;
                    }
                    else {
                        iters = swing_knots + 1;
                    }
                    t = phase_times[t_idx] / static_cast<double>(iters);
                    for (size_t k = 0; k < iters; ++k) {
                        poly_times.push_back(t);
                    }

                    swing_idx++;
                    t_idx++;
                }

                is_stance = !is_stance;
            }

            const VecD zero_vel = VecD::Zero();

            is_stance = _is_stance[0];
            k_idx = -1;
            v_idx = 0;
            swing_idx = 0;

            VecD pos, vel;
            double dt = 0.;

            for (size_t k = 0; k < num_phases; ++k) {
                if (is_stance) {
                    pos = knot_points.segment(v_idx, _Dim);
                    this->add_point(pos, zero_vel, dt);
                    // std::cout << "Adding point: (" << pos.transpose() << "), (" << zeroVel.transpose() << ") -> " << dt << std::endl;

                    dt = poly_times[++k_idx];
                    this->add_point(pos, zero_vel, dt);
                    // std::cout << "Adding point: (" << pos.transpose() << "), (" << zeroVel.transpose() << ") -> " << dt << std::endl;

                    v_idx += _Dim;
                    k_idx++;
                    dt = poly_times[k_idx];
                }
                else {
                    size_t swing_knots = knots_per_swing[swing_idx];
                    swing_idx++;
                    for (size_t j = 0; j < swing_knots; ++j) {
                        pos = knot_points.segment(v_idx, _Dim);
                        vel = knot_points.segment(v_idx + _Dim, _Dim);

                        this->add_point(pos, vel, dt);

                        // std::cout << "Adding point: (" << pos.transpose() << "), (" << vel.transpose() << ") -> " << dt << std::endl;

                        k_idx++;
                        v_idx += 2 * _Dim;
                        dt = poly_times[k_idx];
                    }
                }

                is_stance = !is_stance;
            }
            //////////////////////

            // for (auto& item : _var_start) {
            //     std::cout << item << ", ";
            // }
            // std::cout << std::endl;
        }

        ~PhasedTrajectory() {}

        void clear() override
        {
            Trajectory<_Dim>::clear();

            _is_stance.clear();
            _var_start.clear();
        }

        Jacobian jac_block(Time t, size_t order) const override
        {
            std::pair<SplineIndex, Time> pair = this->normalise_time(t);
            SplineIndex idx = pair.first;
            Time t_norm = pair.second;

            return jac_block(idx, t_norm, order);
        }

        Jacobian jac_block(SplineIndex idx, Time t_norm,
            size_t order) const override
        {
            Jacobian jac(this->_num_vars, _Dim);

            Jacobian spline_jac_T = this->_splines[idx]->jac_block(t_norm, order).transpose();

            // If current index is stance, keep 1st column.
            // If previous index is stance, keep 1st, 3rd and 4th.
            // If previous index is swing, keep all.
            //
            bool is_stance = _is_stance[idx];
            size_t s_idx = _var_start[idx];

            if (is_stance) {
                // If current spline is stance, keep p0, p1.
                // std::cout << "Stance " << _var_start[idx] << std::endl;
                jac.middleRows(s_idx, _Dim) = spline_jac_T.middleRows(0, _Dim);
                jac.middleRows(s_idx, _Dim) += spline_jac_T.middleRows(2 * _Dim, _Dim);
            }
            else {
                bool has_prev = (idx > 0);
                bool has_next = (idx < (_is_stance.size() - 1));
                bool prev_stance = (idx == 0) ? false : _is_stance[idx - 1];
                bool next_stance = (idx >= (_is_stance.size() - 1)) ? false : _is_stance[idx + 1];

                bool first_node = !has_prev && has_next;
                bool last_node = has_prev && !has_next;
                bool middle_node = has_prev && has_next;

                size_t p_idx = has_prev ? _var_start[idx - 1] : s_idx;
                size_t n_idx = has_next ? _var_start[idx + 1] : s_idx;

                // First node cases
                if (first_node && !next_stance)
                    jac.middleRows(s_idx, 4 * _Dim) = spline_jac_T;
                else if (first_node && next_stance) {
                    jac.middleRows(s_idx, 2 * _Dim) = spline_jac_T.middleRows(0, 2 * _Dim); // keep p0 v0
                    jac.middleRows(n_idx, _Dim) = spline_jac_T.middleRows(2 * _Dim, _Dim); // keep p1
                }

                // Last node cases
                if (last_node && prev_stance) {
                    jac.middleRows(p_idx, _Dim) = spline_jac_T.middleRows(0, _Dim); // keep p0
                    jac.middleRows(s_idx, 2 * _Dim) = spline_jac_T.middleRows(2 * _Dim, 2 * _Dim); // keep p1, v1
                }
                else if (last_node && !prev_stance) {
                    jac.middleRows(s_idx, 4 * _Dim) = spline_jac_T; // keep all.
                }

                if (middle_node) {
                    if (prev_stance) {
                        // If previous spline is stance, keep p0, p1 and v1.
                        jac.middleRows(p_idx, _Dim) = spline_jac_T.middleRows(0, _Dim); // keep p0
                        jac.middleRows(n_idx, 2 * _Dim) = spline_jac_T.middleRows(2 * _Dim, 2 * _Dim); // keep p1, v1
                    }
                    else if (!prev_stance && !next_stance) {
                        jac.middleRows(s_idx, 4 * _Dim) = spline_jac_T;
                    }
                    else if (!prev_stance && next_stance) {
                        jac.middleRows(s_idx, 3 * _Dim) = spline_jac_T.middleRows(0, 3 * _Dim); // jac.block(0, 0, jac.rows(), 9).transpose();
                    }
                }
            }

            return jac.transpose();
        }

        inline rspl::Phase spline_phase(size_t idx)
        {
            return (_is_stance[idx]) ? Phase::Stance : Phase::Swing;
        }
        size_t var_start(size_t idx) { return _var_start[idx]; }

        rspl::Phase phase_at(Time t) const
        {
            std::pair<SplineIndex, Time> pair = this->normalise_time(t);
            SplineIndex idx = pair.first;

            return (_is_stance[idx]) ? Phase::Stance : Phase::Swing;
            ;
        }

        size_t var_start_at(Time t) const
        {
            std::pair<SplineIndex, Time> pair = this->normalise_time(t);
            SplineIndex idx = pair.first;

            return _var_start[idx];
        }

    protected:
        // void _debug_print_indices(bool init_phase, size_t k_idx, size_t t_idx,
        // size_t p_idx) const
        // {
        //   std::string phase = (init_phase == true) ? "Stance" : "Swing";
        //   std::cout << "Phase: " + phase + "\t";
        //   std::cout << ", Knot points idx: " << k_idx << "\t";
        //   std::cout << ", Spline times idx: " << t_idx << "\t";
        //   std::cout << ", Knots per swing phase idx: " << p_idx << std::endl;
        // }

    protected:
        std::vector<bool> _is_stance;
        std::vector<size_t> _var_start; // Store in which index each spline's knot point starts.
    };
} // namespace trajopt::rspl
