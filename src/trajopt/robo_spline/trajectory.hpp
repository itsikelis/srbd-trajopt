#pragma once

#include <memory>

#include <cassert>

#include <trajopt/robo_spline/component.hpp>
#include <trajopt/robo_spline/cubic_hermite_spline.hpp>
#include <trajopt/robo_spline/types.hpp>

namespace trajopt::rspl {
    template <size_t _Dim>
    class Trajectory : public Component<_Dim> {
    public:
        using VecD = typename Component<_Dim>::VecD;
        using Spline = CubicHermiteSpline<_Dim>;

    public:
        Trajectory() : Component<_Dim>(-1.) {}

        Trajectory(size_t num_vars) : Component<_Dim>(), _num_vars(num_vars), _num_knot_points(num_vars / (2 * _Dim)) {}

        Trajectory(const Vector& knot_points, const Vector& spline_times) : Component<_Dim>(spline_times.sum()), _num_vars(knot_points.rows()), _num_knot_points(spline_times.rows() + 1)
        {
            // Assert knot_points and times size are correct.
            // if (static_cast<size_t>(knot_points.rows()) != _Dim * 2 * (spline_times.rows() + 1)) {
            //   std::cerr << "Error: Knot points and spline times mismatch!" << std::endl;
            //   std::cerr << "Knot points vector should be: " << _Dim * 2 * (_num_knot_points) << " rows long, but is " << knot_points.rows() << " instead." << std::endl;
            // }

            // std::string message = "Knot points vector should be: " + std::to_string(_Dim * 2 * (_num_knot_points)) + " rows long, but is " + std::to_string(knot_points.rows()) + " instead.";
            assert(static_cast<size_t>(knot_points.rows()) == _Dim * 2 * (spline_times.rows() + 1));

            // Generate Trajectory from knot points.
            const size_t num_splines = _num_knot_points - 1;
            for (size_t i = 0; i < num_splines; ++i) {
                Eigen::VectorXd spline_knots = knot_points.segment(i * 2 * _Dim, 4 * _Dim);
                Time dt = spline_times[i];
                _splines.push_back(std::make_shared<Spline>(spline_knots, dt));
            }
        }

        ~Trajectory() {}

        void add_point(const VecD& next_pos, const VecD& next_vel, double dt = 0.)
        {

            // std::cout << "Adding point: (" << next_pos.transpose() << "), (" << next_vel.transpose() << ") -> " << dt << std::endl;

            // Check if it's the first point added to trajectory.
            if (this->_T < 0.) {

                assert(dt == 0. && "You cannot have a duration > 0. for the initial point!");

                _last_pos = next_pos;
                _last_vel = next_vel;
                this->_T = 0.;

                return;
            }

            Vector spline_knots = Vector::Zero(4 * _Dim);
            spline_knots << _last_pos, _last_vel, next_pos, next_vel;
            this->_splines.push_back(std::make_shared<Spline>(spline_knots, dt));
            this->_T += dt;

            _last_pos = next_pos;
            _last_vel = next_vel;
        }

        /// @brief Evaluate trajectory at time t.
        /// @param t time
        /// @param order 0, 1, 2, ... for position, velocity, acceleration etc.
        inline VecD eval(double t, size_t order) const override
        {
            std::pair<SplineIndex, Time> pair = normalise_time(t);
            SplineIndex idx = pair.first;
            Time t_norm = pair.second;

            return _splines[idx]->eval(t_norm, order);
        }

        /// @brief Clear trajectory splines.
        virtual void clear()
        {
            _splines.clear();
            this->_T = -1.;

            _num_knot_points = 0;
            _num_vars = 0;
        }

        inline size_t num_knot_points() const
        {
            assert(_num_knot_points > 0 && "Knot points not set!");

            return _num_knot_points;
        }

        inline size_t num_vars() const
        {
            assert(_num_vars > 0 && "Num vars not set!");

            return _num_vars;
        }

        inline std::shared_ptr<Spline> spline(size_t idx) const
        {
            assert(_splines.size() > 0 && "Splines vector empty!");

            return _splines.at(idx);
        }

        inline size_t dim() const { return _Dim; }

        std::vector<std::shared_ptr<Spline>>& splines() { return _splines; }

        virtual Jacobian jac_block(Time t, size_t order) const override
        {
            std::pair<SplineIndex, Time> pair = normalise_time(t);
            SplineIndex idx = pair.first;
            Time t_norm = pair.second;

            return jac_block(idx, t_norm, order);
        }

        virtual Jacobian jac_block(SplineIndex idx, Time t_norm, size_t order) const
        {
            Jacobian jac(_num_vars, _Dim);

            Jacobian spline_jac = _splines[idx]->jac_block(t_norm, order);

            jac.middleRows(idx * 2 * _Dim, spline_jac.cols()) = spline_jac.transpose();

            return jac.transpose();
        }

    protected:
        std::pair<SplineIndex, Time> normalise_time(Time t) const
        {
            assert(this->_T > 0. && "No splines added before evaluation!");

            double sum = 0.;
            double prev_sum = 0.;
            size_t iters = _splines.size();
            for (size_t i = 0; i < iters; ++i) {
                sum += _splines[i]->duration();

                if (t <= sum - _epsilon) {
                    Time t_norm = (t - prev_sum);
                    return std::make_pair(i, t_norm);
                }

                prev_sum = sum;
            }

            // If t > this->_T, return final time of last spline.
            return std::make_pair(static_cast<size_t>(_splines.size() - 1), _splines.back()->duration());
        }

    protected:
        static constexpr double _epsilon = 1e-12;

        size_t _num_vars{0};
        size_t _num_knot_points{0};
        std::vector<std::shared_ptr<Spline>> _splines;

        VecD _last_pos{VecD::Zero()};
        VecD _last_vel{VecD::Zero()};
    };
} // namespace trajopt::rspl
