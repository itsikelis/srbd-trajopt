#pragma once

#include <cassert>

#include <Eigen/Dense>

#include <trajopt/robo_spline/types.hpp>

namespace trajopt::rspl {
    template <size_t _Dim>
    class Component {
    public:
        using VecD = Eigen::Matrix<double, _Dim, 1>;

        Component() : _T(-1.) {}
        Component(Time duration) : _T(duration) {}

        ~Component() {}

        inline size_t dim() const { return _Dim; };

        inline Time duration() const { return this->_T; };

        inline void set_duration(Time duration)
        {
            assert(("Duration should be a non-zero positive value!", duration > 0.));

            _T = duration;
        }

        virtual VecD eval(Time, size_t) const = 0;
        virtual Jacobian jac_block(Time, size_t) const = 0;

    protected:
        Time _T{-1.};
    };
} // namespace trajopt::rspl
