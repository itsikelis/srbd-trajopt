#pragma once

#include <iostream>

#include <trajopt/robo_spline/component.hpp>
#include <trajopt/robo_spline/types.hpp>

namespace trajopt::rspl {
    template <size_t _Dim>
    class CubicHermiteSpline : public Component<_Dim> {
    public:
        using VecD = typename Component<_Dim>::VecD;

    public:
        CubicHermiteSpline(const Vector& knot_points, Time duration) : Component<_Dim>(duration)
        {
            // TODO: Assert knot points is 4 * _Dim in size.
            // TODO: Assert duration > 0.
            _p0 = knot_points.head(_Dim); // p0
            _v0 = knot_points.segment(_Dim, _Dim); // v0
            _p1 = knot_points.segment(2 * _Dim, _Dim); // p1
            _v1 = knot_points.tail(_Dim); // v1

            calc_coeffs_from_knot_points();
        }

        ~CubicHermiteSpline() {}

        VecD eval(Time t, size_t order) const override
        {
            switch (order) {
            case 0:
                return pos(t);
            case 1:
                return vel(t);
            case 2:
                return acc(t);
            default:
                std::cerr << "Invalid derivative order!" << std::endl;
                return VecD::Zero();
            }
        }

        Jacobian jac_block(Time t, size_t order) const override
        {
            Jacobian jac(_Dim, _Dim * 4);

            Vector deriv;
            switch (order) {
            case 0:
                deriv = deriv_pos(t);
                break;
            case 1:
                deriv = deriv_vel(t);
                break;
            case 2:
                deriv = deriv_acc(t);
                break;
            default:
                std::cerr << "Invalid derivative order!" << std::endl;
                return Jacobian();
            }

            for (size_t i = 0; i < _Dim; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    jac.coeffRef(i, i + j * _Dim) = deriv[j];
                }
            }

            return jac;
        }

    protected:
        inline void calc_coeffs_from_knot_points()
        {
            const double o_T = 1. / this->_T;
            const double o_T2 = 1. / (this->_T * this->_T);
            const double o_T3 = 1. / (this->_T * this->_T * this->_T);

            _c0 = _p0;
            _c1 = _v0;
            _c2 = 3. * _p1 * o_T2 - 3. * _p0 * o_T2 - 2. * _v0 * o_T - _v1 * o_T;
            _c3 = -2. * _p1 * o_T3 + 2. * _p0 * o_T3 + _v0 * o_T2 + _v1 * o_T2;
        }

        inline VecD pos(Time t) const
        {
            const double t2 = t * t;
            const double t3 = t * t2;

            return _c0 + (_c1 * t) + (_c2 * t2) + (_c3 * t3);
        }

        inline VecD vel(Time t) const
        {
            return _c1 + (2. * _c2 * t) + (3. * _c3 * t * t);
        }

        inline VecD acc(Time t) const
        {
            return (2. * _c2) + (6. * _c3 * t);
        }

        inline Vector deriv_pos(Time t) const
        {
            Vector deriv = Vector::Zero(4);

            const double t2 = t * t;
            const double t3 = t * t2;

            const double o_T = 1. / this->_T;
            const double o_T2 = 1. / (this->_T * this->_T);
            const double o_T3 = 1. / (this->_T * this->_T * this->_T);

            // Position spline derivatives w.r.t. p0, v0, p1, v1.
            deriv[0] = 1. - 3. * t2 * o_T2 + 2. * t3 * o_T3;
            deriv[1] = t - 2. * t2 * o_T + t3 * o_T2;
            deriv[2] = 3. * t2 * o_T2 - 2. * t3 * o_T3;
            deriv[3] = -t2 * o_T + t3 * o_T2;

            return deriv;
        }

        inline Vector deriv_vel(Time t) const
        {
            const double t2 = t * t;
            Vector deriv = Vector::Zero(4);

            const double o_T = 1. / this->_T;
            const double o_T2 = 1. / (this->_T * this->_T);
            const double o_T3 = 1. / (this->_T * this->_T * this->_T);

            // Velocity spline derivatives w.r.t. x0, v0, x1, v1.
            deriv[0] = -6. * t * o_T2 + 6. * t2 * o_T3;
            deriv[1] = 1. - 4. * t * o_T + 3. * t2 * o_T2;
            deriv[2] = 6. * t * o_T2 - 6. * t2 * o_T3;
            deriv[3] = -2. * t * o_T + 3. * t2 * o_T2;

            return deriv;
        }

        inline Vector deriv_acc(Time t) const
        {
            Vector deriv = Vector::Zero(4);

            const double o_T = 1. / this->_T;
            const double o_T2 = 1. / (this->_T * this->_T);
            const double o_T3 = 1. / (this->_T * this->_T * this->_T);

            // Acceleration spline derivatives w.r.t. x0, v0, x1, v1.
            deriv[0] = -6. * o_T2 + 12. * t * o_T3;
            deriv[1] = -4. * o_T + 6. * t * o_T2;
            deriv[2] = 6. * o_T2 - 12. * t * o_T3;
            deriv[3] = -2. * o_T + 6. * t * o_T2;

            return deriv;
        }

    protected:
        // Initial and final knot points.
        VecD _p0;
        VecD _v0;
        VecD _p1;
        VecD _v1;

        // Spline coefficients.
        VecD _c0;
        VecD _c1;
        VecD _c2;
        VecD _c3;
    };
} // namespace trajopt::rspl
