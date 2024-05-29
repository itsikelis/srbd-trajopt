#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace trajopt {
    class TerrainGrid {
    public:
        using Grid = std::vector<double>;

    public:
        TerrainGrid(size_t rows,
            size_t cols,
            double mu,
            size_t min_x,
            size_t min_y,
            size_t max_x,
            size_t max_y) : _rows(rows),
                            _cols(cols),
                            _mu(mu),
                            _min_x(min_x),
                            _min_y(min_y),
                            _max_x(max_x),
                            _max_y(max_y)
        {
            _grid.resize(_rows * _cols);
        }

        ~TerrainGrid() {}

        void set_grid(const Grid& new_grid)
        {
            _grid = new_grid;
        }

        void read_from_csv(const std::string& filename)
        {
            std::ifstream file(filename);

            if (!file.is_open()) {
                std::cerr << "Failed to open csv file." << std::endl;
                return;
            }

            std::string line;
            for (size_t i = 0; i < _rows; ++i) {
                std::getline(file, line);
                auto tokens = split(line, ',');

                _grid.insert(_grid.begin() + i * _cols, tokens.begin(), tokens.end());
            }

            file.close();
        }

        inline double grid_height(size_t x_idx, size_t y_idx) const
        {
            return _grid[x_idx * _cols + y_idx];
        }

        inline double height(double x, double y) const
        {
            double x_norm = _rows * (x - static_cast<double>(_min_x)) / (static_cast<double>(_max_x) - static_cast<double>(_min_x));
            double y_norm = _cols * (y - _min_y) / (_max_y - _min_y);

            // std::cout << x_norm << "," << y_norm << std::endl;

            return (height_pixel(x_norm, y_norm));
        }

        double height_pixel(double x, double y) const
        {
            // Check for edge cases.
            if (near_edge(x, y)) {
                // std::cout << "Near Edge!" << std::endl;
                return grid_height(std::floor(x), std::floor(y));
            }

            // Determine interpolation indices.
            double x_1 = std::floor(x);
            double x_2 = std::ceil(x);
            double y_1 = std::floor(y);
            double y_2 = std::ceil(y);

            return bilinear_interpolation(x, y, x_1, y_1, x_2, y_2);
        }

        Eigen::Vector2d jacobian(double x, double y) const
        {
            Eigen::Vector2d jac;
            // Determine interpolation indices.
            double x_norm = _rows * (x - static_cast<double>(_min_x)) / (static_cast<double>(_max_x) - static_cast<double>(_min_x));
            double y_norm = _cols * (y - _min_y) / (_max_y - _min_y);
            double x_1 = std::floor(x_norm);
            double x_2 = std::ceil(x_norm);
            double y_1 = std::floor(y_norm);
            double y_2 = std::ceil(y_norm);

            if (x_1 == x_2 || y_1 == y_2) {
                return Eigen::Vector2d::Zero();
            }

            double denom = (x_2 - x_1) * (y_2 - y_1);

            Eigen::Matrix2d Q;
            Q.coeffRef(0, 0) = grid_height(x_1, y_1);
            Q.coeffRef(0, 1) = grid_height(x_1, y_2);
            Q.coeffRef(1, 0) = grid_height(x_2, y_1);
            Q.coeffRef(1, 1) = grid_height(x_2, y_2);

            // dfdx
            Eigen::Vector2d Ax;
            Ax[0] = x_2;
            Eigen::Vector2d Cx;
            Cx[0] = y_2 - y;
            Cx[1] = y - y_1;

            // std::cout << "Res 1: " << Q * Cx << std::endl;
            // std::cout << "Res 2: " << Ax.dot(Q * Cx) << std::endl;

            double dfdx;
            if (denom == 0) {
                dfdx = 0;
            }
            else {
                dfdx = (1 / denom) * Ax.dot(Q * Cx);
            }
            jac[0] = dfdx;

            // dfdy
            Eigen::Vector2d Ay;
            Ay[0] = x_2 - x;
            Ay[1] = x - x_1;

            Eigen::Vector2d Cy;
            Cy[0] = y_2;
            Cy[1] = -y_1;

            double dfdy;
            if (denom == 0) {
                dfdy = 0;
            }
            else {
                dfdy = (1 / denom) * Ay.dot(Q * Cy);
            }
            jac[1] = dfdy;

            // std::cout << "Jacobian: " << jac.transpose() << std::endl;

            return jac;
        }

        inline double mu() const
        {
            return _mu;
        }

        Eigen::Vector3d n(double x, double y) const
        {
            Eigen::Vector3d n;

            auto deriv = jacobian(x, y);
            double dx = deriv[0];
            double dy = deriv[1];
            n << -dx, -dy, 1.;
            // n << 0., 0., 1.;

            return n.normalized();
        }

        Eigen::Vector3d t(double x, double y) const
        {
            Eigen::Vector3d t;

            auto deriv = jacobian(x, y);
            t << 1., 0., deriv[0];
            // t << 1., 0., 0.;

            return t.normalized();
        }

        Eigen::Vector3d b(double x, double y) const
        {
            Eigen::Vector3d b;

            auto deriv = jacobian(x, y);
            b << 0., 1., deriv[1];
            // b << 0., 1., 0.;

            return b.normalized();
        }

    protected:
        double bilinear_interpolation(double x, double y, double x_1, double y_1, double x_2, double y_2) const
        {
            double f_1_1 = grid_height(x_1, y_1);
            double f_1_2 = grid_height(x_1, y_2);
            double f_2_1 = grid_height(x_2, y_1);
            double f_2_2 = grid_height(x_2, y_2);

            // std::cout << "Heights: " << f_1_1 << ", " << f_1_2 << ", " << f_2_1 << ", " << f_2_2 << std::endl;

            // Interpolate in the x direction.
            // f(x, y_l)
            double f_y_l;
            double f_y_h;
            if (x_1 == x_2) {
                f_y_l = f_1_1;
                f_y_h = f_2_1;
            }
            else {
                f_y_l = ((x_2 - x) / (x_2 - x_1)) * f_1_1 + ((x - x_1) / (x_2 - x_1)) * f_2_1;
                f_y_h = ((x_2 - x) / (x_2 - x_1)) * f_1_2 + ((x - x_1) / (x_2 - x_1)) * f_2_2;
            }

            // Interpolate in the y direction.
            double res;
            if (y_1 == y_2) {
                res = f_1_1;
            }
            else {
                res = ((y_2 - y) / (y_2 - y_1)) * f_y_l + ((y - y_1) / (y_2 - y_1)) * f_y_h;
            }

            // std::cout << "Height: " << res << std::endl;

            return res;
        }

        bool near_edge(double x, double y) const
        {
            return (std::ceil(x) <= 0 || std::ceil(x) >= _rows || std::ceil(y) <= 0 || std::ceil(y) >= _cols);
        }

        std::vector<double> split(const std::string& s, char delimiter) const
        {
            std::vector<double> tokens;
            tokens.resize(_cols);
            std::string token;
            std::istringstream token_stream(s);
            for (auto& item : tokens) {
                std::getline(token_stream, token, delimiter);
                item = std::stod(token);
            }
            return tokens;
        }

    protected:
        size_t _rows, _cols;
        double _mu;
        int _min_x, _min_y, _max_x, _max_y;
        Grid _grid;
    };
} // namespace trajopt
