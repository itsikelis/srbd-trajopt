#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <fstream>
#include <iostream>

#include <eigen3/Eigen/Dense>

namespace trajopt {
    class TerrainGrid {
    public:
        using Grid = std::vector<double>;

        enum Direction { Normal,
            Tangent1,
            Tangent2 };

        enum Dim2D { X_ = 0,
            Y_ };

        TerrainGrid(size_t rows, size_t cols, double mu, double min_x, double min_y, double max_x, double max_y);

        ~TerrainGrid() {}

        void SetGrid(const Grid& new_grid) { grid_ = new_grid; }

        void SetZero();

        void ReadFromCsv(const std::string& filename);

        double GetHeight(double x, double y) const;

        Eigen::Vector3d GetNormalizedBasis(Direction direction, double x, double y) const;

        double GetDerivativeOfHeightWrt(Dim2D dim, double x, double y) const;

        Eigen::Vector3d GetDerivativeOfNormalizedBasisWrt(Direction direction, Dim2D dim, double x, double y) const;

        double GetFrictionCoeff() const { return mu_; }

    protected:
        inline double BilinearInterpolation(double x_norm, double y_norm) const;

        double GetSecondDerivativeOfHeightWrt(Dim2D dim1, Dim2D dim2, double x, double y) const;

        inline double GetHeightDerivWrtXX(double x, double y) const;
        inline double GetHeightDerivWrtXY(double x, double y) const;

        inline double GetHeightDerivWrtYX(double x, double y) const;
        inline double GetHeightDerivWrtYY(double x, double y) const;

        std::vector<double> Split(const std::string& s, char delimiter) const;

    protected:
        const double eps_ = 1e-6;
        size_t rows_, cols_;
        double mu_;
        double min_x_, min_y_, max_x_, max_y_;
        Grid grid_;
    };
} // namespace trajopt
