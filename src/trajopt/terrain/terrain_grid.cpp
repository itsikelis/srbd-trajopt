#include "terrain_grid.hpp"

#include <fstream>
#include <iostream>

namespace trajopt {
    TerrainGrid::TerrainGrid(size_t rows, size_t cols, double mu, double min_x, double min_y, double max_x, double max_y)
        : rows_(rows), cols_(cols), mu_(mu), min_x_(min_x), min_y_(min_y), max_x_(max_x), max_y_(max_y)
    {
        grid_.resize(rows_ * cols_);
    }

    double TerrainGrid::GetHeight(double x, double y) const
    {
        return BilinearInterpolation(x, y);
    }

    double TerrainGrid::GetDerivativeOfHeightWrt(Dim2D dim, double x, double y) const
    {
        return BilinearInterpolationDerivWrt(dim, x, y);
    }

    Eigen::Vector3d TerrainGrid::GetNormalizedBasis(Direction direction, double x, double y) const
    {
        Eigen::Vector3d vec = Eigen::Vector3d::Zero();
        double dx = 0.;
        double dy = 0.;
        switch (direction) {
        case Normal:
            dx = GetDerivativeOfHeightWrt(X_, x, y);
            dy = GetDerivativeOfHeightWrt(Y_, x, y);
            vec << -dx, -dy, 1.;
            break;
        case Tangent1:
            dx = GetDerivativeOfHeightWrt(X_, x, y);
            vec << 1., 0., dx;
            break;
        case Tangent2:
            dy = GetDerivativeOfHeightWrt(Y_, x, y);
            vec << 0., 1., dy;
            break;
        default:
            break;
        }
        return vec.normalized();
    }

    Eigen::Vector3d TerrainGrid::GetDerivativeOfNormalizedBasisWrt(Direction direction, Dim2D dim, double x, double y) const
    {
        Eigen::Vector3d deriv = Eigen::Vector3d::Zero();
        switch (direction) {
        case Normal:
            if (dim == X_) {
                double dxx = BilinearInterpolationSecondDerivWrt(X_, X_, x, y);
                double dyx = BilinearInterpolationSecondDerivWrt(Y_, X_, x, y);

                deriv << -dxx, -dyx, 0.;
                return deriv;
            }
            else {
                double dxy = BilinearInterpolationSecondDerivWrt(X_, Y_, x, y);
                double dyy = BilinearInterpolationSecondDerivWrt(Y_, Y_, x, y);

                deriv << -dxy, -dyy, 0.;
                return deriv;
            }
            break;
        case Tangent1:
            if (dim == X_) {
                double dxx = BilinearInterpolationSecondDerivWrt(X_, X_, x, y);
                deriv << 0., 0., dxx;
                return deriv;
            }
            else {
                double dxy = BilinearInterpolationSecondDerivWrt(X_, Y_, x, y);
                deriv << 0., 0., dxy;
                return deriv;
            }
            break;
        case Tangent2:
            if (dim == X_) {
                double dyx = BilinearInterpolationSecondDerivWrt(Y_, X_, x, y);
                deriv << 0., 0., dyx;
                return deriv;
            }
            else {
                double dyy = BilinearInterpolationSecondDerivWrt(Y_, Y_, x, y);
                deriv << 0., 0., dyy;
                return deriv;
            }
            break;
        default:
            break;
        }

        return deriv;
    }

    void TerrainGrid::SetZero()
    {
        for (auto& item : grid_) {
            item = 0.;
        }
    }

    void TerrainGrid::ReadFromCsv(const std::string& filename)
    {
        std::ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Failed to open csv file." << std::endl;
            return;
        }

        std::string line;
        for (size_t i = 0; i < rows_; ++i) {
            std::getline(file, line);
            auto tokens = Split(line, ',');

            grid_.insert(grid_.begin() + i * cols_, tokens.begin(), tokens.end());
        }

        file.close();
    }

    double TerrainGrid::BilinearInterpolation(double x, double y) const
    {
        double x_norm = std::max(0., std::min(static_cast<double>(rows_) - 1., rows_ * (x - min_x_) / (max_x_ - min_x_)));
        double y_norm = std::max(0., std::min(static_cast<double>(cols_) - 1., cols_ * (y - min_y_) / (max_y_ - min_y_)));

        double height = 0.;

        size_t x1 = static_cast<size_t>(std::floor(x_norm));
        size_t x2 = static_cast<size_t>(std::ceil(x_norm));
        size_t y1 = static_cast<size_t>(std::floor(y_norm));
        size_t y2 = static_cast<size_t>(std::ceil(y_norm));

        if (x1 == x2 || y1 == y2) {
            return 0.;
        }

        double fx1y1 = grid_.at(x1 * cols_ + y1);
        double fx1y2 = grid_.at(x1 * cols_ + y2);
        double fx2y1 = grid_.at(x2 * cols_ + y1);
        double fx2y2 = grid_.at(x2 * cols_ + y2);

        double fxy1 = 0.; // Interpolate y1 in the x direction.
        double fxy2 = 0.; // Interpolate y2 in the x direction.

        fxy1 = ((x2 - x_norm) / (x2 - x1)) * fx1y1 + ((x_norm - x1) / (x2 - x1)) * fx2y1;
        fxy2 = ((x2 - x_norm) / (x2 - x1)) * fx1y2 + ((x_norm - x1) / (x2 - x1)) * fx2y2;

        height = ((y2 - y_norm) / (y2 - y1)) * fxy1 + ((y_norm - y1) / (y2 - y1)) * fxy2;

        // Sanity check
        if (std::isnan(height)) {
            std::cout << "NaN Encountered!" << std::endl;
            height = 0.;
        }

        return height;
    }

    double TerrainGrid::BilinearInterpolationDerivWrt(Dim2D dim, double x, double y) const
    {
        double deriv = 0;
        if (dim == X_) {
            deriv = BilinearInterpolationDerivWrtX(x, y);
        }
        else {
            deriv = BilinearInterpolationDerivWrtY(x, y);
        }
        return deriv;
    }

    double TerrainGrid::BilinearInterpolationDerivWrtX(double x, double y) const
    {
        double x_norm = std::max(0., std::min(static_cast<double>(rows_) - 1., rows_ * (x - min_x_) / (max_x_ - min_x_)));
        double y_norm = std::max(0., std::min(static_cast<double>(cols_) - 1., cols_ * (y - min_y_) / (max_y_ - min_y_)));

        // std::cout << "x_norm: " << x_norm << " y_norm: " << y_norm << std::endl;

        size_t x1 = static_cast<size_t>(std::floor(x_norm));
        size_t x2 = static_cast<size_t>(std::ceil(x_norm));
        size_t y1 = static_cast<size_t>(std::floor(y_norm));
        size_t y2 = static_cast<size_t>(std::ceil(y_norm));

        if (x1 == x2 || y1 == y2) {
            return 0.;
        }

        double fx1y1 = grid_.at(x1 * cols_ + y1);
        double fx1y2 = grid_.at(x1 * cols_ + y2);
        double fx2y1 = grid_.at(x2 * cols_ + y1);
        double fx2y2 = grid_.at(x2 * cols_ + y2);

        double dR1dx_norm = (-fx1y1 + fx2y1) / (x2 - x1);
        double dR2dx_norm = (-fx1y2 + fx2y2) / (x2 - x1);

        double dfdx_norm = (dR1dx_norm * (y2 - y_norm) + dR2dx_norm * (y_norm - y1)) / (y2 - y1);

        double dx_normdx = rows_ / (max_x_ - min_x_);

        return dfdx_norm * dx_normdx;
    }

    double TerrainGrid::BilinearInterpolationDerivWrtY(double x, double y) const
    {
        double x_norm = std::max(0., std::min(static_cast<double>(rows_) - 1., rows_ * (x - min_x_) / (max_x_ - min_x_)));
        double y_norm = std::max(0., std::min(static_cast<double>(cols_) - 1., cols_ * (y - min_y_) / (max_y_ - min_y_)));

        // std::cout << "x_norm: " << x_norm << " y_norm: " << y_norm << std::endl;

        size_t x1 = static_cast<size_t>(std::floor(x_norm));
        size_t x2 = static_cast<size_t>(std::ceil(x_norm));
        size_t y1 = static_cast<size_t>(std::floor(y_norm));
        size_t y2 = static_cast<size_t>(std::ceil(y_norm));

        if (x1 == x2 || y1 == y2) {
            return 0.;
        }

        double fx1y1 = grid_.at(x1 * cols_ + y1);
        double fx1y2 = grid_.at(x1 * cols_ + y2);
        double fx2y1 = grid_.at(x2 * cols_ + y1);
        double fx2y2 = grid_.at(x2 * cols_ + y2);

        double R1 = (fx1y1 * (x2 - x_norm) + fx2y1 * (x_norm - x1)) / (x2 - x1);
        double R2 = (fx1y2 * (x2 - x_norm) + fx2y2 * (x_norm - x1)) / (x2 - x1);

        double dfdy_norm = (-R1 + R2) / (y2 - y1);

        double dy_normdy = cols_ / (max_y_ - min_y_);

        return dfdy_norm * dy_normdy;
    }

    double TerrainGrid::BilinearInterpolationSecondDerivWrt(Dim2D dim1, Dim2D dim2, double x, double y) const
    {
        if (dim1 == X_) {
            if (dim2 == X_) {
                return BilinearInterpolationSecondDerivWrtXX();
            }
            else {
                return BilinearInterpolationSecondDerivWrtXY(x, y);
            }
        }
        else {
            if (dim2 == X_) {
                return BilinearInterpolationSecondDerivWrtYX(x, y);
            }
            else {
                return BilinearInterpolationSecondDerivWrtYY();
            }
        }
    }

    inline double TerrainGrid::BilinearInterpolationSecondDerivWrtXX() const
    {
        return 0.;
    }

    inline double TerrainGrid::BilinearInterpolationSecondDerivWrtXY(double x, double y) const
    {
        double x_norm = std::max(0., std::min(static_cast<double>(rows_) - 1., rows_ * (x - min_x_) / (max_x_ - min_x_)));
        double y_norm = std::max(0., std::min(static_cast<double>(cols_) - 1., cols_ * (y - min_y_) / (max_y_ - min_y_)));

        size_t x1 = static_cast<size_t>(std::floor(x_norm));
        size_t x2 = static_cast<size_t>(std::ceil(x_norm));
        size_t y1 = static_cast<size_t>(std::floor(y_norm));
        size_t y2 = static_cast<size_t>(std::ceil(y_norm));

        if (x1 == x2 || y1 == y2) {
            return 0.;
        }

        double fx1y1 = grid_.at(x1 * cols_ + y1);
        double fx1y2 = grid_.at(x1 * cols_ + y2);
        double fx2y1 = grid_.at(x2 * cols_ + y1);
        double fx2y2 = grid_.at(x2 * cols_ + y2);

        double dR1dx_norm = (-fx1y1 + fx2y1) / (x2 - x1);
        double dR2dx_norm = (-fx1y2 + fx2y2) / (x2 - x1);

        double dfdx_normy_norm = (-dR1dx_norm + dR2dx_norm) / (y2 - y1);

        double dy_normdy = cols_ / (max_y_ - min_y_);
        double dx_normdx = rows_ / (max_x_ - min_x_);

        return dfdx_normy_norm * dy_normdy * dx_normdx;
    }

    inline double TerrainGrid::BilinearInterpolationSecondDerivWrtYX(double x, double y) const
    {
        double x_norm = std::max(0., std::min(static_cast<double>(rows_) - 1., rows_ * (x - min_x_) / (max_x_ - min_x_)));
        double y_norm = std::max(0., std::min(static_cast<double>(cols_) - 1., cols_ * (y - min_y_) / (max_y_ - min_y_)));

        size_t x1 = static_cast<size_t>(std::floor(x_norm));
        size_t x2 = static_cast<size_t>(std::ceil(x_norm));
        size_t y1 = static_cast<size_t>(std::floor(y_norm));
        size_t y2 = static_cast<size_t>(std::ceil(y_norm));

        if (x1 == x2 || y1 == y2) {
            return 0.;
        }

        double fx1y1 = grid_.at(x1 * cols_ + y1);
        double fx1y2 = grid_.at(x1 * cols_ + y2);
        double fx2y1 = grid_.at(x2 * cols_ + y1);
        double fx2y2 = grid_.at(x2 * cols_ + y2);

        double dR1dx_norm = (-fx1y1 + fx2y1) / (x2 - x1);
        double dR2dx_norm = (-fx1y2 + fx2y2) / (x2 - x1);

        double dfdx_normy_norm = (-dR1dx_norm + dR2dx_norm) / (y2 - y1);

        double dx_normdx = rows_ / (max_x_ - min_x_);
        double dy_normdy = cols_ / (max_y_ - min_y_);

        return dfdx_normy_norm * dx_normdx * dy_normdy;
    }

    inline double TerrainGrid::BilinearInterpolationSecondDerivWrtYY() const
    {
        return 0.;
    }

    std::vector<double> TerrainGrid::Split(const std::string& s, char delimiter) const
    {
        std::vector<double> tokens;
        tokens.resize(cols_);
        std::string token;
        std::istringstream token_stream(s);
        for (auto& item : tokens) {
            std::getline(token_stream, token, delimiter);
            item = std::stod(token);
        }
        return tokens;
    }

} // namespace trajopt
