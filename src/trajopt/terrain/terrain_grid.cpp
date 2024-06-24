#include "terrain_grid.hpp"

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

    double TerrainGrid::GetDerivativeOfHeightWrt(Dim2D dim, double x, double y) const
    {
        double fp = 0.;
        double fm = 0.;

        if (dim == X_) {
            double xm = x - eps_;
            double xp = x + eps_;

            fm = BilinearInterpolation(xm, y);
            fp = BilinearInterpolation(xp, y);
        }
        else {
            double xm = x - eps_;
            double xp = x + eps_;

            fm = BilinearInterpolation(xm, y);
            fp = BilinearInterpolation(xp, y);
        }

        return (fp - fm) / 2 * eps_;
    }

    Eigen::Vector3d TerrainGrid::GetDerivativeOfNormalizedBasisWrt(Direction direction, Dim2D dim, double x, double y) const
    {
        Eigen::Vector3d deriv = Eigen::Vector3d::Zero();
        switch (direction) {
        case Normal:
            if (dim == X_) {
                double dxx = GetSecondDerivativeOfHeightWrt(X_, X_, x, y);
                double dyx = GetSecondDerivativeOfHeightWrt(Y_, X_, x, y);

                deriv << -dxx, -dyx, 0.;
                return deriv;
            }
            else {
                double dxy = GetSecondDerivativeOfHeightWrt(X_, Y_, x, y);
                double dyy = GetSecondDerivativeOfHeightWrt(Y_, Y_, x, y);

                deriv << -dxy, -dyy, 0.;
                return deriv;
            }
            break;
        case Tangent1:
            if (dim == X_) {
                double dxx = GetSecondDerivativeOfHeightWrt(X_, X_, x, y);
                deriv << 0., 0., dxx;
                return deriv;
            }
            else {
                double dxy = GetSecondDerivativeOfHeightWrt(X_, Y_, x, y);
                deriv << 0., 0., dxy;
                return deriv;
            }
            break;
        case Tangent2:
            if (dim == X_) {
                double dyx = GetSecondDerivativeOfHeightWrt(Y_, X_, x, y);
                deriv << 0., 0., dyx;
                return deriv;
            }
            else {
                double dyy = GetSecondDerivativeOfHeightWrt(Y_, Y_, x, y);
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

    double TerrainGrid::GetSecondDerivativeOfHeightWrt(Dim2D dim1, Dim2D dim2, double x, double y) const
    {
        if (dim1 == X_) {
            if (dim2 == X_) {
                return GetHeightDerivWrtXX(x, y);
            }
            else {
                return GetHeightDerivWrtXY(x, y);
            }
        }
        else {
            if (dim2 == X_) {
                return GetHeightDerivWrtYX(x, y);
            }
            else {
                return GetHeightDerivWrtYY(x, y);
            }
        }
    }

    inline double TerrainGrid::GetHeightDerivWrtXX(double x, double y) const
    {
        double xp = x + eps_;
        double xm = x - eps_;

        double dp = GetDerivativeOfHeightWrt(X_, xp, y);
        double dm = GetDerivativeOfHeightWrt(X_, xm, y);

        return (dp - dm) / (2. * eps_);
    }

    inline double TerrainGrid::GetHeightDerivWrtXY(double x, double y) const
    {
        double yp = y + eps_;
        double ym = y - eps_;

        double dp = GetDerivativeOfHeightWrt(X_, x, yp);
        double dm = GetDerivativeOfHeightWrt(X_, x, ym);

        return (dp - dm) / (2. * eps_);
    }

    inline double TerrainGrid::GetHeightDerivWrtYX(double x, double y) const
    {
        double xp = x + eps_;
        double xm = x - eps_;

        double dp = GetDerivativeOfHeightWrt(Y_, xp, y);
        double dm = GetDerivativeOfHeightWrt(Y_, xm, y);

        return (dp - dm) / (2. * eps_);
    }

    inline double TerrainGrid::GetHeightDerivWrtYY(double x, double y) const
    {
        double yp = y + eps_;
        double ym = y - eps_;

        double dp = GetDerivativeOfHeightWrt(Y_, x, yp);
        double dm = GetDerivativeOfHeightWrt(Y_, x, ym);

        return (dp - dm) / (2. * eps_);
    }

    inline double TerrainGrid::BilinearInterpolation(double x, double y) const
    {
        // Normalise x, y;
        x = std::max(0., std::min(static_cast<double>(rows_) - 1., rows_ * (x - min_x_) / (max_x_ - min_x_)));
        y = std::max(0., std::min(static_cast<double>(cols_) - 1., cols_ * (y - min_y_) / (max_y_ - min_y_)));

        double height = 0.;

        size_t x1 = static_cast<size_t>(std::floor(x));
        size_t x2 = static_cast<size_t>(std::ceil(x));
        size_t y1 = static_cast<size_t>(std::floor(y));
        size_t y2 = static_cast<size_t>(std::ceil(y));

        double fx1y1 = grid_.at(x1 * cols_ + y1);
        double fx1y2 = grid_.at(x1 * cols_ + y2);
        double fx2y1 = grid_.at(x2 * cols_ + y1);
        double fx2y2 = grid_.at(x2 * cols_ + y2);

        double fxy1 = 0.; // Interpolate y1 in the x direction.
        double fxy2 = 0.; // Interpolate y2 in the x direction.

        if (x1 == x2) {
            fxy1 = fx1y1;
            fxy2 = fx2y1;
        }
        else {
            fxy1 = ((x2 - x) / (x2 - x1)) * fx1y1 + ((x - x1) / (x2 - x1)) * fx2y1;
            fxy2 = ((x2 - x) / (x2 - x1)) * fx1y2 + ((x - x1) / (x2 - x1)) * fx2y2;
        }

        if (y1 == y2) {
            height = fxy1;
        }
        else {
            height = ((y2 - y) / (y2 - y1)) * fxy1 + ((y - y1) / (y2 - y1)) * fxy2;
        }

        // Sanity check
        if (std::isnan(height)) {
            std::cout << "NaN Encountered!" << std::endl;
            height = 0.;
        }

        return height;
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
