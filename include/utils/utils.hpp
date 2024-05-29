#pragma once

#include <ifopt/bounds.h>
#include <ifopt/composite.h>

namespace trajopt {

    // Return 3D inertia tensor from 6D vector
    inline Eigen::Matrix3d inertiaTensor(double Ixx, double Iyy, double Izz, double Ixy, double Ixz, double Iyz)
    {
        Eigen::Matrix3d I;
        I << Ixx, -Ixy, -Ixz, -Ixy, Iyy, -Iyz, -Ixz, -Iyz, Izz;
        return I;
    }

    inline ifopt::Component::VecBound fillBoundVector(Eigen::Vector3d init, Eigen::Vector3d target, ifopt::Bounds intermediate, size_t size)
    {
        ifopt::Component::VecBound bounds(size, intermediate);
        bounds.at(0) = ifopt::Bounds(init[0], init[0]);
        bounds.at(1) = ifopt::Bounds(init[1], init[1]);
        bounds.at(2) = ifopt::Bounds(init[2], init[2]);
        bounds.at(3) = ifopt::Bounds(init[0], init[0]);
        bounds.at(4) = ifopt::Bounds(init[1], init[1]);
        bounds.at(5) = ifopt::Bounds(init[2], init[2]);
        bounds.at(size - 6) = ifopt::Bounds(target[0], target[0]);
        bounds.at(size - 5) = ifopt::Bounds(target[1], target[1]);
        bounds.at(size - 4) = ifopt::Bounds(target[2], target[2]);
        bounds.at(size - 3) = ifopt::Bounds(target[0], target[0]);
        bounds.at(size - 2) = ifopt::Bounds(target[1], target[1]);
        bounds.at(size - 1) = ifopt::Bounds(target[2], target[2]);

        return bounds;
    }
} // namespace trajopt
