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

    inline ifopt::Component::VecBound fillBoundVector(Eigen::Vector3d initPos, Eigen::Vector3d initVel, Eigen::Vector3d targetPos, Eigen::Vector3d targetVel, ifopt::Bounds intermediate, size_t size)
    {
        ifopt::Component::VecBound bounds(size, intermediate);
        bounds.at(0) = ifopt::Bounds(initPos[0], initPos[0]);
        bounds.at(1) = ifopt::Bounds(initPos[1], initPos[1]);
        bounds.at(2) = ifopt::Bounds(initPos[2], initPos[2]);

        bounds.at(3) = ifopt::Bounds(initVel[0], initVel[0]);
        bounds.at(4) = ifopt::Bounds(initVel[1], initVel[1]);
        bounds.at(5) = ifopt::Bounds(initVel[2], initVel[2]);

        bounds.at(size - 6) = ifopt::Bounds(targetPos[0], targetPos[0]);
        bounds.at(size - 5) = ifopt::Bounds(targetPos[1], targetPos[1]);
        bounds.at(size - 4) = ifopt::Bounds(targetPos[2], targetPos[2]);

        bounds.at(size - 3) = ifopt::Bounds(targetVel[0], targetVel[0]);
        bounds.at(size - 2) = ifopt::Bounds(targetVel[1], targetVel[1]);
        bounds.at(size - 1) = ifopt::Bounds(targetVel[2], targetVel[2]);

        return bounds;
    }

    inline ifopt::Component::Jacobian eulerZYXToMatrix(const Eigen::Vector3d& eulerZYX)
    {
        const double x = eulerZYX[2];
        const double y = eulerZYX[1];
        const double z = eulerZYX[0];

        const double cx = std::cos(x);
        const double sx = std::sin(x);

        const double cy = std::cos(y);
        const double sy = std::sin(y);

        const double cz = std::cos(z);
        const double sz = std::sin(z);

        Eigen::Matrix3d R;
        R << cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy, cy * sz,
            cx * cz + sx * sy * sz, cx * sy * sz - cz * sx, -sy, cy * sx, cx * cy;

        return R.sparseView(1., -1.);
    }

    inline ifopt::Component::Jacobian derivRotationTransposeVector(const Eigen::Vector3d& eulerZYX, const Eigen::Vector3d& v)
    {
        const double x = eulerZYX[2];
        const double y = eulerZYX[1];
        const double z = eulerZYX[0];

        const double cx = std::cos(x);
        const double sx = std::sin(x);

        const double cy = std::cos(y);
        const double sy = std::sin(y);

        const double cz = std::cos(z);
        const double sz = std::sin(z);

        // out = R.T * v
        // out[0] = R.T.row(0) * v = cy * cz * v[0] + cy * sz * v[1] + -sy * v[2]
        // out[1] = R.T.row(1) * v = (cz * sx * sy - cx * sz) * v[0] + (cx * cz + sx
        // * sy * sz) * v[1] + cy * sx * v[2] out[2] = R.T.row(2) * v = (sx * sz +
        // cx * cz * sy) * v[0] + (cx * sy * sz - cz * sx) * v[1] + cx * cy * v[2]

        ifopt::Component::Jacobian jac(3, 3);
        // out[0] wrt Z(0)
        jac.coeffRef(0, 0) = -cy * sz * v[0] + cy * cz * v[1];
        // out[0] wrt Y(1)
        jac.coeffRef(0, 1) = -sy * cz * v[0] - sy * sz * v[1] - cy * v[2];
        // out[0] wrt X(2)
        // jac.coeffRef(0, 2) = 0.;

        // out[1] wrt Z(0)
        jac.coeffRef(1, 0) = (-sz * sx * sy - cx * cz) * v[0] + (-cx * sz + sx * sy * cz) * v[1];
        // out[1] wrt Y(1)
        jac.coeffRef(1, 1) = cz * sx * cy * v[0] + sx * cy * sz * v[1] - sy * sx * v[2];
        // out[1] wrt X(2)
        jac.coeffRef(1, 2) = (cz * cx * sy + sx * sz) * v[0] + (-sx * cz + cx * sy * sz) * v[1] + cy * cx * v[2];

        // out[2] wrt Z(0)
        jac.coeffRef(2, 0) = (sx * cz - cx * sz * sy) * v[0] + (cx * sy * cz + sz * sx) * v[1];
        // out[2] wrt Y(1)
        jac.coeffRef(2, 1) = cx * cz * cy * v[0] + cx * cy * sz * v[1] - cx * sy * v[2];
        // out[2] wrt X(2)
        jac.coeffRef(2, 2) = (cx * sz - sx * cz * sy) * v[0] + (-sx * sy * sz - cz * cx) * v[1] - sx * cy * v[2];

        return jac;
    }

} // namespace trajopt
