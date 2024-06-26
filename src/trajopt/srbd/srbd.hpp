#pragma once

#include <vector>

#include <Eigen/Core>

#include <trajopt/utils/utils.hpp>

namespace trajopt {
    struct SingleRigidBodyDynamicsModel {
        SingleRigidBodyDynamicsModel() = default;
        SingleRigidBodyDynamicsModel(double m, const Eigen::Matrix3d& I, double nFeet, const std::vector<Eigen::Vector3d>& fPoses)
            : mass(m), inertia(I), numFeet(nFeet), feetPoses(fPoses)
        {
            inertia_inv = inertia.inverse();
        }

        const Eigen::Vector3d gravity = Eigen::Vector3d(0., 0., -9.81);

        double mass = 1.;
        Eigen::Matrix3d inertia = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d inertia_inv = Eigen::Matrix3d::Identity();
        unsigned int numFeet = 1;
        std::vector<Eigen::Vector3d> feetPoses;
        std::vector<Eigen::Vector3d> feetMinBounds;
        std::vector<Eigen::Vector3d> feetMaxBounds;
    };

    inline void init_model_biped(SingleRigidBodyDynamicsModel& model)
    {
        double W = 0.4;
        double D = 0.4;
        double H = 0.1;
        double m_b = 5; // 0.5kg
        // Inertia Matrix (https://en.wikipedia.org/wiki/List_of_moments_of_inertia)
        Eigen::Matrix3d I = Eigen::Matrix3d::Zero();
        I(0, 0) = (1. / 12.) * m_b * (H * H + W * W);
        I(1, 1) = (1. / 12.) * m_b * (D * D + H * H);
        I(2, 2) = (1. / 12.) * m_b * (W * W + D * D);

        model.mass = m_b;
        model.inertia = I;
        model.numFeet = 2;

        const double dx = 0.15;
        const double dy = 0.1;
        const double dz = 0.1;

        const double x_nominal_b = 0.15;
        const double y_nominal_b = 0.0;
        const double z_nominal_b = -0.5;

        // Right
        model.feetPoses.push_back(Eigen::Vector3d(x_nominal_b, -y_nominal_b, 0.));
        model.feetMinBounds.push_back(Eigen::Vector3d(x_nominal_b - dx, -y_nominal_b - dy, z_nominal_b - dz));
        model.feetMaxBounds.push_back(Eigen::Vector3d(x_nominal_b + dx, -y_nominal_b + dy, z_nominal_b + dz));

        // Left
        model.feetPoses.push_back(Eigen::Vector3d(x_nominal_b, y_nominal_b, 0.));
        model.feetMinBounds.push_back(Eigen::Vector3d(x_nominal_b - dx, y_nominal_b - dy, z_nominal_b - dz));
        model.feetMaxBounds.push_back(Eigen::Vector3d(x_nominal_b + dx, y_nominal_b + dy, z_nominal_b + dz));
    }

    // Initialise and SRBD model with the Anybotics Anymal parameters.
    inline void init_model_anymal(SingleRigidBodyDynamicsModel& model)
    {
        //   Anymal characteristics
        Eigen::Matrix3d inertia = inertiaTensor(0.88201174, 1.85452968, 1.97309185, 0.00137526, 0.00062895, 0.00018922);
        const double m_b = 30.4213964625;
        const double x_nominal_b = 0.34;
        const double y_nominal_b = 0.19;
        const double z_nominal_b = -0.42;

        const double dx = 0.15;
        const double dy = 0.1;
        const double dz = 0.1;

        model.mass = m_b;
        model.inertia = inertia;
        model.numFeet = 4;

        // Right fore
        model.feetPoses.push_back(Eigen::Vector3d(x_nominal_b, -y_nominal_b, 0.));
        model.feetMinBounds.push_back(Eigen::Vector3d(x_nominal_b - dx, -y_nominal_b - dy, z_nominal_b - dz));
        model.feetMaxBounds.push_back(Eigen::Vector3d(x_nominal_b + dx, -y_nominal_b + dy, z_nominal_b + dz));

        // Left fore
        model.feetPoses.push_back(Eigen::Vector3d(x_nominal_b, y_nominal_b, 0.));
        model.feetMinBounds.push_back(Eigen::Vector3d(x_nominal_b - dx, y_nominal_b - dy, z_nominal_b - dz));
        model.feetMaxBounds.push_back(Eigen::Vector3d(x_nominal_b + dx, y_nominal_b + dy, z_nominal_b + dz));

        // Right hind
        model.feetPoses.push_back(Eigen::Vector3d(-x_nominal_b, -y_nominal_b, 0.));
        model.feetMinBounds.push_back(Eigen::Vector3d(-x_nominal_b - dx, -y_nominal_b - dy, z_nominal_b - dz));
        model.feetMaxBounds.push_back(Eigen::Vector3d(-x_nominal_b + dx, -y_nominal_b + dy, z_nominal_b + dz));

        // Left hind
        model.feetPoses.push_back(Eigen::Vector3d(-x_nominal_b, y_nominal_b, 0.));
        model.feetMinBounds.push_back(Eigen::Vector3d(-x_nominal_b - dx, y_nominal_b - dy, z_nominal_b - dz));
        model.feetMaxBounds.push_back(Eigen::Vector3d(-x_nominal_b + dx, y_nominal_b + dy, z_nominal_b + dz));
    }

    struct Terrain {
        Terrain() = default;
        Terrain(std::string strType) : type(strType) {}

        double z(double x, double y) const
        {
            if (type == "step") {
                if (x > 0.5 || y > 0.5) {
                    return 0.2;
                }
                else {
                    return 0.;
                }
            }
            else {
                return 0.;
            }
        }

        const std::string type = "none";

        Eigen::Vector3d n = Eigen::Vector3d(0., 0., 1.);
        Eigen::Vector3d t = Eigen::Vector3d(1., 0., 0.);
        Eigen::Vector3d b = Eigen::Vector3d(0., 1., 0.);
        double mu = 1.; // friction coefficient
    };
} // namespace trajopt
