#pragma once

#include <Eigen/Core>

namespace trajopt {
struct SingleRigidBodyDynamicsModel {
  SingleRigidBodyDynamicsModel() = default;
  SingleRigidBodyDynamicsModel(double m, Eigen::Matrix3d I, double nFeet,
                               std::vector<Eigen::Vector3d> fPoses)
      : mass(m), inertia(I), numFeet(nFeet), feetPoses(fPoses) {}

  const Eigen::Vector3d gravity = Eigen::Vector3d(0., 0., -9.81);

  double mass = 1.;
  Eigen::Matrix3d inertia = Eigen::Matrix3d::Identity();
  unsigned int numFeet = 1;
  std::vector<Eigen::Vector3d> feetPoses;
  std::vector<Eigen::Vector3d> feetMinBounds;
  std::vector<Eigen::Vector3d> feetMaxBounds;
};

struct Terrain {
  Terrain() = default;
  Terrain(std::string strType) : type(strType) {}

  double z(double x, double y) const {
    if (type == "step") {
      if (x > 0.5 || y > 0.5) {
        return 0.2;
      } else {
        return 0.;
      }
    } else {
      return 0.;
    }
  }

  const std::string type = "none";

  Eigen::Vector3d n = Eigen::Vector3d(0., 0., 1.);
  Eigen::Vector3d t = Eigen::Vector3d(1., 0., 0.);
  Eigen::Vector3d b = Eigen::Vector3d(0., 1., 0.);
  double mu = 1.; // friction coefficient
};

struct TrajOptArguments {
  size_t numKnots;
  size_t numSamples;
  size_t numSteps;
  size_t numKnotsPerSwing;

  double stepPhaseTime;
  double swingPhaseTime;

  Eigen::Vector3d initPos;
  Eigen::Vector3d targetPos;
  Eigen::Vector3d initVel = Eigen::Vector3d::Zero();
  Eigen::Vector3d targetVel = Eigen::Vector3d::Zero();

  Eigen::Vector3d initRot = Eigen::Vector3d::Zero();
  Eigen::Vector3d targetRot = Eigen::Vector3d::Zero();
  Eigen::Vector3d initAngVel = Eigen::Vector3d::Zero();
  Eigen::Vector3d targetAngVel = Eigen::Vector3d::Zero();

  std::string gait;
};
} // namespace trajopt
