#include <ctime>
#include <fstream>
#include <iostream>

#include <Eigen/Dense>

#include "traj_opt.hpp"
#include "utils/srbd.hpp"

// Return 3D inertia tensor from 6D vector.
inline Eigen::Matrix3d InertiaTensor(double Ixx, double Iyy, double Izz,
                                     double Ixy, double Ixz, double Iyz);

int main() {
  std::srand(std::time(0));

  // Create an SRBD Model
  // Anymal characteristics
  Eigen::Matrix3d inertia = InertiaTensor(0.88201174, 1.85452968, 1.97309185,
                                          0.00137526, 0.00062895, 0.00018922);
  const double m_b = 30.4213964625;
  const double x_nominal_b = 0.34;
  const double y_nominal_b = 0.19;
  const double z_nominal_b = -0.42;

  const double dx = 0.15;
  const double dy = 0.1;
  const double dz = 0.1;

  trajopt::SingleRigidBodyDynamicsModel model;

  model.mass = m_b;
  model.inertia = inertia;
  model.numFeet = 4;

  std::vector<Eigen::Vector3d> feet_positions;

  // Right fore
  model.feetPoses.push_back(Eigen::Vector3d(x_nominal_b, -y_nominal_b, 0.));
  model.feetMinBounds.push_back(
      Eigen::Vector3d(x_nominal_b - dx, -y_nominal_b - dy, z_nominal_b - dz));
  model.feetMaxBounds.push_back(
      Eigen::Vector3d(x_nominal_b + dx, -y_nominal_b + dy, z_nominal_b + dz));

  // Left fore
  model.feetPoses.push_back(Eigen::Vector3d(x_nominal_b, y_nominal_b, 0.));
  model.feetMinBounds.push_back(
      Eigen::Vector3d(x_nominal_b - dx, y_nominal_b - dy, z_nominal_b - dz));
  model.feetMaxBounds.push_back(
      Eigen::Vector3d(x_nominal_b + dx, y_nominal_b + dy, z_nominal_b + dz));

  // Right hind
  model.feetPoses.push_back(Eigen::Vector3d(-x_nominal_b, -y_nominal_b, 0.));
  model.feetMinBounds.push_back(
      Eigen::Vector3d(-x_nominal_b - dx, -y_nominal_b - dy, z_nominal_b - dz));
  model.feetMaxBounds.push_back(
      Eigen::Vector3d(-x_nominal_b + dx, -y_nominal_b + dy, z_nominal_b + dz));

  // Left hind
  model.feetPoses.push_back(Eigen::Vector3d(-x_nominal_b, y_nominal_b, 0.));
  model.feetMinBounds.push_back(
      Eigen::Vector3d(-x_nominal_b - dx, y_nominal_b - dy, z_nominal_b - dz));
  model.feetMaxBounds.push_back(
      Eigen::Vector3d(-x_nominal_b + dx, y_nominal_b + dy, z_nominal_b + dz));

  // Create a Terrain Model (Select from predefined ones)
  trajopt::Terrain terrain("step");

  trajopt::TrajOptArguments args;
  args.numKnots = 180;
  args.numSamples = 180;
  args.numSteps = 15;
  args.numKnotsPerSwing = 3;
  args.stepPhaseTime = 0.25;
  args.swingPhaseTime = 0.25;
  args.initPos = Eigen::Vector3d(0., 0., 0.5);
  args.targetPos = Eigen::Vector3d(3., 3., 0.5 + terrain.z(1., 1.));
  args.gait = "trot"; // trot, pace, bound, jumping supported

  trajopt::TrajOpt to(model, terrain, args);

  // Initialise problem
  to.Init();

  to.nlp()->PrintCurrent();

  // Solve the TO Problem.
  to.Solve();

  to.StoreSamplesToCsv("quad_trot_huge.csv");

  return 0;
}

inline Eigen::Matrix3d InertiaTensor(double Ixx, double Iyy, double Izz,
                                     double Ixy, double Ixz, double Iyz) {
  Eigen::Matrix3d I;
  I << Ixx, -Ixy, -Ixz, -Ixy, Iyy, -Iyz, -Ixz, -Iyz, Izz;
  return I;
}
