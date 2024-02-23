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
  double W = 0.4;
  double D = 0.4;
  double H = 0.1;
  double m_b = 20;
  // Inertia Matrix (https://en.wikipedia.org/wiki/List_of_moments_of_inertia)
  Eigen::DiagonalMatrix<double, 3, 3> inertia(
      (1. / 12.) * m_b * (H * H + W * W), (1. / 12.) * m_b * (D * D + H * H),
      (1. / 12.) * m_b * (W * W + D * D));

  const double x_nominal_b = 0.1;
  const double y_nominal_b = 0.1;
  const double z_nominal_b = -0.42;

  const double dx = 0.25;
  const double dy = 0.2;
  const double dz = 0.2;

  // Anymal characteristics
  //   Eigen::Matrix3d inertia =
  //   InertiaTensor(0.88201174, 1.85452968, 1.97309185,
  //                                           0.00137526, 0.00062895,
  //                                           0.00018922);
  //   const double m_b = 30.4213964625;
  //   const double x_nominal_b = 0.34;
  //   const double y_nominal_b = 0.19;
  //   const double z_nominal_b = -0.42;

  //   const double dx = 0.15;
  //   const double dy = 0.1;
  //   const double dz = 0.1;

  trajopt::SingleRigidBodyDynamicsModel model;

  model.mass = m_b;
  model.inertia = inertia;
  model.numFeet = 2;

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

  //   // Right hind
  //   model.feetPoses.push_back(Eigen::Vector3d(-x_nominal_b, -y_nominal_b,
  //   0.)); model.feetMinBounds.push_back(
  //       Eigen::Vector3d(-x_nominal_b - dx, -y_nominal_b - dy, z_nominal_b -
  //       dz));
  //   model.feetMaxBounds.push_back(
  //       Eigen::Vector3d(-x_nominal_b + dx, -y_nominal_b + dy, z_nominal_b +
  //       dz));

  //   // Left hind
  //   model.feetPoses.push_back(Eigen::Vector3d(-x_nominal_b, y_nominal_b,
  //   0.)); model.feetMinBounds.push_back(
  //       Eigen::Vector3d(-x_nominal_b - dx, y_nominal_b - dy, z_nominal_b -
  //       dz));
  //   model.feetMaxBounds.push_back(
  //       Eigen::Vector3d(-x_nominal_b + dx, y_nominal_b + dy, z_nominal_b +
  //       dz));

  // Create a Terrain Model (Select from predefined ones)
  trajopt::Terrain terrain("step");

  // Create an SRBD Trajectory Generation Object(model, terrain, gait_type,
  // numSteps, stepPhaseTime, swingPhaseTime, numKnotPoints, numSamples)

  trajopt::TrajOptArguments args;
  args.numKnots = 250;
  args.numSamples = 250;
  args.numSteps = 5;
  args.numKnotsPerSwing = 3;
  args.stepPhaseTime = 0.25;
  args.swingPhaseTime = 0.25;
  args.initPos = Eigen::Vector3d(0., 0., 0.5);
  args.targetPos = Eigen::Vector3d(1., 1., 0.5 + terrain.z(1., 1.));
  args.gait = "trot"; // trot, pace, bound, jumping supported

  trajopt::TrajOpt to(model, terrain, args);

  // Initialise problem
  to.Init();

  // Solve the TO Problem.
  to.Solve();

  to.StoreSamplesToCsv("biped_trot_step.csv");

  return 0;
}

inline Eigen::Matrix3d InertiaTensor(double Ixx, double Iyy, double Izz,
                                     double Ixy, double Ixz, double Iyz) {
  Eigen::Matrix3d I;
  I << Ixx, -Ixy, -Ixz, -Ixy, Iyy, -Iyz, -Ixz, -Iyz, Izz;
  return I;
}
