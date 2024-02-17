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

  Eigen::Matrix3d inertia = InertiaTensor(0.88201174, 1.85452968, 1.97309185,
                                          0.00137526, 0.00062895, 0.00018922);
  const double anymalMass = 30.4213964625;
  const double x_nominal_b = 0.34;
  const double y_nominal_b = 0.19;
  const double z_nominal_b = -0.42;

  const double dx = 0.15;
  const double dy = 0.1;
  const double dz = 0.1;

  trajopt::SingleRigidBodyDynamicsModel model;

  model.mass = anymalMass;
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

  // Create an SRBD Model(numFeet, mass, gravity, inertia)
  // double W = 0.4;
  // double D = 0.4;
  // double H = 0.1;
  // double m_b = 20; // 0.5kg
  // Inertia Matrix (https://en.wikipedia.org/wiki/List_of_moments_of_inertia)
  // Eigen::DiagonalMatrix/ RotMat<double, 3, 3> I((1. / 12.) * m_b * (H * H + W
  // * W), (1. / 12.) * m_b * (D * D + H * H), (1. / 12.) * m_b * (W * W + D *
  // D));

  // size_t nFeet = 4;

  // std::vector<Eigen::Vector3d> fPoses;
  // fPoses.push_back(Eigen::Vector3d(0.2, -0.2, -0.5)); // r_i right front
  // fPoses.push_back(Eigen::Vector3d(0.2, 0.2, -0.5)); // r_i left front
  // fPoses.push_back(Eigen::Vector3d(-0.2, -0.2, -0.5)); // r_i right hind
  // fPoses.push_back(Eigen::Vector3d(-0.2, 0.2, -0.5)); // r_i left hind

  // trajopt::SingleRigidBodyDynamicsModel model(m_b, I, nFeet, fPoses);

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

  // Retrieve the generated trajectories.
  // GetBodyPosTrajectory()
  // auto posTrajectory = to.GetBodyTrajectory("pos");
  // GetBodyRotTrajectory()
  // GetFootPosTrajectory(footIdx)
  // GetFootForceTrajectory(footIdx)

  // Store results in a CSV for plotting.
  // std::cout << "Writing output to CSV File" << std::endl;
  std::ofstream csv("output.csv");

  if (!csv.is_open()) {
    std::cerr << "Error opening the CSV file!" << std::endl;
    return 1;
  }

  csv << "index,time,x_b,y_b,z_b,th_z, th_y, th_x";

  for (size_t k = 0; k < model.numFeet; ++k) {
    csv << ","
        << "x" << std::to_string(k) << ","
        << "y" << std::to_string(k) << ","
        << "z" << std::to_string(k) << ","
        << "fx" << std::to_string(k) << ","
        << "fy" << std::to_string(k) << ","
        << "fz" << std::to_string(k);
  }

  csv << std::endl;

  double t = 0.;
  for (size_t i = 1; i < args.numSamples; ++i) {

    csv << i << "," << t;

    // Body Positions
    double x_b = to.GetBodyTrajectory("pos").position(t)(0);
    double y_b = to.GetBodyTrajectory("pos").position(t)(1);
    double z_b = to.GetBodyTrajectory("pos").position(t)(2);

    double th_z = to.GetBodyTrajectory("rot").position(t)(0);
    double th_y = to.GetBodyTrajectory("rot").position(t)(1);
    double th_x = to.GetBodyTrajectory("rot").position(t)(2);

    csv << "," << x_b << "," << y_b << "," << z_b << "," << th_z << "," << th_y
        << "," << th_x;

    for (size_t k = 0; k < model.numFeet; ++k) {
      double x = to.GetFootPosTrajectory(k).position(t)(0);
      double y = to.GetFootPosTrajectory(k).position(t)(1);
      double z = to.GetFootPosTrajectory(k).position(t)(2);

      double fx = to.GetFootForceTrajectory(k).position(t)(0);
      double fy = to.GetFootForceTrajectory(k).position(t)(1);
      double fz = to.GetFootForceTrajectory(k).position(t)(2);

      csv << "," << x << "," << y << "," << z << "," << fx << "," << fy << ","
          << fz;
    }
    csv << std::endl;
    t += to.SampleTime();
  }

  // Close the CSV file
  csv.close();
  std::cout << "CSV file created successfully!" << std::endl;

  return 0;
}

inline Eigen::Matrix3d InertiaTensor(double Ixx, double Iyy, double Izz,
                                     double Ixy, double Ixz, double Iyz) {
  Eigen::Matrix3d I;
  I << Ixx, -Ixy, -Ixz, -Ixy, Iyy, -Iyz, -Ixz, -Iyz, Izz;
  return I;
}
