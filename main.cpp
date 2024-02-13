#include <ctime>
#include <fstream>
#include <iostream>

#include <Eigen/Dense>

#include "traj_opt.hpp"
#include "utils/srbd.hpp"

int main() {
  std::srand(std::time(0));

  // Create an SRBD Model(numFeet, mass, gravity, inertia)
  double W = 0.4;
  double D = 0.4;
  double H = 0.2;
  double m_b = 0.5; // 0.5kg
  // Inertia Matrix (https://en.wikipedia.org/wiki/List_of_moments_of_inertia)
  Eigen::DiagonalMatrix<double, 3, 3> I((1. / 12.) * m_b * (H * H + W * W),
                                        (1. / 12.) * m_b * (D * D + H * H),
                                        (1. / 12.) * m_b * (W * W + D * D));

  size_t nFeet = 4;

  std::vector<Eigen::Vector3d> fPoses;
  fPoses.push_back(Eigen::Vector3d(0.2, -0.2, -0.5));  // r_i right front
  fPoses.push_back(Eigen::Vector3d(0.2, 0.2, -0.5));   // r_i left front
  fPoses.push_back(Eigen::Vector3d(-0.2, -0.2, -0.5)); // r_i right hind
  fPoses.push_back(Eigen::Vector3d(-0.2, 0.2, -0.5));  // r_i left hind

  trajopt::SingleRigidBodyDynamicsModel model(m_b, I, nFeet, fPoses);

  // Create a Terrain Model (Select from predefined ones)
  trajopt::Terrain terrain;

  // Create an SRBD Trajectory Generation Object(model, terrain, gait_type,
  // numSteps, stepPhaseTime, swingPhaseTime, numKnotPoints, numSamples)

  trajopt::TrajOptArguments args;
  args.numKnots = 50;
  args.numSamples = 50;
  args.numSteps = 2;
  args.numKnotsPerSwing = 9;
  args.stepPhaseTime = 0.5;
  args.swingPhaseTime = 0.5; // Make these the same when gait == default.
  args.initPos = Eigen::Vector3d(0., 0., 0.5);
  args.targetPos = Eigen::Vector3d(1., 1., 0.5);
  args.gait = "trot"; // trot, pace, bound, jumping supported

  trajopt::TrajOpt to(model, terrain, args);

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

  csv << "index,time,x_b,y_b,z_b";

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

    csv << "," << x_b << "," << y_b << "," << z_b;

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