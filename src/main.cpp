#include "acados_cpp_wrapper.hpp"
#include "matplotlibcpp.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;
namespace plt = matplotlibcpp;

const int NX = DIFFERENTIAL_DRIVE_NX;
const int NU = DIFFERENTIAL_DRIVE_NU;
const int NY = NX + NU + NU;  // state + control + control rate
const int N = DIFFERENTIAL_DRIVE_N;

std::vector<double> dynamics_model(const std::vector<double>& x, const std::vector<double>& u) {
    double dx = u[0] * std::cos(x[2]);
    double dy = u[0] * std::sin(x[2]);
    double dtheta = u[1];
    return {dx, dy, dtheta};
}

std::vector<double> runge_kutta(const std::vector<double>& x, const std::vector<double>& u, double dt) {
    auto f = [&](const std::vector<double>& x, const std::vector<double>& u) {
        return dynamics_model(x, u);
    };

    std::vector<double> k1 = f(x, u);
    std::vector<double> k2 = f({x[0] + dt/2 * k1[0], x[1] + dt/2 * k1[1], x[2] + dt/2 * k1[2]}, u);
    std::vector<double> k3 = f({x[0] + dt/2 * k2[0], x[1] + dt/2 * k2[1], x[2] + dt/2 * k2[2]}, u);
    std::vector<double> k4 = f({x[0] + dt * k3[0], x[1] + dt * k3[1], x[2] + dt * k3[2]}, u);

    return {
        x[0] + dt/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]),
        x[1] + dt/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]),
        x[2] + dt/6 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    };
}

std::vector<std::vector<double>> generate_cubic_trajectory(const std::vector<double>& start, const std::vector<double>& end, int num_points) {
    std::vector<std::vector<double>> trajectory(num_points, std::vector<double>(3));
    for (int i = 0; i < num_points; ++i) {
        double t = static_cast<double>(i) / (num_points - 1);
        for (int j = 0; j < 3; ++j) {
            trajectory[i][j] = start[j] + (3 * t * t - 2 * t * t * t) * (end[j] - start[j]);
        }
    }
    return trajectory;
}

std::vector<std::vector<double>> calculate_reference_control(const std::vector<std::vector<double>>& trajectory, double total_time) {
    int num_points = trajectory.size();
    double dt = total_time / (num_points - 1);
    std::vector<std::vector<double>> u_ref(num_points, std::vector<double>(2));

    for (int i = 0; i < num_points - 1; ++i) {
        double dx = trajectory[i+1][0] - trajectory[i][0];
        double dy = trajectory[i+1][1] - trajectory[i][1];
        u_ref[i][0] = std::sqrt(dx*dx + dy*dy) / dt; 
        u_ref[i][1] = (trajectory[i+1][2] - trajectory[i][2]) / dt;  
    }
    u_ref[num_points-1] = u_ref[num_points-2]; 
    return u_ref;
}

void plot_results(const std::vector<double>& time,
                  const std::vector<double>& plot_times,
                  const std::vector<std::vector<double>>& x_ref,
                  const std::vector<std::vector<double>>& x_feedback,
                  const std::vector<std::vector<double>>& u_ref,
                  const std::vector<std::vector<double>>& u_feedback,
                  const std::vector<double>& x_target) {
    std::vector<double> x, y, theta, v, omega;
    std::vector<double> x_ref_plot, y_ref_plot, theta_ref_plot;
    std::vector<double> v_ref_plot, omega_ref_plot;

    for (const auto& state : x_feedback) {
        x.push_back(state[0]);
        y.push_back(state[1]);
        theta.push_back(state[2]);
    }
    for (const auto& control : u_feedback) {
        v.push_back(control[0]);
        omega.push_back(control[1]);
    }
    for (const auto& state : x_ref) {
        x_ref_plot.push_back(state[0]);
        y_ref_plot.push_back(state[1]);
        theta_ref_plot.push_back(state[2]);
    }
    for (const auto& control : u_ref) {
        v_ref_plot.push_back(control[0]);
        omega_ref_plot.push_back(control[1]);
    }

    // Create a "plots" directory if it doesn't exist
    fs::create_directory("plots");

    // Plot states
    plt::figure_size(1200, 800);
    plt::named_plot("x ref", plot_times, x_ref_plot, "r");
    plt::named_plot("x feedback", time, x, "r--");
    plt::named_plot("y ref", plot_times, y_ref_plot, "g");
    plt::named_plot("y feedback", time, y, "g--");
    plt::named_plot("theta ref", plot_times, theta_ref_plot, "b");
    plt::named_plot("theta feedback", time, theta, "b--");
    plt::title("States");
    plt::xlabel("Time");
    plt::ylabel("State Value");
    plt::legend();
    plt::save("plots/states.png");
    plt::close();

    // Plot controls
    plt::figure_size(1200, 800);
    plt::named_plot("v ref", plot_times, v_ref_plot, "r");
    plt::named_plot("v feedback", time, v, "r--");
    plt::named_plot("omega ref", plot_times, omega_ref_plot, "g");
    plt::named_plot("omega feedback", time, omega, "g--");
    plt::title("Controls");
    plt::xlabel("Time");
    plt::ylabel("Control Value");
    plt::legend();
    plt::save("plots/controls.png");
    plt::close();

    // Plot x-y trajectory
    plt::figure_size(1200, 800);
    plt::named_plot("Reference", x_ref_plot, y_ref_plot, "r");
    plt::named_plot("Feedback", x, y, "r--");
    plt::plot({x_target[0]}, {x_target[1]}, "ro");
    plt::title("X-Y Trajectory");
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::legend();
    plt::save("plots/trajectory.png");
    plt::close();

    // Plot theta over time
    plt::figure_size(1200, 800);
    plt::named_plot("theta ref", plot_times, theta_ref_plot, "r");
    plt::named_plot("theta feedback", time, theta, "r--");
    plt::title("Theta over Time");
    plt::xlabel("Time");
    plt::ylabel("Theta");
    plt::legend();
    plt::save("plots/theta.png");
    plt::close();

    // Plot velocity over time
    plt::figure_size(1200, 800);
    plt::named_plot("v ref", plot_times, v_ref_plot, "r");
    plt::named_plot("v feedback", time, v, "r--");
    plt::title("Velocity over Time");
    plt::xlabel("Time");
    plt::ylabel("Velocity");
    plt::legend();
    plt::save("plots/velocity.png");
    plt::close();

    // Plot angular velocity over time
    plt::figure_size(1200, 800);
    plt::named_plot("omega ref", plot_times, omega_ref_plot, "r");
    plt::named_plot("omega feedback", time, omega, "r--");
    plt::title("Angular Velocity over Time");
    plt::xlabel("Time");
    plt::ylabel("Angular Velocity");
    plt::legend();
    plt::save("plots/angular_velocity.png");
    plt::close();

    std::cout << "Plots saved in the 'plots' directory." << std::endl;
}

int main() {
    plt::backend("Agg");

    try {
        // Initial state and target
        std::vector<double> x_init = {0.0, 0.0, 0.0};
        std::vector<double> x_target = {2.0, 1.5, M_PI/2};

        // Simulation parameters
        double dt = 0.05;
        double distance_threshold = 0.05;
        int num_trajectory_points = 100;

        // Constraints
        std::vector<double> lbx = {-2.0, -2.0, -M_PI};
        std::vector<double> ubx = {2.0, 2.0, M_PI};
        std::vector<double> lbu = {-0.5, -0.5};
        std::vector<double> ubu = {0.5, 0.5};

        AcadosSolver solver(lbx, ubx, lbu, ubu);

        // Generate cubic trajectory and reference controls
        auto x_ref_traj = generate_cubic_trajectory(x_init, x_target, num_trajectory_points);
        double total_time = (num_trajectory_points - 1) * dt;
        auto u_ref_traj = calculate_reference_control(x_ref_traj, total_time);

        // Main simulation loop
        std::vector<double> x_curr = x_init;
        std::vector<double> u_prev = {0.0, 0.0};
        std::vector<double> time_steps;
        std::vector<std::vector<double>> x_feedback;
        std::vector<std::vector<double>> u_feedback;

        int i = 0;

        while (true) {
            double current_time = i * dt;
            time_steps.push_back(current_time);
            x_feedback.push_back(x_curr);

            // Set current state
            solver.setInitialState(x_curr);

            // Set reference trajectory for prediction horizon
            for (int j = 0; j <= N; ++j) {
                int ref_index = std::min(i + j, static_cast<int>(x_ref_traj.size()) - 1);
                std::vector<double> yref(NY, 0.0);
                std::copy(x_ref_traj[ref_index].begin(), x_ref_traj[ref_index].end(), yref.begin());
                std::copy(u_ref_traj[ref_index].begin(), u_ref_traj[ref_index].end(), yref.begin() + NX);
                
                if (j == 0) {
                    for (int k = 0; k < NU; ++k) {
                        yref[NX + NU + k] = u_ref_traj[ref_index][k] - u_prev[k];
                    }
                } else {
                    int prev_ref_index = std::min(i + j - 1, static_cast<int>(x_ref_traj.size()) - 1);
                    for (int k = 0; k < NU; ++k) {
                        yref[NX + NU + k] = u_ref_traj[ref_index][k] - u_ref_traj[prev_ref_index][k];
                    }
                }
                
                solver.setReference(j, yref);
            }

            // Solve OCP
            int status = solver.solve();
            if (status != 0) {
                std::cerr << "Solver failed with status " << status << std::endl;
                return 1;
            }

            // Get optimal control input
            std::vector<double> u = solver.getControl(0);
            u_feedback.push_back(u);

            // Simulate system dynamics
            x_curr = runge_kutta(x_curr, u, dt);

            std::cout << "Time: " << current_time << ", State: (" 
                      << x_curr[0] << ", " << x_curr[1] << ", " << x_curr[2] 
                      << "), Control: (" << u[0] << ", " << u[1] << ")" << std::endl;

            // Check if target is reached
            double distance = std::sqrt(std::pow(x_curr[0] - x_target[0], 2) + 
                                        std::pow(x_curr[1] - x_target[1], 2) +
                                        std::pow(x_curr[2] - x_target[2], 2));
            if (distance < distance_threshold) {
                std::cout << "Reached target at iteration " << i+1 << std::endl;
                break;
            }

            u_prev = u;
            i++;
        }

        solver.printSolverInfo();

        // Plot results
        std::cout << "Plotting results..." << std::endl;
        plot_results(time_steps, time_steps, x_ref_traj, x_feedback, u_ref_traj, u_feedback, x_target);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}