#include "acados_mpc_solver.hpp"
#include "matplotlibcpp.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "json.hpp"
#include "spline.h"

namespace fs = std::filesystem;
namespace plt = matplotlibcpp;
using json = nlohmann::json;

std::vector<double> runge_kutta(
    const std::vector<double>& x,
    const std::vector<double>& u,
    double dt,
    const std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)>& dynamics)
{
    auto f = [&](const std::vector<double>& state, const std::vector<double>& control) {
        return dynamics(state, control);
    };

    std::vector<double> k1 = f(x, u);
    std::vector<double> x2(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x2[i] = x[i] + dt/2 * k1[i];
    }

    std::vector<double> k2 = f(x2, u);
    std::vector<double> x3(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x3[i] = x[i] + dt/2 * k2[i];
    }

    std::vector<double> k3 = f(x3, u);
    std::vector<double> x4(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x4[i] = x[i] + dt * k3[i];
    }

    std::vector<double> k4 = f(x4, u);
    std::vector<double> x_next(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x_next[i] = x[i] + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }

    return x_next;
}

std::vector<std::vector<double>> generate_figure8_trajectory(
    const std::vector<double>& start,
    const std::vector<double>& end,
    int num_points)
{
    std::vector<std::vector<double>> trajectory(num_points, std::vector<double>(start.size()));

    double x0 = start[0], y0 = start[1], z0 = start[2];
    double xf = end[0], yf = end[1], zf = end[2];

    double cx = (x0 + xf) / 2.0;
    double cy = (y0 + yf) / 2.0;
    double scale_x = (xf - x0) / 2.0;
    double scale_y = (yf - y0) / 2.0;

    for (int i = 0; i < num_points; ++i) {
        double t = static_cast<double>(i) / (num_points - 1);
        double angle = t * 4.0 * M_PI - M_PI / 2.0;

        trajectory[i][0] = cx + scale_x * std::sin(angle);
        trajectory[i][1] = cy + scale_y * std::sin(2.0 * angle) / 2.0;
        trajectory[i][2] = z0 + (zf - z0) * t;

        for (size_t j = 3; j < start.size(); ++j) {
            trajectory[i][j] = 0.0;
        }
    }

    return trajectory;
}

std::vector<std::vector<double>> generate_helix_trajectory(
    const std::vector<double>& start,
    const std::vector<double>& end,
    int num_points)
{
    std::vector<std::vector<double>> trajectory(num_points, std::vector<double>(start.size()));

    double x0 = start[0], y0 = start[1], z0 = start[2];
    double xf = end[0], yf = end[1], zf = end[2];

    double cx = (x0 + xf) / 2.0;
    double cy = (y0 + yf) / 2.0;
    double radius = std::sqrt(std::pow(xf - x0, 2) + std::pow(yf - y0, 2)) / 2.0;

    for (int i = 0; i < num_points; ++i) {
        double t = static_cast<double>(i) / (num_points - 1);
        double angle = std::atan2(y0 - cy, x0 - cx) + t * 3.0 * M_PI;

        trajectory[i][0] = cx + radius * std::cos(angle);
        trajectory[i][1] = cy + radius * std::sin(angle);
        trajectory[i][2] = z0 + (zf - z0) * t;

        for (size_t j = 3; j < start.size(); ++j) {
            trajectory[i][j] = start[j] + (end[j] - start[j]) * t;
        }
    }

    return trajectory;
}

std::vector<std::vector<double>> generate_trajectory(
    const std::vector<double>& start,
    const std::vector<double>& end,
    int num_points)
{
    std::vector<std::vector<double>> trajectory(num_points, std::vector<double>(start.size()));

    std::vector<double> param_points = {0.0, 0.5, 1.0};

    std::vector<tk::spline> splines;
    for (size_t j = 0; j < start.size(); ++j) {
        double mid_value = (j < 2) ? (start[j] + end[j]) / 2.0 : start[j] + (end[j] - start[j]) * 0.5;
        std::vector<double> values = {start[j], mid_value, end[j]};
        tk::spline s;
        s.set_points(param_points, values);
        splines.push_back(s);
    }

    for (int i = 0; i < num_points; ++i) {
        double t = static_cast<double>(i) / (num_points - 1);
        for (size_t j = 0; j < start.size(); ++j) {
            trajectory[i][j] = splines[j](t);
        }
    }

    return trajectory;
}

void draw_differential_drive(double x, double y, double theta)
{
    double L = 0.4;
    double W = 0.3;

    std::vector<double> corners_x = {-L/2, L/2, L/2, -L/2, -L/2};
    std::vector<double> corners_y = {-W/2, -W/2, W/2, W/2, -W/2};

    std::vector<double> robot_x, robot_y;
    for (size_t i = 0; i < corners_x.size(); ++i) {
        double rx = x + corners_x[i] * std::cos(theta) - corners_y[i] * std::sin(theta);
        double ry = y + corners_x[i] * std::sin(theta) + corners_y[i] * std::cos(theta);
        robot_x.push_back(rx);
        robot_y.push_back(ry);
    }
    plt::plot(robot_x, robot_y, "b-");

    double arrow_len = 0.3;
    std::vector<double> arrow_x = {x, x + arrow_len * std::cos(theta)};
    std::vector<double> arrow_y = {y, y + arrow_len * std::sin(theta)};
    plt::plot(arrow_x, arrow_y, "r-");
}

void draw_bicycle(double x, double y, double theta)
{
    double L = 0.5;
    double W = 0.2;

    std::vector<double> corners_x = {-L/2, L/2, L/2, -L/2, -L/2};
    std::vector<double> corners_y = {-W/2, -W/2, W/2, W/2, -W/2};

    std::vector<double> robot_x, robot_y;
    for (size_t i = 0; i < corners_x.size(); ++i) {
        double rx = x + corners_x[i] * std::cos(theta) - corners_y[i] * std::sin(theta);
        double ry = y + corners_x[i] * std::sin(theta) + corners_y[i] * std::cos(theta);
        robot_x.push_back(rx);
        robot_y.push_back(ry);
    }
    plt::plot(robot_x, robot_y, "b-");

    double arrow_len = 0.3;
    std::vector<double> arrow_x = {x, x + arrow_len * std::cos(theta)};
    std::vector<double> arrow_y = {y, y + arrow_len * std::sin(theta)};
    plt::plot(arrow_x, arrow_y, "r-");
}

void draw_quadrotor(double x, double y, double z)
{
    double arm_len = 0.3;

    std::vector<double> arm1_x = {x - arm_len, x + arm_len};
    std::vector<double> arm1_y = {y, y};
    plt::plot(arm1_x, arm1_y, "b-");

    std::vector<double> arm2_x = {x, x};
    std::vector<double> arm2_y = {y - arm_len, y + arm_len};
    plt::plot(arm2_x, arm2_y, "b-");

    plt::plot({x}, {y}, "ro");

    std::vector<double> prop_x = {x - arm_len, x + arm_len, x, x};
    std::vector<double> prop_y = {y, y, y - arm_len, y + arm_len};
    for (size_t i = 0; i < prop_x.size(); ++i) {
        double r = 0.1;
        std::vector<double> circle_x, circle_y;
        for (int j = 0; j < 12; ++j) {
            double angle = 2.0 * M_PI * j / 12;
            circle_x.push_back(prop_x[i] + r * std::cos(angle));
            circle_y.push_back(prop_y[i] + r * std::sin(angle));
        }
        circle_x.push_back(circle_x[0]);
        circle_y.push_back(circle_y[0]);
        plt::plot(circle_x, circle_y, "k-");
    }
}

void draw_rocket(double x, double y, double theta, double thrust, double gimbal_y, double gimbal_z)
{
    double height = 1.0;
    double width = 0.3;

    std::vector<double> body_x = {
        x - width/2, x + width/2, x + width/2, x + width/3,
        x, x - width/3, x - width/2, x - width/2
    };
    std::vector<double> body_y = {
        y, y, y + height*0.7, y + height*0.7,
        y + height, y + height*0.7, y + height*0.7, y
    };

    for (size_t i = 0; i < body_x.size(); ++i) {
        double dx = body_x[i] - x;
        double dy = body_y[i] - y;
        body_x[i] = x + dx * std::cos(theta) - dy * std::sin(theta);
        body_y[i] = y + dx * std::sin(theta) + dy * std::cos(theta);
    }
    plt::plot(body_x, body_y, "b-");

    double thrust_scale = thrust / 245250.0;
    double flame_len = 0.5 * thrust_scale;

    double nozzle_x = x;
    double nozzle_y = y;

    double flame_angle = theta - M_PI/2 + gimbal_y;
    double flame_end_x = nozzle_x + flame_len * std::cos(flame_angle);
    double flame_end_y = nozzle_y + flame_len * std::sin(flame_angle);

    std::vector<double> flame_x = {nozzle_x - 0.1, flame_end_x, nozzle_x + 0.1};
    std::vector<double> flame_y = {nozzle_y, flame_end_y, nozzle_y};
    plt::plot(flame_x, flame_y, "r-");

    plt::plot({x}, {y + height/2}, "ko");
}

void create_animation(
    const std::vector<std::vector<double>>& x_feedback,
    const std::vector<std::vector<double>>& u_feedback,
    const std::vector<double>& x_target,
    const std::string& model_name,
    const std::vector<std::vector<double>>& x_ref_traj,
    const std::vector<std::vector<std::vector<double>>>& predictions = {})
{
    fs::create_directory("plots");

    int skip = std::max(1, static_cast<int>(x_feedback.size() / 100));

    double min_x = x_feedback[0][0], max_x = x_feedback[0][0];
    double min_y = x_feedback[0][1], max_y = x_feedback[0][1];

    for (const auto& state : x_feedback) {
        min_x = std::min(min_x, state[0]);
        max_x = std::max(max_x, state[0]);
        min_y = std::min(min_y, state[1]);
        max_y = std::max(max_y, state[1]);
    }

    min_x = std::min(min_x, x_target[0]) - 1.5;
    max_x = std::max(max_x, x_target[0]) + 1.5;
    min_y = std::min(min_y, x_target[1]) - 1.5;
    max_y = std::max(max_y, x_target[1]) + 1.5;

    std::vector<double> traj_x, traj_y;
    std::vector<std::string> frame_files;

    for (size_t i = 0; i < x_feedback.size(); i += skip) {
        plt::clf();

        traj_x.push_back(x_feedback[i][0]);
        traj_y.push_back(x_feedback[i][1]);

        plt::figure_size(1200, 600);

        std::vector<double> ref_x, ref_y, ref_z;
        for (const auto& ref : x_ref_traj) {
            ref_x.push_back(ref[0]);
            ref_y.push_back(ref[1]);
            if (model_name == "quadrotor") {
                ref_z.push_back(ref[2]);
            }
        }

        std::vector<double> boundary_x = {min_x, max_x, max_x, min_x, min_x};
        std::vector<double> boundary_y = {min_y, min_y, max_y, max_y, min_y};
        plt::plot(boundary_x, boundary_y, "w-");

        size_t pred_idx = i * skip;
        bool has_predictions = !predictions.empty() && pred_idx < predictions.size();

        if (model_name == "quadrotor") {
            plt::xlim(min_x, max_x);
            plt::ylim(min_y, max_y);
            plt::axis("equal");

            std::map<std::string, std::string> ref_keywords;
            ref_keywords["color"] = "gray";
            ref_keywords["linestyle"] = ":";
            ref_keywords["linewidth"] = "1.5";
            ref_keywords["label"] = "Reference";
            plt::plot(ref_x, ref_y, ref_keywords);

            std::map<std::string, std::string> traj_keywords;
            traj_keywords["color"] = "blue";
            traj_keywords["linestyle"] = "--";
            traj_keywords["linewidth"] = "2.0";
            traj_keywords["label"] = "Actual";
            plt::plot(traj_x, traj_y, traj_keywords);

            std::map<std::string, std::string> target_keywords;
            target_keywords["color"] = "red";
            target_keywords["marker"] = "*";
            target_keywords["markersize"] = "15";
            target_keywords["label"] = "Target";
            plt::plot({x_target[0]}, {x_target[1]}, target_keywords);
        } else {
            std::map<std::string, std::string> ref_keywords;
            ref_keywords["color"] = "gray";
            ref_keywords["linestyle"] = ":";
            ref_keywords["linewidth"] = "1.5";
            ref_keywords["label"] = "Reference";
            plt::plot(ref_x, ref_y, ref_keywords);

            std::map<std::string, std::string> traj_keywords;
            traj_keywords["color"] = "blue";
            traj_keywords["linestyle"] = "--";
            traj_keywords["linewidth"] = "2.0";
            traj_keywords["label"] = "Actual";
            plt::plot(traj_x, traj_y, traj_keywords);

            std::map<std::string, std::string> target_keywords;
            target_keywords["color"] = "red";
            target_keywords["marker"] = "*";
            target_keywords["markersize"] = "15";
            target_keywords["label"] = "Target";
            plt::plot({x_target[0]}, {x_target[1]}, target_keywords);
        }

        double x = x_feedback[i][0];
        double y = x_feedback[i][1];

        if (has_predictions && model_name == "quadrotor") {
            std::vector<double> pred_x, pred_y;
            for (const auto& pred_state : predictions[pred_idx]) {
                if (pred_state.size() >= 2) {
                    pred_x.push_back(pred_state[0]);
                    pred_y.push_back(pred_state[1]);
                }
            }
            if (!pred_x.empty()) {
                plt::plot(pred_x, pred_y, "g--");
            }
        }

        if (model_name == "rocket") {
            double theta = x_feedback[i][6];
            double thrust = i < u_feedback.size() ? u_feedback[i][0] : 0.0;
            double gimbal_y = i < u_feedback.size() ? u_feedback[i][1] : 0.0;
            double gimbal_z = i < u_feedback.size() ? u_feedback[i][2] : 0.0;
            draw_rocket(x, y, theta, thrust, gimbal_y, gimbal_z);
        } else if (model_name == "quadrotor") {
            double z = x_feedback[i][2];
            draw_quadrotor(x, y, z);
        } else if (model_name == "differential_drive") {
            double theta = x_feedback[i][2];
            draw_differential_drive(x, y, theta);
        } else if (model_name == "bicycle") {
            double theta = x_feedback[i][2];
            draw_bicycle(x, y, theta);
        } else {
            double theta = x_feedback[i].size() > 2 ? x_feedback[i][2] : 0.0;
            draw_differential_drive(x, y, theta);
        }

        plt::legend();

        std::ostringstream filename;
        filename << "plots/frame_" << model_name << "_" << std::setfill('0') << std::setw(4) << frame_files.size() << ".png";
        plt::save(filename.str());
        frame_files.push_back(filename.str());
        plt::close();
    }

    std::string gif_path = "plots/animation_" + model_name + ".gif";
    std::ostringstream convert_cmd;
    convert_cmd << "convert -delay 5 -loop 0 plots/frame_" << model_name << "_*.png " << gif_path;

    int result = system(convert_cmd.str().c_str());

    if (result == 0) {
        std::cout << "Animation saved to " << gif_path << std::endl;
        for (const auto& f : frame_files) {
            remove(f.c_str());
        }
    } else {
        std::cout << "Frames saved. To create GIF: " << convert_cmd.str() << std::endl;
    }
}

void plot_results(
    const std::vector<double>& time,
    const std::vector<std::vector<double>>& x_feedback,
    const std::vector<std::vector<double>>& u_feedback,
    const std::vector<double>& x_target,
    const std::string& model_name)
{
    fs::create_directory("plots");

    int nx = x_feedback[0].size();
    int nu = u_feedback[0].size();

    std::vector<std::vector<double>> states(nx);
    for (const auto& state : x_feedback) {
        for (int i = 0; i < nx; ++i) {
            states[i].push_back(state[i]);
        }
    }

    std::vector<std::vector<double>> controls(nu);
    for (const auto& control : u_feedback) {
        for (int i = 0; i < nu; ++i) {
            controls[i].push_back(control[i]);
        }
    }

    plt::figure_size(1200, 800);
    for (int i = 0; i < nx; ++i) {
        plt::plot(time, states[i]);
    }
    plt::save("plots/states_" + model_name + ".png");
    plt::clf();

    plt::figure_size(1200, 800);
    for (int i = 0; i < nu; ++i) {
        plt::plot(time, controls[i]);
    }
    plt::save("plots/controls_" + model_name + ".png");
    plt::clf();

    if (nx >= 2) {
        plt::figure_size(1200, 800);
        plt::plot(states[0], states[1], "r");
        plt::plot({x_target[0]}, {x_target[1]}, "go");
        plt::save("plots/trajectory_" + model_name + ".png");
        plt::clf();
    }

    std::cout << "Plots saved in 'plots' directory." << std::endl;
}

int main(int argc, char** argv) {
    plt::backend("Agg");

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>" << std::endl;
        return 1;
    }

    std::ifstream config_file(argv[1]);
    if (!config_file.is_open()) {
        std::cerr << "Failed to open config file: " << argv[1] << std::endl;
        return 1;
    }

    json config;
    config_file >> config;

    std::string model_name = config["model_name"];

    // Handle multi-quadrotor separately
    if (model_name == "multi_quadrotor") {
        int n_agents = config["n_agents"];
        double dt    = config["dt"];
        double distance_threshold   = config.value("distance_threshold", 0.2);
        int max_iterations          = config.value("max_iterations", 3000);

        std::cout << "Multi-Agent Quadrotor MPC with Obstacle Avoidance\n";
        std::cout << "Agents: " << n_agents << "\n";
        std::cout << "Obstacles: " << config["obstacles"].size() << "\n\n";

        // Load all agent solvers
        std::vector<std::unique_ptr<AcadosMPCSolver>> solvers;
        std::vector<std::vector<double>> x_curr(n_agents);
        std::vector<std::vector<double>> x_target(n_agents);
        std::vector<std::vector<double>> u_prev(n_agents);

        // Initialize each agent
        for (int i = 0; i < n_agents; ++i) {
            auto& agent_config = config["agents"][i];
            x_curr[i] = agent_config["x_init"].get<std::vector<double>>();
            x_target[i] = agent_config["x_target"].get<std::vector<double>>();

            // Create solver for this agent
            std::vector<double> lbx = {-10, -10, 0, -5, -5, -5, -1.047, -1.047, -3.14159, -2, -2, -2};
            std::vector<double> ubx = {10, 10, 5, 5, 5, 5, 1.047, 1.047, 3.14159, 2, 2, 2};
            std::vector<double> lbu = {2.943, -1, -1, -0.5};
            std::vector<double> ubu = {19.62, 1, 1, 0.5};
            std::vector<double> Q_diag = {150, 150, 100, 15, 15, 10, 5, 5, 2, 1, 1, 1};
            std::vector<double> R_diag = {0.1, 0.5, 0.5, 0.5};
            std::vector<double> R_rate_diag = {0.5, 0.3, 0.3, 0.3};

            std::string model_name = "quadrotor_" + std::to_string(i);
            solvers.push_back(std::make_unique<AcadosMPCSolver>(
                model_name, lbx, ubx, lbu, ubu, Q_diag, R_diag, R_rate_diag));

            u_prev[i] = std::vector<double>(4, 0.0);

            std::cout << "Agent " << i << ": (" << x_curr[i][0] << "," << x_curr[i][1] << "," << x_curr[i][2]
                      << ") -> (" << x_target[i][0] << "," << x_target[i][1] << "," << x_target[i][2] << ")\n";
        }

        std::cout << "\nRunning distributed MPC...\n";

        // Storage for trajectories
        std::vector<std::vector<std::vector<double>>> trajectories(n_agents);

        // Main MPC loop
        for (int iter = 0; iter < max_iterations; ++iter) {
            bool all_reached = true;

            // Solve for each agent
            for (int i = 0; i < n_agents; ++i) {
                // Check if reached target
                double error = std::sqrt(std::pow(x_curr[i][0] - x_target[i][0], 2) +
                                         std::pow(x_curr[i][1] - x_target[i][1], 2) +
                                         std::pow(x_curr[i][2] - x_target[i][2], 2));

                if (error >= distance_threshold) {
                    all_reached = false;

                    // Set initial state
                    solvers[i]->setInitialState(x_curr[i]);

                    // Set reference to target
                    int N = solvers[i]->getN();
                    int nx = solvers[i]->getNX();
                    int nu = solvers[i]->getNU();

                    for (int j = 0; j <= N; ++j) {
                        std::vector<double> yref(nx + nu, 0.0);
                        for (int k = 0; k < nx; ++k) {
                            yref[k] = x_target[i][k];
                        }
                        // Set hover thrust as reference
                        yref[nx] = 9.81;
                        solvers[i]->setReference(j, yref);
                    }

                    // Solve MPC
                    int status = solvers[i]->solve();

                    // Only update if solver succeeded
                    if (status == 0) {
                        x_curr[i] = solvers[i]->getState(1);
                        u_prev[i] = solvers[i]->getControl(0);
                    }
                }

                // Store trajectory
                trajectories[i].push_back({x_curr[i][0], x_curr[i][1], x_curr[i][2]});
            }

            if (all_reached) {
                std::cout << "All agents reached targets at iteration " << iter << "\n";
                break;
            }

            if (iter % 50 == 0) {
                std::cout << "Iteration " << iter << "\n";
                for (int i = 0; i < n_agents; ++i) {
                    double err = std::sqrt(std::pow(x_curr[i][0] - x_target[i][0], 2) +
                                           std::pow(x_curr[i][1] - x_target[i][1], 2) +
                                           std::pow(x_curr[i][2] - x_target[i][2], 2));
                    std::cout << "  Agent " << i << " pos: (" << x_curr[i][0] << "," << x_curr[i][1] << "," << x_curr[i][2]
                              << ") error: " << err << " m\n";
                }
            }
        }

        std::cout << "\nCreating 3D animation with obstacles...\n";

        // Create Python animation script
        std::ofstream script("../build/animate_multi_quad.py");
        script << "import numpy as np\n";
        script << "import matplotlib.pyplot as plt\n";
        script << "from mpl_toolkits.mplot3d import Axes3D\n";
        script << "from matplotlib.animation import FuncAnimation\n\n";

        // Write trajectories
        script << "trajectories = [\n";
        for (int i = 0; i < n_agents; ++i) {
            script << "    [";
            for (size_t j = 0; j < trajectories[i].size(); ++j) {
                script << "[" << trajectories[i][j][0] << "," << trajectories[i][j][1] << "," << trajectories[i][j][2] << "]";
                if (j < trajectories[i].size() - 1) script << ",";
            }
            script << "]";
            if (i < n_agents - 1) script << ",\n";
        }
        script << "]\n\n";

        // Write targets
        script << "targets = [";
        for (int i = 0; i < n_agents; ++i) {
            script << "[" << x_target[i][0] << "," << x_target[i][1] << "," << x_target[i][2] << "]";
            if (i < n_agents - 1) script << ",";
        }
        script << "]\n\n";

        // Write obstacles
        script << "obstacles = [";
        for (size_t i = 0; i < config["obstacles"].size(); ++i) {
            auto& obs = config["obstacles"][i];
            script << "{'x':" << obs["x"] << ",'y':" << obs["y"] << ",'z':" << obs["z"] << ",'r':" << obs["radius"] << "}";
            if (i < config["obstacles"].size() - 1) script << ",";
        }
        script << "]\n\n";

        script << R"(
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Initialize lines and points
lines = [ax.plot([], [], [], '-', color=colors[i], linewidth=2, alpha=0.6, label=f'Agent {i}')[0] for i in range(len(trajectories))]
points = [ax.plot([], [], [], 'o', color=colors[i], markersize=10)[0] for i in range(len(trajectories))]
target_points = [ax.plot([targets[i][0]], [targets[i][1]], [targets[i][2]], '*', color=colors[i], markersize=20, alpha=0.5)[0] for i in range(len(targets))]

# Draw obstacles as spheres
for obs in obstacles:
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = obs['r'] * np.outer(np.cos(u), np.sin(v)) + obs['x']
    y = obs['r'] * np.outer(np.sin(u), np.sin(v)) + obs['y']
    z = obs['r'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obs['z']
    ax.plot_surface(x, y, z, color='red', alpha=0.3)

def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
    return lines + points

def update(frame):
    for i in range(len(trajectories)):
        if frame < len(trajectories[i]):
            traj = np.array(trajectories[i][:frame+1])
            lines[i].set_data(traj[:, 0], traj[:, 1])
            lines[i].set_3d_properties(traj[:, 2])
            points[i].set_data([traj[-1, 0]], [traj[-1, 1]])
            points[i].set_3d_properties([traj[-1, 2]])
    return lines + points

max_len = max(len(t) for t in trajectories)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 6])
ax.set_zlim([0, 4])
ax.set_title('Multi-Agent Quadrotor with Obstacle Avoidance')
ax.legend()
ax.view_init(elev=25, azim=45)

anim = FuncAnimation(fig, update, frames=max_len, init_func=init, interval=50, blit=True, repeat=True)
anim.save('../build/plots/multi_quad.gif', writer='pillow', fps=20)
print('Saved to build/plots/multi_quad.gif')
plt.show()
)";
        script.close();

        system("cd ../build && python3 animate_multi_quad.py");

        return 0;
    }

    std::vector<double> x_init = config["x_init"];
    std::vector<double> x_target = config["x_target"];
    std::vector<double> lbx = config["lbx"];
    std::vector<double> ubx = config["ubx"];
    std::vector<double> lbu = config["lbu"];
    std::vector<double> ubu = config["ubu"];
    std::vector<double> Q_diag = config["Q_diag"];
    std::vector<double> R_diag = config["R_diag"];
    std::vector<double> R_rate_diag = config["R_rate_diag"];
    double dt = config["dt"];
    double distance_threshold = config.value("distance_threshold", 0.1);
    int max_iterations = config.value("max_iterations", 10000);

    try {
        AcadosMPCSolver solver(model_name, lbx, ubx, lbu, ubu, Q_diag, R_diag, R_rate_diag);

        int nx = solver.getNX();
        int nu = solver.getNU();
        int N = solver.getN();
        int ny = nx + nu + nu;

        std::cout << "Model: " << model_name << std::endl;
        std::cout << "States: " << nx << ", Controls: " << nu << ", Horizon: " << N << std::endl;

        std::vector<std::vector<double>> x_ref_traj;
        if (model_name == "quadrotor") {
            x_ref_traj = generate_helix_trajectory(x_init, x_target, 150);
        } else {
            x_ref_traj = generate_trajectory(x_init, x_target, 100);
        }

        std::vector<double> x_curr = x_init;
        std::vector<double> u_prev(nu, 0.0);
        std::vector<double> time_steps;
        std::vector<std::vector<double>> x_feedback;
        std::vector<std::vector<double>> u_feedback;
        std::vector<std::vector<std::vector<double>>> predictions;

        for (int i = 0; i < max_iterations; ++i) {
            double error = 0.0;
            size_t num_pos_states = (model_name == "quadrotor" || model_name == "rocket") ? 3 : 2;
            for (size_t j = 0; j < std::min(num_pos_states, x_curr.size()); ++j) {
                error += std::pow(x_curr[j] - x_target[j], 2);
            }
            error = std::sqrt(error);

            if (error < distance_threshold) {
                std::cout << "Reached target at time " << i * dt << "s" << std::endl;
                std::cout << "Final error: " << error << std::endl;
                break;
            }

            double current_time = i * dt;
            time_steps.push_back(current_time);
            x_feedback.push_back(x_curr);

            solver.setInitialState(x_curr);

            for (int j = 0; j <= N; ++j) {
                int ref_index = std::min(i + j, static_cast<int>(x_ref_traj.size()) - 1);
                std::vector<double> yref(ny, 0.0);

                for (int k = 0; k < nx; ++k) {
                    yref[k] = x_ref_traj[ref_index][k];
                }

                solver.setReference(j, yref);
            }

            int status = solver.solve();
            if (status != 0) {
                std::cerr << "Solver failed with status " << status << std::endl;
                return 1;
            }

            std::vector<double> u = solver.getControl(0);
            u_feedback.push_back(u);

            std::vector<std::vector<double>> prediction_horizon;
            for (int j = 0; j <= N; ++j) {
                prediction_horizon.push_back(solver.getState(j));
            }
            predictions.push_back(prediction_horizon);

            std::vector<double> x_next = solver.getState(1);
            x_curr = x_next;

            if (i % 20 == 0) {
                std::cout << "Time: " << current_time << "s, Error: " << error << std::endl;
            }

            u_prev = u;
        }

        solver.printSolverInfo();

        std::cout << "Plotting results..." << std::endl;
        plot_results(time_steps, x_feedback, u_feedback, x_target, model_name);

        std::cout << "Creating animation..." << std::endl;
        create_animation(x_feedback, u_feedback, x_target, model_name, x_ref_traj, predictions);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
