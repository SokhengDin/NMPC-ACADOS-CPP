#include "acados_cpp_wrapper.hpp"
#include <stdexcept>
#include <cstring>
#include <iostream>

AcadosSolver::AcadosSolver(const std::vector<double>& lbx, const std::vector<double>& ubx,
                           const std::vector<double>& lbu, const std::vector<double>& ubu) 
{
    capsule = differential_drive_acados_create_capsule();
    if (capsule == nullptr) {
        throw std::runtime_error("Failed to create acados capsule");
    }

    int status = differential_drive_acados_create(capsule);
    if (status != 0) {
        throw std::runtime_error("Failed to create acados solver");
    }

    nlp_config = differential_drive_acados_get_nlp_config(capsule);
    nlp_dims = differential_drive_acados_get_nlp_dims(capsule);
    nlp_in = differential_drive_acados_get_nlp_in(capsule);
    nlp_out = differential_drive_acados_get_nlp_out(capsule);
    nlp_solver = differential_drive_acados_get_nlp_solver(capsule);
    nlp_opts = differential_drive_acados_get_nlp_opts(capsule);

    // Set state constraints
    for (int i = 0; i <= DIFFERENTIAL_DRIVE_N; ++i) {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbx", const_cast<double*>(lbx.data()));
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubx", const_cast<double*>(ubx.data()));
    }

    // Set control constraints
    for (int i = 0; i < DIFFERENTIAL_DRIVE_N; ++i) {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbu", const_cast<double*>(lbu.data()));
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubu", const_cast<double*>(ubu.data()));
    }
}


AcadosSolver::~AcadosSolver() {
    if (capsule) {
        differential_drive_acados_free(capsule);
        differential_drive_acados_free_capsule(capsule);
    }
}

int AcadosSolver::solve() {
    return differential_drive_acados_solve(capsule);
}

void AcadosSolver::setInitialState(const std::vector<double>& x0) {
    if (x0.size() != DIFFERENTIAL_DRIVE_NX) {
        throw std::invalid_argument("Invalid initial state size");
    }
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", const_cast<double*>(x0.data()));
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", const_cast<double*>(x0.data()));
}

void AcadosSolver::setReference(int stage, const std::vector<double>& yref) {
    if (stage > DIFFERENTIAL_DRIVE_N) {
        throw std::invalid_argument("Invalid stage");
    }
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, stage, "yref", const_cast<double*>(yref.data()));
}

std::vector<double> AcadosSolver::getState(int stage) {
    std::vector<double> state(DIFFERENTIAL_DRIVE_NX);
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, stage, "x", state.data());
    return state;
}

std::vector<double> AcadosSolver::getControl(int stage) {
    if (stage >= DIFFERENTIAL_DRIVE_N) {
        throw std::invalid_argument("Invalid stage for control");
    }
    std::vector<double> control(DIFFERENTIAL_DRIVE_NU);
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, stage, "u", control.data());
    return control;
}

double AcadosSolver::getSolveTime() {
    double solve_time;
    ocp_nlp_get(nlp_config, nlp_solver, "time_tot", &solve_time);
    return solve_time;
}

int AcadosSolver::getSQPIterations() {
    int sqp_iter;
    ocp_nlp_get(nlp_config, nlp_solver, "sqp_iter", &sqp_iter);
    return sqp_iter;
}

void AcadosSolver::printSolverInfo() {
    std::cout << "\nSolver info:\n";
    std::cout << " SQP iterations " << getSQPIterations() << std::endl;
    std::cout << " Solve time " << getSolveTime() * 1000 << " [ms]\n";
}