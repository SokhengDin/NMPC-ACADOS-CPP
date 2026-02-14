#include "acados_mpc_solver.hpp"
#include <dlfcn.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

AcadosMPCSolver::AcadosMPCSolver(
    const std::string& model_name,
    const std::vector<double>& lbx,
    const std::vector<double>& ubx,
    const std::vector<double>& lbu,
    const std::vector<double>& ubu,
    const std::vector<double>& Q_diag,
    const std::vector<double>& R_diag,
    const std::vector<double>& R_rate_diag)
{
    std::string lib_path = "libacados_ocp_solver_" + model_name + ".dylib";

    lib_handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib_handle) {
        throw std::runtime_error("Failed to load library: " + std::string(dlerror()));
    }

    create_capsule_fn = reinterpret_cast<void*(*)()>(
        dlsym(lib_handle, (model_name + "_acados_create_capsule").c_str()));
    create_fn = reinterpret_cast<int(*)(void*)>(
        dlsym(lib_handle, (model_name + "_acados_create").c_str()));
    solve_fn = reinterpret_cast<int(*)(void*)>(
        dlsym(lib_handle, (model_name + "_acados_solve").c_str()));
    free_capsule_fn = reinterpret_cast<int(*)(void*)>(
        dlsym(lib_handle, (model_name + "_acados_free_capsule").c_str()));
    get_nlp_config_fn = reinterpret_cast<void*(*)(void*)>(
        dlsym(lib_handle, (model_name + "_acados_get_nlp_config").c_str()));
    get_nlp_dims_fn = reinterpret_cast<void*(*)(void*)>(
        dlsym(lib_handle, (model_name + "_acados_get_nlp_dims").c_str()));
    get_nlp_in_fn = reinterpret_cast<void*(*)(void*)>(
        dlsym(lib_handle, (model_name + "_acados_get_nlp_in").c_str()));
    get_nlp_out_fn = reinterpret_cast<void*(*)(void*)>(
        dlsym(lib_handle, (model_name + "_acados_get_nlp_out").c_str()));
    get_nlp_solver_fn = reinterpret_cast<void*(*)(void*)>(
        dlsym(lib_handle, (model_name + "_acados_get_nlp_solver").c_str()));
    get_nlp_opts_fn = reinterpret_cast<void*(*)(void*)>(
        dlsym(lib_handle, (model_name + "_acados_get_nlp_opts").c_str()));
    update_params_fn = reinterpret_cast<int(*)(void*, unsigned int, double*, int)>(
        dlsym(lib_handle, (model_name + "_acados_update_params").c_str()));

    if (!create_capsule_fn || !create_fn || !solve_fn || !free_capsule_fn ||
        !get_nlp_config_fn || !get_nlp_dims_fn || !get_nlp_in_fn ||
        !get_nlp_out_fn || !get_nlp_solver_fn || !get_nlp_opts_fn) {
        dlclose(lib_handle);
        throw std::runtime_error("Failed to load solver functions for model: " + model_name);
    }

    capsule = create_capsule_fn();
    create_fn(capsule);

    nlp_config = static_cast<ocp_nlp_config*>(get_nlp_config_fn(capsule));
    nlp_dims   = static_cast<ocp_nlp_dims*>(get_nlp_dims_fn(capsule));
    nlp_in     = static_cast<ocp_nlp_in*>(get_nlp_in_fn(capsule));
    nlp_out    = static_cast<ocp_nlp_out*>(get_nlp_out_fn(capsule));
    nlp_solver = static_cast<ocp_nlp_solver*>(get_nlp_solver_fn(capsule));
    nlp_opts   = get_nlp_opts_fn(capsule);

    nx = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nlp_out, 0, "x");
    nu = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nlp_out, 0, "u");
    N  = nlp_dims->N;

    for (int i = 0; i <= N; ++i) {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbx", const_cast<double*>(lbx.data()));
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubx", const_cast<double*>(ubx.data()));
    }

    for (int i = 0; i < N; ++i) {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbu", const_cast<double*>(lbu.data()));
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubu", const_cast<double*>(ubu.data()));
    }

    int ny   = nx + nu + nu;
    int ny_e = nx;

    std::vector<double> W(ny * ny, 0.0);
    std::vector<double> W_e(ny_e * ny_e, 0.0);

    for (int i = 0; i < nx; ++i) {
        W[i * ny + i] = Q_diag[i];
    }
    for (int i = 0; i < nu; ++i) {
        W[(nx + i) * ny + (nx + i)] = R_diag[i];
    }
    for (int i = 0; i < nu; ++i) {
        W[(nx + nu + i) * ny + (nx + nu + i)] = R_rate_diag[i];
    }

    for (int i = 0; i < nx; ++i) {
        W_e[i * ny_e + i] = Q_diag[i];
    }

    for (int i = 0; i < N; ++i) {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W.data());
    }
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e.data());
}

AcadosMPCSolver::~AcadosMPCSolver() {
    if (capsule) {
        free_capsule_fn(capsule);
    }
    if (lib_handle) {
        dlclose(lib_handle);
    }
}

int AcadosMPCSolver::solve() {
    return solve_fn(capsule);
}

void AcadosMPCSolver::setInitialState(const std::vector<double>& x0) {
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", const_cast<double*>(x0.data()));
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", const_cast<double*>(x0.data()));
}

void AcadosMPCSolver::setReference(int stage, const std::vector<double>& yref) {
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, stage, "yref", const_cast<double*>(yref.data()));
}

void AcadosMPCSolver::setParameter(int stage, const std::vector<double>& p) {
    if (update_params_fn) {
        update_params_fn(capsule, stage, const_cast<double*>(p.data()), p.size());
    }
}

std::vector<double> AcadosMPCSolver::getState(int stage) {
    std::vector<double> x(nx);
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, stage, "x", x.data());
    return x;
}

std::vector<double> AcadosMPCSolver::getControl(int stage) {
    std::vector<double> u(nu);
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, stage, "u", u.data());
    return u;
}

double AcadosMPCSolver::getSolveTime() {
    double solve_time = 0.0;
    ocp_nlp_get(nlp_solver, "time_tot", &solve_time);
    return solve_time;
}

int AcadosMPCSolver::getSQPIterations() {
    int sqp_iter = 0;
    ocp_nlp_get(nlp_solver, "sqp_iter", &sqp_iter);
    return sqp_iter;
}

void AcadosMPCSolver::printSolverInfo() {
    std::cout << "Solver info:" << std::endl;
    std::cout << " SQP iterations " << getSQPIterations() << std::endl;
    std::cout << " Solve time " << getSolveTime() << " [ms]" << std::endl;
}
