#pragma once

#include "acados_c/ocp_nlp_interface.h"
#include <vector>
#include <string>
#include <functional>

class AcadosMPCSolver {
public:
    AcadosMPCSolver(
        const std::string& model_name,
        const std::vector<double>& lbx,
        const std::vector<double>& ubx,
        const std::vector<double>& lbu,
        const std::vector<double>& ubu,
        const std::vector<double>& Q_diag,
        const std::vector<double>& R_diag,
        const std::vector<double>& R_rate_diag
    );

    ~AcadosMPCSolver();

    int solve();
    void setInitialState(const std::vector<double>& x0);
    void setReference(int stage, const std::vector<double>& yref);
    void setParameter(int stage, const std::vector<double>& p);
    std::vector<double> getState(int stage);
    std::vector<double> getControl(int stage);
    double getSolveTime();
    int getSQPIterations();
    void printSolverInfo();

    int getNX() const { return nx; }
    int getNU() const { return nu; }
    int getN() const { return N; }

private:
    void* capsule;
    void* lib_handle;

    std::function<void*()> create_capsule_fn;
    std::function<int(void*)> create_fn;
    std::function<int(void*)> solve_fn;
    std::function<int(void*)> free_capsule_fn;
    std::function<void*(void*)> get_nlp_config_fn;
    std::function<void*(void*)> get_nlp_dims_fn;
    std::function<void*(void*)> get_nlp_in_fn;
    std::function<void*(void*)> get_nlp_out_fn;
    std::function<void*(void*)> get_nlp_solver_fn;
    std::function<void*(void*)> get_nlp_opts_fn;
    std::function<int(void*, unsigned int, double*, int)> update_params_fn;

    ocp_nlp_config* nlp_config;
    ocp_nlp_dims* nlp_dims;
    ocp_nlp_in* nlp_in;
    ocp_nlp_out* nlp_out;
    ocp_nlp_solver* nlp_solver;
    void* nlp_opts;

    int nx;
    int nu;
    int N;

    std::vector<std::vector<double>> params_storage;  // Storage for parameters at each stage
};
