#pragma once

#include "acados_solver_differential_drive.h"
#include <vector>

class AcadosSolver {
    public:
        AcadosSolver(const std::vector<double>& lbx, const std::vector<double>& ubx,
                    const std::vector<double>& lbu, const std::vector<double>& ubu);
        ~AcadosSolver();

        int solve();
        void setInitialState(const std::vector<double>& x0);
        void setReference(int stage, const std::vector<double>& yref);
        std::vector<double> getState(int stage);
        std::vector<double> getControl(int stage);
        double getSolveTime();
        int getSQPIterations();
        void printSolverInfo();

    private:
        differential_drive_solver_capsule* capsule;
        ocp_nlp_config* nlp_config;
        ocp_nlp_dims* nlp_dims;
        ocp_nlp_in* nlp_in;
        ocp_nlp_out* nlp_out;
        ocp_nlp_solver* nlp_solver;
        void* nlp_opts;
};