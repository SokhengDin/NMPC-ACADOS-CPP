#!/usr/bin/env python3
import numpy as np
import casadi as ca

from acados_template import AcadosOcp, AcadosOcpSolver
from bicycle_model import export_bicycle_model


def main():
    ocp = AcadosOcp()

    model = export_bicycle_model()
    ocp.model = model

    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    N  = 20

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf        = Tf

    Q_mat      = np.diag([25.0, 25.0, 1.0, 0.1])
    R_mat      = np.diag([0.1, 0.01])
    R_rate_mat = np.diag([1.0, 2.0])

    u_prev               = ca.SX.sym('u_prev', nu)
    ocp.model.p          = u_prev
    ocp.parameter_values = np.zeros(nu)

    ocp.cost.cost_type       = 'NONLINEAR_LS'
    ocp.model.cost_y_expr    = ca.vertcat(model.x, model.u, model.u - u_prev)
    ocp.cost.yref            = np.zeros((nx + nu + nu,))
    ocp.cost.W               = ca.diagcat(Q_mat, R_mat, R_rate_mat).full()

    ocp.cost.cost_type_e     = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e  = model.x
    ocp.cost.yref_e          = np.zeros((nx,))
    ocp.cost.W_e             = Q_mat

    ocp.constraints.lbx      = np.array([-100.0, -100.0, -np.pi, -5.0])
    ocp.constraints.ubx      = np.array([100.0, 100.0, np.pi, 15.0])
    ocp.constraints.idxbx    = np.array([0, 1, 2, 3])

    ocp.constraints.lbu      = np.array([-3.0, -0.6])
    ocp.constraints.ubu      = np.array([3.0, 0.6])
    ocp.constraints.idxbu    = np.array([0, 1])

    ocp.constraints.x0       = np.zeros(nx)

    ocp.solver_options.qp_solver         = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hpipm_mode        = 'SPEED'
    ocp.solver_options.hessian_approx    = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type   = 'ERK'
    ocp.solver_options.nlp_solver_type   = 'SQP_RTI'

    ocp.code_export_directory   = '../acados_generated_bicycle'

    ocp_solver                  = AcadosOcpSolver(ocp, json_file='../acados_generated_bicycle/acados_ocp.json')

    print(f"Generated solver: {model.name}")
    print(f"States (nx)     : {nx}")
    print(f"Controls (nu)   : {nu}")
    print(f"Horizon (N)     : {N}")
    print(f"Prediction time : {Tf}s")

    return ocp_solver


if __name__ == '__main__':
    main()
