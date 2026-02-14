#!/usr/bin/env python3
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from quadrotor_model import export_quadrotor_model


def main():
    ocp = AcadosOcp()

    model     = export_quadrotor_model()
    ocp.model = model

    Tf = 2.0
    nx = model.x.rows()
    nu = model.u.rows()
    N  = 30

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf

    # Cost matrices
    Q_mat = np.diag([
        100.0, 100.0, 80.0,  # x, y, z
        15.0, 15.0, 10.0,    # vx, vy, vz 
        5.0, 5.0, 2.0,       # phi, theta, psi
        1.0, 1.0, 1.0        # p, q, r
    ])
    R_mat       = np.diag([0.1, 0.5, 0.5, 0.5])
    R_rate_mat  = np.diag([0.5, 0.3, 0.3, 0.3])

    u_prev                  = ca.SX.sym('u_prev', nu)
    ocp.model.p             = u_prev
    ocp.parameter_values    = np.zeros(nu)

    # Path cost
    ocp.cost.cost_type      = 'NONLINEAR_LS'
    ocp.model.cost_y_expr   = ca.vertcat(model.x, model.u, model.u - u_prev)
    ocp.cost.yref           = np.zeros((nx + nu + nu,))
    ocp.cost.W              = ca.diagcat(Q_mat, R_mat, R_rate_mat).full()

    # Terminal cost
    ocp.cost.cost_type_e    = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref_e         = np.zeros((nx,))
    ocp.cost.W_e            = Q_mat * 2.0

    # State constraints
    ocp.constraints.lbx = np.array([
        -20.0, -20.0, 0.0,        # position bounds
        -5.0, -5.0, -5.0,         # velocity bounds
        -np.pi/3, -np.pi/3, -np.pi,  # attitude bounds
        -2.0, -2.0, -2.0          # angular velocity bounds
    ])
    ocp.constraints.ubx = np.array([
        20.0, 20.0, 15.0,
        5.0, 5.0, 5.0,
        np.pi/3, np.pi/3, np.pi,
        2.0, 2.0, 2.0
    ])
    ocp.constraints.idxbx = np.arange(nx)

    # Control constraints
    m = 1.0
    g = 9.81
    ocp.constraints.lbu = np.array([m*g*0.3, -1.0, -1.0, -0.5])
    ocp.constraints.ubu = np.array([m*g*2.0, 1.0, 1.0, 0.5])
    ocp.constraints.idxbu = np.arange(nu)

    # Initial state
    ocp.constraints.x0 = np.zeros(nx)

    # Solver options
    ocp.solver_options.qp_solver    = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hpipm_mode   = 'SPEED'
    ocp.solver_options.hessian_approx   = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type  = 'ERK'
    ocp.solver_options.nlp_solver_type  = 'SQP_RTI'

    ocp.code_export_directory           = '../acados_generated_quadrotor'

    ocp_solver = AcadosOcpSolver(ocp, json_file='../acados_generated_quadrotor/acados_ocp.json')

    print(f"Generated solver: {model.name}")
    print(f"States (nx)     : {nx}")
    print(f"Controls (nu)   : {nu}")
    print(f"Horizon (N)     : {N}")
    print(f"Prediction time : {Tf}s")

    return ocp_solver


if __name__ == '__main__':
    main()
