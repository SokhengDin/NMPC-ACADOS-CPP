#!/usr/bin/env python3
"""
Generate Acados C code for differential drive robot
"""
import numpy as np
import casadi as ca

from acados_template import AcadosOcp, AcadosOcpSolver
from differential_drive_model import export_differential_drive_model


def main():
    # Create OCP
    ocp = AcadosOcp()

    # Set model
    model = export_differential_drive_model()
    ocp.model = model

    # Dimensions
    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    N  = 30

    # Set prediction horizon
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf        = Tf

    # Cost matrices
    Q_mat = np.diag([25.0, 25.0, 25.0])      # x, y, theta
    R_mat = np.diag([0.1, 0.1])            # v, omega
    R_rate_mat = np.diag([2.0, 2.0])        # v_rate, omega_rate

    # Previous control as parameter
    u_prev               = ca.SX.sym('u_prev', nu)
    ocp.model.p          = u_prev
    ocp.parameter_values = np.zeros(nu)

    # Path cost
    ocp.cost.cost_type       = 'NONLINEAR_LS'
    ocp.model.cost_y_expr    = ca.vertcat(model.x, model.u, model.u - u_prev)
    ocp.cost.yref            = np.zeros((nx + nu + nu,))
    ocp.cost.W               = ca.diagcat(Q_mat, R_mat, R_rate_mat).full()

    # Terminal cost
    ocp.cost.cost_type_e     = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e  = model.x
    ocp.cost.yref_e          = np.zeros((nx,))
    ocp.cost.W_e             = Q_mat

    # State constraints
    ocp.constraints.lbx      = np.array([-10.0, -10.0, -np.pi])
    ocp.constraints.ubx      = np.array([10.0, 10.0, np.pi])
    ocp.constraints.idxbx    = np.array([0, 1, 2])

    # Control constraints
    ocp.constraints.lbu      = np.array([-1.5, -1.5])
    ocp.constraints.ubu      = np.array([1.5, 1.5])
    ocp.constraints.idxbu    = np.array([0, 1])

    # Initial state
    ocp.constraints.x0       = np.zeros(nx)

    # Solver options
    ocp.solver_options.qp_solver         = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hpipm_mode        = 'SPEED'
    ocp.solver_options.hessian_approx    = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type   = 'ERK'
    ocp.solver_options.nlp_solver_type   = 'SQP_RTI'

    # Set output directory
    ocp.code_export_directory   = '../acados_generated'

    # Generate solver
    ocp_solver                  = AcadosOcpSolver(ocp, json_file='../acados_generated/acados_ocp.json')

    print(f"Generated solver: {model.name}")
    print(f"States (nx)     : {nx}")
    print(f"Controls (nu)   : {nu}")
    print(f"Horizon (N)     : {N}")
    print(f"Prediction time : {Tf}s")

    return ocp_solver


if __name__ == '__main__':
    main()
