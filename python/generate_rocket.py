#!/usr/bin/env python3
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from rocket_model import export_rocket_model


def main():
    ocp = AcadosOcp()

    model     = export_rocket_model()
    ocp.model = model

    Tf = 2.0
    nx = model.x.rows()
    nu = model.u.rows()
    N  = 20

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf

    # Cost matrices 
    Q_mat = np.diag([
        100.0, 100.0, 80.0,  # x, y, z
        15.0, 15.0, 10.0     # vx, vy, vz
    ])

    R_mat      = np.diag([0.1, 0.5, 0.5])
    R_rate_mat = np.diag([0.5, 0.3, 0.3])

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
        -50.0, -50.0, 0.0,           # position bounds (z >= 0)
        -15.0, -15.0, -20.0          # velocity bounds
    ])
    ocp.constraints.ubx = np.array([
        50.0, 50.0, 100.0,
        15.0, 15.0, 20.0
    ])
    ocp.constraints.idxbx = np.arange(nx)

    # Control constraints - 3 thrust components
    m = 25000.0
    g = 9.81
    max_thrust_xy = m * g * 0.5  # Lateral thrust
    max_thrust_z = m * g * 2.0   # Vertical thrust

    ocp.constraints.lbu = np.array([-max_thrust_xy, -max_thrust_xy, 0.0])
    ocp.constraints.ubu = np.array([max_thrust_xy, max_thrust_xy, max_thrust_z])
    ocp.constraints.idxbu = np.arange(nu)

    # Initial state 
    ocp.constraints.x0 = np.zeros(nx)

    # Solver options 
    ocp.solver_options.qp_solver        = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hpipm_mode       = 'SPEED'
    ocp.solver_options.hessian_approx   = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type  = 'ERK'
    ocp.solver_options.nlp_solver_type  = 'SQP_RTI'

    ocp.code_export_directory = '../acados_generated_rocket'

    ocp_solver  = AcadosOcpSolver(ocp, json_file='../acados_generated_rocket/acados_ocp.json')

    # Dynamically consistent warm start
    m  = 25000.0
    g  = 9.81
    z0 = 10.0
    dt_stage = Tf / N

    # Constant thrust slightly less than hover for gentle descent
    thrust_z    = m * g * 0.95
    az          = thrust_z / m - g  # net acceleration (should be ~-0.5 m/s^2)

    z   = z0
    vz  = 0.0

    for i in range(N):
        ocp_solver.set(i, 'x', np.array([0.0, 0.0, z, 0.0, 0.0, vz]))
        ocp_solver.set(i, 'u', np.array([0.0, 0.0, thrust_z]))

        # Forward simulate with constant thrust
        vz = vz + az * dt_stage
        z  = z + vz * dt_stage
        z  = max(0.0, z)  # clamp to ground

    ocp_solver.set(N, 'x', np.zeros(nx))

    print(f"Generated solver: {model.name}")
    print(f"States (nx)     : {nx}")
    print(f"Controls (nu)   : {nu}")
    print(f"Horizon (N)     : {N}")
    print(f"Prediction time : {Tf}s")

    return ocp_solver


if __name__ == '__main__':
    main()
