#!/usr/bin/env python3
"""
Differential drive robot model for Acados
"""

from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos


def export_differential_drive_model():
    """Export differential drive dynamics model"""

    model_name = 'differential_drive'

    # States
    x     = SX.sym('x')
    y     = SX.sym('y')
    theta = SX.sym('theta')
    x_sym = vertcat(x, y, theta)

    # State derivatives
    x_dot     = SX.sym('x_dot')
    y_dot     = SX.sym('y_dot')
    theta_dot = SX.sym('theta_dot')
    xdot      = vertcat(x_dot, y_dot, theta_dot)

    # Controls
    v     = SX.sym('v')
    omega = SX.sym('omega')
    u     = vertcat(v, omega)

    # Dynamics
    f_expl = vertcat(
        v * cos(theta),
        v * sin(theta),
        omega
    )

    # Create model
    model           = AcadosModel()
    model.name      = model_name
    model.x         = x_sym
    model.xdot      = xdot
    model.u         = u
    model.f_expl_expr = f_expl

    return model
