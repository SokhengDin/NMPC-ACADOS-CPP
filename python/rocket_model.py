#!/usr/bin/env python3
import numpy as np
import casadi as ca
from acados_template import AcadosModel


def export_rocket_model() -> AcadosModel:
    model_name = 'rocket'

    # State: [x, y, z, vx, vy, vz] 
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    z = ca.SX.sym('z')

    vx = ca.SX.sym('vx')
    vy = ca.SX.sym('vy')
    vz = ca.SX.sym('vz')

    state = ca.vertcat(x, y, z, vx, vy, vz)

    # Controls: [thrust_x, thrust_y, thrust_z] - direct 3D thrust
    thrust_x = ca.SX.sym('thrust_x')
    thrust_y = ca.SX.sym('thrust_y')
    thrust_z = ca.SX.sym('thrust_z')

    control = ca.vertcat(thrust_x, thrust_y, thrust_z)

    # Parameters
    m = 25000.0  # mass (kg) - Falcon 9 first stage ~25 tons dry
    g = 9.81     # gravity (m/s^2)

    # Dynamics 
    dx = vx
    dy = vy
    dz = vz

    dvx = thrust_x / m
    dvy = thrust_y / m
    dvz = thrust_z / m - g

    f_expl = ca.vertcat(dx, dy, dz, dvx, dvy, dvz)

    x_dot = ca.SX.sym('x_dot', state.shape)
    f_impl = x_dot - f_expl

    model = AcadosModel()
    model.name = model_name
    model.x = state
    model.xdot = x_dot
    model.u = control
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl

    return model
