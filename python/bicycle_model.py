#!/usr/bin/env python3
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, tan


def export_bicycle_model():
    model_name = 'bicycle'

    x     = SX.sym('x')        # [m] x position
    y     = SX.sym('y')        # [m] y position
    theta = SX.sym('theta')    # [rad] heading angle
    v     = SX.sym('v')        # [m/s] velocity
    x_sym = vertcat(x, y, theta, v)

    a     = SX.sym('a')        # [m/s^2] acceleration
    delta = SX.sym('delta')    # [rad] steering angle
    u     = vertcat(a, delta)

    L = 2.5                    # [m] wheelbase length

    xdot     = SX.sym('xdot')
    ydot     = SX.sym('ydot')
    thetadot = SX.sym('thetadot')
    vdot     = SX.sym('vdot')
    xdot_sym = vertcat(xdot, ydot, thetadot, vdot)

    f_expl = vertcat(
        v * cos(theta),
        v * sin(theta),
        (v / L) * tan(delta),
        a
    )

    model             = AcadosModel()
    model.name        = model_name
    model.x           = x_sym
    model.xdot        = xdot_sym
    model.u           = u
    model.f_expl_expr = f_expl

    return model
