#!/usr/bin/env python3
import numpy as np
import casadi as ca
from acados_template import AcadosModel


def export_quadrotor_model() -> AcadosModel:
    model_name = 'quadrotor'

    # State: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    z = ca.SX.sym('z')

    vx = ca.SX.sym('vx')
    vy = ca.SX.sym('vy')
    vz = ca.SX.sym('vz')

    phi = ca.SX.sym('phi')      # roll
    theta = ca.SX.sym('theta')  # pitch
    psi = ca.SX.sym('psi')      # yaw

    p = ca.SX.sym('p')          # roll rate
    q = ca.SX.sym('q')          # pitch rate
    r = ca.SX.sym('r')          # yaw rate

    state = ca.vertcat(x, y, z, vx, vy, vz, phi, theta, psi, p, q, r)

    # Controls: [thrust, tau_phi, tau_theta, tau_psi]
    thrust      = ca.SX.sym('thrust')
    tau_phi     = ca.SX.sym('tau_phi')
    tau_theta   = ca.SX.sym('tau_theta')
    tau_psi     = ca.SX.sym('tau_psi')

    control     = ca.vertcat(thrust, tau_phi, tau_theta, tau_psi)

    # Parameters
    m = 1.0      # mass (kg)
    g = 9.81     # gravity (m/s^2)
    Ixx = 0.01   # moment of inertia
    Iyy = 0.01
    Izz = 0.02

    # Dynamics
    dx = vx
    dy = vy
    dz = vz

    dvx = (ca.sin(psi) * ca.sin(phi) + ca.cos(psi) * ca.sin(theta) * ca.cos(phi)) * thrust / m
    dvy = (-ca.cos(psi) * ca.sin(phi) + ca.sin(psi) * ca.sin(theta) * ca.cos(phi)) * thrust / m
    dvz = ca.cos(theta) * ca.cos(phi) * thrust / m - g

    dphi = p + ca.sin(phi) * ca.tan(theta) * q + ca.cos(phi) * ca.tan(theta) * r
    dtheta = ca.cos(phi) * q - ca.sin(phi) * r
    dpsi = ca.sin(phi) / ca.cos(theta) * q + ca.cos(phi) / ca.cos(theta) * r

    dp = tau_phi / Ixx + (Iyy - Izz) / Ixx * q * r
    dq = tau_theta / Iyy + (Izz - Ixx) / Iyy * p * r
    dr = tau_psi / Izz + (Ixx - Iyy) / Izz * p * q

    f_expl = ca.vertcat(dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr)

    x_dot = ca.SX.sym('x_dot', state.shape)
    f_impl = x_dot - f_expl

    # Create model
    model = AcadosModel()
    model.name = model_name
    model.x = state
    model.xdot = x_dot
    model.u = control
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl

    return model
