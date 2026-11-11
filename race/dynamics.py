"""Car dynamics in JAX: single-track Pacejka model, blended with a kinematic model at low speed.

The simulator and Auto-Tuning MPPI use the blend (``ODE:ks_pacejka``); the MPPI baseline predicts with the
pure Pacejka model (``ODE:pacejka``), as in the paper's runs.

State: [yaw rate, v_x, v_y, heading, cos, sin, x, y, slip angle, steering angle] (see ``state.py``).
Control: [desired steering angle (rad), acceleration command (m/s²)].
"""

from functools import partial

import jax
import jax.numpy as jnp


def _wrap_angle_rad(angle):
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def _steering_constraints(delta, delta_dot, s_min, s_max, sv_min, sv_max):
    at_min = jnp.logical_and(delta <= s_min, delta_dot <= 0.0)
    at_max = jnp.logical_and(delta >= s_max, delta_dot >= 0.0)
    allowed = jnp.logical_not(jnp.logical_or(at_min, at_max))
    delta_dot = delta_dot * allowed.astype(delta_dot.dtype)
    return jnp.clip(delta_dot, sv_min, sv_max)


def _accl_constraints(v_x, accl, a_min, a_max, v_min, v_max, v_switch, mu, g_):
    vel = jnp.where(v_x == 0.0, 1.0e-4, v_x)
    pos_limit = jnp.where(vel > v_switch, a_max * v_switch / vel, a_max)
    v_too_low = jnp.logical_and(v_x < v_min, accl < 0.0)
    v_too_high = jnp.logical_and(v_x > v_max, accl > 0.0)
    accl = jnp.where(jnp.logical_or(v_too_low, v_too_high), 0.0, accl)
    accl = jnp.clip(accl, a_min, pos_limit)
    max_a_friction = mu * g_
    return jnp.clip(accl, -max_a_friction, max_a_friction)


def _servo_proportional(desired_steering_angle, delta, servo_p, steering_diff_low, sv_min, sv_max):
    steering_angle_difference = desired_steering_angle - delta
    active = jnp.abs(steering_angle_difference) > steering_diff_low
    delta_dot = active.astype(jnp.float32) * (steering_angle_difference * servo_p)
    return jnp.clip(delta_dot, sv_min, sv_max)


def _smooth_sign_vx(v_x, v_dead):
    return v_x / jnp.sqrt(v_x * v_x + v_dead * v_dead)


def _apply_longitudinal_multipliers(v_x_dot, brake_multiplier, accel_multiplier):
    """Brake / accel scaling of the command (smooth in the sign of the command)."""
    tanh_term = jnp.tanh(25.0 * v_x_dot)
    sig_neg = 0.5 * (1.0 - tanh_term)
    sig_pos = 0.5 * (1.0 + tanh_term)
    neutral = jnp.maximum(0.0, 1.0 - sig_neg - sig_pos)
    scale = sig_neg * brake_multiplier + sig_pos * accel_multiplier + neutral
    return v_x_dot * scale


def _rolling_resistance_accel(v_x, c_rr, g_, v_dead):
    return -c_rr * g_ * _smooth_sign_vx(v_x, v_dead)


def _cornering_resistance_accel(v_x, F_yf, F_yr, m, curve_resistance_factor, v_dead):
    lateral_force_magnitude = jnp.sqrt(F_yf * F_yf + F_yr * F_yr)
    return -curve_resistance_factor * lateral_force_magnitude / m * _smooth_sign_vx(v_x, v_dead)


def _pacejka_lateral_forces(v_x, v_y, psi_dot, delta, v_x_dot,
                            lf, lr, h_cg, m, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, mu):
    # |v_x| and sign(v_x) * delta keep the slip angles valid when driving in reverse.
    v_x_safe = jnp.maximum(jnp.abs(v_x), 1.0e-3)
    alpha_f = -jnp.arctan((v_y + psi_dot * lf) / v_x_safe) + jnp.sign(v_x) * delta
    alpha_r = -jnp.arctan((v_y - psi_dot * lr) / v_x_safe)

    F_zf = m * (-v_x_dot * h_cg + g_ * lr) / (lr + lf)
    F_zr = m * (v_x_dot * h_cg + g_ * lf) / (lr + lf)

    F_yf = mu * F_zf * D_f * jnp.sin(
        C_f * jnp.arctan(B_f * alpha_f - E_f * (B_f * alpha_f - jnp.arctan(B_f * alpha_f))))
    F_yr = mu * F_zr * D_r * jnp.sin(
        C_r * jnp.arctan(B_r * alpha_r - E_r * (B_r * alpha_r - jnp.arctan(B_r * alpha_r))))

    # Friction ellipse: the drive/brake force uses part of the tyre grip, the rest is left for cornering.
    lat_scale = jnp.sqrt(jnp.maximum(1.0 - (v_x_dot / (mu * g_)) ** 2, 1.0e-4))
    return F_yf * lat_scale, F_yr * lat_scale


def _pacejka_step(s_x, s_y, delta, v_x, v_y, psi, psi_dot, delta_dot, v_x_dot,
                  lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r,
                  mu, s_min, s_max, dt_sub, curve_resistance_factor=0.0, v_dead=0.05):
    F_yf, F_yr = _pacejka_lateral_forces(
        v_x, v_y, psi_dot, delta, v_x_dot, lf, lr, h_cg, m, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, mu)
    v_x_dot = v_x_dot + jax.lax.stop_gradient(
        _cornering_resistance_accel(v_x, F_yf, F_yr, m, curve_resistance_factor, v_dead))
    max_a_friction = mu * g_
    v_x_dot = jnp.clip(v_x_dot, -max_a_friction, max_a_friction)

    d_s_x = v_x * jnp.cos(psi) - v_y * jnp.sin(psi)
    d_s_y = v_x * jnp.sin(psi) + v_y * jnp.cos(psi)
    d_psi = psi_dot
    d_v_x = v_x_dot - F_yf * jnp.sin(delta) / m + v_y * psi_dot
    d_v_y = (F_yr + F_yf * jnp.cos(delta)) / m - v_x * psi_dot
    d_psi_dot = (-lr * F_yr + lf * F_yf * jnp.cos(delta)) / I_z

    s_x = s_x + dt_sub * d_s_x
    s_y = s_y + dt_sub * d_s_y
    delta = jnp.clip(delta + dt_sub * delta_dot, s_min, s_max)
    v_x = v_x + dt_sub * d_v_x
    v_y = v_y + dt_sub * d_v_y
    psi = _wrap_angle_rad(psi + dt_sub * d_psi)
    psi_dot = psi_dot + dt_sub * d_psi_dot
    return s_x, s_y, delta, v_x, v_y, psi, psi_dot


def _ks_step(s_x, s_y, delta, v_x, v_y, psi, delta_dot, v_x_dot, lr, l_wb, s_min, s_max, dt_sub):
    # Kinematic bicycle at the CoG: the steering sets only the direction of the speed.
    v = jnp.where(v_x < 0.0, -1.0, 1.0) * jnp.sqrt(v_x * v_x + v_y * v_y)
    beta = jnp.arctan(jnp.tan(delta) * lr / l_wb)
    s_x_dot = v * jnp.cos(psi + beta)
    s_y_dot = v * jnp.sin(psi + beta)
    psi_dot = v * jnp.cos(beta) * jnp.tan(delta) / l_wb

    s_x = s_x + dt_sub * s_x_dot
    s_y = s_y + dt_sub * s_y_dot
    delta = jnp.clip(delta + dt_sub * delta_dot, s_min, s_max)
    v = v + dt_sub * v_x_dot
    psi = _wrap_angle_rad(psi + dt_sub * psi_dot)
    beta = jnp.arctan(jnp.tan(delta) * lr / l_wb)
    v_x = v * jnp.cos(beta)
    v_y = v * jnp.sin(beta)
    angular_vel_z = v_x * jnp.tan(delta) / l_wb
    return s_x, s_y, delta, v_x, v_y, psi, angular_vel_z


def _next_step_output(psi_dot, v_x, v_y, psi, s_x, s_y, delta):
    v_x_safe = jnp.where(v_x < 1.0e-3, 1.0e-3, v_x)
    slip_angle = jnp.arctan(v_y / v_x_safe)
    return jnp.array([
        psi_dot, v_x, v_y, psi, jnp.cos(psi), jnp.sin(psi),
        s_x, s_y, slip_angle, delta,
    ], dtype=jnp.float32)


def _blend_states_circular_yaw(s_ks, s_pacejka, weight):
    """Linear blend of the states, circular blend of the heading."""
    w = weight
    blended = (1.0 - w) * s_ks + w * s_pacejka
    sin_yaw = (1.0 - w) * jnp.sin(s_ks[3]) + w * jnp.sin(s_pacejka[3])
    cos_yaw = (1.0 - w) * jnp.cos(s_ks[3]) + w * jnp.cos(s_pacejka[3])
    psi = jnp.arctan2(sin_yaw, cos_yaw)
    blended = blended.at[3].set(psi)
    blended = blended.at[4].set(jnp.cos(psi))
    blended = blended.at[5].set(jnp.sin(psi))
    return blended


@partial(jax.jit, static_argnames=['intermediate_steps', 'ode_model'])
def car_dynamics(state, control, car_params, dt, intermediate_steps=1, ode_model='ODE:ks_pacejka'):
    """Advance ``state`` by ``dt`` under ``control``, in ``intermediate_steps`` Euler steps.

    ``car_params`` is ``CarParameters.to_array()``. ``ode_model``: 'ODE:pacejka', 'ODE:ks' or 'ODE:ks_pacejka'.
    """
    mu, lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, \
        servo_p, s_min, s_max, sv_min, sv_max, a_min, a_max, v_min, v_max, v_switch = car_params[:25]
    steering_diff_low = car_params[25]
    c_rr = car_params[26]
    v_dead = car_params[27]
    curve_resistance_factor = car_params[28]
    brake_multiplier = car_params[29]
    accel_multiplier = car_params[30]
    steering_multiplier = car_params[31]
    l_wb = lf + lr
    dt_sub = dt / intermediate_steps
    use_ks = ode_model in ('ODE:ks', 'ODE:ks_pacejka')
    use_pacejka = ode_model in ('ODE:pacejka', 'ODE:ks_pacejka')
    use_blend = ode_model == 'ODE:ks_pacejka'

    def single_step(state_in, _):
        psi_dot, v_x, v_y, psi, _, _, s_x, s_y, _, delta = state_in
        desired_steering_angle, translational_control = control
        desired_steering_angle = desired_steering_angle * steering_multiplier

        delta_dot = _servo_proportional(desired_steering_angle, delta, servo_p, steering_diff_low, sv_min, sv_max)
        delta_dot = _steering_constraints(delta, delta_dot, s_min, s_max, sv_min, sv_max)

        v_x_dot = _apply_longitudinal_multipliers(translational_control, brake_multiplier, accel_multiplier)
        v_x_dot = _accl_constraints(v_x, v_x_dot, a_min, a_max, v_min, v_max, v_switch, mu, g_)
        v_x_dot = v_x_dot + jax.lax.stop_gradient(_rolling_resistance_accel(v_x, c_rr, g_, v_dead))
        max_a_friction = mu * g_
        v_x_dot = jnp.clip(v_x_dot, -max_a_friction, max_a_friction)

        if use_pacejka:
            p_s_x, p_s_y, p_delta, p_v_x, p_v_y, p_psi, p_psi_dot = _pacejka_step(
                s_x, s_y, delta, v_x, v_y, psi, psi_dot, delta_dot, v_x_dot,
                lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r,
                mu, s_min, s_max, dt_sub, curve_resistance_factor, v_dead)
            s_pacejka = _next_step_output(p_psi_dot, p_v_x, p_v_y, p_psi, p_s_x, p_s_y, p_delta)
        else:
            s_pacejka = state_in

        if use_ks:
            k_s_x, k_s_y, k_delta, k_v_x, k_v_y, k_psi, k_angular_vel_z = _ks_step(
                s_x, s_y, delta, v_x, v_y, psi, delta_dot, v_x_dot, lr, l_wb, s_min, s_max, dt_sub)
            s_ks = _next_step_output(k_angular_vel_z, k_v_x, k_v_y, k_psi, k_s_x, k_s_y, k_delta)
        else:
            s_ks = state_in

        if use_blend:
            # Kinematic model only when slow and not sliding.
            speed = jnp.sqrt(v_x * v_x + v_y * v_y)
            slide_f = (v_y + lf * psi_dot) * jnp.cos(delta) - v_x * jnp.sin(delta)
            slide_r = v_y - lr * psi_dot
            slide = jnp.maximum(jnp.abs(slide_f), jnp.abs(slide_r))
            w_speed = 1.0 / (1.0 + jnp.exp(-4.8 * (speed - 1.75)))
            w_slide = 1.0 / (1.0 + jnp.exp(-10.0 * (slide - 0.5)))
            weight = 1.0 - (1.0 - w_speed) * (1.0 - w_slide)
            next_state = _blend_states_circular_yaw(s_ks, s_pacejka, weight)
        elif use_pacejka:
            next_state = s_pacejka
        else:
            next_state = s_ks
        return next_state, None

    final_state, _ = jax.lax.scan(single_step, state, None, length=intermediate_steps)
    return final_state


@partial(jax.jit, static_argnames=["dt", "horizon", "intermediate_steps"])
def car_steps_sequential(s0, Q_sequence, car_params, dt, horizon, intermediate_steps=1):
    """``(H, 10)`` states after each of the ``H`` controls ``Q_sequence``, pure Pacejka model (MPPI baseline)."""
    def rollout_fn(state, control):
        next_state = car_dynamics(state, control, car_params, dt, intermediate_steps, ode_model='ODE:pacejka')
        return next_state, next_state
    _, trajectory = jax.lax.scan(rollout_fn, s0, Q_sequence)
    return trajectory
