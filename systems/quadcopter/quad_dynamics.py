import jax
import jax.numpy as jnp
import jax.random as jr
import chex

import numpy as np
from brax.envs.base import State, Env
from munch import Munch
from scipy.spatial.transform import Rotation

def _compute_motor_pwm(params: Munch,
                       base_thrust: chex.Array,
                       motor_variation: chex.Array) -> chex.Array:
    temp = jnp.array([[-0.5, -0.5, -1.0],
                            [-0.5, +0.5, +1.0],
                            [+0.5, +0.5, -1.0],
                            [+0.5, -0.5, +1.0]])
    adjustment = temp.dot(motor_variation)
    motor_pwm = base_thrust + adjustment
    motor_pwm = jnp.maximum(motor_pwm, jnp.asarray(params.quad.thrust_min))
    motor_pwm = jnp.minimum(motor_pwm, jnp.asarray(params.quad.thrust_max))

    return motor_pwm

def _motor_state_update(params: Munch,
                        pwm: chex. Array) -> chex.Array:
    """ computes motor squared rpm from pwm """
    motor_rates = params.quad.pwm2rpm_scale * pwm + params.quad.pwm2rpm_const
    squared_motor_rates = motor_rates ** 2
    return squared_motor_rates

def _quad_state_update(params: Munch,
                       state: State,
                       sq_mot_rates: chex. Array,
                       disturb: bool) -> State:
    R_w2b = state.metrics["R"]

    # compute thrust
    thrust_b = params.quad.ct * jnp.sum(sq_mot_rates)
    thrust_w = R_w2b.dot(jnp.array([0, 0, thrust_b]))

    # compute net force
    grav_force = -params.quad.m * params.quad.g
    net_force = thrust_w + jnp.array([0, 0, grav_force])

    # compute acceleration
    acc = net_force / params.quad.m

    # compute moment
    alpha = params.quad.l * params.quad.ct / jnp.sqrt(2)
    beta = params.quad.cd

    moment_x = alpha * jnp.sum(jnp.array([-1, -1, 1, 1]) * sq_mot_rates)
    moment_y = alpha * jnp.sum(jnp.array([-1, 1, 1, -1]) * sq_mot_rates)
    moment_z = beta * jnp.sum(jnp.array([-1, 1, -1, 1]) * sq_mot_rates)

    moment = jnp.array([moment_x, moment_y, moment_z])

    # compute inertia
    I = jnp.diag(jnp.array([params.quad.Ixx, params.quad.Iyy, params.quad.Izz]))
    I_inv = jnp.linalg.inv(I)
    pqr = state.obs[9:]
    temp0 = moment - jnp.cross(pqr, I.dot(pqr))
    angular_rate_derivative = I_inv.dot(temp0)

    dt = params.sim.dt
    # compute rpy angles
    omega = pqr + dt * angular_rate_derivative

    # update based on quaternion
    quat_next = _quat_update(state.metrics["quat"], pqr, dt)
    R_next = _quat2rotation(quat_next)
    rpy_next = _rotation2euler(R_next)

    # update state
    obs = jnp.array([
        state.obs[0] + dt * state.obs[3],
        state.obs[1] + dt * state.obs[4],
        state.obs[2] + dt * state.obs[5],
        state.obs[3] + dt * acc[0],
        state.obs[4] + dt * acc[1],
        state.obs[5] + dt * acc[2],
        rpy_next[0],
        rpy_next[1],
        rpy_next[2],
        omega[0],
        omega[1],
        omega[2]
    ])

    metrics = {
        'time': state.metrics["time"] + dt,
        'R': R_next,
        'quat': quat_next,
        'acc': acc
    }

    # Check if done
    condition1 = state.metrics["time"] > params.sim.t1
    condition2 = jnp.any(jnp.abs(obs[:3]) > params.sim.pos_lim)
    
    done = jnp.where(condition1 | condition2, 1.0, 0.0)

    return State(pipeline_state=None,
                    obs=obs,
                    reward=0.0,
                    done=done,
                    metrics=metrics)

def _quat_update(cur_quat: chex.Array,
                 pqr: chex.Array,
                 dt: float) -> chex.Array:
    """Update quaternion based on angular rates pqr"""
    vector_cross = jnp.array([[cur_quat[0], -cur_quat[3], cur_quat[2]],
                                [cur_quat[3], cur_quat[0], -cur_quat[1]],
                                [-cur_quat[2], cur_quat[1], cur_quat[0]]])

    # update angle and axis
    quat_w_next = -0.5 * cur_quat[1:].dot(pqr) * dt
    quat_axis_next = 0.5 * vector_cross.dot(pqr) * dt

    # update quaternion
    quat_next = jnp.append(quat_w_next, quat_axis_next) + cur_quat
    quat_next /= jnp.linalg.norm(quat_next)
    return quat_next

def _quat2rotation(quat: chex.Array) -> chex.Array:
    """ convert quaternion to rotation matrix """
    w, x, y, z = quat

    # Compute the elements of the rotation matrix
    r11 = 1 - 2 * (y**2 + z**2)
    r12 = 2 * (x * y - z * w)
    r13 = 2 * (x * z + y * w)
    
    r21 = 2 * (x * y + z * w)
    r22 = 1 - 2 * (x**2 + z**2)
    r23 = 2 * (y * z - x * w)
    
    r31 = 2 * (x * z - y * w)
    r32 = 2 * (y * z + x * w)
    r33 = 1 - 2 * (x**2 + y**2)
    
    # Assemble the rotation matrix
    rotation_matrix = jnp.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])
    
    return rotation_matrix

def _rotation2euler(R: chex.Array) -> chex.Array:
    """ convert rotation matrix to euler (xyz convention)"""
    # Extract the angles from the rotation matrix
    sy = jnp.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    alpha = jnp.where(sy >= 1e-6,
                      jnp.arctan2(R[2, 1], R[2, 2]),
                      jnp.arctan2(-R[1, 2], R[1, 1]))
    
    beta = jnp.where(sy >= 1e-6,
                     jnp.arctan2(-R[2, 0], sy),
                     jnp.arctan2(-R[2, 0], sy))
    
    gamma = jnp.where(sy >= 1e-6,
                      jnp.arctan2(R[1, 0], R[0, 0]),
                      jnp.zeros_like(R[0, 0]))  # gamma is 0 when sy is small
    
    return jnp.array([alpha, beta, gamma])

def _euler2quat(roll: float,
                pitch: float,
                yaw: float) -> chex.Array:
    """ convert euler angles (xyz convention) to quaternion """
    
    half_roll = roll / 2.0
    half_pitch = pitch / 2.0
    half_yaw = yaw / 2.0
    
    cos_half_roll = jnp.cos(half_roll)
    sin_half_roll = jnp.sin(half_roll)
    
    cos_half_pitch = jnp.cos(half_pitch)
    sin_half_pitch = jnp.sin(half_pitch)
    
    cos_half_yaw = jnp.cos(half_yaw)
    sin_half_yaw = jnp.sin(half_yaw)
    
    w = cos_half_roll * cos_half_pitch * cos_half_yaw + sin_half_roll * sin_half_pitch * sin_half_yaw
    x = sin_half_roll * cos_half_pitch * cos_half_yaw - cos_half_roll * sin_half_pitch * sin_half_yaw
    y = cos_half_roll * sin_half_pitch * cos_half_yaw + sin_half_roll * cos_half_pitch * sin_half_yaw
    z = cos_half_roll * cos_half_pitch * sin_half_yaw - sin_half_roll * sin_half_pitch * cos_half_yaw
    
    return jnp.array([w, x, y, z])