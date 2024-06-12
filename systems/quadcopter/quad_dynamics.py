# general
import math
import numpy as np

# utilities
from helper import rotation_x, rotation_y, rotation_z
from helper import euler2quat, quat2rotation, rotation2euler
from helper import update_full_state

def rotation_matrix(roll, pitch, yaw):
    """ find rotation matrix """

    rot_x = rotation_x(roll)
    rot_y = rotation_y(pitch)
    rot_z = rotation_z(yaw)

    R_b2w = rot_z.dot(rot_y).dot(rot_x)

    rot_x = rotation_x(-roll)
    rot_y = rotation_y(-pitch)
    rot_z = rotation_z(-yaw)

    R_w2b = rot_x.dot(rot_y).dot(rot_z)

    return R_w2b, R_b2w

def quat_update(cur_quat, pqr, dt):
    vector_cross = np.array([[cur_quat[0], -cur_quat[3], cur_quat[2]],
                    [cur_quat[3], cur_quat[0], -cur_quat[1]],
                    [-cur_quat[2], cur_quat[1], cur_quat[0]]])

    # update angle and axis
    quat_w_next = -0.5 * cur_quat[1:].dot(pqr) * dt
    quat_axis_next = 0.5 * vector_cross.dot(pqr) * dt

    # update quaternion
    quat_next = np.append(quat_w_next, quat_axis_next) + cur_quat

    # normalize quaternion
    quat_next /= np.linalg.norm(quat_next)

    return quat_next

# TODO: check this agianst the cpp simulator
def euler_update(rpy, pqr, dt):
    # update euler angles
    rpy_next = rpy + dt * pqr

    # wrap angles
    for i in range(len(rpy_next)):
        rpy_next[i] = rpy_next[i] % (2 * math.pi)
        if rpy_next[i] > math.pi:
            rpy_next[i] -= (2 * math.pi)

    return rpy_next

def quad_state_update(curr_state, squared_motor_rates, params, rot_update_type, disturb):
    # compute rotation matrix
    # R_w2b, R_b2w = rotation_matrix(curr_state.roll, curr_state.pitch, curr_state.yaw)
    R_w2b = curr_state.R

    # compute thurst
    thrust_b = params.quad.ct * np.sum(squared_motor_rates)
    thrust_w = R_w2b.dot(np.array([0, 0, thrust_b]))

    # compute net force
    force = -params.quad.m * params.quad.g
    net_force = thrust_w + np.array([0, 0, force])

    # compute acceleration
    acc = net_force / params.quad.m

    # compute moment
    alpha = params.quad.l * params.quad.ct / math.sqrt(2)
    beta = params.quad.cd

    moment_x = alpha * np.sum(np.array([-1, -1, 1, 1]) * squared_motor_rates)
    moment_y = alpha * np.sum(np.array([-1, 1, 1, -1]) * squared_motor_rates)
    moment_z = beta * np.sum(np.array([-1, 1, -1, 1]) * squared_motor_rates)

    moment = np.array([moment_x, moment_y, moment_z])

    # compute inertia
    J = np.diag([params.quad.Ixx, params.quad.Iyy, params.quad.Izz])
    J_inv = np.linalg.inv(J)
    pqr = np.array([curr_state.p, curr_state.q, curr_state.r])
    temp0 = moment - np.cross(pqr, J.dot(pqr))
    angular_rate_derivative = J_inv.dot(temp0)

    # compute rpy angles
    omega = pqr + params.sim.dt * angular_rate_derivative

    if rot_update_type == "quat":
        # update based on quaternion
        quat_next = quat_update(curr_state.quat, pqr, params.sim.dt)
        R_next = quat2rotation(quat_next)
        rpy_next = rotation2euler(R_next)

    elif rot_update_type == "euler":
        # update based on euler angle
        rpy = np.array([curr_state.roll, curr_state.pitch, curr_state.yaw])
        rpy_next = euler_update(rpy, pqr, params.sim.dt)
        quat_next = euler2quat(rpy_next[0], rpy_next[1], rpy_next[2])
        R_next = quat2rotation(quat_next)

    else:
        print("Invalid update type for rotational dynamics")

    # populate next_state
    next_state = curr_state

    next_state.x = curr_state.x + params.sim.dt * curr_state.vx
    next_state.y = curr_state.y + params.sim.dt * curr_state.vy
    next_state.z = curr_state.z + params.sim.dt * curr_state.vz

    if disturb:
        curr_state.az += 3000

    next_state.vx = curr_state.vx + params.sim.dt * curr_state.ax
    next_state.vy = curr_state.vy + params.sim.dt * curr_state.ay
    next_state.vz = curr_state.vz + params.sim.dt * curr_state.az

    next_state.ax = acc[0]
    next_state.ay = acc[1]
    next_state.az = acc[2]

    next_state.p = omega[0]
    next_state.q = omega[1]
    next_state.r = omega[2]

    next_state.quat = quat_next
    next_state.R = R_next
    next_state.roll = rpy_next[0]
    next_state.pitch = rpy_next[1]
    next_state.yaw = rpy_next[2]

    next_state.full_state = update_full_state(next_state)

    return next_state

def main():
    quad_state_update()

if __name__ == "__main__":
    main()