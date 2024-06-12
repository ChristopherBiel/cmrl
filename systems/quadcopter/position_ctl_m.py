
import numpy as np
import math
import pdb
from helper import rotation2euler
from helper import clamp

def position_controller_m(curr_state, set_point, quad):
    params = quad.params
    controller = quad.controller

    measured_pos = np.array([curr_state.x, curr_state.y, curr_state.z])
    measured_vel = np.array([curr_state.vx, curr_state.vy, curr_state.vz])
    current_R = curr_state.R

    desired_pos = np.array([set_point.x, set_point.y, set_point.z])
    desired_vel = np.array([set_point.vx, set_point.vy, set_point.vz])
    desired_acc = np.array([set_point.ax, set_point.ay, set_point.az])
    desired_yaw = set_point.yaw

    # compute position and velocity error
    pos_error = desired_pos - measured_pos
    vel_error = desired_vel - measured_vel

    # update integral error

    #TODO where should I keep this var?
    controller.pos_ctl.i_error += pos_error * params.sim.dt
    controller.pos_ctl.i_error = clamp(controller.pos_ctl.i_error, np.array(params.pos_ctl.i_range))

    # compute target thrust
    target_thrust = np.zeros(3)

    target_thrust += params.pos_ctl.kp * pos_error
    target_thrust += params.pos_ctl.ki * controller.pos_ctl.i_error
    target_thrust += params.pos_ctl.kd * vel_error
    target_thrust += params.quad.m * desired_acc
    # pdb.set_trace()
    target_thrust[2] += params.quad.m * params.quad.g
    # pdb.set_trace()
    # update z_axis
    z_axis = current_R[:,2]
    # z_axis = current_R[2,:]

    # update current thrust
    current_thrust = target_thrust.dot(z_axis)
    current_thrust = max(current_thrust, 0.0)

    # update z_axis_desired
    z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
    x_c_des = np.array([math.cos(desired_yaw), math.sin(desired_yaw), 0.0])
    y_axis_desired = np.cross(z_axis_desired, x_c_des)
    y_axis_desired /= np.linalg.norm(y_axis_desired)
    x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

    R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
    euler_desired = rotation2euler(R_desired)
    if current_thrust < 0.00:
        pdb.set_trace()
    #thrust_desired = (1 / params.quad.pwm2rpm_scale) * (np.sign(current_thrust) * math.sqrt(abs(current_thrust) / (4 * params.quad.ct)) - params.quad.pwm2rpm_const)
    thrust_desired = (1 / params.quad.pwm2rpm_scale) * (math.sqrt(current_thrust / (4 * params.quad.ct)) - params.quad.pwm2rpm_const)

    # pdb.set_trace()

    return thrust_desired, euler_desired

def position_controller_reset(params):
    params.pos_ctl.i_error = np.zeros(3)

    return

def main():
    position_controller_m()

if __name__ == "__main__":
    main()
