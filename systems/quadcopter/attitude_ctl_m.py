# general
import numpy as np
from helper import clamp

def attitude_controller_m(curr_state, set_point, quad):
    params = quad.params
    controller = quad.controller
    
    measured_pqr = np.array([curr_state.p, curr_state.q, curr_state.r])
    current_R = curr_state.R
    desired_euler = np.array([set_point.roll, set_point.pitch, set_point.yaw])
    desired_pqr = np.array([set_point.p, set_point.q, set_point.r])
    desired_R = set_point.R

    # eR error
    eRM = desired_R.T.dot(current_R) - current_R.T.dot(desired_R)

    eR = np.zeros(3)
    eR[0] = eRM[2,1]
    eR[1] = eRM[0,2]
    eR[2] = eRM[1,0]

    controller.att_ctl.prev_pqr = measured_pqr[:]
    controller.att_ctl.prev_pqr_des = desired_pqr[:]

    # ew error
    ew = desired_pqr - measured_pqr

    # compute i error
    controller.att_ctl.i_error += -eR * params.sim.dt
    controller.att_ctl.i_error = clamp(controller.att_ctl.i_error, np.array(params.att_ctl.i_range))

    # compute moment
    M = np.zeros(3)
    M -= params.att_ctl.kR * eR
    M += params.att_ctl.kw * ew
    M += params.att_ctl.ki * controller.att_ctl.i_error
    # not using kd term
    # M += params.att_ctl.kd * err_d

    # motor variation
    motor_var_desired = clamp(M, np.array([32000.0, 32000.0, 32000.0]))
    # pdb.set_trace()
    return motor_var_desired

def attitude_controller_reset(quad):
	controller = quad.controller
	controller.att_ctl.i_error = np.zeros(3)

def main():
    attitude_controller_m()

if __name__ == "__main__":
    main()
