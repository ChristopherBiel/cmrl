# general
import numpy as np
import os

# utilities
from helper import *

# controller
from attitude_ctl_m import *
from position_ctl_m import *

# dynamics
from power_distrib import *
from motor_dynamics import *
from quad_dynamics import *

# quadrotor class
from quadrotor import quadrotor

# for logging
import pickle

# timer
import time

# printout and plotting flags
verbose = False
plot_state_flag = True
plot_tracking_flag = True

# logging
save_log = True
save_dir = './results_temp/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# simulation flags
add_disturbance_flag = False

# sample delay
sample_delay = 35

def main():
    # timer
    time_start = time.time()

    # instantiate quadrotor
    quad = quadrotor()

    # initialization
    params = quad.params
    curr_state = quad.state
    reference = quad.reference
    desired = quad.desired

    # simulation paramseters
    t0 = params.sim.t0
    dt = params.sim.dt
    t1 = params.sim.t1

    # simulation time
    times = np.arange(t0, t1, dt)
    num_times = len(times.tolist())

    # position trajectories
    x_traj = np.sin(times) #np.append(np.zeros(int(np.floor(num_times/2))), np.ones(int(np.floor(num_times/2)+1)))
    y_traj = np.zeros(num_times)
    z_traj = np.zeros(num_times)

    # velocity trajectories
    vx_traj = np.zeros(num_times)
    vy_traj = np.zeros(num_times)
    vz_traj = np.zeros(num_times)

    # logging
    log_times = np.zeros(1)
    log_state = np.zeros(16)
    log_reference = np.zeros(6)
    log_desired = np.zeros(6)
    log_error = np.zeros(1)
    log_yaw_des = np.zeros(1)

    # simulation
    for k in range(num_times - sample_delay):
        desired.x = x_traj[k]
        desired.y = y_traj[k]
        desired.z = z_traj[k]
        desired.vx = vx_traj[k]
        desired.vy = vy_traj[k]
        desired.vz = vz_traj[k]

        # switch cases
        print("baseline %d/%d" % (k, num_times))
        reference.x = desired.x
        reference.y = desired.y
        reference.z = desired.z
        reference.vx = desired.vx
        reference.vy = desired.vy
        reference.vz = desired.vz

        # position control
        reference.yaw = (4 * np.pi - 0) * times[k] / params.sim.t1
        print("ref yaw %.2f" % reference.yaw)
        thrust, euler = position_controller_m(curr_state, reference, quad)

        # set desired euler angle
        reference.roll = euler[0]
        reference.pitch = euler[1]
        reference.yaw = euler[2]

        # compute corresponding quaternion
        reference.quat = euler2quat(euler[0], euler[1], euler[2])
        reference.R = quat2rotation(reference.quat)
        reference.full_state = update_full_state(reference)

        # attitude control
        variation = attitude_controller_m(curr_state, reference, quad)

        # pwm
        motor_pwm = compute_motor_pwm(thrust, variation, params)

        # motor rates
        smr = motor_state_update(motor_pwm, params)

        # quad dynamics
        rot_update_type = 'quat' # quat or euler
        disturb = False
        if k == 2000 and add_disturbance_flag:
            disturb = True
            print("---> perturb")
        curr_state = quad_state_update(curr_state, smr, params, rot_update_type, disturb)

        # tracking error (yd - y)
        e_y = desired.x - curr_state.x
        print('error', e_y)

        # print yaw
        print("ref yaw %.2f, act yaw %.2f" % (reference.yaw, curr_state.yaw))

        # log simulation data
        curr_log = times[k]
        log_times = np.vstack([log_times, curr_log])

        curr_log = np.append(curr_state.full_state, motor_pwm)
        log_state = np.vstack([log_state, curr_log])

        curr_log = np.array([reference.x, reference.y, reference.z, reference.vx, reference.vy, reference.vz])
        log_reference = np.vstack([log_reference, curr_log])

        curr_log = np.array([desired.x, desired.y, desired.z, desired.vx, desired.vy, desired.vz])
        log_desired = np.vstack([log_desired, curr_log])

        curr_log = np.array([e_y])
        log_error = np.vstack([log_error, curr_log])

        curr_log = np.array([reference.yaw])
        log_yaw_des = np.vstack([log_yaw_des, curr_log])

        # print out
        if k % 100 == 0 and verbose:
            print("finished %d seconds" % (int(k/500)))
            print(curr_state.full_state)

    # compute rmse
    rmse_x = np.sqrt(np.mean((log_error) ** 2))
    print('>> rmse_x: ', rmse_x)

    # sim run time
    sim_runtime = time.time() - time_start
    print('>> sim runtime: %.2f sec' % sim_runtime)

    # plot states
    if plot_state_flag:
        plot_states(log_times, log_state)

    # plot tracking results
    if plot_tracking_flag:
        plot_tracking(log_times, log_state, log_reference, log_desired)

    # save log
    if save_log:
        file_dir = save_dir + '.pkl'
        print(file_dir)

        with open(file_dir, 'wb') as f:
            pickle.dump([log_times, log_state, log_reference, log_desired, log_error, rmse_x, log_yaw_des], f)

if __name__ == "__main__":
    main()
