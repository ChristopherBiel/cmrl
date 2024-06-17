import jax
import chex
import jax.numpy as jnp
import jax.random as jr
from brax.envs.base import State, Env

import json
from munch import Munch

class QuadCopterEnv(Env):
    def __init__(self,
                 init_state: chex.Array | None = None,
                 dynamics_type: str = 'quat',
                 disturb: bool = False,
                 seed: int = 0,) -> None:
        assert dynamics_type in ['quat', 'euler'], "Dynamics type can only be quat or euler"
        
        # Load parameters
        load_params = json.load(open("systems/quadcopter/parameters.json"))
        self.params = Munch.fromDict(load_params)

        # Initialize quadrotor state
        self.key = jr.PRNGKey(seed=seed)
        self._dynamics_type = dynamics_type
        self._disturb = disturb

    def reset(self,
              rng: jr.PRNGKey,
              init_state: chex.Array | None = None,
              ) -> State:
        pipeline_state = None

        sys_state = self._init_zero_state(init_state)
        obs = sys_state.full_state
        # Possibility to add domain randomization here

        reward, done ,zero = jnp.zeros(3)
        # Metrics
        metrics = {
            'time': zero,
            'R': jnp.eye(3),
            'quat': jnp.array([1.0, 0.0, 0.0, 0.0]),

        }
        return State(pipeline_state, obs, reward, done, metrics)
    
    def reward(self,
               x: chex.Array,
               u: chex.Array) -> chex.Array:
        # Calculate the reward based on the state and the action
        # Can also use the time from metrics so week can have an adaptive trajectory
        return 0.0
    
    def step(self,
             state: State,
             action: chex.Array) -> State:
        pipeline_state = None

        # Calculate the motor pwm from action
        motor_pwn = self._compute_motor_pwm(action[0], action[1:], self.quad.params)
        sq_mot_rates = self._motor_state_update(motor_pwn, self.quad.params)
        
        # quadrotor dynamics
        state = self._quad_state_update(state, sq_mot_rates, self.params, self._dynamics_type, self._disturb)

        # Calculate the reward
        reward = self.reward(state, action)

        return State(pipeline_state, state.obs, reward, state.done, state.metrics)

    @property
    def dt(self):
        return self.quad.params.sim.dt
    
    @property
    def observation_size(self) -> int:
        # Observations: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        return self.quad.state.full_state.shape[0]
    
    @property
    def action_size(self) -> int:
        # Actions: [thrust, p, q, r]
        # Where p, q, r are the roll, pitch, and yaw rates respectively
        return 4
    
    def _init_zero_state(self) -> State:
        
        observation = jnp.zeros(12)
        state = State(pipeline_state=None,
                      obs=observation,
                      reward=0.0,
                      done=0.0,
                      metrics={
                          'time': 0.0,
                          'R': jnp.eye(3),
                          'quat': jnp.array([1.0, 0.0, 0.0, 0.0]),
                          'acc': jnp.zeros(3),
                      })

        return state
    
    def _init_state(self,
                    init_state: chex.Array | None = None) -> Munch:
        if init_state is None:
            state = self._init_zero_state()
            return state
        
        R = self._quat2rotation(self._euler2quat(init_state[6], init_state[7], init_state[8]))
        state = State(pipeline_state=None,
                      obs=init_state,
                      reward=0.0,
                      done=0.0,
                      metrics={
                          'time': 0.0,
                          'R': jnp.eye(3),
                          'quat': jnp.array([1.0, 0.0, 0.0, 0.0]),
                          'acc': jnp.zeros(3),
                      })

    
    def _compute_motor_pwm(self,
                           base_thrust: chex.Array,
                           motor_variation: chex.Array) -> chex.Array:
        temp = jnp.array([[-0.5, -0.5, -1.0],
                             [-0.5, +0.5, +1.0],
                             [+0.5, +0.5, -1.0],
                             [+0.5, -0.5, +1.0]])
        adjustment = temp.dot(motor_variation)
        motor_pwm = base_thrust + adjustment
        motor_pwm = jnp.maximum(motor_pwm, self.params.quad.thrust_min)
        motor_pwm = jnp.minimum(motor_pwm, self.params.quad.thrust_max)

        return motor_pwm
    
    def motor_state_update(self,
                           pwm: chex. Array) -> chex.Array:
        """ computes motor squared rpm from pwm """

        # assume pwm is a jax numpy array
        motor_rates = self.params.quad.pwm2rpm_scale * pwm + self.params.quad.pwm2rpm_const
        squared_motor_rates = motor_rates ** 2

        return squared_motor_rates
    
    def _quad_state_update(self,
                           state: State,
                           sq_mot_rates: chex. Array,
                           params: Munch,
                           dynamic_type: str,
                           disturb: bool) -> Munch:
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
        pqr = state.full_state[9:]
        temp0 = moment - jnp.cross(pqr, I.dot(pqr))
        angular_rate_derivative = I_inv.dot(temp0)

        # compute rpy angles
        omega = pqr + params.sim.dt * angular_rate_derivative

        if dynamic_type == "quat":
            # update based on quaternion
            quat_next = self._quat_update(state.metrics["quat"], pqr, params.sim.dt)
            R_next = self._quat2rotation(quat_next)
            rpy_next = self._rotation2euler(R_next)

        elif dynamic_type == "euler":
            # update based on euler angle
            rpy = state.full_state[6:9]
            rpy_next = self._euler_update(rpy, pqr, params.sim.dt)
            quat_next = self._euler2quat(rpy_next[0], rpy_next[1], rpy_next[2])
            R_next = self._quat2rotation(quat_next)

        # update state
        x = state.x + params.sim.dt * state.vx
        y = state.y + params.sim.dt * state.vy
        z = state.z + params.sim.dt * state.vz
        vx = state.vx + params.sim.dt * state.ax
        state.vy = state.vy + params.sim.dt * state.ay
        state.vz = state.vz + params.sim.dt * state.az
        state.roll = rpy_next[0]
        state.pitch = rpy_next[1]
        state.yaw = rpy_next[2]
        state.p = omega[0]
        state.q = omega[1]
        state.r = omega[2]
        state.R = R_next
        state.quat = quat_next
        state.full_state = jnp.array([state.x, state.y, state.z,
                                      state.vx, state.vy, state.vz,
                                      state.roll, state.pitch, state.yaw,
                                      state.p, state.q, state.r])
        


    def _quat_update(self):
        pass

    def _quat2rotation(self):
        pass

    def _rotation2euler(self):
        pass

    def _euler_update(self):
        pass

    def _euler2quat(self):
        pass
