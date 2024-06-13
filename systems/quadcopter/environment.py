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
        sys_state = self._quad_state_update(state, sq_mot_rates, self.params, self._dynamics_type, self._disturb)
        # quadrotor dynamics
        obs = sys_state.full_state
        # Calculate the reward
        reward = self.reward(state.obs, action)

        # Done
        if state.full_state[2] < 0.0:
            done = jnp.array(1.0)
        return State(pipeline_state, obs, reward, done, metrics)

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
    
    def _init_zero_state(self,
                         init_state: chex.Array | None = None) -> Munch:
        state = {
            "x" : 0.0,
            "y" : 0.0,
            "z" : 0.0,
            "vx" : 0.0,
            "vy" : 0.0,
            "vz" : 0.0,
            "ax" : 0.0,
            "ay" : 0.0,
            "az" : 0.0,
            "roll"  : 0.0,
            "pitch" : 0.0,
            "yaw"   : 0.0,
            "p" : 0.0,
            "q" : 0.0,
            "r" : 0.0,
            "R" : jnp.eye(3),
            "quat" : jnp.array([1.0, 0.0, 0.0, 0.0])
        }

        # convert dict to . dict format
        state = Munch.fromDict(state)
        # construct full state array
        state.full_state = jnp.array([state.x, state.y, state.z,
                           state.vx, state.vy, state.vz,
                           state.roll, state.pitch, state.yaw,
                           state.p, state.q, state.r])
        return state
    
    def _compute_motor_pwm(self,
                           base_thrust: chex.Array,
                           motor_variation: chex.Array) -> chex.Array:
        temp = jnp.array([[-0.5, -0.5, -1.0],
                             [-0.5, +0.5, +1.0],
                             [+0.5, +0.5, -1.0],
                             [+0.5, -0.5, +1.0]])
        """
        temp = np.array([[+0.5, -0.5, -1.0],
                             [-0.5, -0.5, +1.0],
                             [-0.5, +0.5, -1.0],
                             [+0.5, +0.5, +1.0]])
        """
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
                           disturb: bool) -> Munch