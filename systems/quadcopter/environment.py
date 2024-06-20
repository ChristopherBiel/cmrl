import jax
import chex
import jax.numpy as jnp
import jax.random as jr
from brax.envs.base import State, Env

import json
import os

from munch import Munch

from systems.quadcopter.quad_dynamics import \
    _compute_motor_pwm, _motor_state_update, _quad_state_update, \
    _euler2quat, _quat2rotation

class QuadCopterEnv(Env):
    def __init__(self,
                 disturb: bool = False,
                 seed: int = 0,
                 params_file: str = "systems/quadcopter/parameters.json") -> None:
        """Initialize the quadcopter environment. Load parameters from a json file."""
        # Check if the parameters file exists
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Parameters file {params_file} not found")
        
        # Load parameters
        load_params = json.load(open(params_file))
        self.params = Munch.fromDict(load_params)

        # Initialize quadrotor state
        self.key = jr.PRNGKey(seed=seed)
        self._disturb = disturb

    def reset(self,
              rng: jr.PRNGKey = None,
              init_state: chex.Array = None,
              ) -> State:
        """Reset the quadrotor state. If init_state is None, initialize to zero.
        Args:
            rng: random number generator (jt.PRNGKey)
            init_state: initial state of the quadrotor [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r] """
        
        pipeline_state = None

        sys_state = self._init_state(init_state)
        # Possibility to add domain randomization here

        reward, done ,zero = jnp.zeros(3)
        return State(pipeline_state, sys_state.obs, reward, done, sys_state.metrics)
    
    def reward(self,
               x: chex.Array,
               u: chex.Array) -> chex.Array:
        # Calculate the reward based on the state and the action
        # Can also use the time from metrics so week can have an adaptive trajectory
        return 0.0
    
    def step(self,
             input_state: State,
             action: chex.Array) -> State:
        """Step the quadrotor dynamics forward in time."""

        # Calculate the motor pwm from action
        motor_pwn = _compute_motor_pwm(self.params, action[0], action[1:])
        sq_mot_rates = _motor_state_update(self.params, motor_pwn)
        
        # quadrotor dynamics
        next_state = _quad_state_update(self.params, input_state, sq_mot_rates, self._disturb)
        # Combine the state metrics
        input_state.metrics['time'] = next_state.metrics['time']
        input_state.metrics['R'] = next_state.metrics['R']
        input_state.metrics['quat'] = next_state.metrics['quat']
        input_state.metrics['acc'] = next_state.metrics['acc']

        # Calculate the reward
        reward = self.reward(next_state, action)

        return State(pipeline_state=input_state.pipeline_state,
                     obs=next_state.obs,
                     reward=reward,
                     done=next_state.done,
                     metrics=input_state.metrics,
                     info=input_state.info)

    @property
    def dt(self):
        return self.params.sim.dt

    @property
    def observation_size(self) -> int:
        # Observations: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        return 12

    @property
    def action_size(self) -> int:
        # Actions: [thrust, p, q, r]
        # Where p, q, r are the roll, pitch, and yaw rates respectively
        return 4

    def _init_zero_state(self) -> State:
        """Initialize the quadrotor state to zero."""
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
                    init_state: chex.Array = None) -> State:
        """Initialize the quadrotor state. If init_state is None, initialize to zero."""
        if init_state is None:
            state = self._init_zero_state()
            return state
        
        quat = self._euler2quat(init_state[6], init_state[7], init_state[8])
        R = self._quat2rotation(quat)
        state = State(pipeline_state=None,
                      obs=init_state,
                      reward=0.0,
                      done=0.0,
                      metrics={
                          'time': 0.0,
                          'R': R,
                          'quat': quat,
                          'acc': jnp.zeros(3),
                      })
        
        return state

    def backend(self) -> str:
        return "jax"
    