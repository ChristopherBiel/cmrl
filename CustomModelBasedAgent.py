import jax
import jax.numpy as jnp
import jax.random as jr

from brax.training.replay_buffers import ReplayBufferState
from mbrl.model_based_agent.base_model_based_agent import BaseModelBasedAgent
from mbrl.model_based_agent.optimizer_wrapper import Actor, OptimisticActor
from mbrl.model_based_agent.system_wrapper import OptimisticSystem, OptimisticDynamics
from mbpo.optimizers.base_optimizer import BaseOptimizer


from differentiators.nn_smoother.smoother_net import SmootherNet
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
from bsm.utils.normalization import Data

class ContModelBasedAgent(BaseModelBasedAgent):
    def __init__(self,
                 smoother_net = SmootherNet,
                 *args, **kwargs):
        self.smoother = smoother_net
        super().__init__(*args, **kwargs)

    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        dynamics, system, actor = OptimisticDynamics, OptimisticSystem, OptimisticActor
        dynamics = dynamics(statistical_model=self.statistical_model,
                            x_dim=self.env.observation_size,
                            u_dim=self.env.action_size)
        system = system(dynamics=dynamics,
                        reward=self.reward_model, )
        actor = actor(env_observation_size=self.env.observation_size,
                        env_action_size=self.env.action_size,
                        optimizer=optimizer)
        actor.set_system(system=system)
        return actor

    def _collected_buffer_to_train_data(self, collected_buffer_state: ReplayBufferState):
        """We have to use the differential smoothing library to convert the collected buffer to training data."""
        idx = jnp.arange(start=collected_buffer_state.sample_position, stop=collected_buffer_state.insert_position)
        all_data = jnp.take(collected_buffer_state.data, idx, axis=0, mode='wrap')
        all_transitions = self.collected_data_buffer._unflatten_fn(all_data)
        obs = all_transitions.observation
        act = all_transitions.action

        smoother_key = jr.split(self.key)
        predictions, derivatives = self._smoothe_and_differentiate(obs, smoother_key)

        inputs = jnp.concatenate([predictions, act], axis=1)
        outputs = derivatives
        return Data(inputs=inputs, outputs=outputs)

    def _smoothe_and_differentiate(self, obs, smoother_key):
        """Calculate the derivative of the observations."""
        # Create smoother data
        t = jnp.arange(0, obs.shape[0])
        smoother_data = Data(inputs=t, outputs=obs)

        model_states = self.smoother.train_new_smoother(smoother_key, smoother_data)
        predictions = self.smoother.predict_batch(t, model_states)
        derivatives = self.smoother.derivative_batch(t, model_states)

        return predictions, derivatives

        

