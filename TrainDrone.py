import jax.numpy as jnp
import jax.random as jr
from jax.nn import swish
import wandb

from brax.training.types import Transition
from brax.training.replay_buffers import UniformSamplingQueue
from mbpo.optimizers import SACOptimizer
from mbpo.systems.rewards.base_rewards import Reward
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from systems.quadcopter.environment import QuadCopterEnv
from mbrl.model_based_agent.optimistic_model_based_agent import OptimisticModelBasedAgent

class DroneReward(Reward):
    def __init__(self):
        super().__init__(x_dim=12, u_dim=4)

    def __call__(self,
                 x: jnp.ndarray,
                 u: jnp.ndarray,
                 reward_params: dict,
                 x_next: jnp.ndarray | None = None
                 ):
        assert x.shape == (12,) and u.shape == (4,)
        # Reward is the negative distance to the origin
        reward = -jnp.linalg.norm(x[:3]) + 5
        return reward, reward_params

    def init_params(self, key: jnp.ndarray) -> dict:
        return {}

def experiment():
    #wandb.init(project='CMRL-Test_Drone')

    env = QuadCopterEnv()
    cur_state = env.reset()

    dyn_model = BNNStatisticalModel(
        input_dim = env.observation_size + env.action_size,
        output_dim = env.observation_size,
        num_training_steps = 8_000,
        output_stds = 1e-3 * jnp.ones(env.observation_size),
        features = (64, 64, 64),
        num_particles = 5,
        logging_wandb = True,
        return_best_model = True,
        eval_batch_size = 64,
        train_share = 0.8,
        eval_frequency = 1_000,
    )

    sac_kwargs = {
        'num_timesteps': 20_000,
        'episode_length': 64,
        'num_env_steps_between_updates': 10,
        'num_envs': 16,
        'num_eval_envs': 4,
        'lr_alpha': 3e-4,
        'lr_policy': 3e-4,
        'lr_q': 3e-4,
        'wd_alpha': 0.,
        'wd_policy': 0.,
        'wd_q': 0.,
        'max_grad_norm': 1e5,
        'discounting': 0.99,
        'batch_size': 32,
        'num_evals': 20,
        'normalize_observations': True,
        'reward_scaling': 1.,
        'tau': 0.005,
        'min_replay_size': 10 ** 4,
        'max_replay_size': 10 ** 5,
        'grad_updates_per_step': 10 * 16,  # should be num_envs * num_env_steps_between_updates
        'deterministic_eval': True,
        'init_log_alpha': 0.,
        'policy_hidden_layer_sizes': (64, 64),
        'policy_activation': swish,
        'critic_hidden_layer_sizes': (64, 64),
        'critic_activation': swish,
        'wandb_logging': True,
        'return_best_model': True,
    }

    max_replay_size_true_data_buffer = 10 ** 4
    dummy_sample = Transition(observation=jnp.ones(env.observation_size),
                              action=jnp.zeros(shape=(env.action_size,)),
                              reward=jnp.array(0.0),
                              discount=jnp.array(0.99),
                              next_observation=jnp.ones(env.observation_size))

    sac_buffer = UniformSamplingQueue(
        max_replay_size=max_replay_size_true_data_buffer,
        dummy_data_sample=dummy_sample,
        sample_batch_size=1)

    optimizer = SACOptimizer(system=None,
                             true_buffer=sac_buffer,
                             **sac_kwargs)

    horizon = 64
    agent = OptimisticModelBasedAgent(
        env=env,
        eval_env=env,
        statistical_model=dyn_model,
        optimizer=optimizer,
        reward_model=DroneReward(),
        episode_length=horizon,
        offline_data=None,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=True,
    )

    agent_state = agent.run_episodes(num_episodes=20,
                                     start_from_scratch=True,
                                     key=jr.PRNGKey(0))

if __name__ == '__main__':
    experiment()