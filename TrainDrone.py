import jax.numpy as jnp

from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from systems.quadcopter.environment import QuadCopterEnv


def experiment():

    env = QuadCopterEnv()

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

if __name__ == '__main__':
    experiment()