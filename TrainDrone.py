import jax.numpy as jnp

from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from systems.quadcopter.environment import QuadCopterEnv
from matplotlib import pyplot as plt

def experiment():

    env = QuadCopterEnv()
    cur_state = env.reset()

    if env.action_size != 4:
        print("Action size should be 4")
        exit()
    if env.observation_size != 12:
        print("Observation size should be 12")

    # Run the environment for a few steps
    states = []
    for _ in range(10):
        action = jnp.zeros(env.action_size)
        cur_state = env.step(cur_state, action)
        states.append(cur_state.obs)

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    states = jnp.array(states)
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    plt.show()
    plt.savefig("quadcopter.png")
    

    exit()

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