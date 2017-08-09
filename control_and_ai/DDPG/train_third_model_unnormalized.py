import sys
import os
import shutil
import gym
import argparse
import sklearn.preprocessing
sys.path.append('C://Users//REUBS_LEN//PycharmProjects//RocketLanding')
from rocketlander_v2 import get_state_sample
from constants import *
from .utils import Utils


def train(env, agent, FLAGS):
    print("Fuel Cost = 0, Max Steps = Unlimited, Episode Training = 1000, RANDOM FORCE = 20000, RANDOM X_FORCE = 0.2*RANDOM FORCE")
    print("4th Model with get_state_with_barge_and_landing_coordinates - UnNormalized")
    print("Using Normal State nor Untransformed. Otherwise same settings as previous unnormalized test.")
    #print("Fuel Cost = 0, Max Steps = Unlimited, Episode Training = 2000")
    obs_size = env.observation_space.shape[0]

    util = Utils()
    state_samples = get_state_sample(samples=5000, normal_state=False, untransformed_state=False)
    util.create_normalizer(state_sample=state_samples)

    for episode in range(1, FLAGS.num_episodes + 1):
        old_state = None
        done = False
        total_reward = 0

        s = env.reset()
        state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False)
        max_steps = 1000

        left_or_right_barge_movement = np.random.randint(0, 2)
        epsilon = 0.05

        for t in range(max_steps): # env.spec.max_episode_steps
            if FLAGS.show or episode % 10 == 0:
                env.refresh(render=True)

            old_state = state

            # infer an action
            action = agent.get_action(np.reshape(state, (1, obs_size)), not FLAGS.test)

            # take it
            s, reward, done, _ = env.step(action[0])
            state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False)
            total_reward += reward

            if s[LEFT_GROUND_CONTACT] == 0 and s[RIGHT_GROUND_CONTACT] == 0:
                env.move_barge_randomly(epsilon, left_or_right_barge_movement)
                env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
                env.apply_random_y_disturbance(epsilon=0.005)

            if not FLAGS.test:
                # update q vals
                agent.update(old_state, action[0], np.array(reward), state, done)

            if done:
                break

        agent.log_data(total_reward, episode)

        if episode % 50 == 0 and not FLAGS.test:
            print('Saved model at episode', episode)
            agent.save_model(episode)
        print("Reward:\t{0}".format(total_reward))

def set_up():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_episodes',
        type=int,
        default=1000,
        help='How many episodes to train for'
    )

    parser.add_argument(
        '--show',
        default=False,
        action='store_true',
        help='At what point to render the cart environment'
    )

    parser.add_argument(
        '--wipe_logs',
        default=False,
        action='store_true',
        help='Wipe logs or not'
    )

    parser.add_argument(
        '--log_dir',
        default='logs',
        help='Where to store logs'
    )

    parser.add_argument(
        '--retrain',
        default=False,
        action='store_true',
        help='Whether to start training from scratch again or not'
    )

    parser.add_argument(
        '--test',
        default=False,
        action='store_true',
        help='Test more or no (true = no training updates)'
    )

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.wipe_logs and os.path.exists(os.getcwd() + '/' + FLAGS.log_dir):
        shutil.rmtree(os.getcwd() + '/' + FLAGS.log_dir)

    return FLAGS
