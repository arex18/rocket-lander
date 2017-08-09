import numpy as np

from control_and_ai.DDPG.train import Utils
from rocketlander_v2 import get_state_sample


def test(env, agent, simulation_settings):

    obs_size = env.observation_space.shape[0]
    util = Utils()
    state_samples = get_state_sample(samples=5000, normal_state=True)
    util.create_normalizer(state_sample=state_samples)

    for episode in range(1, simulation_settings['Episodes']):

        done = False
        total_reward = 0

        state = env.reset()
        state = util.normalize(state)

        for i in range(1000): #
            if simulation_settings['Render']:
                #env.refresh(render=True)
                env.render()

            # infer an action
            action = agent.get_action(np.reshape(state, (1, obs_size)), explore=False)

            # take it
            state, reward, done, _ = env.step(action[0])
            state = util.normalize(state)
            total_reward += reward

            if done:
                break

        agent.log_data(total_reward, episode)
        print("Reward:\t{0}".format(total_reward))