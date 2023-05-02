##################################################
# script: ddpg_main.py
# purpose: top level script for training and testing DDPG algorithm
#
# dependencies:
#   - simulation environment:   environments/rocketlander.py
#   - test framework:           evaluation_main.py and evaluation_parameters.py
#   - DDPG agent:               DDPG_update/DDPG.py
#
# input args:
#   - phase (optional): 'test' or 'continue'    | sets agent to test mode or continue training provided model.  Leave blank to train model from scratch
#
# Example command to run the evaluation:
#   python ddpg_main.py 'test'
#
##################################################

from DDPG import Agent
import gym
import numpy as np
import os
import sys
from environments.rocketlander import RocketLander
from evaluation_scripts.evaluation_parameters import *
import evaluation_main as eval


phase = None
if len(sys.argv) > 1:
    phase = sys.argv[1]

# add model name to input args
if phase.lower() == 'test':
    test_phase = True
    continue_training = False
    render = True
    model_name = '_ddpg_1400_final.chkpt' # This model works great, but I think I had the target critic network updating incorrectly (tau instead of 1-tau)
elif phase.lower() == 'continue':
    test_phase = False
    continue_training = True
    render = False
    model_name = '_ddpg.chkpt' # at the moment, this is a copy of the 1400_final model
else:
    test_phase = False
    continue_training = False
    render = False
    model_name = '_ddpg.chkpt'

if test_phase:
    simulation_settings = {'Side Engines': True,
                            'Clouds': True,
                            'Vectorized Nozzle': True,
                            'Starting Y-Pos Constant': 1,
                            'Initial Force': (0, 0),
                            'Render': render,
                            'Evaluation Averaging Loops': 3,
                            'Gather Stats': True,
                            'Episodes': 50}
else: # training settings
    simulation_settings = {'Side Engines': True,
                            'Clouds': False,
                            'Vectorized Nozzle': True,
                            'Starting Y-Pos Constant': 1,
                            'Initial x-Coord': (0.5, 0.2), # Randomize X
                            'Initial Force': 'Random',#(0, 0),
                            'Initial Theta': 'Random',
                            'Render': render,
                            'Evaluation Averaging Loops': 3,
                            'Gather Stats': True,
                            'Episodes': 2000}


rocket_lander_sim = True

if rocket_lander_sim:
    env = RocketLander(simulation_settings)
    chkpt_dir = os.path.realpath('./lander/ddpg')
    n_actions = 3 # rocketlander
else:
    env = gym.make('LunarLanderContinuous-v2')
    chkpt_dir = 'tmp/ddpg'
    n_actions = 2 # lunar lander

if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)


agent = Agent(alpha=0.0001, beta=0.001, input_dims=[8], tau=0.005, env=env,
              batch_size=32, layer1_size=400, layer2_size=300, n_actions=n_actions, chkpt_dir=chkpt_dir, model_name=model_name, test_phase=test_phase)

if continue_training or test_phase:
    agent.load_models()

np.random.seed(0)
score_history = []

def evaluate_ddpg():
    file_path = chkpt_dir

    testing_framework = eval.Evaluation_Framework(simulation_settings)
    reward_results, final_state_history, action_history = testing_framework.execute_evaluation(env, agent,
                                                                                               INITIAL_STATES,
                                                                                               INITIAL_FORCES,
                                                                                               DISTURBANCES, IMPULSES)

def main():
    with open('ddpg_log.txt', 'w') as log:
        for i in range(800):
            log.write(f'Episode {i+1}:\n')
            done = False
            score = 0
            obs = env.reset()
            if not rocket_lander_sim:
                act_limits = np.array([env.action_space.low, env.action_space.high])
            step = 0
            while not done:
                step += 1
                act = agent.choose_action(obs)
                # if not rocket_lander_sim:
                #     act = np.clip(act,act_limits[0], act_limits[1])
                new_state, reward, done, info = env.step(act)
                log.write(f'step {step}: actions: {act}\n')
                # if env.CONTACT_FLAG:
                #     log.write(f'Y-Vel: {obs[3]}, Crash: {env.crash}\n')
                #     if env.crash: print('Crashed!')
                if done:
                    log.write(f'Final State: {obs}\nNext State: {new_state}\n\n')
                agent.remember(obs, act, reward, new_state, int(done))
                agent.learn()
                score += reward
                obs = new_state

                if render:
                    # Optional render
                    env.render()
                    # Draw the target
                    env.draw_marker(env.landing_coordinates[0], env.landing_coordinates[1])
                    # Refresh render
                    env.refresh(render=False)


            score_history.append(score)

            print(f'Episode: {i+1}, score: {score}, 100 game average: {np.mean(score_history[-100:])}')

            if (i+1) % 100 == 0 and not test_phase:
                agent.save_models()
        stats = env.get_terminal_stats()
        log.write('Terminal Stats:\n')
        for k,v in stats.items():
            log.write(f'\t{k}: {v}\n')


if __name__ == '__main__':
    try:
        # main()
        if test_phase:
            evaluate_ddpg()
        else:
            main()
    finally:
        np.save('reward_history.npy', score_history)