from DDPG.ddpg import DDPG
from DDPG.train import set_up
from DDPG.train_third_model_normalized import train as train_third_model_normalized

from constants import DEGTORAD
from control_and_ai.DDPG.exploration import OUPolicy
from main_simulation import RocketLander

FLAGS = set_up()

action_bounds = [1, 1, 15*DEGTORAD]

eps = []
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))

simulation_settings = {'Side Engines': True,
                       'Clouds': True,
                       'Vectorized Nozzle': True,
                       'Graph': False,
                       'Render': False,
                       'Starting Y-Pos Constant': 1,
                       'Initial Force': 'random',
                       'Rows': 1,
                       'Columns': 2,
                       'Episodes': 300}
env = RocketLander(simulation_settings)
#env = wrappers.Monitor(env, '/tmp/contlunarlander', force=True, write_upon_reset=True)

FLAGS.retrain = True # Restore weights if False
FLAGS.test = False
FLAGS.num_episodes = 300
model_dir = 'C://Users//REUBS_LEN//PycharmProjects//RocketLanding//DDPG//models_unlimited_episodes_full_normalized_normal_state'

agent = DDPG(
    action_bounds,
    eps,
    env.observation_space.shape[0],
    actor_learning_rate=0.0001,
    critic_learning_rate=0.001,
    retrain=FLAGS.retrain,
    log_dir=FLAGS.log_dir,
    model_dir=model_dir)

#test(env, agent, simulation_settings)
train_third_model_normalized(env, agent, FLAGS)
#train_second_model(env, agent, FLAGS)

