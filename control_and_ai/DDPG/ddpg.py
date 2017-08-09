# replication of the deep deterministic policy gradient paper by Lillicrap et al.
import tensorflow as tf
import numpy as np
import os
from .actor import Actor
from .critic import Critic

class ExperienceBuffer():

    def __init__(self, max_buffer_size):
        self.size = 0
        self.max_buffer_size = max_buffer_size
        self.experiences = []

    def add(self, experience):
        assert len(experience) == 5, 'Experience must be of form (s, a, r, s, t\')'
        assert type(experience[4]) == bool

        self.experiences.append(experience)
        self.size += 1
        if self.size >= self.max_buffer_size:
            self.experiences.pop(0)
            self.size -= 1

    def get_batch(self, batch_size):
        states, actions, rewards, new_states, is_terminals = [], [], [], [], []
        dist = np.random.randint(0, high=self.size, size=batch_size)
        
        for i in dist:
            states.append(self.experiences[i][0])
            actions.append(self.experiences[i][1])
            rewards.append(self.experiences[i][2])
            new_states.append(self.experiences[i][3])
            is_terminals.append(self.experiences[i][4])

        return states, actions, rewards, new_states, is_terminals

class DDPG():
    def __init__(self, 
            action_space_bounds,
            exploration_policies,
            env_space_size,
            batch_size=100,
            buffer_size=1000000,
            actor_learning_rate=0.0001,
            critic_learning_rate=0.001,
            gamma=0.99,
            retrain=False,
            log_dir=None,
            model_dir=None):

        self.sess = tf.Session() 

        self.experience_buffer = ExperienceBuffer(buffer_size)
        self.batch_size = batch_size
        self.actor = Actor(self.sess, action_space_bounds, exploration_policies, env_space_size, actor_learning_rate, optimizer=tf.train.AdamOptimizer)
        self.critic = Critic(self.sess, len(action_space_bounds), env_space_size, critic_learning_rate, gamma, optimizer=tf.train.AdamOptimizer)

        model_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(model_vars)

        # directories for saving models, etc
        if model_dir is None:
            model_dir = os.getcwd() + '/models'
        self.model_dir = model_dir
        self.model_loc = self.model_dir + '/DDPG.ckpt'
        self.log_dir = os.getcwd() + '/' + log_dir

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())

        self.sess.run(tf.global_variables_initializer())

        # if we are not retraining from scratch, just restore weights
        if retrain == False:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))

    def get_action(self, state, explore=True):
        return self.actor.get_action(state, explore)

    def update(self, old_state, action, reward, new_state, done):
        self.experience_buffer.add([old_state, action, reward, new_state, done])

        if self.experience_buffer.size >= self.batch_size:
            old_states, actions, rewards, new_states, is_terminals = self.experience_buffer.get_batch(self.batch_size)

            self.critic.update(old_states, actions, rewards, new_states, self.actor.get_target_action(new_states), is_terminals)
            action_derivs = self.critic.get_gradients(old_states, self.actor.get_action(old_states, explore=False))
            self.actor.update(old_states, action_derivs)
    
    def save_model(self, episode):
        self.saver.save(self.sess, self.model_loc, global_step=episode)

    def log_data(self, episode_reward, episode):
        def val_to_summary(tag, value):
            return tf.Summary(value=[
                tf.Summary.Value(tag=tag, simple_value=value), 
            ])

        self.writer.add_summary(val_to_summary('reward', episode_reward), episode)
        self.writer.add_summary(val_to_summary('loss', self.critic.loss_val), episode)
        self.writer.flush()

